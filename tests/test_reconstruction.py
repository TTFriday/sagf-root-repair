import math

import numpy as np
import pytest
from scipy import ndimage
from skimage.graph import MCP_Geometric

from toposeg.constants import HYPOCOTYL, IGNORE_INDEX, LATERAL_ROOT, LEAF, MAIN_ROOT, SEED
from toposeg.metrics import compute_metrics, compute_metrics_fast, target_features
from toposeg.reconstruction import (
    ReconstructionParams,
    _minimax_attachment,
    anchor_filter,
    bottleneck_gaps,
    bridge_gaps,
    class_probabilities,
    close_gaps,
    decode,
    fragment_graph,
    geodesic_forest,
    reconstruct,
    reconstruct_many,
    size_filter,
)


def _same(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all((math.isnan(a[k]) and math.isnan(b[k])) or a[k] == b[k] for k in a)


def _random_label(rng: np.random.Generator, shape=(96, 128)) -> np.ndarray:
    label = np.zeros(shape, dtype=np.uint8)
    for cls in (MAIN_ROOT, LATERAL_ROOT, SEED, HYPOCOTYL, 5, 6):
        for _ in range(rng.integers(0, 4)):
            y, x = rng.integers(0, shape[0] - 20), rng.integers(0, shape[1] - 20)
            label[y:y + rng.integers(2, 20), x:x + rng.integers(1, 6)] = cls
    label[rng.random(shape) < 0.002] = IGNORE_INDEX
    return label


@pytest.mark.parametrize("seed", range(6))
def test_fast_metrics_match_reference(seed):
    rng = np.random.default_rng(seed)
    target, prediction = _random_label(rng), _random_label(rng)
    prediction[prediction == IGNORE_INDEX] = 0
    assert _same(compute_metrics(prediction, target), compute_metrics_fast(prediction, target_features(target)))


def test_fast_metrics_empty_cases():
    empty = np.zeros((40, 40), dtype=np.uint8)
    root = empty.copy()
    root[5:30, 10:13] = MAIN_ROOT
    for prediction, target in ((empty, empty), (root, empty), (empty, root), (root, root)):
        assert _same(compute_metrics(prediction, target), compute_metrics_fast(prediction, target_features(target)))


def _plate(gap_prob: int) -> tuple[np.ndarray, np.ndarray]:
    """Hypocotyl, a root broken by a 6-px gap, and a remote detached fragment, as stored uint8 probabilities."""
    h, w = 120, 160
    q = np.zeros((6, h, w), dtype=np.uint8)
    q[HYPOCOTYL - 1, 5:20, 38:43] = 250
    q[MAIN_ROOT - 1, 20:50, 39:42] = 250
    q[MAIN_ROOT - 1, 50:56, 39:42] = gap_prob
    q[MAIN_ROOT - 1, 56:90, 39:42] = 250
    q[MAIN_ROOT - 1, 30:60, 130:133] = 250
    return q, np.where(q.max(0) > 127, q.argmax(0) + 1, 0).astype(np.uint8)


def test_geodesic_forest_bridges_likely_gap_and_drops_remote_fragment():
    q, _ = _plate(gap_prob=100)
    params = ReconstructionParams(forest_budget=8.0, forest_step=0.1)
    out = reconstruct(q, params)
    root = np.isin(out, (MAIN_ROOT, LATERAL_ROOT))
    labels, count = ndimage.label(root, np.ones((3, 3)))
    assert count == 1
    assert root[20:90, 39:42].sum() >= root[20:90, 39:42].size - 3 * 6
    assert not root[:, 120:].any()


def test_geodesic_forest_budget_rejects_unlikely_gap():
    q, _ = _plate(gap_prob=1)
    out = reconstruct(q, ReconstructionParams(forest_budget=4.0, forest_step=0.1))
    root = np.isin(out, (MAIN_ROOT, LATERAL_ROOT))
    assert root[20:50, 39:42].all()
    assert not root[56:90].any()


def test_geodesic_forest_keep_area_retains_large_unanchored_fragment():
    q, _ = _plate(gap_prob=1)
    out = reconstruct(q, ReconstructionParams(forest_budget=4.0, forest_step=0.1, forest_keep_area=80))
    assert np.isin(out[30:60, 130:133], (MAIN_ROOT, LATERAL_ROOT)).all()


def test_geodesic_forest_without_shoot_removes_all_root():
    label = np.zeros((50, 50), dtype=np.uint8)
    label[10:40, 20:23] = MAIN_ROOT
    prob = np.where(label > 0, 250, 0).astype(np.int16)
    out = geodesic_forest(label, prob, np.full_like(label, MAIN_ROOT), budget=5.0, step_cost=0.1)
    assert not out.any()


def test_geodesic_forest_blocks_paths_through_other_organs():
    label = np.zeros((30, 60), dtype=np.uint8)
    label[13:17, 2:5] = SEED
    label[14:16, 5:10] = MAIN_ROOT
    label[14:16, 30:35] = MAIN_ROOT
    label[:, 20:22] = LEAF
    prob = np.where(label == MAIN_ROOT, 250, 0).astype(np.int16)
    out = geodesic_forest(label, prob, np.full_like(label, MAIN_ROOT), budget=10.0, step_cost=0.1, radius=50, lam=0.0)
    assert (out[14:16, 5:10] == MAIN_ROOT).all()
    assert not np.isin(out[:, 30:35], (MAIN_ROOT, LATERAL_ROOT)).any()
    assert (out[:, 20:22] == LEAF).all()


def test_endpoint_bridge_blocks_paths_through_other_organs():
    label = np.zeros((30, 60), dtype=np.uint8)
    label[14:16, 5:10] = MAIN_ROOT
    label[14:16, 30:35] = MAIN_ROOT
    label[:, 20:22] = LEAF
    out = bridge_gaps(label, np.full_like(label, 250, dtype=np.int16), np.full_like(label, MAIN_ROOT),
                      radius=50, min_prob=0.0, max_angle=90.0)
    _, count = ndimage.label(np.isin(out, (MAIN_ROOT, LATERAL_ROOT)), np.ones((3, 3)))
    assert count == 2
    assert (out[:, 20:22] == LEAF).all()


def test_geodesic_forest_handles_boundaryless_full_image_node():
    label = np.full((20, 20), MAIN_ROOT, dtype=np.uint8)
    out = geodesic_forest(label, np.full_like(label, 250, dtype=np.int16), np.full_like(label, MAIN_ROOT),
                          budget=1.0, step_cost=0.1, anchor="largest")
    assert np.array_equal(out, label)


def test_anchor_filter_chain_semantics():
    label = np.zeros((40, 80), dtype=np.uint8)
    label[18:22, 0:4] = SEED
    label[19:21, 4:20] = MAIN_ROOT
    label[19:21, 30:40] = MAIN_ROOT
    label[19:21, 70:78] = MAIN_ROOT
    near = anchor_filter(label, anchor_distance=1, chain_distance=12)
    assert (near[19:21, 30:40] == MAIN_ROOT).all() and not near[:, 70:].any()
    strict = anchor_filter(label, anchor_distance=1, chain_distance=0)
    assert (strict[19:21, 4:20] == MAIN_ROOT).all() and not strict[:, 30:].any()


def test_union_decoding_prefers_root_union():
    q = np.zeros((6, 1, 1), dtype=np.uint8)
    q[MAIN_ROOT - 1], q[LATERAL_ROOT - 1], q[4] = 60, 60, 100
    probabilities = np.concatenate([np.full((1, 1, 1), 35, dtype=np.int16), q.astype(np.int16)])
    assert decode(probabilities, "argmax")[0, 0] == 5
    assert decode(probabilities, "union")[0, 0] == MAIN_ROOT


def test_size_filter_keep_largest():
    label = np.zeros((40, 40), dtype=np.uint8)
    label[2:30, 2:5] = MAIN_ROOT
    label[2:12, 20:22] = LATERAL_ROOT
    label[30:33, 30:33] = MAIN_ROOT
    out = size_filter(label, np.full(label.shape, 250, dtype=np.int16), 0, 0.0, keep_largest=1)
    assert (out[2:30, 2:5] == MAIN_ROOT).all() and not out[:, 20:].any()


def _chain(gap_prob: int) -> np.ndarray:
    """Hypocotyl above a root broken into four pieces by three equal 4-px gaps."""
    q = np.zeros((6, 140, 40), dtype=np.uint8)
    q[HYPOCOTYL - 1, 2:10, 18:23] = 250
    for top in (10, 44, 78, 112):
        q[MAIN_ROOT - 1, top:top + 30, 19:22] = 250
    for top in (40, 74, 108):
        q[MAIN_ROOT - 1, top:top + 4, 19:22] = gap_prob
    return q


def test_minimax_attachment_keeps_chain_that_additive_cuts():
    q = _chain(gap_prob=60)
    minimax = reconstruct(q, ReconstructionParams(forest_budget=8.0, forest_width=0))
    additive = reconstruct(q, ReconstructionParams(forest_budget=8.0, forest_width=0, forest_mode="additive"))
    assert np.isin(minimax[112:142, 19:22], (MAIN_ROOT, LATERAL_ROOT)).all()
    assert (additive[10:40, 19:22] == MAIN_ROOT).all() and not additive[112:].any()


def test_largest_anchor_keeps_largest_fragment_without_shoot():
    label = np.zeros((60, 60), dtype=np.uint8)
    label[5:50, 10:13] = MAIN_ROOT
    label[5:15, 40:42] = MAIN_ROOT
    prob = np.where(label > 0, 250, 0).astype(np.int16)
    out = geodesic_forest(label, prob, np.full_like(label, MAIN_ROOT), budget=2.0, step_cost=0.1, anchor="largest")
    assert (out[5:50, 10:13] == MAIN_ROOT).all() and not out[:, 30:].any()



def test_voronoi_graph_matches_complete_geodesic_bottlenecks():
    label = np.zeros((30, 40), dtype=np.uint8)
    label[2:5, 2:5] = SEED
    label[2:5, 35:38] = HYPOCOTYL
    label[5:11, 3:5] = MAIN_ROOT
    label[14:20, 7:9] = LATERAL_ROOT
    label[12:17, 27:29] = MAIN_ROOT
    label[22:25, 19:24] = LATERAL_ROOT
    label[8:24, 16:18] = LEAF
    rng = np.random.default_rng(7)
    prob = rng.integers(1, 200, label.shape, dtype=np.int16)
    prob[np.isin(label, (MAIN_ROOT, LATERAL_ROOT))] = 250
    step, lam, radius = 0.2, 0.7, 30
    graph = fragment_graph(label, prob, step, radius, lam=lam)
    assert graph is not None and len(graph.anchor_nodes) == 2

    nodes = graph.fragments.astype(np.int64)
    anchors, _ = ndimage.label(np.isin(label, (SEED, HYPOCOTYL)), np.ones((3, 3), dtype=bool))
    anchor_pixels = anchors > 0
    nodes[anchor_pixels] = anchors[anchor_pixels] + graph.n_fragments
    occupied = nodes > 0
    boundary = occupied & ((ndimage.maximum_filter(nodes, 3) != nodes) | (ndimage.minimum_filter(nodes, 3) != nodes))
    boundary[[0, -1], :] |= occupied[[0, -1], :]
    boundary[:, [0, -1]] |= occupied[:, [0, -1]]
    band = (label == 0) & (ndimage.distance_transform_edt(~occupied) <= radius)
    cost = np.zeros(label.shape)
    cost[band] = step - lam * np.log(np.maximum(prob[band].astype(float), 1.0) / 255.0)
    allowed = np.where(band | boundary, cost, -1.0)
    count = graph.n_fragments + len(graph.anchor_nodes)
    complete = np.full((count + 1, count + 1), np.inf)
    np.fill_diagonal(complete, 0.0)
    for node in range(1, count + 1):
        cumulative, _ = MCP_Geometric(allowed).find_costs([tuple(p) for p in np.argwhere(boundary & (nodes == node))])
        for other in range(node + 1, count + 1):
            complete[node, other] = complete[other, node] = cumulative[boundary & (nodes == other)].min()
    minimax = complete.copy()
    for intermediate in range(1, count + 1):
        minimax = np.minimum(minimax, np.maximum(minimax[:, intermediate, None], minimax[intermediate, None, :]))
    expected = np.min(minimax[1:graph.n_fragments + 1, graph.anchor_nodes], axis=1)
    assert np.allclose(bottleneck_gaps(graph)[1:], expected)


@pytest.mark.parametrize("seed", range(5))
def test_bottleneck_gaps_agree_with_budgeted_kruskal(seed):
    rng = np.random.default_rng(seed)
    label = _random_label(rng, (120, 160))
    label[label == IGNORE_INDEX] = 0
    prob = rng.integers(0, 256, label.shape).astype(np.int16)
    graph = fragment_graph(label, prob, step_cost=0.2, radius=15)
    if graph is None:
        return
    beta = bottleneck_gaps(graph)
    for budget in (0.0, 1.0, 3.0, 8.0, 1e9):
        anchored, _ = _minimax_attachment(graph, budget)
        assert np.array_equal(anchored[1:], beta[1:] <= budget)


def _sequential(q: np.ndarray, p: ReconstructionParams) -> np.ndarray:
    probabilities = class_probabilities(q)
    root_prob = probabilities[MAIN_ROOT] + probabilities[LATERAL_ROOT]
    root_class = np.where(probabilities[MAIN_ROOT] >= probabilities[LATERAL_ROOT], MAIN_ROOT, LATERAL_ROOT).astype(np.uint8)
    label = decode(probabilities, p.decode, p.root_threshold)
    if p.closing_radius:
        label = close_gaps(label, root_class, p.closing_radius)
    if p.min_area or p.keep_largest:
        label = size_filter(label, root_prob, p.min_area, p.min_peak, p.keep_largest)
    if p.anchor_distance >= 0:
        label = anchor_filter(label, p.anchor_distance, p.chain_distance)
    if p.forest_budget:
        label = geodesic_forest(label, root_prob, root_class, p.forest_budget, p.forest_step, p.forest_width,
                                p.forest_keep_area, p.anchor_min_area, p.forest_radius, p.forest_lambda,
                                p.forest_anchor, p.forest_mode)
    return label


def test_reconstruct_many_matches_sequential_pipeline():
    q, _ = _plate(gap_prob=100)
    grid = [
        ReconstructionParams(),
        ReconstructionParams(min_area=20),
        ReconstructionParams(forest_budget=4.0),
        ReconstructionParams(forest_budget=8.0, forest_width=3),
        ReconstructionParams(decode="union", forest_budget=8.0),
        ReconstructionParams(min_area=20, forest_budget=8.0),
        ReconstructionParams(anchor_distance=2, chain_distance=10),
        ReconstructionParams(closing_radius=3),
        ReconstructionParams(keep_largest=1),
        ReconstructionParams(keep_largest=1, forest_budget=8.0),
        ReconstructionParams(forest_budget=2.0, forest_width=0),
        ReconstructionParams(forest_budget=8.0, forest_mode="additive"),
        ReconstructionParams(forest_budget=8.0, forest_lambda=0.0, forest_step=0.2),
        ReconstructionParams(forest_budget=8.0, forest_anchor="largest"),
    ]
    results = dict(reconstruct_many(q, grid))
    for index, params in enumerate(grid):
        np.testing.assert_array_equal(results[index], _sequential(q, params))
