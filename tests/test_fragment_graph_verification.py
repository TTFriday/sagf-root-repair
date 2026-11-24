"""Independent-oracle, fixed-source and forest-drawing regressions."""
import inspect
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import verify_fragment_graph as verification
from toposeg.constants import HYPOCOTYL, LATERAL_ROOT, LEAF, MAIN_ROOT, PETIOLE, SEED
from toposeg.reconstruction import bottleneck_gaps, fragment_graph, grow_forest


@pytest.mark.parametrize("name", verification.EXPLICIT_NAMES)
def test_explicit_complete_network_bottlenecks_and_fixed_sources(name):
    label, prob, settings = verification.explicit_case(name)
    row, detail = verification.compare_case(label, prob, settings)
    assert row["status"] == "pass", (name, row, detail)
    assert row["owner_mismatch"] == 0
    assert row["finite_mismatch"] == row["budgetmismatch"] == 0
    oracle = verification.complete_oracle(label, prob, settings)
    graph = fragment_graph(label, prob, settings.step, settings.radius,
                           settings.anchor_min_area, settings.lam, settings.anchor)
    assert np.array_equal(graph.owner[oracle.domain.boundary],
                          oracle.domain.nodes[oracle.domain.boundary])
    if name == "fixed_sources_zero_tie":
        assert np.all(oracle.beta[1:] == 0)
        assert row["complete_finite_edges"] == row["n_nodes"] * (row["n_nodes"] - 1) // 2
    elif name in ("no_anchors", "shoot_filters_all"):
        assert np.all(np.isinf(oracle.beta[1:]))
        assert row["n_anchors"] == 0
    elif name in ("leaf_obstacle", "petiole_obstacle", "disconnect_band"):
        assert np.isfinite(oracle.beta[1]) and np.isinf(oracle.beta[2])
        assert row["complete_finite_edges"] < row["n_nodes"] * (row["n_nodes"] - 1) // 2
    elif name == "multi_anchors":
        assert row["n_anchors"] == 2 and np.all(oracle.beta[1:] == 0)
    elif name == "shoot_filters_gappy":
        assert oracle.domain.anchors == [oracle.domain.n_roots + 2]
        assert not oracle.domain.allowed[1, 1]  # Filtered shoot remains an obstacle.
    elif name in ("largest_anchor", "full_image_boundary"):
        assert oracle.domain.anchors == [1] and oracle.beta[1] == 0
        if name == "full_image_boundary":
            assert oracle.domain.boundary.sum() == 4 * 24 - 4


@pytest.mark.parametrize("spec", verification.random_specs()[:20], ids=lambda spec: spec["case_id"])
def test_fixed_random_selftest_subset(spec):
    label, prob, settings = verification.random_case(spec)
    row, detail = verification.compare_case(label, prob, settings)
    assert row["n_nodes"] == spec["requested_nodes"]
    assert row["status"] == "pass", (spec, row, detail)


@pytest.mark.parametrize("step,lam,probability", [(1, 0, 255), (0, 0, 0), (0.2, 0.25, 128)])
def test_oracle_analytic_one_dimensional_path_including_zero_edges(step, lam, probability):
    label = np.zeros((1, 7), dtype=np.uint8)
    label[0, 0], label[0, 6] = SEED, MAIN_ROOT
    oracle = verification.complete_oracle(label, np.full_like(label, probability),
                                          verification.Settings(step=step, lam=lam))
    pixel_cost = step - lam * np.log(max(probability, 1) / 255)
    # Two endpoint half-costs and four whole interior costs, all sources cost zero.
    assert oracle.complete[0, 1] == pytest.approx(5 * pixel_cost)
    assert oracle.beta[1] == pytest.approx(5 * pixel_cost)
    assert oracle.pixel_arcs == 12  # Zero-cost arcs must not disappear from CSR.


@pytest.mark.parametrize("side", ["production", "oracle", "both"])
@pytest.mark.parametrize("invalid", [np.nan, -np.inf])
def test_invalid_bottlenecks_cannot_pass_as_unreachable(monkeypatch, side, invalid):
    if side in ("production", "both"):
        original_beta = verification.bottleneck_gaps

        def invalid_beta(graph):
            beta = original_beta(graph)
            beta[1:] = invalid
            return beta

        monkeypatch.setattr(verification, "bottleneck_gaps", invalid_beta)
    if side in ("oracle", "both"):
        original_oracle = verification.complete_oracle

        def invalid_oracle(label, prob, settings):
            oracle = original_oracle(label, prob, settings)
            oracle.beta[1:] = invalid
            return oracle

        monkeypatch.setattr(verification, "complete_oracle", invalid_oracle)
    label, prob, settings = verification.explicit_case("no_anchors")
    row, _ = verification.compare_case(label, prob, settings)
    assert row["status"] == "fail"
    assert row["finite_mismatch"] > 0


def test_oracle_has_no_production_graph_dependency(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("production graph/forest cannot be part of the oracle")
    monkeypatch.setattr(verification, "fragment_graph", forbidden)
    monkeypatch.setattr(verification, "bottleneck_gaps", forbidden)
    monkeypatch.setattr(verification, "grow_forest", forbidden)
    label, prob, settings = verification.explicit_case("fixed_sources_zero_tie")
    assert np.all(verification.complete_oracle(label, prob, settings).beta[1:] == 0)
    source = inspect.getsource(verification.complete_oracle) + inspect.getsource(verification.independent_domain)
    for token in ("MCP_Geometric", "_geodesic_owner", "graph.edges", "graph.owner", "bottleneck_gaps("):
        assert token not in source


def test_boundary_interior_and_obstacle_domain_are_independent():
    label = np.zeros((20, 24), dtype=np.uint8)
    label[0:6, 0:6] = MAIN_ROOT
    label[8:11, 10:13] = SEED
    label[3:6, 15:18] = LEAF
    label[15:18, 15:18] = PETIOLE
    domain = verification.independent_domain(label, np.full_like(label, 128), verification.Settings(radius=2))
    assert domain.boundary[0, :6].all() and domain.boundary[:6, 0].all()
    assert not domain.allowed[2, 2]  # Node interior is forbidden, not a zero-cost shortcut.
    assert domain.boundary[5, 5] and domain.cost[5, 5] == 0
    assert not domain.allowed[np.isin(label, (LEAF, PETIOLE))].any()
    assert not domain.allowed[-1, -1]


@pytest.mark.parametrize("budget", verification.BUDGETS)
@pytest.mark.parametrize("width", [0, 1, 21])
def test_wide_drawing_never_changes_budget_retention_or_original_organs(budget, width):
    label, prob, settings = verification.explicit_case("wide_forest")
    graph = fragment_graph(label, prob, settings.step, settings.radius, lam=settings.lam)
    oracle = verification.complete_oracle(label, prob, settings)
    out = grow_forest(graph, np.full_like(label, MAIN_ROOT), budget, bridge_width=width)
    for node in range(1, graph.n_fragments + 1):
        pixels = graph.fragments == node
        if oracle.beta[node] <= budget:
            assert np.array_equal(out[pixels], label[pixels])
        else:
            assert np.all(out[pixels] == 0)  # Dilation must not repaint removed roots.
    other_organs = np.isin(label, (SEED, HYPOCOTYL, LEAF, PETIOLE))
    assert np.array_equal(out[other_organs], label[other_organs])
    assert np.array_equal(graph.label, label)


def test_forest_removed_fragment_is_inside_wide_paint_but_not_repainted():
    label, prob, settings = verification.explicit_case("wide_forest")
    graph = fragment_graph(label, prob, settings.step, settings.radius, lam=settings.lam)
    assert bottleneck_gaps(graph)[1] == pytest.approx(4)
    assert bottleneck_gaps(graph)[2] > 4
    thin = grow_forest(graph, np.full_like(label, MAIN_ROOT), 4, bridge_width=1)
    wide = grow_forest(graph, np.full_like(label, MAIN_ROOT), 4, bridge_width=21)
    assert thin[10, 5] == wide[10, 5] == MAIN_ROOT  # An accepted bridge is actually drawn.
    assert thin[18, 4] == 0 and wide[18, 4] == MAIN_ROOT
    assert label[18, 5] == LATERAL_ROOT and wide[18, 5] == 0


def test_frozen_plan_counts_settings_and_reproducibility():
    plan = verification.protocol(True)
    random = [s for s in plan["cases"] if s["kind"] == "random"]
    assert len(random) == 200 and random == verification.random_specs()
    assert plan["seed"] == 20261124 and plan["budgets"] == [0, 1, 4, 10, 1000]
    assert {s["size"] for s in random} == {24, 40, 64}
    assert all(8 <= s["requested_nodes"] <= 20 for s in random)
    assert {(s["settings"]["step"], s["settings"]["lam"], s["settings"]["radius"]) for s in random} == set(
        itertools.product((0, 0.2, 1), (0, 0.25, 1), (2, 8, 40)))
    real = [s for s in plan["cases"] if s["kind"] == "validation_roi"]
    assert [s["input"] for s in real] == list(verification.PROB_NAMES)
    assert all(s["settings"] == {"step": 0.2, "lam": 0.25, "radius": 40,
                                 "anchor_min_area": 0, "anchor": "shoot"} for s in real)
    assert len([s for s in plan["cases"] if s["kind"] == "benchmark"]) == 9
    assert len(verification.protocol(False)["cases"]) == len(plan["cases"]) - 5
    assert str(verification.SOURCE) not in json.dumps(plan)
    a = verification.random_case(random[19])
    b = verification.random_case(random[19])
    assert verification.input_digest(*a[:2]) == verification.input_digest(*b[:2])


def test_npz_actual_key_roi_clipping_minarea_and_no_gt_access(tmp_path):
    q = np.zeros((6, 300, 320), dtype=np.uint8)
    q[SEED - 1, 5:8, 4:7] = 250
    q[HYPOCOTYL - 1, 280:290, 300:310] = 250
    q[MAIN_ROOT - 1, 20:25, 20:25] = 250
    q[LATERAL_ROOT - 1, 40:42, 40:42] = 250
    path = tmp_path / "input.npz"
    # Any attempt to read the decoy GT with allow_pickle=False would fail.
    np.savez_compressed(path, p=q, gt=np.array([object()], dtype=object))
    label, prob, metadata = verification.load_roi(path)
    assert label.shape == prob.shape == (256, 256)
    assert metadata["roi_coordinates"] == [0, 0, 256, 256]
    assert metadata["roi_center_yx"] == [6.0, 5.0]
    assert metadata["probability_key"] == "p" and metadata["input"] == path.name
    assert metadata["input_sha256"] == verification.sha256(path)
    assert np.all(label[20:25, 20:25] == MAIN_ROOT) and not label[40:42, 40:42].any()
    assert str(tmp_path) not in json.dumps(metadata)


def test_no_shoot_roi_uses_image_center(tmp_path):
    q = np.zeros((6, 300, 320), dtype=np.uint8)
    path = tmp_path / "empty.npz"
    np.savez_compressed(path, p=q)
    label, prob, metadata = verification.load_roi(path)
    assert metadata["roi_center_rule"] == "image_center"
    assert metadata["roi_coordinates"] == [21, 31, 277, 287]
    assert not label.any() and not prob.any()


def test_cli_rejects_existing_output_without_writing(tmp_path):
    before = list(tmp_path.iterdir())
    with pytest.raises(SystemExit) as exc:
        verification.main(["--output", str(tmp_path)])
    assert exc.value.code == 2
    assert list(tmp_path.iterdir()) == before


def test_json_is_strict_and_write_once(tmp_path):
    path = tmp_path / "numeric.json"
    verification.write_json(path, {"finite": 1.5, "unreachable": float("inf")})
    assert json.loads(path.read_text(encoding="utf-8")) == {"finite": 1.5, "unreachable": "inf"}
    with pytest.raises(FileExistsError):
        verification.write_json(path, {"finite": 2})
