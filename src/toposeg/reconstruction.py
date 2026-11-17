from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.graph import MCP_Geometric
from skimage.morphology import disk, skeletonize

from .constants import HYPOCOTYL, MAIN_ROOT, LATERAL_ROOT, SEED

EIGHT = np.ones((3, 3), dtype=bool)
NEIGHBOURS = np.ones((3, 3), dtype=np.uint8)


@dataclass(frozen=True)
class ReconstructionParams:
    decode: str = "argmax"
    root_threshold: float = 1.0
    hysteresis_low: float = 0.0
    closing_radius: int = 0
    bridge_radius: int = 0
    bridge_min_prob: float = 0.2
    bridge_max_angle: float = 60.0
    anchor_distance: float = -1.0
    chain_distance: float = 0.0
    min_area: int = 0
    min_peak: float = 0.0
    keep_largest: int = 0
    forest_budget: float = 0.0
    forest_step: float = 0.1
    forest_width: int = 1
    forest_keep_area: int = 0
    forest_radius: int = 40
    forest_lambda: float = 1.0
    forest_anchor: str = "shoot"
    forest_mode: str = "minimax"
    anchor_min_area: int = 0
    spur_length: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def class_probabilities(quantised: np.ndarray) -> np.ndarray:
    """Return (7, H, W) uint8-scale probabilities including background from the six stored foreground channels."""
    foreground = quantised.astype(np.int16)
    background = np.clip(255 - foreground.sum(0), 0, 255)
    return np.concatenate([background[None], foreground], 0)


def decode(probabilities: np.ndarray, mode: str = "argmax", root_threshold: float = 1.0) -> np.ndarray:
    """Argmax over the seven classes, or hierarchical decoding that first scores the root union.

    In "union" mode main and lateral root are merged into one tubular class for the decision and split
    again by their relative probability; pixels whose root-union probability reaches root_threshold
    are root even when another single class is larger.
    """
    if mode == "argmax":
        return probabilities.argmax(0).astype(np.uint8)
    if mode != "union":
        raise ValueError(f"Unknown decode mode: {mode}")
    root = probabilities[MAIN_ROOT] + probabilities[LATERAL_ROOT]
    others = probabilities.copy()
    others[MAIN_ROOT] = -1
    others[LATERAL_ROOT] = -1
    best_other = others.argmax(0)
    best_other_value = np.take_along_axis(others, best_other[None], 0)[0]
    is_root = root > best_other_value
    if root_threshold < 1.0:
        is_root |= root >= root_threshold * 255
    root_class = np.where(probabilities[MAIN_ROOT] >= probabilities[LATERAL_ROOT], MAIN_ROOT, LATERAL_ROOT)
    return np.where(is_root, root_class, best_other).astype(np.uint8)


def _root_class(probabilities: np.ndarray) -> np.ndarray:
    return np.where(probabilities[MAIN_ROOT] >= probabilities[LATERAL_ROOT], MAIN_ROOT, LATERAL_ROOT).astype(np.uint8)


def hysteresis(label: np.ndarray, root_prob: np.ndarray, low: float, root_class: np.ndarray) -> np.ndarray:
    """Grow the decoded root into connected pixels whose root-union probability is at least low."""
    strong = np.isin(label, (MAIN_ROOT, LATERAL_ROOT))
    weak = (root_prob >= low * 255) | strong
    components, _ = ndimage.label(weak, EIGHT)
    keep_ids = np.unique(components[strong])
    grown = np.isin(components, keep_ids[keep_ids > 0]) & ~strong & (label == 0)
    out = label.copy()
    out[grown] = root_class[grown]
    return out


def close_gaps(label: np.ndarray, root_class: np.ndarray, radius: int) -> np.ndarray:
    """Morphological closing of the root union; only background pixels are converted."""
    root = np.isin(label, (MAIN_ROOT, LATERAL_ROOT))
    added = ndimage.binary_closing(root, structure=disk(radius), border_value=0) & (label == 0)
    out = label.copy()
    out[added] = root_class[added]
    return out


def _endpoints(skeleton: np.ndarray) -> np.ndarray:
    count = ndimage.convolve(skeleton.astype(np.uint8), NEIGHBOURS, mode="constant") - skeleton
    return np.argwhere(skeleton & (count == 1))


def _tangent(skeleton: np.ndarray, point: np.ndarray, reach: int = 8) -> np.ndarray | None:
    y, x = point
    y0, x0 = max(0, y - reach), max(0, x - reach)
    local = np.argwhere(skeleton[y0:y + reach + 1, x0:x + reach + 1]) + (y0, x0)
    if len(local) < 3:
        return None
    direction = point - local.mean(0)
    norm = np.linalg.norm(direction)
    return direction / norm if norm > 0 else None


def bridge_gaps(label: np.ndarray, root_prob: np.ndarray, root_class: np.ndarray, radius: int, min_prob: float, max_angle: float) -> np.ndarray:
    """Link each fragment end to a different root fragment along the minimum-cost path through the root probability.

    A link is accepted when the path stays inside the search window, its mean root probability reaches
    min_prob and it leaves the end within max_angle of the fragment's own direction.
    """
    root = np.isin(label, (MAIN_ROOT, LATERAL_ROOT))
    if not root.any():
        return label
    components, count = ndimage.label(root, EIGHT)
    if count < 2:
        return label
    parent = np.arange(count + 1)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    skeleton = skeletonize(root)
    cos_limit = np.cos(np.deg2rad(max_angle))
    traversable = (label == 0) | root
    cost_full = 1.0 - root_prob.astype(np.float32) / 255.0 + 1e-3
    cost_full[~traversable] = np.inf
    out = label.copy()
    h, w = root.shape
    for point in _endpoints(skeleton):
        y, x = int(point[0]), int(point[1])
        own = components[y, x]
        if own == 0:
            continue
        y0, y1, x0, x1 = max(0, y - radius), min(h, y + radius + 1), max(0, x - radius), min(w, x + radius + 1)
        window = components[y0:y1, x0:x1]
        targets = (window > 0) & (window != own)
        if not targets.any():
            continue
        own_root = find(own)
        ids = window[targets]
        roots_of_targets = np.array([find(int(i)) for i in ids])
        valid = roots_of_targets != own_root
        if not valid.any():
            continue
        end_points = np.argwhere(targets)[valid]
        tangent = _tangent(skeleton, point)
        if tangent is None:
            continue
        mcp = MCP_Geometric(cost_full[y0:y1, x0:x1])
        costs, _ = mcp.find_costs([(y - y0, x - x0)], ends=[tuple(p) for p in end_points], find_all_ends=False)
        end_costs = costs[end_points[:, 0], end_points[:, 1]]
        best = int(np.argmin(end_costs))
        if not np.isfinite(end_costs[best]):
            continue
        target = end_points[best]
        path = np.array(mcp.traceback(tuple(target)))
        offset = target - np.array([y - y0, x - x0])
        norm = np.linalg.norm(offset)
        if norm == 0 or float(offset @ tangent) / norm < cos_limit:
            continue
        path_global = path + (y0, x0)
        interior = path_global[1:-1]
        if len(interior) and float(root_prob[interior[:, 0], interior[:, 1]].mean()) / 255.0 < min_prob:
            continue
        gap = interior[label[interior[:, 0], interior[:, 1]] == 0] if len(interior) else interior
        if len(gap):
            out[gap[:, 0], gap[:, 1]] = root_class[gap[:, 0], gap[:, 1]]
        target_id = int(window[target[0], target[1]])
        parent[find(target_id)] = own_root
    return out


def anchor_filter(label: np.ndarray, anchor_distance: float, chain_distance: float) -> np.ndarray:
    """Keep root fragments that reach a shoot organ directly or through a chain of nearby fragments.

    Fragments closer than chain_distance are grouped by single linkage (their chain_distance/2
    neighbourhoods overlap); a group survives when any member lies within anchor_distance of a shoot.
    """
    root = np.isin(label, (MAIN_ROOT, LATERAL_ROOT))
    shoot = np.isin(label, (SEED, HYPOCOTYL))
    if not root.any() or not shoot.any():
        return label
    grouped = ndimage.distance_transform_edt(~root) <= chain_distance / 2 if chain_distance > 0 else root
    groups, count = ndimage.label(grouped, EIGHT)
    to_shoot = ndimage.distance_transform_edt(~shoot)
    nearest = np.asarray(ndimage.minimum(to_shoot, np.where(root, groups, 0), np.arange(1, count + 1)))
    kept = np.flatnonzero(nearest <= anchor_distance) + 1
    out = label.copy()
    out[root & ~np.isin(groups, kept)] = 0
    return out


def _geodesic_owner(traceback: np.ndarray, offsets: np.ndarray, seeds: np.ndarray, reached: np.ndarray, seed_owner: np.ndarray) -> np.ndarray:
    """Owner node of every reached pixel: the node whose boundary its minimum-cost path starts from."""
    width = traceback.shape[1]
    flat = np.flatnonzero(reached)
    position = np.full(traceback.size, -1, dtype=np.int64)
    position[flat] = np.arange(flat.size)
    step = traceback.ravel()[flat]
    shift = offsets[:, 0].astype(np.int64) * width + offsets[:, 1].astype(np.int64)
    predecessor = np.where(step >= 0, flat - shift[np.maximum(step, 0)], flat)
    pointer = position[predecessor]
    seed_index = position[np.ravel_multi_index(seeds.T, traceback.shape)]
    pointer[seed_index] = seed_index
    while True:
        jumped = pointer[pointer]
        if np.array_equal(jumped, pointer):
            break
        pointer = jumped
    owner_of_seed = np.zeros(flat.size, dtype=np.int64)
    owner_of_seed[seed_index] = seed_owner
    owner = np.zeros(traceback.size, dtype=np.int64)
    owner[flat] = owner_of_seed[pointer]
    return owner.reshape(traceback.shape)


@dataclass
class FragmentGraph:
    """Root fragments and shoot anchors with the cheapest geodesic gap between neighbouring nodes."""

    label: np.ndarray
    root: np.ndarray
    fragments: np.ndarray
    n_fragments: int
    anchor_nodes: list[int]
    edges: np.ndarray
    paths: MCP_Geometric
    owner: np.ndarray
    distance: np.ndarray


def fragment_graph(label: np.ndarray, root_prob: np.ndarray, step_cost: float, radius: int = 40, anchor_min_area: int = 0,
                   lam: float = 1.0, anchor: str = "shoot") -> FragmentGraph | None:
    """Build the fragment graph used by the shoot-anchored geodesic forest.

    Root fragments and shoot organs (seed, hypocotyl) are nodes. Gaps are measured by the geodesic
    distance under the metric step_cost - lam * log p_root, i.e. the negative log-likelihood that the path
    is root plus a length penalty, inside a background-pixel band of the given radius around the nodes. Neighbouring cells
    of the geodesic Voronoi partition of that band give one candidate edge per node pair, weighted by
    the cheapest path through their common border. With anchor="largest" the largest root fragment
    replaces the shoot organs as the only anchor.
    """
    root = np.isin(label, (MAIN_ROOT, LATERAL_ROOT))
    if not root.any():
        return None
    fragments, n_fragments = ndimage.label(root, EIGHT)
    if anchor == "shoot":
        anchors, n_anchors = ndimage.label(np.isin(label, (SEED, HYPOCOTYL)), EIGHT)
    elif anchor == "largest":
        anchors, n_anchors = np.zeros_like(fragments), 0
    else:
        raise ValueError(f"Unknown anchor: {anchor}")
    if anchor_min_area > 0 and n_anchors:
        area = np.bincount(anchors.ravel(), minlength=n_anchors + 1)
        anchors[area[anchors] < anchor_min_area] = 0
    nodes = fragments.astype(np.int64)
    anchor_pixels = anchors > 0
    nodes[anchor_pixels] = anchors[anchor_pixels] + n_fragments
    occupied = nodes > 0
    boundary = occupied & ((ndimage.maximum_filter(nodes, 3) != nodes) | (ndimage.minimum_filter(nodes, 3) != nodes))
    boundary[[0, -1], :] |= occupied[[0, -1], :]
    boundary[:, [0, -1]] |= occupied[:, [0, -1]]
    band = (label == 0) & (ndimage.distance_transform_edt(~occupied) <= radius)
    pixel_cost = np.zeros(label.shape)
    pixel_cost[band] = step_cost - lam * np.log(np.maximum(root_prob[band].astype(np.float64), 1.0) / 255.0)
    seeds = np.argwhere(boundary)
    mcp = MCP_Geometric(np.where(band | boundary, pixel_cost, -1.0))
    cumulative, traceback = mcp.find_costs([tuple(p) for p in seeds])
    reached = np.isfinite(cumulative) & (band | boundary)
    owner = _geodesic_owner(traceback, np.asarray(mcp.offsets), seeds, reached, nodes[boundary])
    pairs = []
    h, w = label.shape
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        xs = max(0, -dx)
        a_sl = (slice(0, h - dy), slice(xs, w - max(0, dx)))
        b_sl = (slice(dy, h), slice(xs + dx, w - max(0, dx) + dx))
        oa, ob = owner[a_sl], owner[b_sl]
        meet = (oa > 0) & (ob > 0) & (oa != ob)
        if not meet.any():
            continue
        cost = cumulative[a_sl][meet] + cumulative[b_sl][meet] + 0.5 * np.hypot(dy, dx) * (pixel_cost[a_sl][meet] + pixel_cost[b_sl][meet])
        ya, xa = np.nonzero(meet)
        pairs.append(np.column_stack([np.minimum(oa[meet], ob[meet]), np.maximum(oa[meet], ob[meet]), cost,
                                      ya, xa + xs, ya + dy, xa + xs + dx]))
    edges = np.zeros((0, 7))
    if pairs:
        edges = np.concatenate(pairs)
        edges = edges[np.lexsort((edges[:, 2], edges[:, 1], edges[:, 0]))]
        first = np.ones(len(edges), dtype=bool)
        first[1:] = (edges[1:, 0] != edges[:-1, 0]) | (edges[1:, 1] != edges[:-1, 1])
        edges = edges[first]
        edges = edges[np.argsort(edges[:, 2], kind="stable")]
    if anchor == "largest":
        anchor_nodes = [int(np.argmax(np.bincount(fragments.ravel())[1:])) + 1]
    else:
        anchor_nodes = [int(a) + n_fragments for a in np.unique(anchors[anchor_pixels])]
    return FragmentGraph(label, root, fragments, n_fragments, anchor_nodes, edges, mcp, owner, cumulative)


def _additive_attachment(graph: FragmentGraph, budget: float) -> tuple[np.ndarray, list]:
    """Shortest-path (sum of gap costs) attachment to the merged anchor set instead of the minimax gap."""
    n = graph.n_fragments + max(graph.anchor_nodes, default=graph.n_fragments) + 1
    remap = np.arange(n)
    remap[graph.anchor_nodes] = 0
    best: dict[tuple[int, int], tuple] = {}
    for a, b, cost, y1, x1, y2, x2 in graph.edges:
        u, v = int(remap[int(a)]), int(remap[int(b)])
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key not in best or cost < best[key][0]:
            best[key] = (max(float(cost), 1e-9), (int(y1), int(x1)), (int(y2), int(x2)))
    if not best:
        anchored = np.zeros(graph.n_fragments + 1, dtype=bool)
        anchored[[a for a in graph.anchor_nodes if a <= graph.n_fragments]] = True
        return anchored, []
    rows, cols = np.array(list(best)).T
    weights = np.array([v[0] for v in best.values()])
    matrix = csr_matrix((weights, (rows, cols)), shape=(n, n))
    distance, predecessor = dijkstra(matrix, directed=False, indices=0, return_predecessors=True)
    anchored = np.zeros(graph.n_fragments + 1, dtype=bool)
    anchored[1:] = distance[1:graph.n_fragments + 1] <= budget
    anchored[[a for a in graph.anchor_nodes if a <= graph.n_fragments]] = True
    accepted = []
    for node in np.flatnonzero(anchored):
        parent = int(predecessor[node])
        if parent >= 0:
            _, pos1, pos2 = best[(min(parent, int(node)), max(parent, int(node)))]
            accepted.append((pos1, pos2))
    return anchored, accepted


def grow_forest(graph: FragmentGraph | None, root_class: np.ndarray, budget: float, bridge_width: int = 1, keep_area: int = 0,
                mode: str = "minimax") -> np.ndarray | None:
    """Kruskal's algorithm with all shoot nodes pre-merged attaches each fragment to the anchored forest
    through its minimax gap; fragments whose bottleneck gap exceeds budget are removed unless they cover
    at least keep_area pixels, and the accepted gaps are painted as root (bridge_width 0 filters only).
    mode="additive" attaches by the summed gap cost of the shortest path instead."""
    if graph is None:
        return None
    if mode == "additive":
        anchored, accepted = _additive_attachment(graph, budget)
    elif mode == "minimax":
        anchored, accepted = _minimax_attachment(graph, budget)
    else:
        raise ValueError(f"Unknown forest mode: {mode}")
    if keep_area > 0:
        area = np.bincount(graph.fragments.ravel(), minlength=graph.n_fragments + 1)
        anchored |= area >= keep_area
        anchored[0] = False
    out = graph.label.copy()
    out[graph.root & ~anchored[graph.fragments]] = 0
    if bridge_width == 0:
        return out
    bridge = np.zeros_like(graph.root)
    for pos1, pos2 in accepted:
        for end in (pos1, pos2):
            path = np.array(graph.paths.traceback(end))
            bridge[path[:, 0], path[:, 1]] = True
    if bridge_width > 1:
        bridge = ndimage.binary_dilation(bridge, np.ones((bridge_width, bridge_width), dtype=bool))
    bridge &= graph.label == 0
    out[bridge] = root_class[bridge]
    return out


def bottleneck_gaps(graph: FragmentGraph) -> np.ndarray:
    """Bottleneck gap beta(i) of every fragment to the anchors (index 0 unused; inf if unreachable), in one Kruskal pass."""
    size = graph.n_fragments + max(graph.anchor_nodes, default=graph.n_fragments) + 1
    parent = list(range(size))
    members = {i: [i] if 1 <= i <= graph.n_fragments else [] for i in range(size)}
    beta = np.full(graph.n_fragments + 1, np.inf)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> int:
        if len(members[a]) < len(members[b]):
            a, b = b, a
        parent[b] = a
        members[a].extend(members.pop(b))
        return a

    anchor = find(0)
    for node in graph.anchor_nodes:
        root = find(node)
        if root != anchor:
            anchor = union(anchor, root)
    beta[members[anchor]] = 0.0
    for a, b, cost, *_ in graph.edges:
        ra, rb = find(int(a)), find(int(b))
        if ra == rb:
            continue
        if anchor in (ra, rb):
            beta[members[rb if ra == anchor else ra]] = cost
        merged = union(ra, rb)
        if anchor in (ra, rb):
            anchor = merged
    return beta


def _minimax_attachment(graph: FragmentGraph, budget: float) -> tuple[np.ndarray, list]:
    parent = list(range(graph.n_fragments + max(graph.anchor_nodes, default=graph.n_fragments) + 1))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for node in graph.anchor_nodes:
        parent[find(node)] = find(0)
    accepted = []
    for a, b, cost, y1, x1, y2, x2 in graph.edges:
        if cost > budget:
            break
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
            accepted.append((int(a), (int(y1), int(x1)), (int(y2), int(x2))))
    anchored = np.array([False] + [find(i) == find(0) for i in range(1, graph.n_fragments + 1)])
    return anchored, [(p1, p2) for node, p1, p2 in accepted if find(node) == find(0)]


def geodesic_forest(label: np.ndarray, root_prob: np.ndarray, root_class: np.ndarray, budget: float, step_cost: float,
                    bridge_width: int = 1, keep_area: int = 0, anchor_min_area: int = 0, radius: int = 40,
                    lam: float = 1.0, anchor: str = "shoot", mode: str = "minimax") -> np.ndarray:
    graph = fragment_graph(label, root_prob, step_cost, radius, anchor_min_area, lam, anchor)
    return label if graph is None else grow_forest(graph, root_class, budget, bridge_width, keep_area, mode)


def size_filter(label: np.ndarray, root_prob: np.ndarray, min_area: int, min_peak: float, keep_largest: int = 0) -> np.ndarray:
    root = np.isin(label, (MAIN_ROOT, LATERAL_ROOT))
    if not root.any():
        return label
    components, count = ndimage.label(root, EIGHT)
    index = np.arange(1, count + 1)
    area = np.asarray(ndimage.sum(np.ones_like(components), components, index))
    peak = np.asarray(ndimage.maximum(root_prob, components, index)) / 255.0
    drop = (area < min_area) | (peak < min_peak)
    if keep_largest > 0:
        drop[np.argsort(-area, kind="stable")[keep_largest:]] = True
    out = label.copy()
    out[np.isin(components, index[drop])] = 0
    return out


def _skeleton_neighbours(skeleton: np.ndarray, point: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = skeleton.shape
    y, x = point
    return [
        (y + dy, x + dx)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dy or dx) and 0 <= y + dy < h and 0 <= x + dx < w and skeleton[y + dy, x + dx]
    ]


def _branch_pixels(skeleton: np.ndarray, start: np.ndarray, limit: int) -> list[tuple[int, int]] | None:
    """Pixels of the terminal branch starting at an end if it meets a junction within limit steps.

    Isolated short segments return None: they are fragments, not spurs.
    """
    current = (int(start[0]), int(start[1]))
    path = [current]
    visited = {current}
    for _ in range(limit):
        ahead = [p for p in _skeleton_neighbours(skeleton, current) if p not in visited]
        if not ahead:
            return None
        if len(ahead) > 1:
            return path[:-1] if len(path) > 1 else None
        nxt = ahead[0]
        if len(_skeleton_neighbours(skeleton, nxt)) >= 3:
            return path
        path.append(nxt)
        visited.add(nxt)
        current = nxt
    return None


def prune_spurs(label: np.ndarray, spur_length: int) -> np.ndarray:
    """Remove the mask lobes that generate short terminal skeleton branches."""
    root = np.isin(label, (MAIN_ROOT, LATERAL_ROOT))
    if not root.any():
        return label
    skeleton = skeletonize(root)
    removed = np.zeros_like(skeleton)
    for point in _endpoints(skeleton):
        branch = _branch_pixels(skeleton, point, spur_length)
        if branch:
            ys, xs = zip(*branch)
            removed[list(ys), list(xs)] = True
    if not removed.any():
        return label
    kept = skeleton & ~removed
    radius = ndimage.distance_transform_edt(root)
    support = np.zeros_like(root)
    values = np.ceil(radius[kept]).astype(int) + 1
    for r in np.unique(values):
        seeds = np.zeros_like(root)
        sel = values == r
        coords = np.argwhere(kept)[sel]
        seeds[coords[:, 0], coords[:, 1]] = True
        support |= ndimage.binary_dilation(seeds, structure=disk(int(r)))
    out = label.copy()
    out[root & ~support] = 0
    return out


_STAGES = (
    ("decode", "root_threshold"),
    ("hysteresis_low",),
    ("closing_radius",),
    ("min_area", "min_peak", "keep_largest"),
    ("bridge_radius", "bridge_min_prob", "bridge_max_angle"),
    ("anchor_distance", "chain_distance"),
    ("forest_step", "forest_radius", "anchor_min_area", "forest_lambda", "forest_anchor"),
    ("forest_budget", "forest_width", "forest_keep_area", "forest_mode"),
    ("spur_length",),
)
_GRAPH_STAGE = 6


def _stage_keys(params: ReconstructionParams) -> list[tuple]:
    values = asdict(params)
    keys, prefix = [], ()
    for stage, fields in enumerate(_STAGES):
        prefix = prefix + tuple(values[f] for f in fields)
        if stage == _GRAPH_STAGE:
            # the graph does not depend on the budget, only on whether a forest is grown at all
            prefix = prefix + (params.forest_budget > 0,)
        keys.append(prefix)
    return keys


def _apply_stage(stage: int, label, params: ReconstructionParams, root_prob: np.ndarray, root_class: np.ndarray):
    p = params
    if stage == 1 and p.hysteresis_low > 0:
        return hysteresis(label, root_prob, p.hysteresis_low, root_class)
    if stage == 2 and p.closing_radius > 0:
        return close_gaps(label, root_class, p.closing_radius)
    if stage == 3 and (p.min_area > 0 or p.min_peak > 0 or p.keep_largest > 0):
        return size_filter(label, root_prob, p.min_area, p.min_peak, p.keep_largest)
    if stage == 4 and p.bridge_radius > 0:
        return bridge_gaps(label, root_prob, root_class, p.bridge_radius, p.bridge_min_prob, p.bridge_max_angle)
    if stage == 5 and p.anchor_distance >= 0:
        return anchor_filter(label, p.anchor_distance, p.chain_distance)
    if stage == 6:
        graph = None
        if p.forest_budget > 0:
            graph = fragment_graph(label, root_prob, p.forest_step, p.forest_radius, p.anchor_min_area, p.forest_lambda, p.forest_anchor)
        return label, graph
    if stage == 7:
        label, graph = label
        return label if graph is None else grow_forest(graph, root_class, p.forest_budget, p.forest_width, p.forest_keep_area, p.forest_mode)
    if stage == 8 and p.spur_length > 0:
        return prune_spurs(label, p.spur_length)
    return label


def reconstruct_many(quantised: np.ndarray, params_list: list[ReconstructionParams]):
    """Yield (index, label) for every parameter set, sharing all common pipeline prefixes.

    Candidates are processed in prefix order and only the latest output of every stage is kept, so
    memory stays at one label image per stage.
    """
    probabilities = class_probabilities(quantised)
    root_prob = probabilities[MAIN_ROOT] + probabilities[LATERAL_ROOT]
    root_class = _root_class(probabilities)
    keys = [_stage_keys(p) for p in params_list]
    cache: list[tuple | None] = [None] * len(_STAGES)
    for index in sorted(range(len(params_list)), key=lambda i: keys[i]):
        params = params_list[index]
        label = None
        for stage in range(len(_STAGES)):
            if cache[stage] is not None and cache[stage][0] == keys[index][stage]:
                label = cache[stage][1]
                continue
            if stage == 0:
                label = decode(probabilities, params.decode, params.root_threshold)
            else:
                label = _apply_stage(stage, label, params, root_prob, root_class)
            cache[stage] = (keys[index][stage], label)
            for later in range(stage + 1, len(_STAGES)):
                cache[later] = None
        yield index, label


def reconstruct(quantised: np.ndarray, params: ReconstructionParams) -> np.ndarray:
    return next(reconstruct_many(quantised, [params]))[1]
