"""Independent complete-network verification of fragment-graph bottlenecks."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import sys
import time
from dataclasses import asdict, dataclass

# Must precede numpy/scipy imports, including when launched without shell flags.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"
sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / "src"))

import numpy as np
import scipy
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import skimage

from toposeg.constants import HYPOCOTYL, LATERAL_ROOT, LEAF, MAIN_ROOT, PETIOLE, SEED
from toposeg.reconstruction import (
    bottleneck_gaps, class_probabilities, decode, fragment_graph, grow_forest,
    size_filter,
)

SEED_VALUE = 20261124
BUDGETS = (0.0, 1.0, 4.0, 10.0, 1000.0)
ATOL, RTOL = 1e-8, 1e-9
PROB_NAMES = ("image_1.npz", "image_211.npz", "image_368.npz",
              "image_655.npz", "image_857.npz")


@dataclass(frozen=True)
class Settings:
    step: float = 0.2
    lam: float = 0.25
    radius: int = 40
    anchor_min_area: int = 0
    anchor: str = "shoot"


@dataclass
class Domain:
    nodes: np.ndarray
    fragments: np.ndarray
    n_roots: int
    anchors: list[int]
    ids: np.ndarray
    boundary: np.ndarray
    allowed: np.ndarray
    cost: np.ndarray


@dataclass
class Oracle:
    domain: Domain
    complete: np.ndarray
    beta: np.ndarray
    pixel_vertices: int
    pixel_arcs: int


def independent_domain(label: np.ndarray, prob: np.ndarray, settings: Settings) -> Domain:
    """Reconstruct from labels without production graph, owner or cost arrays."""
    if label.shape != prob.shape or label.ndim != 2:
        raise ValueError("label/prob must be matching two-dimensional arrays")
    if settings.step < 0 or settings.lam < 0 or np.any(prob < 0) or not np.all(np.isfinite(prob)):
        raise ValueError("oracle requires finite nonnegative probabilities and cost parameters")
    root = (label == MAIN_ROOT) | (label == LATERAL_ROOT)
    fragments, n_roots = ndimage.label(root, structure=np.ones((3, 3), dtype=bool))
    nodes = fragments.astype(np.int64)
    if settings.anchor == "shoot":
        shoot, _ = ndimage.label((label == SEED) | (label == HYPOCOTYL),
                                structure=np.ones((3, 3), dtype=bool))
        sizes = np.bincount(shoot.ravel())
        accepted = (shoot > 0) & (sizes[shoot] >= settings.anchor_min_area)
        nodes[accepted] = n_roots + shoot[accepted]
        anchors = [int(v) for v in np.unique(nodes[accepted])]
    elif settings.anchor == "largest":
        anchors = [int(np.argmax(np.bincount(fragments.ravel())[1:])) + 1] if n_roots else []
    else:
        raise ValueError("unsupported anchor policy")
    occupied = nodes != 0
    h, w = label.shape
    padded = np.pad(nodes, 1, mode="edge")
    boundary = np.zeros_like(occupied)
    for dy, dx in itertools.product((-1, 0, 1), repeat=2):
        if dy or dx:
            neighbour = padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            boundary |= occupied & (neighbour != nodes)
    boundary[0, :] |= occupied[0, :]
    boundary[-1, :] |= occupied[-1, :]
    boundary[:, 0] |= occupied[:, 0]
    boundary[:, -1] |= occupied[:, -1]
    band = (label == 0) & (ndimage.distance_transform_edt(~occupied) <= settings.radius)
    cost = np.zeros(label.shape, dtype=np.float64)
    cost[band] = settings.step - settings.lam * np.log(np.maximum(prob[band].astype(float), 1) / 255)
    if np.any(cost[band] < 0):
        raise ValueError("Dijkstra oracle requires nonnegative pixel costs")
    return Domain(nodes, fragments, int(n_roots), anchors, np.unique(nodes[occupied]),
                  boundary, band | boundary, cost)


def complete_oracle(label: np.ndarray, prob: np.ndarray, settings: Settings) -> Oracle:
    domain = independent_domain(label, prob, settings)
    h, w = label.shape
    pixel_id = np.full(label.shape, -1, dtype=np.int64)
    pixel_id[domain.allowed] = np.arange(np.count_nonzero(domain.allowed))
    rows, cols, weights = [], [], []
    # All eight directed arcs are built independently. No ownership partition.
    for dy, dx in itertools.product((-1, 0, 1), repeat=2):
        if not (dy or dx):
            continue
        y0, y1, x0, x1 = max(0, -dy), min(h, h - dy), max(0, -dx), min(w, w - dx)
        here = np.s_[y0:y1, x0:x1]
        there = np.s_[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        valid = domain.allowed[here] & domain.allowed[there]
        rows.append(pixel_id[here][valid])
        cols.append(pixel_id[there][valid])
        weights.append(np.hypot(dy, dx) * (domain.cost[here][valid] + domain.cost[there][valid]) / 2)
    n_pixels = int(np.count_nonzero(domain.allowed))
    matrix = csr_matrix((np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
                        shape=(n_pixels, n_pixels))
    # Do NOT eliminate_zeros: a cost-zero arc is an edge, not missing data.
    boundaries = [pixel_id[domain.boundary & (domain.nodes == node)] for node in domain.ids]
    complete = np.full((len(domain.ids), len(domain.ids)), np.inf)
    for i, starts in enumerate(boundaries):
        if starts.size == 0:
            raise AssertionError("every occupied node must have boundary sources")
        distances = dijkstra(matrix, directed=True, indices=starts, min_only=True)
        for j, targets in enumerate(boundaries):
            complete[i, j] = distances[targets].min()
    if not np.allclose(complete, complete.T, atol=ATOL, rtol=RTOL):
        raise AssertionError("independent symmetric pixel network produced asymmetric distances")
    # Keep the computed matrix, not a sparse Kruskal/MST substitute.
    minimax = complete.copy()
    for k in range(len(domain.ids)):
        minimax = np.minimum(minimax, np.maximum(minimax[:, k, None], minimax[k, None, :]))
    beta = np.full(domain.n_roots + 1, np.inf)
    if domain.anchors:
        positions = {int(node): i for i, node in enumerate(domain.ids)}
        anchor_columns = [positions[node] for node in domain.anchors]
        for node in range(1, domain.n_roots + 1):
            beta[node] = minimax[positions[node], anchor_columns].min()
    return Oracle(domain, complete, beta, n_pixels, int(matrix.nnz))


def compare_case(label: np.ndarray, prob: np.ndarray, settings: Settings) -> tuple[dict, dict]:
    start = time.perf_counter()
    graph = fragment_graph(label, prob, settings.step, settings.radius,
                           settings.anchor_min_area, settings.lam, settings.anchor)
    production_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    oracle = complete_oracle(label, prob, settings)
    oracle_ms = (time.perf_counter() - start) * 1000
    domain = oracle.domain
    if graph is None:
        raise ValueError("verification case has no root")
    actual, expected = bottleneck_gaps(graph)[1:], oracle.beta[1:]
    finite_a, finite_e = np.isfinite(actual), np.isfinite(expected)
    both = finite_a & finite_e
    errors = np.abs(actual[both] - expected[both])
    numerical = int(np.count_nonzero(~np.isclose(actual[both], expected[both], atol=ATOL, rtol=RTOL)))
    source_bad = domain.boundary & (graph.owner != domain.nodes)
    # A fixed-source assertion is represented as a failure row, never discarded.
    owner_mismatch = int(np.count_nonzero(source_bad))
    topology_mismatch = int(graph.n_fragments != domain.n_roots
                            or graph.anchor_nodes != domain.anchors
                            or not np.array_equal(graph.fragments, domain.fragments))
    retained = []
    budgetmismatch = 0
    root_class = np.full_like(label, MAIN_ROOT)
    for budget in BUDGETS:
        out = grow_forest(graph, root_class, budget, bridge_width=0)
        production_set = [i for i in range(1, domain.n_roots + 1)
                          if np.all(out[domain.fragments == i] == label[domain.fragments == i])]
        expected_set = (np.flatnonzero(expected <= budget) + 1).tolist()
        beta_set = (np.flatnonzero(actual <= budget) + 1).tolist()
        mismatch = production_set != expected_set or beta_set != expected_set
        budgetmismatch += int(mismatch)
        retained.append({"budget": budget, "production": production_set,
                         "oracle": expected_set, "production_beta": beta_set})
    possible = len(domain.ids) * (len(domain.ids) - 1) // 2
    complete_finite = int(np.count_nonzero(np.isfinite(np.triu(oracle.complete, 1))
                                          & np.triu(np.ones_like(oracle.complete, dtype=bool), 1)))
    invalid = np.isnan(actual) | np.isnan(expected) | np.isneginf(actual) | np.isneginf(expected)
    finite_mismatch = int(np.count_nonzero(
        (finite_a != finite_e) | (np.isposinf(actual) != np.isposinf(expected)) | invalid))
    failed = any((owner_mismatch, topology_mismatch, numerical, finite_mismatch, budgetmismatch))
    row = {"status": "fail" if failed else "pass", "reason": "",
           "n_nodes": len(domain.ids), "n_roots": domain.n_roots, "n_anchors": len(domain.anchors),
           "finite_beta": int(finite_a.sum()), "finite_beta_oracle": int(finite_e.sum()),
           "finite_mismatch": finite_mismatch, "maxerr": float(errors.max()) if errors.size else 0.0,
           "numerical_mismatch": numerical, "budgetmismatch": budgetmismatch,
           "owner_mismatch": owner_mismatch, "topology_mismatch": topology_mismatch,
           "sparse_edges": len(graph.edges), "complete_finite_edges": complete_finite,
           "sparse_density": len(graph.edges) / possible if possible else 0.0,
           "complete_density": complete_finite / possible if possible else 0.0,
           "pixel_vertices": oracle.pixel_vertices, "pixel_arcs": oracle.pixel_arcs,
           "production_ms": production_ms, "oracle_ms": oracle_ms}
    detail = {"production_beta": actual.tolist(), "oracle_beta": expected.tolist(),
              "retained": retained, "source_mismatch_yx": np.argwhere(source_bad).tolist()}
    return row, detail


EXPLICIT_NAMES = (
    "fixed_sources_zero_tie", "touching_sources", "no_anchors", "multi_anchors",
    "leaf_obstacle", "petiole_obstacle", "disconnect_band", "edge_boundary",
    "full_image_boundary", "shoot_filters_gappy", "shoot_filters_all", "largest_anchor",
    "wide_forest",
)


def explicit_case(name: str) -> tuple[np.ndarray, np.ndarray, Settings]:
    label = np.zeros((24, 40), dtype=np.uint8)
    label[10:13, 3:6] = SEED
    label[10:13, 6:9] = MAIN_ROOT
    label[10:13, 30:33] = LATERAL_ROOT
    settings = Settings()
    if name == "fixed_sources_zero_tie":
        label[:] = 0
        label[2:5, 18:21] = SEED
        for x in (4, 12, 30):
            label[12:15, x:x + 3] = MAIN_ROOT
        settings = Settings(step=0, lam=0)
    elif name == "touching_sources":
        settings = Settings(step=1, lam=0)
    elif name == "no_anchors":
        label[label == SEED] = 0
    elif name == "multi_anchors":
        label[10:13, 33:36] = HYPOCOTYL
        label[:, 20:22] = LEAF
    elif name in ("leaf_obstacle", "petiole_obstacle"):
        label[:, 20:22] = LEAF if name == "leaf_obstacle" else PETIOLE
    elif name == "disconnect_band":
        settings = Settings(radius=2)
    elif name == "edge_boundary":
        label[:] = 0
        label[0:5, 0:5] = SEED
        label[5:9, 0:3] = MAIN_ROOT
        label[-4:, -3:] = LATERAL_ROOT
    elif name == "full_image_boundary":
        label = np.full((24, 24), MAIN_ROOT, dtype=np.uint8)
        settings = Settings(anchor="largest")
    elif name == "shoot_filters_gappy":
        label[label == SEED] = 0
        label[1, 1] = SEED
        label[16:20, 30:34] = HYPOCOTYL
        settings = Settings(anchor_min_area=4)
    elif name == "shoot_filters_all":
        settings = Settings(anchor_min_area=20)
    elif name == "largest_anchor":
        label[10:18, 6:9] = MAIN_ROOT
        settings = Settings(anchor="largest")
    elif name == "wide_forest":
        label = np.zeros((32, 40), dtype=np.uint8)
        label[10, 2], label[10, 7], label[18, 5] = SEED, MAIN_ROOT, LATERAL_ROOT
        label[12:16, 3:5], label[4:7, 3:5] = LEAF, PETIOLE
        settings = Settings(step=1, lam=0)
    else:
        raise ValueError("unknown explicit case")
    return label, np.full(label.shape, 255, dtype=np.uint8), settings


def random_specs() -> list[dict]:
    rng = np.random.default_rng(SEED_VALUE)
    grid = list(itertools.product((0.0, 0.2, 1.0), (0.0, 0.25, 1.0), (2, 8, 40)))
    specs = []
    for i in range(200):
        step, lam, radius = grid[i % len(grid)]
        specs.append({"case_id": f"random_{i:03d}", "kind": "random", "size": int(rng.choice((24, 40, 64))),
                      "seed": int(rng.integers(0, 2**32)), "requested_nodes": int(rng.integers(8, 21)),
                      "n_anchors": i % 4, "probability": str(rng.choice(("random", "uniform255", "uniform0"))),
                      "obstacles": ("none", "patches")[i % 2],
                      "settings": asdict(Settings(step, lam, radius))})
    return specs


def random_case(spec: dict) -> tuple[np.ndarray, np.ndarray, Settings]:
    rng = np.random.default_rng(spec["seed"])
    size = spec["size"]
    label = np.zeros((size, size), dtype=np.uint8)
    cells = [(y, x) for y in range(0, size - 3, 4) for x in range(0, size - 3, 4)]
    chosen = rng.choice(len(cells), spec["requested_nodes"], replace=False)
    for i, cell in enumerate(chosen):
        y, x = cells[int(cell)]
        height, width = rng.integers(1, 4, size=2)
        y += int(rng.integers(0, 4 - height))
        x += int(rng.integers(0, 4 - width))
        cls = (SEED if i % 2 == 0 else HYPOCOTYL) if i < spec["n_anchors"] else (
            MAIN_ROOT if i % 2 == 0 else LATERAL_ROOT)
        label[y:y + height, x:x + width] = cls
    if spec["obstacles"] == "patches":
        for cls in (LEAF, PETIOLE):
            y, x = rng.integers(0, size - 6, size=2)
            patch = label[y:y + 6, x:x + 2]
            patch[patch == 0] = cls
    if spec["probability"] == "random":
        prob = rng.integers(0, 256, size=label.shape, dtype=np.uint8)
    else:
        prob = np.full(label.shape, 255 if spec["probability"] == "uniform255" else 0, dtype=np.uint8)
    return label, prob, Settings(**spec["settings"])


def benchmark_case(size: int) -> tuple[np.ndarray, np.ndarray, Settings]:
    rng = np.random.default_rng(SEED_VALUE + size)
    label = np.zeros((size, size), dtype=np.uint8)
    locations = np.linspace(size // 8, size - size // 8 - 3, 4, dtype=int)
    for i, (y, x) in enumerate(itertools.product(locations, repeat=2)):
        label[y:y + 3, x:x + 3] = SEED if i == 0 else (HYPOCOTYL if i == 15 else MAIN_ROOT)
    return label, rng.integers(0, 256, size=label.shape, dtype=np.uint8), Settings(radius=size)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_digest(label: np.ndarray, prob: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (label, prob):
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def frozen_hashes() -> dict:
    names = ("src/toposeg/reconstruction.py", "src/toposeg/constants.py",
             "scripts/verify_fragment_graph.py", "tests/test_fragment_graph_verification.py")
    return {name: sha256(SOURCE / name) for name in names}


def environment() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "scikit_image": skimage.__version__, "system": platform.system(),
            "release": platform.release(), "architecture": platform.machine(),
            "threads": {k: os.environ[k] for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}


def protocol(include_prob: bool) -> dict:
    explicit = [{"case_id": name, "kind": "explicit", "shape": list(explicit_case(name)[0].shape),
                 "settings": asdict(explicit_case(name)[2])} for name in EXPLICIT_NAMES]
    bench = [{"case_id": f"benchmark_{size}_{repeat}", "kind": "benchmark", "size": size,
              "repeat": repeat, "n_nodes": 16, "n_anchors": 2,
              "settings": asdict(Settings(radius=size))}
             for size in (128, 256, 512) for repeat in range(3)]
    real = [{"case_id": name.removesuffix(".npz"), "kind": "validation_roi", "input": name,
             "settings": asdict(Settings())} for name in PROB_NAMES] if include_prob else []
    return {"protocol_version": 2, "seed": SEED_VALUE, "budgets": list(BUDGETS),
            "tolerance": {"atol": ATOL, "rtol": RTOL, "budget_comparison": "exact <=, no tolerance"},
            "oracle": "symmetric explicit-zero CSR eight-neighbour pixel arcs; per-node all-boundary multi-source Dijkstra; Floyd min-max",
            "distance_checks": "root beta finite/+inf status and numeric values; NaN/-inf are failures; not each pair or drawing path",
            "owner_check": "every independent boundary source must retain its original node owner",
            "random_generator": "independent recorded per-case seeds; 4-pixel cells with a background moat; 1..3-pixel rectangles; no discarded cases",
            "roi": {"size": [256, 256], "center": "first predicted shoot 8CC centroid; image center if no shoot",
                    "rounding": "floor centroid, subtract 128, clip window origin to image bounds",
                    "decode": "production seven-class argmax", "min_area": 20, "min_peak": 0,
                    "other_filters": "disabled", "anchor_min_area": 0,
                    "skip": "more than 80 occupied nodes or no root; no replacement",
                    "gt": "not read", "purpose": "cropped graph algorithm validation, not full-image performance"},
            "benchmark": {"repeats": 3, "warmup": 0, "layout": "same seeded input in each repeat; 4x4 grid of 3x3 nodes",
                          "radius": "size (full pixel domain), separate synthetic scaling protocol",
                          "production_ms": "fragment_graph total only, excluding beta/forest",
                          "oracle_ms": "independent domain + CSR + all source propagations + complete distances + Floyd total",
                          "interpretation": "paired implementations, not system speedup; no stage timing or memory estimates"},
            "environment": environment(), "frozen_source_sha256": frozen_hashes(),
            "cases": explicit + random_specs() + real + bench}


def load_roi(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    metadata = {"input": path.name, "input_sha256": sha256(path)}
    with np.load(path, allow_pickle=False) as stored:
        metadata["npz_keys"] = list(stored.files)
        # The actual supplied format has been inspected; never read any GT key.
        if "p" not in stored.files:
            raise ValueError("missing inspected foreground probability key p")
        q = stored["p"]
    if q.ndim != 3 or q.shape[0] != 6 or q.dtype != np.uint8:
        raise ValueError("expected six uint8 foreground channels")
    metadata["probability_key"] = "p"
    metadata["input_shape"] = list(q.shape)
    predicted = decode(class_probabilities(q))
    h, w = predicted.shape
    if h < 256 or w < 256:
        raise ValueError("input is smaller than the fixed ROI")
    shoot, n_shoot = ndimage.label((predicted == SEED) | (predicted == HYPOCOTYL),
                                 structure=np.ones((3, 3), dtype=bool))
    if n_shoot:
        center = ndimage.center_of_mass(shoot == 1)
        metadata["roi_center_rule"] = "first_shoot_8cc"
    else:
        center = ((h - 1) / 2, (w - 1) / 2)
        metadata["roi_center_rule"] = "image_center"
    y0 = int(np.clip(np.floor(center[0]) - 128, 0, h - 256))
    x0 = int(np.clip(np.floor(center[1]) - 128, 0, w - 256))
    metadata["roi_center_yx"] = [float(v) for v in center]
    metadata["roi_coordinates"] = [y0, x0, y0 + 256, x0 + 256]
    crop = q[:, y0:y0 + 256, x0:x0 + 256].copy()
    probabilities = class_probabilities(crop)
    root_prob = probabilities[MAIN_ROOT] + probabilities[LATERAL_ROOT]
    label = size_filter(decode(probabilities), root_prob, min_area=20, min_peak=0, keep_largest=0)
    return label, root_prob, metadata


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return "inf" if value > 0 else ("-inf" if value < 0 else "nan")
    return value


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, ensure_ascii=True, indent=2, allow_nan=False)
        handle.write("\n")


CSV_FIELDS = (
    "case_id", "kind", "status", "reason", "height", "width", "input", "input_sha256",
    "case_sha256", "roi_y0", "roi_x0", "roi_y1", "roi_x1", "step", "lam", "radius", "anchor",
    "anchor_min_area", "n_nodes", "n_roots", "n_anchors", "finite_beta", "finite_beta_oracle",
    "finite_mismatch", "maxerr", "numerical_mismatch", "budgetmismatch", "owner_mismatch",
    "topology_mismatch", "sparse_edges", "complete_finite_edges", "sparse_density", "complete_density",
    "pixel_vertices", "pixel_arcs", "production_ms", "oracle_ms",
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new write-once directory outside source")
    parser.add_argument("--prob-dir", type=Path, help="optional inspected validation probability directory")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.is_relative_to(SOURCE):
        parser.error("output must be outside the source tree")
    if output.exists():
        parser.error("output already exists; write-once protocol forbids reuse")
    if not output.parent.is_dir():
        parser.error("output parent must already exist")
    output.mkdir(exist_ok=False)
    plan = protocol(args.prob_dir is not None)
    write_json(output / "protocol.json", plan)  # Prewritten before any case execution.
    details, rows = [], []
    wall_start = time.perf_counter()
    with (output / "cases.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        handle.flush()
        for spec in plan["cases"]:
            row = {"case_id": spec["case_id"], "kind": spec["kind"]}
            metadata, detail = {}, {}
            settings = Settings(**spec["settings"])
            row.update(asdict(settings))
            try:
                if spec["kind"] == "explicit":
                    label, prob, settings = explicit_case(spec["case_id"])
                elif spec["kind"] == "random":
                    label, prob, settings = random_case(spec)
                elif spec["kind"] == "benchmark":
                    label, prob, settings = benchmark_case(spec["size"])
                else:
                    row["input"] = spec["input"]
                    label, prob, metadata = load_roi(args.prob_dir.resolve() / spec["input"])
                    row.update({"input_sha256": metadata["input_sha256"]})
                    for field, value in zip(("roi_y0", "roi_x0", "roi_y1", "roi_x1"), metadata["roi_coordinates"]):
                        row[field] = value
                row.update(height=label.shape[0], width=label.shape[1], case_sha256=input_digest(label, prob))
                if spec["kind"] == "validation_roi":
                    domain = independent_domain(label, prob, settings)
                    row.update(n_nodes=len(domain.ids), n_roots=domain.n_roots, n_anchors=len(domain.anchors))
                    if domain.n_roots == 0 or len(domain.ids) > 80:
                        row.update(status="skipped", reason="no_root" if domain.n_roots == 0 else "nodes_gt_80")
                if row.get("status") != "skipped":
                    measured, detail = compare_case(label, prob, settings)
                    row.update(measured)
                    if spec["kind"] == "random" and row["n_nodes"] != spec["requested_nodes"]:
                        row.update(status="fail", reason="random_node_count_not_requested")
            except Exception as exc:
                # Never silently replace/drop a case; avoid absolute paths in public artifacts.
                row.update(status="error", reason=type(exc).__name__)
                detail["error"] = type(exc).__name__
            writer.writerow(row)
            handle.flush()
            rows.append(row)
            details.append({"case_id": spec["case_id"], "status": row["status"],
                            "metadata": metadata, "comparison": detail})
            print(f"{row['case_id']}: {row['status']}", flush=True)
    after_hashes = frozen_hashes()
    source_unchanged = after_hashes == plan["frozen_source_sha256"]
    failed = [r["case_id"] for r in rows if r["status"] in ("fail", "error")]
    groups = {}
    for kind in ("explicit", "random", "validation_roi", "benchmark"):
        subset = [r for r in rows if r["kind"] == kind]
        groups[kind] = {status: sum(r["status"] == status for r in subset)
                        for status in ("pass", "fail", "error", "skipped")}
    bench_summary = {}
    for size in (128, 256, 512):
        subset = [r for r in rows if r["kind"] == "benchmark" and r.get("height") == size
                  and "production_ms" in r]
        bench_summary[str(size)] = {"measured_repeats": len(subset),
                                   **{field + "_median": float(np.median([r[field] for r in subset]))
                                      for field in ("production_ms", "oracle_ms", "sparse_density", "complete_density")
                                      if subset}}
    summary = {"seed": SEED_VALUE, "total_cases": len(rows), "groups": groups, "failed_cases": failed,
               "skipped_cases": [{"case_id": r["case_id"], "reason": r["reason"]}
                                 for r in rows if r["status"] == "skipped"],
               "maxerr": max((r.get("maxerr", 0.0) for r in rows), default=0.0),
               "finite_mismatch": sum(r.get("finite_mismatch", 0) for r in rows),
               "budgetmismatch": sum(r.get("budgetmismatch", 0) for r in rows),
               "owner_mismatch": sum(r.get("owner_mismatch", 0) for r in rows),
               "source_unchanged": source_unchanged, "frozen_source_sha256_after": after_hashes,
               "benchmark": bench_summary, "verification_wall_seconds": time.perf_counter() - wall_start,
               "limitations": ["No GT, full-image performance, path equality, memory or stage timing claims.",
                               "Passing sampled cases is numerical evidence, not a proof of the proposition.",
                               "Benchmark radius=size is a separate full-domain synthetic protocol."]}
    write_json(output / "details.json", details)
    write_json(output / "summary.json", summary)
    print(json.dumps(json_safe(summary), ensure_ascii=True), flush=True)
    return int(bool(failed) or not source_unchanged)


if __name__ == "__main__":
    raise SystemExit(main())
