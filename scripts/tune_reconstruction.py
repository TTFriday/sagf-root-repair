"""Evaluate inference-time root reconstruction candidates on stored class probabilities.

Validation split only: test-split probabilities are refused unless --allow-test is given, which the
frozen confirmatory evaluation (evaluate_reconstruction.py) is the only intended user of.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from toposeg.dataset import SeedlingDataset, load_label
from toposeg.metrics import compute_metrics_fast, target_features
from toposeg.reconstruction import ReconstructionParams as P
from toposeg.reconstruction import reconstruct_many

KEEP = ["complete_root_dice", "complete_root_cldice", "endpoint_f1", "branchpoint_f1", "component_count_error",
        "root_hd95_px", "root_asd_px", "skeleton_length_error_px", "skeleton_length_absolute_error_px",
        "largest_component_recall", "endpoint_precision", "endpoint_recall", "branchpoint_precision", "branchpoint_recall"]
ORGANS = range(1, 7)


def stage1_grid() -> dict[str, P]:
    grid = {"raw": P()}
    for t in (1.0, 0.4, 0.3):
        grid[f"union_t{t}"] = P(decode="union", root_threshold=t)
    for low in (0.2, 0.3, 0.4):
        grid[f"hyst_{low}"] = P(hysteresis_low=low)
    for r in (1, 2, 3, 5):
        grid[f"close_{r}"] = P(closing_radius=r)
    for area, peak in itertools.product((10, 20, 30, 50, 80, 120), (0.0, 0.8, 0.9)):
        grid[f"size_a{area}_p{peak}"] = P(min_area=area, min_peak=peak)
    for radius, prob in itertools.product((15, 25, 40), (0.05, 0.15, 0.3)):
        grid[f"bridge_r{radius}_p{prob}"] = P(bridge_radius=radius, bridge_min_prob=prob)
    for anchor, chain in itertools.product((3, 10, 20), (0, 10, 20, 40)):
        grid[f"anchor_d{anchor}_c{chain}"] = P(anchor_distance=anchor, chain_distance=chain)
    for area, anchor, chain in itertools.product((20, 30), (10,), (20, 40)):
        grid[f"size{area}+anchor_d{anchor}_c{chain}"] = P(min_area=area, anchor_distance=anchor, chain_distance=chain)
    for mode, area, step, budget, width, keep in itertools.product(
            ("argmax", "union"), (0, 20), (0.05, 0.2, 0.5), (2, 4, 6, 8, 12, 16), (0, 1, 3), (0, 1000)):
        grid[f"forest_{mode}_a{area}_s{step}_b{budget}_w{width}_k{keep}"] = P(
            decode=mode, min_area=area, forest_step=step, forest_budget=float(budget), forest_width=width, forest_keep_area=keep)
    return grid


GRIDS = {"stage1": stage1_grid}


def load_grid(spec: str) -> dict[str, P]:
    if spec in GRIDS:
        return GRIDS[spec]()
    return {name: P(**values) for name, values in json.loads(Path(spec).read_text(encoding="utf-8")).items()}


def evaluate_image(task):
    run, image_id, label_path, event_id, present, split, grid = task
    names = list(grid)
    quantised = np.load(Path(run) / f"probs_{split}" / f"{image_id}.npz")["p"]
    features = target_features(load_label(label_path))
    rows = []
    for index, prediction in reconstruct_many(quantised, [grid[n] for n in names]):
        metrics = compute_metrics_fast(prediction, features)
        row = {"run": run, "image_id": image_id, "event_id": event_id, "candidate": names[index]}
        row.update({key: metrics[key] for key in KEEP})
        dice = [metrics[f"dice_class_{cls}"] for cls in ORGANS if present[cls - 1]]
        row["macro_positive_dice"] = float(np.mean(dice)) if dice else np.nan
        rows.append(row)
    return rows


IMPLEMENTATION_PATHS = (
    Path("src/toposeg/reconstruction.py"),
    Path("src/toposeg/metrics.py"),
    Path("src/toposeg/dataset.py"),
    Path("src/toposeg/constants.py"),
    Path("src/toposeg/phenotyping.py"),
    Path("scripts/tune_reconstruction.py"),
)
FROZEN_IMPLEMENTATIONS = {
    "src/toposeg/reconstruction.py",
    "src/toposeg/metrics.py",
    "src/toposeg/dataset.py",
    "src/toposeg/constants.py",
    "src/toposeg/phenotyping.py",
    "scripts/tune_reconstruction.py",
    "scripts/select_reconstruction.py",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_run(run: str) -> str:
    return str(Path(run.replace("\\", "/"))).replace("\\", "/").rstrip("/").casefold()


def select_images(split: str, per_event: int) -> pd.DataFrame:
    frame = SeedlingDataset("manifests/splits/main.csv", split).frame.sort_values(["event_id", "image_id"])
    if per_event:
        picked = []
        for _, group in frame.groupby("event_id"):
            indices = np.unique(np.linspace(0, len(group) - 1, min(per_event, len(group))).round().astype(int))
            picked.append(group.iloc[indices])
        frame = pd.concat(picked)
    return frame


def verify_frozen(path: Path) -> dict:
    frozen = json.loads(path.read_text(encoding="utf-8"))
    validation = frozen.get("validation", {})
    if (frozen.get("schema_version") != 1 or validation.get("test_used_for_reconstruction_selection") is not False
            or validation.get("images_per_event_cap") != 6 or validation.get("images") != 59
            or validation.get("events") != 11 or validation.get("candidates") != 180):
        raise SystemExit("The frozen file does not describe the pre-declared Stage-2 validation protocol")
    hashes = {name.replace("\\", "/"): value
              for name, value in validation.get("implementation_sha256", {}).items()}
    if (set(hashes) != FROZEN_IMPLEMENTATIONS or not validation.get("grid_sha256")
            or not validation.get("grid_protocol_sha256") or not validation.get("results_sha256")):
        raise SystemExit("The frozen file has an incomplete provenance record")
    for name, expected in hashes.items():
        source = Path(name)
        if not source.is_file() or sha256(source) != expected:
            raise SystemExit(f"Implementation differs from the frozen validation version: {name}")
    return frozen


def protocol(args: argparse.Namespace, grid: dict[str, P], frame: pd.DataFrame) -> dict:
    labels = {}
    for row in frame.itertuples():
        actual = sha256(Path(row.label_path))
        if actual != str(row.label_sha256):
            raise SystemExit(f"Label differs from its manifest hash: {row.image_id}")
        labels[str(row.image_id)] = actual
    probabilities = {
        run: {str(image_id): sha256(Path(run) / f"probs_{args.split}" / f"{image_id}.npz")
              for image_id in frame.image_id}
        for run in args.runs
    }
    record = {
        "schema_version": 1,
        "split": args.split,
        "grid": {name: asdict(params) for name, params in grid.items()},
        "runs": args.runs,
        "per_event": args.per_event,
        "images": [{"image_id": str(row.image_id), "event_id": str(row.event_id)} for row in frame.itertuples()],
        "manifest_sha256": sha256(Path("manifests/splits/main.csv")),
        "label_sha256": labels,
        "probability_sha256": probabilities,
        "implementation_sha256": {str(path).replace("\\", "/"): sha256(path) for path in IMPLEMENTATION_PATHS},
    }
    grid_path = Path(args.grid)
    if grid_path.is_file():
        record["grid_sha256"] = sha256(grid_path)
    if args.split == "test":
        record["frozen_sha256"] = sha256(args.frozen)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--frozen", type=Path, help="Frozen validation selection, required for test evaluation")
    parser.add_argument("--grid", default="stage1", help="Named grid or JSON file {name: params}")
    parser.add_argument("--per-event", type=int, default=0, help="Evenly spaced images per event (0 = all)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.runs = [normalize_run(run) for run in args.runs]
    if len(set(args.runs)) != len(args.runs):
        raise SystemExit("Duplicate runs after path normalization")
    if args.split == "test" and (not args.allow_test or not args.frozen or args.per_event):
        raise SystemExit("Test confirmation requires --allow-test, --frozen and all images (--per-event 0)")
    grid = load_grid(args.grid)
    frame = select_images(args.split, args.per_event)
    if args.split == "val" and args.grid == "stage1":
        runs = [run.replace("\\", "/") for run in args.runs]
        if runs != ["outputs/baseline_region/seed_17"] or args.per_event != 4 or len(frame) != 42 or len(grid) != 486:
            raise SystemExit("Stage 1 requires Region seed 17, --per-event 4, 42 images and 486 candidates")
    if args.split == "test":
        frozen = verify_frozen(args.frozen)
        entries = [frozen["selected"], frozen["overall_best"], *frozen["comparators"].values(), *frozen["ablations"].values()]
        permitted = {entry["name"]: entry["params"] for entry in entries}
        permitted["raw"] = asdict(P())
        if any(name not in permitted or asdict(params) != permitted[name] for name, params in grid.items()):
            raise SystemExit("The test grid may contain only exact configurations from the frozen validation selection")
    run_protocol = protocol(args, grid, frame)
    record = args.output.with_suffix(".protocol.json")
    if record.is_file():
        if json.loads(record.read_text(encoding="utf-8")) != run_protocol:
            raise SystemExit("Cannot resume evaluation with a different protocol")
    else:
        if args.output.exists():
            raise SystemExit("Existing results without a protocol record cannot be resumed")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with record.open("x", encoding="utf-8") as handle:
            json.dump(run_protocol, handle, indent=2)
    done = set()
    expected_pairs = {(run, str(image_id)) for run in args.runs for image_id in frame.image_id}
    if args.output.is_file():
        previous = pd.read_csv(args.output, usecols=["run", "image_id", "candidate"])
        if previous.duplicated(["run", "image_id", "candidate"]).any():
            raise SystemExit("Existing results contain duplicate candidate rows")
        observed_pairs = set(zip(previous.run, previous.image_id.astype(str)))
        counts = previous.groupby(["run", "image_id"]).candidate.nunique()
        if not observed_pairs.issubset(expected_pairs) or not (counts == len(grid)).all():
            raise SystemExit("Existing results contain unexpected or incomplete run-images")
        done = observed_pairs
    tasks = [(run, row.image_id, row.label_path, row.event_id, [int(getattr(row, f"has_class_{c}")) for c in ORGANS], args.split, grid)
             for run in args.runs for row in frame.itertuples() if (run, str(row.image_id)) not in done]
    print(f"{len(grid)} candidates, {len(tasks)} run-images pending, {len(done)} done", flush=True)
    start = time.time()
    with Pool(args.workers) as pool:
        for count, rows in enumerate(pool.imap_unordered(evaluate_image, tasks), 1):
            pd.DataFrame(rows).to_csv(args.output, mode="a", header=not args.output.is_file(), index=False)
            print(f"{count}/{len(tasks)} {rows[0]['run']} {rows[0]['image_id']} {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
