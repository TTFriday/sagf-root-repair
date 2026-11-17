from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from toposeg.constants import ROOT_CLASSES
from toposeg.dataset import SeedlingDataset, load_label
from toposeg.metrics import compute_metrics_fast, target_features
from toposeg.phenotyping import skeleton_points
from toposeg.reconstruction import ReconstructionParams, reconstruct_many

METRICS = ["complete_root_dice", "complete_root_cldice", "endpoint_f1", "branchpoint_f1",
           "component_count_error", "root_hd95_px", "skeleton_length_error_px",
           "skeleton_length_absolute_error_px", "largest_component_recall",
           "endpoint_precision", "endpoint_recall", "branchpoint_precision", "branchpoint_recall"]


def validation_arms(frozen: dict) -> dict[str, ReconstructionParams]:
    base = ReconstructionParams(**frozen["selected"]["params"])
    return {
        "raw": ReconstructionParams(),
        "size": ReconstructionParams(**frozen["comparators"]["size"]["params"]),
        "sagf": base,
        "no_drawing": replace(base, forest_width=0),
        "no_area_guard": replace(base, forest_keep_area=0),
        "no_prefilter": replace(base, min_area=0),
        "geometric_same_budget": replace(base, forest_lambda=0.0),
        "additive_same_budget": replace(base, forest_mode="additive"),
    }


def evaluate(task):
    run, image_id, label_path, event_id, present, grid = task
    probability = Path(run) / "probs_val" / f"{image_id}.npz"
    with np.load(probability) as stored:
        quantised = stored["p"]
    label = load_label(label_path)
    target = target_features(label)
    gt_tips, gt_branches = skeleton_points(np.isin(label, ROOT_CLASSES))
    names = list(grid)
    rows = []
    for index, prediction in reconstruct_many(quantised, [grid[name] for name in names]):
        scores = compute_metrics_fast(prediction, target)
        pred_tips, pred_branches = skeleton_points(np.isin(prediction, ROOT_CLASSES) & target.valid)
        row = {"run": Path(run).parent.name + "/" + Path(run).name,
               "image_id": image_id, "event_id": event_id, "candidate": names[index]}
        row.update({metric: scores[metric] for metric in METRICS})
        row["macro_positive_dice"] = float(np.mean([scores[f"dice_class_{cls}"]
                                                   for cls in range(1, 7) if present[cls - 1]]))
        row.update(pred_root_endpoints=pred_tips, gt_root_endpoints=gt_tips,
                   error_root_endpoints=pred_tips - gt_tips,
                   pred_root_branch_points=pred_branches, gt_root_branch_points=gt_branches,
                   error_root_branch_points=pred_branches - gt_branches)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Fixed-parameter exploratory validation diagnostics; no model selection.")
    parser.add_argument("--runs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--manifest", type=Path, default=Path("manifests/splits/main.csv"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--frozen", type=Path, default=Path("results/reconstruction/frozen.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/supplementary_validation"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--per-event", type=int, default=6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = json.loads(args.frozen.read_text(encoding="utf-8"))
    grid = validation_arms(frozen)
    frame = SeedlingDataset(args.manifest, "val", data_root=args.data_root).frame.sort_values(["event_id", "image_id"])
    chosen = []
    for _, event in frame.groupby("event_id"):
        indices = np.unique(np.linspace(0, len(event) - 1, min(args.per_event, len(event))).round().astype(int))
        chosen.append(event.iloc[indices])
    frame = pd.concat(chosen)
    seeds = (17, 29, 43)
    record = {
        "analysis_type": "post-hoc exploratory validation diagnostics; not independent confirmation",
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "selection": "none; all fixed arms are reported; original frozen configuration remains unchanged",
        "split": "val", "seeds": list(seeds), "images": frame.image_id.tolist(),
        "events": int(frame.event_id.nunique()), "per_event_cap": args.per_event,
        "frozen_sha256": hashlib.sha256(args.frozen.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "arms": {name: asdict(params) for name, params in grid.items()},
        "implementation_sha256": {
            "src/toposeg/reconstruction.py": hashlib.sha256((Path(__file__).resolve().parents[1] / "src/toposeg/reconstruction.py").read_bytes()).hexdigest(),
            "scripts/supplementary_validation.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "caveats": ["Validation images were previously used for tuning.",
                    "The probability-free arm keeps the numerical budget, not an equivalent physical gap length.",
                    "Count surplus/deficit are aggregate errors, not point-matched false/missed tips."]
    }
    (args.output / "protocol.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    tasks = []
    for seed in seeds:
        run = args.runs_root / "baseline_region" / f"seed_{seed}"
        for item in frame.itertuples():
            present = [int(getattr(item, f"has_class_{cls}")) for cls in range(1, 7)]
            tasks.append((str(run), item.image_id, item.label_path, item.event_id, present, grid))
    print(f"{len(tasks)} run-images, {len(grid)} fixed arms, {args.workers} workers", flush=True)
    with Pool(args.workers) as pool, (args.output / "per_image.csv").open("w", encoding="utf-8", newline="") as stream:
        for index, rows in enumerate(pool.imap(evaluate, tasks, chunksize=1)):
            pd.DataFrame(rows).to_csv(stream, index=False, header=index == 0)
            if (index + 1) % 10 == 0 or index + 1 == len(tasks):
                stream.flush()
                print(f"completed {index + 1}/{len(tasks)}", flush=True)


if __name__ == "__main__":
    main()
