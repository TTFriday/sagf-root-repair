from __future__ import annotations

import argparse
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.morphology import skeletonize

from toposeg.constants import IGNORE_INDEX, ROOT_CLASSES
from toposeg.dataset import SeedlingDataset, load_label
from toposeg.metrics import _skeleton_points, tolerant_point_metrics
from toposeg.point_matching import maximum_cardinality_metrics
from toposeg.reconstruction import ReconstructionParams, reconstruct_many


def evaluate(task):
    run, image_id, label_path, event_id, base = task
    target = load_label(label_path)
    valid = target != IGNORE_INDEX
    gt_root = np.isin(target, ROOT_CLASSES) & valid
    gt_end, gt_branch = _skeleton_points(skeletonize(gt_root))
    with np.load(Path(run) / "probs_test" / f"{image_id}.npz") as stored:
        quantised = stored["p"]
    rows = []
    for index, prediction in reconstruct_many(quantised, [ReconstructionParams(), base]):
        pred_end, pred_branch = _skeleton_points(skeletonize(np.isin(prediction, ROOT_CLASSES) & valid))
        row = {"run": Path(run).parent.name + "/" + Path(run).name,
               "image_id": image_id, "event_id": event_id, "candidate": ("raw", "sagf")[index]}
        for kind, predicted, gt in (("endpoint", pred_end, gt_end), ("branchpoint", pred_branch, gt_branch)):
            for tolerance in (2.0, 5.0, 10.0):
                original = tolerant_point_metrics(predicted, gt, tolerance)
                revised = maximum_cardinality_metrics(predicted, gt, tolerance)
                row[f"{kind}_original_{int(tolerance)}px"] = original["f1"]
                row[f"{kind}_maxcard_{int(tolerance)}px"] = revised["f1"]
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Exploratory point-matching sensitivity; frozen confirmatory metrics unchanged.")
    parser.add_argument("--runs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("manifests/splits/main.csv"))
    parser.add_argument("--frozen", type=Path, default=Path("results/reconstruction/frozen.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/point_matching_sensitivity"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frame = SeedlingDataset(args.manifest, "test", data_root=args.data_root).frame
    base = ReconstructionParams(**json.loads(args.frozen.read_text(encoding="utf-8"))["selected"]["params"])
    implementation = Path(__file__).resolve().parents[1] / "src" / "toposeg"
    record = {
        "analysis_type": "post-hoc exploratory sensitivity to point-assignment definition",
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "selection": "none; both methods and all point tolerances reported; original confirmatory metrics unchanged",
        "seeds": [17, 29, 43], "frames": frame.image_id.tolist(), "events": int(frame.event_id.nunique()),
        "frozen_sha256": hashlib.sha256(args.frozen.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "implementation_sha256": {name: hashlib.sha256((implementation / name).read_bytes()).hexdigest()
                                   for name in ("reconstruction.py", "metrics.py", "point_matching.py")},
        "probability_maps_sha256": {
            f"baseline_region/seed_{seed}/{image_id}.npz": hashlib.sha256(
                (args.runs_root / "baseline_region" / f"seed_{seed}" / "probs_test" / f"{image_id}.npz").read_bytes()).hexdigest()
            for seed in (17, 29, 43) for image_id in frame.image_id},
        "metric": "maximum-cardinality bipartite matching within tolerance; compare to distance assignment then rejection",
        "point_tolerances_px": [2, 5, 10]
    }
    (args.output / "protocol.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    tasks = [(str(args.runs_root / "baseline_region" / f"seed_{seed}"), item.image_id,
              item.label_path, item.event_id, base) for seed in (17, 29, 43) for item in frame.itertuples()]
    print(f"{len(tasks)} run-images, two frozen decoders, three matching tolerances", flush=True)
    with Pool(args.workers) as pool, (args.output / "per_image.csv").open("w", encoding="utf-8", newline="") as stream:
        for index, rows in enumerate(pool.imap(evaluate, tasks, chunksize=1)):
            pd.DataFrame(rows).to_csv(stream, index=False, header=index == 0)
            if (index + 1) % 25 == 0 or index + 1 == len(tasks):
                stream.flush()
                print(f"completed {index + 1}/{len(tasks)}", flush=True)
    values = pd.read_csv(args.output / "per_image.csv")
    metrics = [column for column in values if column.startswith(("endpoint_", "branchpoint_"))]
    values.groupby(["candidate", "event_id"])[metrics].mean().groupby(level="candidate").mean().to_csv(args.output / "event_summary.csv")


if __name__ == "__main__":
    main()
