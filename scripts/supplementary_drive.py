from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from skimage.morphology import skeletonize

import drive_transfer
from toposeg.metrics import _skeleton_points, tolerant_point_metrics
from toposeg.reconstruction import ReconstructionParams, reconstruct_many


def drive_arms(frozen: dict) -> dict[str, ReconstructionParams]:
    base = ReconstructionParams(**frozen["selected"]["params"])
    arms = {"raw": ReconstructionParams()}
    for area in (1, 4, 10, 20, 40, 80):
        arms[f"size_{area}"] = ReconstructionParams(min_area=area)
    for radius in (1, 2):
        arms[f"closing_{radius}"] = ReconstructionParams(closing_radius=radius)
    for radius in (4, 8, 16):
        arms[f"sagf_r{radius}"] = replace(base, forest_anchor="largest", forest_radius=radius,
                                         min_area=1, forest_keep_area=40)
    return arms


def main():
    parser = argparse.ArgumentParser(description="Exploratory DRIVE repair comparators, using one shared checkpoint.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/drive_transfer/unet_drive.pt"))
    parser.add_argument("--frozen", type=Path, default=Path("results/reconstruction/frozen.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/supplementary_drive"))
    parser.add_argument("--split", default="val")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    drive_transfer.DATA = args.data_root
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = drive_transfer.SmallUNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()
    frozen = json.loads(args.frozen.read_text(encoding="utf-8"))
    arms = drive_arms(frozen)
    items = drive_transfer.load_split(args.split)
    record = {
        "analysis_type": "post-hoc exploratory single-checkpoint comparator evaluation",
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "selection": "none; all fixed size and closing variants are reported; SAGF remains unchanged",
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "frozen_sha256": hashlib.sha256(args.frozen.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "split_folder": args.split, "images": [name for name, _, _ in items],
        "backend": {"device": str(device), "torch": torch.__version__,
                    "cudnn": torch.backends.cudnn.version(), "deterministic_algorithms": True},
        "arms": {name: asdict(params) for name, params in arms.items()},
        "evaluation_region": "entire supplied image; official field-of-view masks unavailable in this mirror",
        "point_tolerances_px": [2.0, 5.0],
        "caveats": ["One network checkpoint; no retraining-based uncertainty estimate.",
                    "Archived checkpoint initialization seed was not recorded; inference is reproducible conditional on its hash.",
                    "Spatial tolerances and full-frame measurements are not a clinical or standard DRIVE ranking."]
    }
    (args.output / "protocol.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    rows = []
    for name, image, gt in items:
        h, w = image.shape
        padded = np.zeros(((h + 3) // 4 * 4, (w + 3) // 4 * 4), dtype=np.float32)
        padded[:h, :w] = image
        with torch.inference_mode():
            probability = torch.sigmoid(model(torch.from_numpy(padded[None, None]).to(device)))[0, 0].cpu().numpy()[:h, :w]
        quantised = np.zeros((6,) + image.shape, dtype=np.uint8)
        quantised[0] = np.rint(probability * 255).astype(np.uint8)
        gt = gt.astype(bool)
        gt_endpoints, gt_branches = _skeleton_points(skeletonize(gt))
        names = list(arms)
        for index, label in reconstruct_many(quantised, [arms[arm] for arm in names]):
            pred = label.astype(bool)
            row = {"image_id": name, "arm": names[index], **drive_transfer.vessel_metrics(pred, gt)}
            pred_endpoints, pred_branches = _skeleton_points(skeletonize(pred))
            row["endpoint_f1_5px"] = tolerant_point_metrics(pred_endpoints, gt_endpoints, tolerance=5.0)["f1"]
            row["branchpoint_f1_5px"] = tolerant_point_metrics(pred_branches, gt_branches, tolerance=5.0)["f1"]
            rows.append(row)
        print(f"completed image {name}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "per_image.csv", index=False)
    frame.groupby("arm").agg(
        dice=("dice", "mean"), cldice=("cldice", "mean"),
        endpoint_mae=("endpoint_error", lambda values: values.abs().mean()),
        endpoint_bias=("endpoint_error", "mean"), branch_mae=("branch_error", lambda values: values.abs().mean()),
        endpoint_f1=("endpoint_f1", "mean"), endpoint_f1_5px=("endpoint_f1_5px", "mean"),
        branchpoint_f1=("branchpoint_f1", "mean"), branchpoint_f1_5px=("branchpoint_f1_5px", "mean")
    ).reset_index().to_csv(args.output / "summary.csv", index=False)
    print("all fixed comparator results archived", flush=True)


if __name__ == "__main__":
    main()
