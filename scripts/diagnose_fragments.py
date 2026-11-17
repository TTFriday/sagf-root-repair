from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.morphology import skeletonize

from toposeg.constants import HYPOCOTYL, IGNORE_INDEX, ROOT_CLASSES, SEED
from toposeg.dataset import SeedlingDataset, load_label

EIGHT = np.ones((3, 3), dtype=bool)


def describe(prob: np.ndarray, label: np.ndarray) -> tuple[list[dict], dict]:
    valid = label != IGNORE_INDEX
    full = np.concatenate([1.0 - prob.sum(0, keepdims=True), prob], 0)
    pred = full.argmax(0)
    root_prob = prob[0] + prob[1]
    root_pred = np.isin(pred, ROOT_CLASSES) & valid
    root_gt = np.isin(label, ROOT_CLASSES) & valid
    anchor = np.isin(pred, (HYPOCOTYL, SEED))
    dist_gt = ndimage.distance_transform_edt(~root_gt) if root_gt.any() else np.full(label.shape, np.inf)
    dist_anchor = ndimage.distance_transform_edt(~anchor) if anchor.any() else np.full(label.shape, np.inf)
    labels, count = ndimage.label(root_pred, EIGHT)
    rows = []
    if count:
        index = np.arange(1, count + 1)
        sizes = ndimage.sum(np.ones_like(labels), labels, index)
        overlap = ndimage.sum(root_gt, labels, index)
        min_gt = ndimage.minimum(dist_gt, labels, index)
        min_anchor = ndimage.minimum(dist_anchor, labels, index)
        mean_prob = ndimage.mean(root_prob, labels, index)
        for i in range(count):
            rows.append({"size": sizes[i], "overlap_px": overlap[i], "dist_gt": min_gt[i], "dist_anchor": min_anchor[i], "mean_prob": mean_prob[i]})
    gt_count = ndimage.label(root_gt, EIGHT)[1]
    gt_skel = skeletonize(root_gt)
    missed = gt_skel & ~root_pred
    stats = {
        "gt_components": gt_count,
        "pred_components": count,
        "gt_skeleton_px": int(gt_skel.sum()),
        "missed_skeleton_px": int(missed.sum()),
        "missed_prob_ge_0.1": int((missed & (root_prob >= 0.1)).sum()),
        "missed_prob_ge_0.25": int((missed & (root_prob >= 0.25)).sum()),
        "missed_prob_ge_0.4": int((missed & (root_prob >= 0.4)).sum()),
    }
    return rows, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = SeedlingDataset("manifests/splits/main.csv", args.split)
    lookup = dataset.frame.set_index("image_id")
    comp_rows, image_rows = [], []
    for path in sorted((args.run / f"probs_{args.split}").glob("*.npz")):
        image_id = path.stem
        prob = np.load(path)["p"].astype(np.float32) / 255.0
        label = load_label(lookup.loc[image_id, "label_path"])
        rows, stats = describe(prob, label)
        for row in rows:
            row["image_id"] = image_id
        comp_rows.extend(rows)
        image_rows.append({"image_id": image_id, "event_id": lookup.loc[image_id, "event_id"], **stats})
    args.output.mkdir(parents=True, exist_ok=True)
    comps, images = pd.DataFrame(comp_rows), pd.DataFrame(image_rows)
    comps.to_csv(args.output / "components.csv", index=False)
    images.to_csv(args.output / "images.csv", index=False)
    comps["fp"] = comps.overlap_px == 0
    print(images[["gt_components", "pred_components"]].describe().round(1).to_string())
    print("components:", len(comps), "false-positive (no GT overlap):", int(comps.fp.sum()))
    for lo, hi in ((0, 50), (50, 200), (200, 1000), (1000, 10**9)):
        sel = comps[(comps["size"] >= lo) & (comps["size"] < hi)]
        print(f"size [{lo},{hi}): n={len(sel)} fp={int(sel.fp.sum())} mean_prob={sel.mean_prob.mean():.3f}")
    tp = comps[~comps.fp]
    print("TP comps: dist_anchor quantiles", tp.dist_anchor.quantile([0.5, 0.9, 0.99]).round(0).tolist())
    fp = comps[comps.fp]
    print("FP comps: dist_gt quantiles", fp.dist_gt.quantile([0.1, 0.5, 0.9]).round(0).tolist(), "dist_anchor", fp.dist_anchor.quantile([0.1, 0.5, 0.9]).round(0).tolist())
    total = images.sum(numeric_only=True)
    print("missed skeleton fraction", round(total.missed_skeleton_px / total.gt_skeleton_px, 3),
          "of which prob>=0.1/0.25/0.4:", [round(total[k] / max(1, total.missed_skeleton_px), 3) for k in ("missed_prob_ge_0.1", "missed_prob_ge_0.25", "missed_prob_ge_0.4")])


if __name__ == "__main__":
    main()
