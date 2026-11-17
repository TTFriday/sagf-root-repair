"""Count-level (tip/branch) phenotypes for frozen reconstruction candidates, same caliber as paper 16."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import fields
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd

from toposeg.constants import ROOT_CLASSES
from toposeg.dataset import SeedlingDataset, load_label
from toposeg.phenotyping import phenotype, skeleton_points
from toposeg.reconstruction import ReconstructionParams, reconstruct_many


def params(values: dict) -> ReconstructionParams:
    names = {field.name for field in fields(ReconstructionParams)}
    return ReconstructionParams(**{key: value for key, value in values.items() if key in names})


def evaluate_image(task):
    run, image_id, label_path, event_id, split, grid, mode = task
    quantised = np.load(Path(run) / f"probs_{split}" / f"{image_id}.npz")["p"]
    label = load_label(label_path)
    if mode == "counts":
        gt_ep, gt_br = skeleton_points(np.isin(label, ROOT_CLASSES))
        gt = {"root_endpoints": gt_ep, "root_branch_points": gt_br}
    else:
        gt = phenotype(label)
    names = list(grid)
    rows = []
    for index, mask in reconstruct_many(quantised, [grid[n] for n in names]):
        if mode == "counts":
            ep, br = skeleton_points(np.isin(mask, ROOT_CLASSES))
            pred = {"root_endpoints": ep, "root_branch_points": br}
        else:
            pred = phenotype(mask)
        row = {"run": run, "image_id": str(image_id), "event_id": str(event_id), "candidate": names[index]}
        for key in pred:
            row[f"pred_{key}"] = pred[key]
            row[f"gt_{key}"] = gt[key]
            row[f"error_{key}"] = pred[key] - gt[key]
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test", "ood"])
    parser.add_argument("--frozen", type=Path, default=Path("outputs/reconstruction/frozen.json"))
    parser.add_argument("--candidates", nargs="+",
                        default=["raw", "selected", "sagffilter", "sagfgeom", "sagfadd", "bridge"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--phenotypes", default="full", choices=["full", "counts"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frozen = json.loads(args.frozen.read_text(encoding="utf-8"))
    named = {entry["name"]: entry["params"] for entry in
             [frozen["selected"], *frozen["comparators"].values(), *frozen["ablations"].values()]}
    named["raw"] = ReconstructionParams().as_dict()
    alias = {"selected": frozen["selected"]["name"], "sagffilter": "forest_union_a20_s0.2_b2_w0_k1000",
             "sagfgeom": frozen["overall_best"]["name"], "sagfadd": "sagfadd_argmax_a20_s0.2_l1.0_r40_as0_b12_w1_k1000"}
    for key, entry in frozen["comparators"].items():
        alias[key] = entry["name"]
    grid = {}
    for cand in args.candidates:
        name = alias.get(cand, cand)
        grid[cand] = params(named[name])
    split_csv = "manifests/splits/main.csv" if args.split != "ood" else "manifests/splits/ood.csv"
    frame = SeedlingDataset(split_csv, args.split).frame.sort_values(["event_id", "image_id"])
    tasks = [(run, row.image_id, row.label_path, row.event_id, args.split, grid, args.phenotypes)
             for run in args.runs for row in frame.itertuples()]
    print(f"{len(grid)} candidates, {len(tasks)} run-images", flush=True)
    start = time.time()
    with Pool(args.workers) as pool:
        for count, rows in enumerate(pool.imap_unordered(evaluate_image, tasks), 1):
            pd.DataFrame(rows).to_csv(args.output, mode="a", header=not args.output.is_file(), index=False)
            if count % 200 == 0:
                print(f"{count}/{len(tasks)} {time.time() - start:.0f}s", flush=True)
    print(f"done {len(tasks)} in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()