"""Descriptive OOD-split reconstruction evaluation (no confirmatory families)."""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from toposeg.dataset import SeedlingDataset
from toposeg.reconstruction import ReconstructionParams
from tune_reconstruction import evaluate_image, ORGANS


def params(values: dict) -> ReconstructionParams:
    from dataclasses import fields
    names = {field.name for field in fields(ReconstructionParams)}
    return ReconstructionParams(**{key: value for key, value in values.items() if key in names})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--grid", type=Path, default=Path("outputs/reconstruction/test_grid.json"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("outputs/reconstruction/ood.csv"))
    args = parser.parse_args()
    grid = {name: params(values) for name, values in json.loads(args.grid.read_text(encoding="utf-8")).items()}
    frame = SeedlingDataset("manifests/splits/ood.csv", "ood").frame.sort_values(["event_id", "image_id"])
    tasks = [(run, row.image_id, row.label_path, row.event_id,
              [int(getattr(row, f"has_class_{c}")) for c in ORGANS], "ood", grid)
             for run in args.runs for row in frame.itertuples()]
    print(f"{len(grid)} candidates, {len(tasks)} run-images", flush=True)
    start = time.time()
    with Pool(args.workers) as pool:
        for count, rows in enumerate(pool.imap_unordered(evaluate_image, tasks), 1):
            pd.DataFrame(rows).to_csv(args.output, mode="a", header=not args.output.is_file(), index=False)
            if count % 100 == 0:
                print(f"{count}/{len(tasks)} {time.time() - start:.0f}s", flush=True)
    print(f"done {len(tasks)} in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()