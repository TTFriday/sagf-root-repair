"""Measure probability-to-mask reconstruction latency on validation images only."""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

from toposeg.dataset import SeedlingDataset
from toposeg.reconstruction import ReconstructionParams, reconstruct


def cpu_name() -> str:
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name.Trim()"],
            capture_output=True, text=True, timeout=30, check=True)
        return result.stdout.strip()
    except Exception:
        return platform.processor() or "unknown"


def params(values: dict) -> ReconstructionParams:
    names = {field.name for field in fields(ReconstructionParams)}
    return ReconstructionParams(**{key: value for key, value in values.items() if key in names})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/baseline_region/seed_17"))
    parser.add_argument("--frozen", type=Path, default=Path("outputs/reconstruction/frozen.json"))
    parser.add_argument("--per-event", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reconstruction/runtime"))
    args = parser.parse_args()
    frozen = json.loads(args.frozen.read_text(encoding="utf-8"))
    entries = {
        "raw": {field.name: field.default for field in fields(ReconstructionParams)},
        "size filter": frozen["comparators"]["size"]["params"],
        "endpoint bridging": frozen["comparators"]["bridge"]["params"],
        "Euclidean anchoring": frozen["comparators"]["anchor"]["params"],
        "SAGF": frozen["selected"]["params"],
    }
    methods = {name: params(values) for name, values in entries.items()}
    frame = SeedlingDataset("manifests/splits/main.csv", "val").frame.sort_values(["event_id", "image_id"])
    picked = []
    for _, group in frame.groupby("event_id"):
        indices = np.unique(np.linspace(0, len(group) - 1, min(args.per_event, len(group))).round().astype(int))
        picked.append(group.iloc[indices])
    frame = pd.concat(picked)
    rows = []
    checksum = 0
    for row in frame.itertuples():
        quantised = np.load(args.run / "probs_val" / f"{row.image_id}.npz")["p"]
        for name, method in methods.items():
            reconstruct(quantised, method)
            elapsed = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                prediction = reconstruct(quantised, method)
                elapsed.append(1000 * (time.perf_counter() - start))
                checksum ^= int(prediction.sum())
            rows.append({"method": name, "image_id": row.image_id, "event_id": row.event_id,
                         "median_ms": statistics.median(elapsed), "min_ms": min(elapsed), "max_ms": max(elapsed),
                         "repeats": args.repeats})
        print(row.image_id, flush=True)
    per_image = pd.DataFrame(rows)
    grouped = per_image.groupby("method").median(numeric_only=True)[["median_ms", "min_ms", "max_ms"]]
    grouped["images"] = per_image.groupby("method").size()
    grouped = grouped.reset_index()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_image.to_csv(args.output_dir / "per_image.csv", index=False)
    grouped.to_csv(args.output_dir / "summary.csv", index=False)
    hardware = {"cpu": cpu_name(), "logical_processors": __import__("os").cpu_count(),
                "python": platform.python_version(), "platform": platform.platform(), "split": "validation",
                "disk_io_included": False, "warmup_runs_per_image_method": 1, "timed_repeats": args.repeats,
                "images_per_event_cap": args.per_event, "images": len(frame), "checksum": checksum}
    (args.output_dir / "hardware.json").write_text(json.dumps(hardware, indent=2), encoding="utf-8")
    print(grouped.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
