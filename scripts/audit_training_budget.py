from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

EXPERIMENTS = {
    "Region": "baseline_region",
    "clDice": "baseline_cldice",
    "Skeleton Recall": "baseline_skelrecall",
    "MultiTopo": "full",
    "MultiTopo-U": "ablation_uniform",
    "MultiTopo-J": "ablation_no_joint_root",
    "MultiTopo-H": "ablation_no_skeleton_head",
}
SEEDS = (17, 29, 43)
CRITERION = "val_complete_root_cldice"


def audit(runs_root: Path, output: Path) -> list[dict]:
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "analysis": "training-history audit, not checkpoint re-evaluation",
        "max_steps": 30000,
        "patience_validation_epochs": 20,
        "validation_schedule": "after each training epoch",
        "selection": f"first strict maximum of {CRITERION}",
        "seeds": SEEDS,
        "experiments": EXPERIMENTS,
        "checkpoint_hash_binding_checked": False,
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    records = []
    histories = []
    for objective, experiment in EXPERIMENTS.items():
        for seed in SEEDS:
            relative = Path(experiment) / f"seed_{seed}" / "training.csv"
            path = runs_root / relative
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            best_index = max(range(len(rows)), key=lambda index: float(rows[index][CRITERION]))
            best, last = rows[best_index], rows[-1]
            records.append({
                "objective": objective,
                "experiment": experiment,
                "seed": seed,
                "stopping_step": int(last["step"]),
                "selected_step": int(best["step"]),
                "stopping_epoch": int(last["epoch"]),
                "selected_epoch": int(best["epoch"]),
                "selected_val_cldice": float(best[CRITERION]),
                "history_relative_path": relative.as_posix(),
                "history_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
            for row in rows:
                histories.append({
                    "objective": objective,
                    "seed": seed,
                    "epoch": int(row["epoch"]),
                    "step": int(row["step"]),
                    CRITERION: float(row[CRITERION]),
                })
    for filename, data in (("budget.csv", records), ("selection_histories.csv", histories)):
        with (output / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    summary = {
        "runs": len(records),
        "reached_cap": sum(row["stopping_step"] == 30000 for row in records),
        "stopped_early": sum(row["stopping_step"] < 30000 for row in records),
        "stopping_step_range": [min(row["stopping_step"] for row in records), max(row["stopping_step"] for row in records)],
        "selected_step_range": [min(row["selected_step"] for row in records), max(row["selected_step"] for row in records)],
        "checkpoint_hash_binding_checked": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    audit(args.runs_root, args.output)


if __name__ == "__main__":
    main()
