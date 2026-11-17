from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from toposeg.stats import holm_adjust

PRIMARY = ["complete_root_cldice", "endpoint_f1", "branchpoint_f1", "component_count_error", "root_hd95_px"]
LOWER = {"component_count_error", "root_hd95_px", "tip_mae", "branch_mae"}


def exact_sign_flip(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    signs = np.array(list(itertools.product((-1, 1), repeat=len(values))), dtype=float)
    observed = abs(values.mean())
    null = np.abs((signs * values).mean(axis=1))
    return float(np.mean(null >= observed - 1e-12))


def paired_summary(values: np.ndarray, rng: np.random.Generator) -> dict:
    values = np.asarray(values, dtype=float)
    draws = rng.integers(0, len(values), (20000, len(values)))
    lower, upper = np.percentile(values[draws].mean(axis=1), [2.5, 97.5])
    return {"mean_difference": float(values.mean()), "ci95_lower": float(lower), "ci95_upper": float(upper),
            "exact_permutation_p": exact_sign_flip(values), "paired_events": len(values)}


def count_values(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    column = "root_endpoints" if kind == "tip" else "root_branch_points"
    present = frame[frame[f"gt_{column}"] > 0].copy()
    error = present[f"error_{column}"]
    present[f"{kind}_mae"] = error.abs()
    present[f"{kind}_surplus"] = error.clip(lower=0)
    present[f"{kind}_deficit"] = -error.clip(upper=0)
    return present


def aligned_candidates(frame: pd.DataFrame, candidates: list[str]) -> dict[str, pd.DataFrame]:
    keys = ["run", "image_id", "event_id"]
    arms = {candidate: frame[frame.candidate == candidate].set_index(keys).sort_index() for candidate in candidates}
    base = arms[candidates[0]]
    if base.empty or base.index.has_duplicates:
        raise ValueError("Reference coverage is empty or duplicated")
    for candidate, arm in arms.items():
        if arm.index.has_duplicates or not arm.index.equals(base.index):
            raise ValueError(f"Incomplete or mismatched coverage for {candidate}")
    runs = base.index.get_level_values("run").unique()
    for experiment in {run.rsplit("/", 1)[0] for run in runs}:
        seeds = {run.rsplit("/", 1)[1] for run in runs if run.rsplit("/", 1)[0] == experiment}
        if seeds != {"seed_17", "seed_29", "seed_43"}:
            raise ValueError(f"All three seeds are required for {experiment}")
    image_sets = [set(base.xs(run, level="run").index) for run in runs]
    if any(images != image_sets[0] for images in image_sets[1:]):
        raise ValueError("Runs must cover identical image/event keys")
    return arms


def region_differences(arm: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    arm = arm.loc[arm.index.get_level_values("run").str.contains("baseline_region/")]
    raw = raw.loc[raw.index.get_level_values("run").str.contains("baseline_region/")]
    return (arm[PRIMARY] - raw[PRIMARY]).groupby(level="event_id").mean()


def main():
    parser = argparse.ArgumentParser(description="Event-balanced exploratory count and robustness analyses; no selection.")
    parser.add_argument("--input", type=Path, default=Path("results/reconstruction"))
    parser.add_argument("--output", type=Path, default=Path("outputs/count_analysis"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = json.loads((args.input / "frozen.json").read_text(encoding="utf-8"))
    selected = frozen["selected"]["name"]
    size = frozen["comparators"]["size"]["name"]
    count_paths = [args.input / "test_counts.csv", args.input / "test_counts_comparators.csv"]
    counts = pd.concat([pd.read_csv(path) for path in count_paths], ignore_index=True)
    aliases = {selected: "sagf", "selected": "sagf", "size": "size", "bridge": "bridge", "anchor": "anchor"}
    aliases.update({item["name"]: key for key, item in frozen["comparators"].items()})
    counts["candidate"] = counts.candidate.replace(aliases)
    counts = counts[counts.run.str.contains("baseline_region/")].copy()
    count_arms = aligned_candidates(counts, ["raw", "sagf", "size", "bridge", "anchor"])
    for arm in count_arms.values():
        if not arm[["gt_root_endpoints", "gt_root_branch_points"]].equals(count_arms["raw"][["gt_root_endpoints", "gt_root_branch_points"]]):
            raise ValueError("Target counts must agree across candidates")
    if counts.event_id.nunique() != 12:
        raise ValueError("The frozen count analysis requires all 12 test events")
    rng = np.random.default_rng(20261117)
    image_rows, event_rows, paired_rows = [], [], []
    for kind in ("tip", "branch"):
        data = count_values(counts, kind)
        columns = [f"{kind}_mae", f"{kind}_surplus", f"{kind}_deficit"]
        for candidate, group in data.groupby("candidate"):
            means = group.groupby("event_id")[columns].mean()
            image_rows.append({"candidate": candidate, "kind": kind, "run_images": len(group), **group[columns].mean().to_dict()})
            event_rows.append({"candidate": candidate, "kind": kind, "events": len(means), **means.mean().to_dict()})
        wide = data.pivot(index=["run", "image_id", "event_id"], columns="candidate", values=f"{kind}_mae")
        if wide[["raw", "sagf", "size", "bridge", "anchor"]].isna().any().any():
            raise ValueError("Count contrasts contain missing observations")
        for reference in ("raw", "size", "bridge", "anchor"):
            paired = (wide.sagf - wide[reference]).groupby(level="event_id").mean().to_numpy()
            paired_rows.append({"metric": f"{kind}_mae", "reference": reference,
                                **paired_summary(paired, rng), "events_favouring_sagf": int((paired < 0).sum())})
    pd.DataFrame(image_rows).to_csv(args.output / "count_summary_image.csv", index=False)
    pd.DataFrame(event_rows).to_csv(args.output / "count_summary_event.csv", index=False)
    paired_counts = pd.DataFrame(paired_rows)
    paired_counts["holm_p_8_exploratory_contrasts"] = holm_adjust(paired_counts.exact_permutation_p)
    paired_counts.to_csv(args.output / "paired_counts.csv", index=False)
    test = pd.read_csv(args.input / "test.csv")
    test_arms = aligned_candidates(test, [selected, "raw", size])
    arm, raw, tuned_size = test_arms[selected], test_arms["raw"], test_arms[size]
    if arm.index.get_level_values("run").nunique() != 21:
        raise ValueError("Cross-objective analysis requires all 21 trained runs")
    differences = region_differences(arm, raw)
    loo = []
    for omitted in differences.index:
        for metric in PRIMARY:
            values = differences.drop(index=omitted)[metric].dropna().to_numpy()
            loo.append({"omitted_event": omitted, "metric": metric, "mean_difference": values.mean(),
                        "same_direction_as_full": bool((values.mean() > 0) == (differences[metric].mean() > 0))})
    pd.DataFrame(loo).to_csv(args.output / "leave_one_event_out.csv", index=False)
    pooled_rows = []
    for metric in PRIMARY:
        values = (arm[metric] - tuned_size[metric]).groupby(level="event_id").mean().dropna().to_numpy()
        result = paired_summary(values, rng)
        pooled_rows.append({"metric": metric, **result, "events_favouring_sagf": int(((values < 0) if metric in LOWER else (values > 0)).sum())})
    pooled = pd.DataFrame(pooled_rows)
    pooled["holm_p_5_exploratory_endpoints"] = holm_adjust(pooled.exact_permutation_p)
    pooled.to_csv(args.output / "size_across_objectives.csv", index=False)
    ood = pd.read_csv(args.input / "ood.csv")
    ood[ood.candidate.isin(["raw", selected, size])].groupby("candidate")[PRIMARY + ["complete_root_dice", "macro_positive_dice"]].mean().to_csv(args.output / "unmapped_descriptive.csv")
    protocol = {
        "analysis_type": "post-hoc exploratory; original confirmatory inference unchanged",
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "inputs_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in count_paths + [args.input / "test.csv", args.input / "ood.csv", args.input / "frozen.json"]},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "statistics": "seed/image means within event; 20000 event bootstrap resamples; exact 2^n sign flips",
        "holm_families": {"counts": 8, "pooled_objective_size_comparison": 5},
        "seed": 20261117,
        "selection": "none; counts, leave-one-event-out estimates and all five pooled size contrasts reported",
        "unmapped_inference": "none; acquisition dependence unavailable",
        "count_definition": "positive/negative signed count error, not spatially matched false-positive/false-negative points"
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print("event-balanced counts, exact exploratory tests, robustness and unmapped comparator tables saved")
    print(pd.DataFrame(event_rows).round(4).to_string(index=False))
    print(paired_counts.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
