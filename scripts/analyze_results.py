from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from toposeg.stats import (
    LOWER_IS_BETTER, ORGANS, POST_HOC_EXPERIMENTS, REQUIRED_SEEDS, favour_label, load_metrics, paired_by_event, select_evaluation_paths,
)

TOPOLOGY_METRICS = ["complete_root_dice", "complete_root_cldice", "endpoint_f1", "branchpoint_f1", "skeleton_length_absolute_error_px"]
REGION_METRICS = ["macro_positive_dice", "root_hd95_px", "root_asd_px", "largest_component_recall", "component_count_error"]


def check_seeds(data: pd.DataFrame, baseline: str, allow_partial: bool) -> None:
    incomplete = {
        experiment: sorted(REQUIRED_SEEDS - set(group.seed))
        for experiment, group in data.groupby("experiment")
        if set(group.seed) != REQUIRED_SEEDS
    }
    if incomplete and not allow_partial:
        raise RuntimeError(f"Three-seed results are incomplete: {incomplete}")
    if allow_partial:
        print(f"WARNING: debugging mode, partial seeds accepted: {incomplete}")
    if baseline not in set(data.experiment):
        raise RuntimeError(f"Baseline {baseline!r} not found")


def organ_table(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (experiment, seed), group in data.groupby(["experiment", "seed"]):
        for cls, name in ORGANS.items():
            present = group[group[f"has_class_{cls}"] == 1]
            rows.append({
                "experiment": experiment, "seed": seed, "organ": name, "positive_images": len(present),
                "dice_positive": present[f"dice_class_{cls}"].mean(),
                "iou_positive": present[f"iou_class_{cls}"].mean(),
                "precision_positive": present[f"precision_class_{cls}"].mean(),
                "recall_positive": present[f"recall_class_{cls}"].mean(),
                "dice_all_images": group[f"dice_class_{cls}"].mean(),
            })
    frame = pd.DataFrame(rows)
    grouped = frame.groupby(["experiment", "organ"]).agg(
        positive_images=("positive_images", "mean"),
        dice_positive_mean=("dice_positive", "mean"), dice_positive_sd=("dice_positive", "std"),
        iou_positive_mean=("iou_positive", "mean"), iou_positive_sd=("iou_positive", "std"),
        precision_positive_mean=("precision_positive", "mean"), recall_positive_mean=("recall_positive", "mean"),
        dice_all_images_mean=("dice_all_images", "mean"),
    ).reset_index()
    grouped["inflation_from_empty_classes"] = grouped["dice_all_images_mean"] - grouped["dice_positive_mean"]
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate three-seed evaluations and paired event-level statistics")
    parser.add_argument("--input", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis"))
    parser.add_argument("--split-csv", type=Path, default=Path("manifests/splits/main.csv"))
    parser.add_argument("--baseline", default="baseline_region")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--noninferiority-margin", type=float, default=0.005)
    parser.add_argument("--evaluation", default="evaluation_test",
                        help="Only read results from evaluation directories with this name, so that the nominal and out-of-domain diagnostics cannot mix into the protocol analysis")
    parser.add_argument("--exclude", nargs="*", default=list(POST_HOC_EXPERIMENTS),
                        help="Experiments kept out of this analysis; the post-hoc comparator has its own family")
    parser.add_argument("--debug-partial", action="store_true", help="Accept fewer than three seeds; never writes into paper/")
    args = parser.parse_args()
    if args.debug_partial and "paper" in str(args.output).replace("\\", "/").split("/"):
        raise RuntimeError("--debug-partial must not write publication tables")

    paths = select_evaluation_paths(sorted(args.input.glob("**/per_image_metrics.csv")), args.evaluation)
    data = load_metrics(paths, args.split_csv)
    data = data[~data.experiment.isin(args.exclude)]
    check_seeds(data, args.baseline, args.debug_partial)
    metrics = [metric for metric in TOPOLOGY_METRICS + REGION_METRICS if metric in data]
    if not metrics:
        raise RuntimeError("No supported metric columns found")

    seed_means = data.groupby(["experiment", "seed"])[metrics].mean().reset_index()
    summary = seed_means.groupby("experiment")[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()

    events = data.groupby(["experiment", "seed", "event_id"])[metrics].mean().reset_index()
    rng = np.random.default_rng(20261016)
    tests = pd.concat([paired_by_event(events, args.baseline, metric, rng, args.iterations) for metric in metrics], ignore_index=True)
    tests["direction"] = np.where(tests["metric"].isin(LOWER_IS_BETTER), "lower", "higher")
    tests["favors"] = favour_label(tests, LOWER_IS_BETTER)
    tests["noninferior_at_margin"] = np.nan
    region = tests["metric"] == "macro_positive_dice"
    tests.loc[region, "noninferior_at_margin"] = (tests.loc[region, "ci95_lower"] > -args.noninferiority_margin).astype(float)

    organs = organ_table(data)

    args.output.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output / "per_image_all_runs.csv", index=False)
    seed_means.to_csv(args.output / "seed_means.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    organs.to_csv(args.output / "organ_table.csv", index=False)
    tests.to_csv(args.output / "paired_tests.csv", index=False)

    print(args.output)


if __name__ == "__main__":
    main()
