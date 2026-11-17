"""Confirmatory test-split analysis of the frozen reconstruction configuration.

Acquisition events are the unit of inference: per (seed, image) differences between two arms are
averaged within each test event, and the event means are tested with a bootstrap CI and a sign-flip
permutation test. Holm correction is applied within each pre-declared family:

  primary      selected vs raw on the region-loss baseline, five topology endpoints
  models       selected vs raw on every other trained configuration
  comparators  each comparator vs raw, and selected vs each comparator, on the region-loss baseline
  losses       region-loss baseline + selected vs each topology-loss model with raw decoding
  ablations    each ablation vs selected on the region-loss baseline
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from select_reconstruction import load_protocols, load_results
from toposeg.dataset import SeedlingDataset
from toposeg.stats import holm_adjust, paired_test

TOPOLOGY = ["complete_root_cldice", "endpoint_f1", "branchpoint_f1", "component_count_error", "root_hd95_px"]
GUARDS = ["complete_root_dice", "macro_positive_dice"]
EXTRA = ["skeleton_length_error_px", "skeleton_length_absolute_error_px", "largest_component_recall",
         "endpoint_precision", "endpoint_recall", "branchpoint_precision", "branchpoint_recall", "root_asd_px"]
LOWER_IS_BETTER = {"component_count_error", "root_hd95_px", "root_asd_px", "skeleton_length_absolute_error_px"}
BASELINE = "baseline_region"
EXPECTED_EXPERIMENTS = [
    BASELINE,
    "baseline_cldice",
    "baseline_skelrecall",
    "full",
    "ablation_uniform",
    "ablation_no_joint_root",
    "ablation_no_skeleton_head",
]
TOPOLOGY_LOSSES = ["baseline_cldice", "baseline_skelrecall", "full"]
EXPECTED_SEEDS = {17, 29, 43}
ANALYSIS_IMPLEMENTATIONS = (
    Path("scripts/analyze_reconstruction.py"),
    Path("scripts/select_reconstruction.py"),
    Path("src/toposeg/stats.py"),
)


def split_run(data: pd.DataFrame) -> pd.DataFrame:
    parts = data["run"].str.replace("\\", "/", regex=False).str.split("/")
    return data.assign(experiment=parts.str[-2], seed=parts.str[-1].str.replace("seed_", "").astype(int))


def summary(data: pd.DataFrame) -> pd.DataFrame:
    metrics = TOPOLOGY + GUARDS + EXTRA
    per_seed = (data.groupby(["experiment", "candidate", "seed", "event_id"])[metrics].mean()
                .groupby(["experiment", "candidate", "seed"]).mean())
    grouped = per_seed.groupby(["experiment", "candidate"])
    out = grouped.mean().add_suffix("_mean").join(grouped.std(ddof=1).add_suffix("_sd"))
    return out.reset_index()


def compare(data: pd.DataFrame, arm: tuple[str, str], reference: tuple[str, str], metrics: list[str],
            rng: np.random.Generator, iterations: int) -> list[dict]:
    keys = ["seed", "image_id", "event_id"]
    a = data[(data.experiment == arm[0]) & (data.candidate == arm[1])].set_index(keys)
    b = data[(data.experiment == reference[0]) & (data.candidate == reference[1])].set_index(keys)
    if a.empty or b.empty:
        raise RuntimeError(f"Missing arm {arm} or reference {reference}")
    joined = a[metrics].join(b[metrics], lsuffix="_a", rsuffix="_b", how="inner")
    if len(joined) != len(a) or len(joined) != len(b):
        raise RuntimeError(f"Unpaired images between {arm} and {reference}")
    rows = []
    for metric in metrics:
        difference = (joined[f"{metric}_a"] - joined[f"{metric}_b"]).groupby(level="event_id").mean()
        lower, upper, p_value = paired_test(difference.to_numpy(), rng, iterations)
        better = difference < 0 if metric in LOWER_IS_BETTER else difference > 0
        rows.append({"arm": f"{arm[0]}|{arm[1]}", "reference": f"{reference[0]}|{reference[1]}", "metric": metric,
                     "events": len(difference), "events_favouring_arm": int(better.sum()),
                     "mean_difference": float(difference.mean()), "ci95_lower": lower, "ci95_upper": upper,
                     "permutation_p": p_value})
    return rows


def family(name: str, pairs: list[tuple[tuple[str, str], tuple[str, str]]], data: pd.DataFrame,
           rng: np.random.Generator, iterations: int) -> pd.DataFrame:
    rows = []
    for arm, reference in pairs:
        rows += compare(data, arm, reference, TOPOLOGY, rng, iterations)
    frame = pd.DataFrame(rows)
    frame.insert(0, "family", name)
    frame["holm_p"] = holm_adjust(frame["permutation_p"])
    guards = pd.DataFrame([row for arm, reference in pairs
                           for row in compare(data, arm, reference, GUARDS + EXTRA, rng, iterations)])
    guards.insert(0, "family", f"{name}_descriptive")
    return pd.concat([frame, guards], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--frozen", type=Path, default=Path("outputs/reconstruction/frozen.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reconstruction/test_analysis"))
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261113)
    args = parser.parse_args()
    if args.iterations != 10000 or args.seed != 20261113:
        raise SystemExit("Confirmatory analysis requires --iterations 10000 and --seed 20261113")
    frozen = json.loads(args.frozen.read_text(encoding="utf-8"))
    frozen_sha256 = hashlib.sha256(args.frozen.read_bytes()).hexdigest()
    protocols = load_protocols(args.results, "test")
    manifest = SeedlingDataset("manifests/splits/main.csv", "test").frame
    expected_images = {str(row.image_id): str(row.event_id) for row in manifest.itertuples()}
    if len(expected_images) != 201 or len(set(expected_images.values())) != 12:
        raise RuntimeError("The current test manifest does not match the pre-declared split")
    for item in protocols:
        images = item.get("images", [])
        observed_images = {entry["image_id"]: entry["event_id"] for entry in images}
        if (item.get("frozen_sha256") != frozen_sha256 or item.get("per_event") != 0
                or len(images) != len(observed_images) or observed_images != expected_images):
            raise RuntimeError("Confirmatory analysis requires complete test protocols bound to this frozen file")
    selected = frozen["selected"]["name"]
    comparators = [entry["name"] for entry in frozen["comparators"].values()]
    ablations = [entry["name"] for entry in frozen["ablations"].values()]
    region_candidates = {"raw", selected, frozen["overall_best"]["name"], *comparators, *ablations}
    transfer_candidates = {"raw", selected}
    data = split_run(load_results(args.results, "test"))
    if data.duplicated(["run", "image_id", "candidate"]).any():
        raise RuntimeError("Confirmatory inputs contain duplicate candidate rows")
    experiments = set(data.experiment.unique())
    if experiments != set(EXPECTED_EXPERIMENTS):
        raise RuntimeError(f"Expected exactly the seven pre-declared experiments, found {sorted(experiments)}")
    for experiment in EXPECTED_EXPERIMENTS:
        subset = data[data.experiment == experiment]
        if set(subset.seed.unique()) != EXPECTED_SEEDS:
            raise RuntimeError(f"Expected seeds 17, 29 and 43 for {experiment}")
        required = region_candidates if experiment == BASELINE else transfer_candidates
        for seed in EXPECTED_SEEDS:
            group = subset[subset.seed == seed]
            if (group.run.nunique() != 1 or set(group.candidate) != required
                    or group.image_id.nunique() != 201 or group.event_id.nunique() != 12
                    or len(group) != 201 * len(required)):
                raise RuntimeError(f"Incomplete pre-declared candidates for {experiment}, seed {seed}")
    rng = np.random.default_rng(args.seed)
    families = [
        family("primary", [((BASELINE, selected), (BASELINE, "raw"))], data, rng, args.iterations),
        family("models", [((experiment, selected), (experiment, "raw"))
                          for experiment in EXPECTED_EXPERIMENTS if experiment != BASELINE],
               data, rng, args.iterations),
        family("comparators", [((BASELINE, c), (BASELINE, "raw")) for c in comparators]
               + [((BASELINE, selected), (BASELINE, c)) for c in comparators], data, rng, args.iterations),
        family("losses", [((BASELINE, selected), (experiment, "raw")) for experiment in TOPOLOGY_LOSSES],
               data, rng, args.iterations),
        family("ablations", [((BASELINE, a), (BASELINE, selected)) for a in ablations], data, rng, args.iterations),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    analysis_protocol = {
        "schema_version": 1,
        "frozen": str(args.frozen).replace("\\", "/"),
        "frozen_sha256": frozen_sha256,
        "results": [{"path": str(path).replace("\\", "/"),
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "protocol_sha256": hashlib.sha256(path.with_suffix(".protocol.json").read_bytes()).hexdigest()}
                    for path in args.results],
        "iterations": args.iterations,
        "seed": args.seed,
        "experiments": EXPECTED_EXPERIMENTS,
        "seeds": sorted(EXPECTED_SEEDS),
        "implementation_sha256": {str(path).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in ANALYSIS_IMPLEMENTATIONS},
    }
    record = args.output_dir / "analysis.protocol.json"
    paired_path = args.output_dir / "paired_tests.csv"
    summary_path = args.output_dir / "summary.csv"
    if record.is_file():
        if json.loads(record.read_text(encoding="utf-8")) != analysis_protocol:
            raise RuntimeError("Cannot rerun confirmatory analysis with a different protocol")
    else:
        if paired_path.exists() or summary_path.exists():
            raise RuntimeError("Existing confirmatory outputs have no protocol record")
        with record.open("x", encoding="utf-8") as handle:
            json.dump(analysis_protocol, handle, indent=2)
    tests = pd.concat(families, ignore_index=True)
    tests.to_csv(paired_path, index=False)
    table = summary(data)
    table.to_csv(summary_path, index=False)
    pd.set_option("display.width", 250)
    confirm = tests[~tests.family.str.endswith("_descriptive")]
    print(confirm[["family", "arm", "reference", "metric", "events_favouring_arm", "mean_difference",
                   "ci95_lower", "ci95_upper", "holm_p"]].round(4).to_string())


if __name__ == "__main__":
    main()
