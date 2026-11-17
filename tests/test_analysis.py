import numpy as np
import pandas as pd
import pytest
from pathlib import Path

import scripts.analyze_phenotypes as phenotype_analysis
import scripts.analyze_repair_counts as count_analysis
import scripts.analyze_protocol as protocol_analysis
import scripts.analyze_results as result_analysis

ORGANS = (1, 2, 3, 4, 5, 6)
SEEDS = (17, 29, 43)
EVENTS = 6
EXPERIMENTS = ("baseline_region", "full")


def metric_value(experiment: str, gain: float, base_cldice: float, noise: float) -> tuple[float, float]:
    """Return (complete-root clDice, complete-root length absolute error); error falls with topology gain."""
    if experiment == "baseline_region":
        return base_cldice, 200.0 + noise
    return base_cldice + gain, max(1.0, 200.0 - 1500.0 * gain + noise)


def make_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(7)
    metrics, phenotypes, split_rows = [], [], []
    for event in range(EVENTS):
        event_id = f"event_{event}"
        for image in range(2):
            image_id = f"image_{event}_{image}"
            present = {cls: int((event + cls + image) % 3 != 0) for cls in ORGANS}
            split_rows.append({"image_id": image_id, "event_id": event_id, **{f"has_class_{cls}": value for cls, value in present.items()}})
            ground_truth = 2000.0 if image == 0 else 0.0
            for experiment in EXPERIMENTS:
                for seed in SEEDS:
                    gain = 0.01 * (event + 1) + 0.004 * rng.normal(0, 1)
                    base_cldice = 0.40 + 0.001 * event
                    cldice, length_error = metric_value(experiment, gain, base_cldice, float(rng.normal(0, 2)))
                    root_dice = 0.45 + 0.001 * event + (0.2 * gain + 0.003 * rng.normal(0, 1) if experiment == "full" else 0.0)
                    dice = {cls: (0.5 if present[cls] else 1.0) for cls in ORGANS}
                    metrics.append({
                        "image_id": image_id, "experiment": experiment, "seed": seed, "event_id": event_id, "condition": "Roots",
                        **{f"dice_class_{cls}": dice[cls] for cls in ORGANS},
                        **{f"iou_class_{cls}": dice[cls] / 2 for cls in ORGANS},
                        **{f"precision_class_{cls}": dice[cls] for cls in ORGANS},
                        **{f"recall_class_{cls}": dice[cls] for cls in ORGANS},
                        "complete_root_dice": root_dice,
                        "complete_root_cldice": cldice,
                        "endpoint_f1": 0.2 + (0.05 if experiment == "full" else 0.0),
                        "branchpoint_f1": 0.1 + (0.05 if experiment == "full" else 0.0),
                        "skeleton_length_absolute_error_px": length_error,
                        "root_hd95_px": 60.0, "root_asd_px": 12.0,
                        "largest_component_recall": 0.8, "component_count_error": 3.0,
                    })
                    endpoints = (12, 20) if experiment == "baseline_region" else (12, 14)
                    branches = (4, 7) if experiment == "baseline_region" else (4, 5)
                    phenotypes.append({
                        "image_id": image_id, "experiment": experiment, "seed": seed, "event_id": event_id,
                        "pred_main_root_length_px": 700.0 + length_error, "gt_main_root_length_px": 700.0,
                        "error_main_root_length_px": length_error, "absolute_error_main_root_length_px": length_error,
                        "pred_lateral_root_length_px": 300.0 + length_error, "gt_lateral_root_length_px": 300.0 if image == 0 else 0.0,
                        "error_lateral_root_length_px": length_error, "absolute_error_lateral_root_length_px": length_error,
                        "pred_complete_root_length_px": ground_truth + length_error, "gt_complete_root_length_px": ground_truth,
                        "error_complete_root_length_px": length_error, "absolute_error_complete_root_length_px": length_error,
                        "pred_root_endpoints": endpoints[1], "gt_root_endpoints": endpoints[0],
                        "error_root_endpoints": endpoints[1] - endpoints[0], "absolute_error_root_endpoints": abs(endpoints[1] - endpoints[0]),
                        "pred_root_branch_points": branches[1], "gt_root_branch_points": branches[0],
                        "error_root_branch_points": branches[1] - branches[0], "absolute_error_root_branch_points": abs(branches[1] - branches[0]),
                    })
    return pd.DataFrame(metrics), pd.DataFrame(phenotypes), pd.DataFrame(split_rows)


def write_runs(root, frame: pd.DataFrame, filename: str) -> None:
    for (experiment, seed), group in frame.groupby(["experiment", "seed"]):
        directory = root / experiment / f"seed_{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        group.to_csv(directory / filename, index=False)


@pytest.fixture
def prepared(tmp_path):
    metrics, phenotypes, split = make_frames()
    split.to_csv(tmp_path / "main.csv", index=False)
    write_runs(tmp_path, metrics, "per_image_metrics.csv")
    write_runs(tmp_path, phenotypes, "phenotype_errors.csv")
    return tmp_path, metrics, phenotypes, split


def test_macro_positive_dice_ignores_absent_organs(prepared):
    tmp_path, _, _, split = prepared
    frame = result_analysis.load_metrics([tmp_path / "full" / "seed_17" / "per_image_metrics.csv"], tmp_path / "main.csv")
    for index in (0, 3, 7):
        row = frame.iloc[index]
        source = split[split.image_id == row.image_id].iloc[0]
        expected = [row[f"dice_class_{cls}"] for cls in ORGANS if source[f"has_class_{cls}"] == 1]
        assert np.isclose(row.macro_positive_dice, np.mean(expected))
    assert (frame.macro_all_dice >= frame.macro_positive_dice).all()


def test_analyze_results_writes_tables_and_paired_tests(prepared, monkeypatch):
    tmp_path, output = prepared[0], prepared[0] / "analysis"
    monkeypatch.setattr("sys.argv", [
        "analyze_results", "--input", str(tmp_path), "--output", str(output),
        "--split-csv", str(tmp_path / "main.csv"), "--iterations", "500",
    ])
    result_analysis.main()
    tests = pd.read_csv(output / "paired_tests.csv")
    assert set(tests.experiment) == {"full"}
    assert {"holm_p", "ci95_lower", "ci95_upper", "noninferior_at_margin"} <= set(tests)
    topology = tests[tests.metric == "complete_root_cldice"]
    assert (topology.mean_difference > 0).all() and (topology.paired_events == EVENTS).all()
    noninferior = tests[tests.metric == "macro_positive_dice"].noninferior_at_margin
    assert noninferior.notna().all() and set(noninferior) <= {0.0, 1.0}
    region = tests[tests.metric == "macro_positive_dice"].iloc[0]
    assert region.noninferior_at_margin == float(region.ci95_lower > -0.005)
    errors = tests[tests.metric == "skeleton_length_absolute_error_px"]
    assert (errors.direction == "lower").all() and (errors.favors == "experiment").all()
    for name in ("summary.csv", "organ_table.csv", "seed_means.csv", "per_image_all_runs.csv", "paired_tests.csv"):
        assert (output / name).is_file(), name
    assert not list(output.glob("*.tex"))
    assert not list(output.glob("*.png"))
    assert not list(output.glob("*.pdf"))
    organs = pd.read_csv(output / "organ_table.csv")
    assert organs.inflation_from_empty_classes.ge(0).all()


def test_named_evaluation_excludes_other_protocols(prepared, monkeypatch):
    """A published-protocol diagnostic stored beside the test run must not be pooled into it."""
    tmp_path, metrics, phenotypes, _ = prepared
    root = tmp_path / "runs"
    write_runs(root, metrics, "per_image_metrics.csv")
    write_runs(root, phenotypes, "phenotype_errors.csv")
    for directory in sorted(root.glob("*/*/per_image_metrics.csv")) + sorted(root.glob("*/*/phenotype_errors.csv")):
        nominal = directory.parent / "evaluation_nominal"
        nominal.mkdir(exist_ok=True)
        directory.rename(nominal / directory.name)
    for directory in sorted(root.glob("*/*/evaluation_nominal")):
        test = directory.parent / "evaluation_test"
        test.mkdir()
        for name in ("per_image_metrics.csv", "phenotype_errors.csv"):
            (test / name).write_bytes((directory / name).read_bytes())
    expected = len(metrics)
    monkeypatch.setattr("sys.argv", [
        "analyze_results", "--input", str(root), "--output", str(tmp_path / "test_only"),
        "--split-csv", str(tmp_path / "main.csv"), "--iterations", "500",
    ])
    result_analysis.main()
    assert len(pd.read_csv(tmp_path / "test_only" / "per_image_all_runs.csv")) == expected
    monkeypatch.setattr("sys.argv", [
        "analyze_results", "--input", str(root), "--output", str(tmp_path / "nominal_only"),
        "--split-csv", str(tmp_path / "main.csv"), "--iterations", "500", "--evaluation", "evaluation_nominal",
    ])
    result_analysis.main()
    assert len(pd.read_csv(tmp_path / "nominal_only" / "per_image_all_runs.csv")) == expected


def test_incomplete_seed_coverage_is_rejected(prepared, monkeypatch):
    tmp_path, metrics, _, _ = prepared
    partial = tmp_path / "partial"
    partial.mkdir()
    metrics[metrics.seed == 17].to_csv(partial / "per_image_metrics.csv", index=False)
    monkeypatch.setattr("sys.argv", ["x", "--input", str(partial), "--output", str(tmp_path / "out"), "--split-csv", str(tmp_path / "main.csv")])
    with pytest.raises(RuntimeError, match="incomplete"):
        result_analysis.main()
    monkeypatch.setattr("sys.argv", ["x", "--input", str(partial), "--output", str(tmp_path / "out"), "--split-csv", str(tmp_path / "main.csv"), "--debug-partial"])
    result_analysis.main()


def test_debug_partial_cannot_write_publication_tables(prepared, monkeypatch):
    tmp_path = prepared[0]
    monkeypatch.setattr("sys.argv", ["x", "--input", str(tmp_path), "--output", str(tmp_path / "paper" / "tables"), "--debug-partial"])
    with pytest.raises(RuntimeError, match="publication"):
        result_analysis.main()


def test_phenotype_summary_and_association(prepared, monkeypatch):
    tmp_path, output = prepared[0], prepared[0] / "pheno"
    monkeypatch.setattr("sys.argv", ["analyze_phenotypes", "--input", str(tmp_path), "--output", str(output), "--iterations", "500"])
    phenotype_analysis.main()
    summary = pd.read_csv(output / "phenotype_summary.csv")
    assert set(summary.phenotype) == set(phenotype_analysis.PHENO_COLUMNS)
    complete = summary[summary.phenotype == "complete_root_length"].set_index("experiment")
    assert complete.loc["full", "images_without_ground_truth"] == 6 * len(SEEDS)
    assert complete.loc["full", "phantom_rate"] == 1.0
    assert complete.loc["full", "mae"] < complete.loc["baseline_region", "mae"]
    paired = pd.read_csv(output / "phenotype_paired.csv")
    length = paired[paired.metric == "absolute_error_complete_root_length_px"]
    assert (length.mean_difference < 0).all() and (length.paired_events == EVENTS).all()
    link = pd.read_csv(output / "topology_phenotype_association.csv")
    fitted = link[link.status == "ok"]
    assert not fitted.empty, link.status.unique().tolist()
    assert "joint" in set(fitted.model)
    topology = fitted[fitted.term == "delta_cldice"]
    image_unit = topology[topology.unit == "seed_x_image"]
    assert (image_unit.estimate < 0).all() and (image_unit.ci95_upper < 0).any()
    event_unit = topology[topology.unit == "event"]
    assert not event_unit.empty and np.isfinite(event_unit.estimate).all()
    assert not list(output.glob("*.tex"))
    assert not list(output.glob("*.png"))
    assert not list(output.glob("*.pdf"))


def write_evaluation(root: Path, frame: pd.DataFrame, filename: str, evaluation: str) -> None:
    """Write one result file per experiment and seed inside a named evaluation directory."""
    for (experiment, seed), group in frame.groupby(["experiment", "seed"]):
        directory = root / experiment / f"seed_{seed}" / evaluation
        directory.mkdir(parents=True, exist_ok=True)
        group.to_csv(directory / filename, index=False)


def test_protocol_separates_memorised_from_unseen_images(prepared, monkeypatch, capsys):
    tmp_path, metrics, _, split = prepared
    runs = tmp_path / "protocol_runs"
    roles = {"event_0": "train", "event_1": "train", "event_2": "val", "event_3": "val", "event_4": "test", "event_5": "test"}
    split.assign(frozen_split=split.event_id.map(roles)).to_csv(tmp_path / "nominal.csv", index=False)
    unseen_ids = split[split.event_id.isin(["event_4", "event_5"])].image_id
    split[split.image_id.isin(unseen_ids)].to_csv(tmp_path / "main.csv", index=False)
    write_evaluation(runs, metrics, "per_image_metrics.csv", "evaluation_nominal")
    write_evaluation(runs, metrics[metrics.image_id.isin(unseen_ids)], "per_image_metrics.csv", "evaluation_test")
    output = tmp_path / "analysis_protocol"
    monkeypatch.setattr("sys.argv", [
        "analyze_protocol", "--input", str(runs), "--splits", str(tmp_path), "--output", str(output), "--iterations", "500",
    ])
    protocol_analysis.main()
    groups = pd.read_csv(output / "protocol_groups.csv")
    assert set(groups.group) == {"nominal_all", "nominal_trained", "nominal_validation", "nominal_unseen", "leak_free_test"}
    counts = groups[groups.metric == "complete_root_cldice"].set_index("group").images
    assert counts.to_dict() == {"nominal_all": 36, "nominal_trained": 12, "nominal_validation": 12, "nominal_unseen": 12, "leak_free_test": 12}
    assert groups.set_index("group").events.loc[["nominal_trained", "nominal_validation", "nominal_unseen", "leak_free_test"]].eq(2).all()
    trained = groups[(groups.experiment == "full") & (groups.group == "nominal_trained") & (groups.metric == "complete_root_cldice")]["mean"].iloc[0]
    reference = metrics[(metrics.experiment == "full") & metrics.event_id.isin(["event_0", "event_1"])]
    assert np.isclose(trained, reference.complete_root_cldice.mean())
    contrasts = pd.read_csv(output / "protocol_contrasts.csv")
    assert set(contrasts.contrast) == {"nominal_trained - nominal_unseen", "nominal_validation - nominal_unseen"}
    assert set(contrasts.events_a) == {2} and set(contrasts.events_b) == {2}
    errors = contrasts[contrasts.metric == "skeleton_length_absolute_error_px"]
    assert (errors.better_group == np.where(errors.mean_difference > 0, "nominal_unseen", "nominal_trained")).all()
    assert "evaluation_ood" in capsys.readouterr().out


@pytest.mark.parametrize("values,expected", [
    ([1.0, 2.0, 3.0], 0.25),
    ([-1.0, -2.0, -3.0], 0.25),
    ([1.0, -1.0], 1.0),
    ([0.0, 0.0, 0.0], 1.0),
    ([2.0], 1.0),
    ([1.0, 0.0], 1.0),
])
def test_exact_sign_flip_is_two_sided_and_includes_ties(values, expected):
    assert count_analysis.exact_sign_flip(np.array(values)) == expected


@pytest.mark.parametrize("kind,column", [("tip", "root_endpoints"), ("branch", "root_branch_points")])
def test_count_estimands_are_gt_positive_surplus_and_deficit(kind, column):
    frame = pd.DataFrame({
        "image_id": ["phantom", "surplus", "deficit", "exact"],
        f"gt_{column}": [0, 2, 4, 3],
        f"error_{column}": [100, 3, -2, 0],
    })
    original = frame.copy(deep=True)
    result = count_analysis.count_values(frame, kind)
    assert result.image_id.tolist() == ["surplus", "deficit", "exact"]
    assert result[f"{kind}_mae"].tolist() == [3, 2, 0]
    assert result[f"{kind}_surplus"].tolist() == [3, 0, 0]
    assert result[f"{kind}_deficit"].tolist() == [0, 2, 0]
    assert result[f"{kind}_mae"].mean() == pytest.approx(5 / 3)
    assert result[f"{kind}_surplus"].mean() == 1
    assert result[f"{kind}_deficit"].mean() == pytest.approx(2 / 3)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("kind,column", [("tip", "root_endpoints"), ("branch", "root_branch_points")])
def test_count_estimands_with_no_gt_positive_images(kind, column):
    frame = pd.DataFrame({f"gt_{column}": [0], f"error_{column}": [5]})
    result = count_analysis.count_values(frame, kind)
    assert result.empty
    assert {f"{kind}_mae", f"{kind}_surplus", f"{kind}_deficit"} <= set(result)


def test_paired_count_summary_bootstraps_events_not_run_images():
    # Nine images in one event must not outweigh the single image in another event.
    frame = pd.DataFrame({"event_id": ["a"] * 9 + ["b"], "difference": [1.0] * 9 + [3.0]})
    values = frame.groupby("event_id").difference.mean().to_numpy()
    result = count_analysis.paired_summary(values, np.random.default_rng(17))
    assert result == count_analysis.paired_summary(values, np.random.default_rng(17))
    assert result["mean_difference"] == 2.0
    assert result["paired_events"] == 2
    assert result["exact_permutation_p"] == 0.5
    assert result["ci95_lower"] == 1.0 and result["ci95_upper"] == 3.0


@pytest.fixture
def fixed_frozen():
    from dataclasses import asdict
    from toposeg.reconstruction import ReconstructionParams

    return {
        "selected": {"params": asdict(ReconstructionParams(
            min_area=3, forest_budget=12.0, forest_radius=30, forest_lambda=2.5,
            forest_width=2, forest_keep_area=200,
        ))},
        "comparators": {"size": {"params": asdict(ReconstructionParams(min_area=15))}},
    }


def test_validation_arms_change_only_the_declared_parameter(fixed_frozen):
    import copy
    from dataclasses import replace
    from scripts.supplementary_validation import validation_arms
    from toposeg.reconstruction import ReconstructionParams

    original = copy.deepcopy(fixed_frozen)
    base = ReconstructionParams(**fixed_frozen["selected"]["params"])
    expected = {
        "raw": ReconstructionParams(),
        "size": ReconstructionParams(min_area=15),
        "sagf": base,
        "no_drawing": replace(base, forest_width=0),
        "no_area_guard": replace(base, forest_keep_area=0),
        "no_prefilter": replace(base, min_area=0),
        "geometric_same_budget": replace(base, forest_lambda=0.0),
        "additive_same_budget": replace(base, forest_mode="additive"),
    }
    assert validation_arms(fixed_frozen) == expected
    assert fixed_frozen == original


def test_drive_arms_report_every_fixed_variant_without_mutating_frozen(fixed_frozen, monkeypatch):
    import copy
    import importlib
    import sys
    from dataclasses import replace
    import scripts.drive_transfer as drive
    from toposeg.reconstruction import ReconstructionParams

    # The CLI imports its sibling by script name; expose that name only in this test.
    monkeypatch.setitem(sys.modules, "drive_transfer", drive)
    supplementary = importlib.import_module("scripts.supplementary_drive")
    original = copy.deepcopy(fixed_frozen)
    base = ReconstructionParams(**fixed_frozen["selected"]["params"])
    expected = {"raw": ReconstructionParams()}
    expected.update({f"size_{area}": ReconstructionParams(min_area=area) for area in (1, 4, 10, 20, 40, 80)})
    expected.update({f"closing_{radius}": ReconstructionParams(closing_radius=radius) for radius in (1, 2)})
    expected.update({f"sagf_r{radius}": replace(base, forest_anchor="largest", forest_radius=radius,
                                             min_area=1, forest_keep_area=40) for radius in (4, 8, 16)})
    assert supplementary.drive_arms(fixed_frozen) == expected
    assert fixed_frozen == original


def test_count_cli_rejects_missing_comparator_seed(tmp_path, monkeypatch):
    import json

    rows = []
    for candidate in ("raw", "sagf", "size", "bridge", "anchor"):
        for seed in SEEDS:
            for event in range(12):
                rows.append({"candidate": candidate, "run": f"baseline_region/seed_{seed}",
                             "image_id": f"image_{event}", "event_id": f"event_{event}",
                             "gt_root_endpoints": 2, "gt_root_branch_points": 1,
                             "error_root_endpoints": 5, "error_root_branch_points": 2})
    counts = pd.DataFrame(rows)
    counts[counts.candidate.isin(["raw", "sagf"])].to_csv(tmp_path / "test_counts.csv", index=False)
    comparators = counts[counts.candidate.isin(["size", "bridge", "anchor"])]
    comparators = comparators[~((comparators.candidate == "size") & comparators.run.str.endswith("seed_43"))]
    comparators.to_csv(tmp_path / "test_counts_comparators.csv", index=False)
    (tmp_path / "frozen.json").write_text(json.dumps({"selected": {"name": "sagf"},
        "comparators": {name: {"name": name} for name in ("size", "bridge", "anchor")}}))
    monkeypatch.setattr("sys.argv", ["count_analysis", "--input", str(tmp_path), "--output", str(tmp_path / "result")])
    with pytest.raises(ValueError, match="coverage for size"):
        count_analysis.main()
    assert not (tmp_path / "result/paired_counts.csv").exists()


def test_region_differences_are_invariant_to_independent_row_order():
    rows = []
    for experiment in ("baseline_region", "full"):
        for seed in SEEDS:
            for image in range(3):
                rows.append({"run": f"{experiment}/seed_{seed}", "image_id": f"image_{image}",
                             "event_id": f"event_{image}", **{metric: float(image) for metric in count_analysis.PRIMARY}})
    raw = pd.DataFrame(rows)
    arm = raw.copy()
    arm[count_analysis.PRIMARY] += 0.1
    keys = ["run", "image_id", "event_id"]
    expected = count_analysis.region_differences(arm.set_index(keys), raw.set_index(keys))
    actual = count_analysis.region_differences(arm.sample(frac=1, random_state=2).set_index(keys),
                                             raw.sample(frac=1, random_state=3).set_index(keys))
    pd.testing.assert_frame_equal(actual, expected)
    assert np.allclose(actual, 0.1)


def test_validation_counts_exclude_ignored_prediction_fragments(tmp_path, monkeypatch):
    import scripts.supplementary_validation as validation
    from toposeg.constants import IGNORE_INDEX
    from toposeg.reconstruction import ReconstructionParams

    label = np.zeros((12, 12), dtype=np.uint8)
    label[2, 1:7] = 1
    label[8, 2:7] = IGNORE_INDEX
    prediction = np.where(label == IGNORE_INDEX, 1, label).astype(np.uint8)
    probability_dir = tmp_path / "run/probs_val"
    probability_dir.mkdir(parents=True)
    np.savez(probability_dir / "image.npz", p=np.zeros((6, 12, 12), dtype=np.uint8))
    monkeypatch.setattr(validation, "load_label", lambda _: label)
    monkeypatch.setattr(validation, "reconstruct_many", lambda *_: iter([(0, prediction)]))
    rows = validation.evaluate((str(tmp_path / "run"), "image", "label", "event", [1, 0, 0, 0, 0, 0],
                                {"raw": ReconstructionParams()}))
    assert rows[0]["complete_root_dice"] == 1.0
    assert rows[0]["endpoint_f1"] == 1.0
    assert rows[0]["error_root_endpoints"] == 0


def test_drive_checkpoint_inference_is_repeatable(tmp_path, monkeypatch, fixed_frozen):
    import importlib
    import json
    import sys
    import torch
    import scripts.drive_transfer as drive

    monkeypatch.setitem(sys.modules, "drive_transfer", drive)
    supplementary = importlib.import_module("scripts.supplementary_drive")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(drive.SmallUNet().state_dict(), checkpoint)
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps(fixed_frozen))
    image = np.zeros((16, 16), dtype=np.float32)
    label = np.zeros((16, 16), dtype=np.uint8)
    label[8, 2:14] = 1
    monkeypatch.setattr(drive, "load_split", lambda _: [("01", image, label)])
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        for index in (1, 2):
            monkeypatch.setattr("sys.argv", ["drive", "--data-root", str(tmp_path), "--checkpoint", str(checkpoint),
                                             "--frozen", str(frozen), "--output", str(tmp_path / f"output_{index}")])
            supplementary.main()
        pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "output_1/per_image.csv"),
                                      pd.read_csv(tmp_path / "output_2/per_image.csv"))
        assert json.loads((tmp_path / "output_2/protocol.json").read_text())["backend"]["deterministic_algorithms"]
    finally:
        torch.use_deterministic_algorithms(previous)


def test_drive_mirror_loads_official_train_and_test_ids(tmp_path, monkeypatch):
    from PIL import Image
    import scripts.drive_transfer as drive

    for split, ids, suffix in (("train", range(21, 41), "training"), ("val", range(1, 21), "test")):
        inputs, labels = tmp_path / split / "input", tmp_path / split / "label"
        inputs.mkdir(parents=True)
        labels.mkdir(parents=True)
        for image_id in ids:
            Image.fromarray(np.full((8, 8), image_id, dtype=np.uint8)).save(inputs / f"{image_id:02d}_{suffix}.png")
            Image.fromarray(np.full((8, 8), 255, dtype=np.uint8)).save(labels / f"{image_id:02d}_manual1.png")
    monkeypatch.setattr(drive, "DATA", tmp_path)
    for split, ids in (("train", range(21, 41)), ("val", range(1, 21))):
        items = drive.load_split(split)
        assert [int(name.split("_")[0]) for name, _, _ in items] == list(ids)
        for name, image, label in items:
            assert image.dtype == np.float32 and label.dtype == np.uint8
            assert np.allclose(image, int(name.split("_")[0]) / 255)
            assert label.all()


def test_drive_new_training_seeds_numpy_and_torch_repeatably(monkeypatch):
    import torch
    import scripts.drive_transfer as drive

    image = np.linspace(0, 1, 512 * 512, dtype=np.float32).reshape(512, 512)
    monkeypatch.setattr(drive, "load_split", lambda split: [("21_training", image, (image > 0.5).astype(np.uint8))])
    monkeypatch.setattr(drive, "SmallUNet", lambda: torch.nn.Conv2d(1, 1, 1))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", torch.backends.cudnn.benchmark)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", torch.backends.cudnn.deterministic)
    np_state, torch_state = np.random.get_state(), torch.random.get_rng_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        first = drive.train_model(torch.device("cpu"), epochs=2, seed=17)
        np_first, torch_first = np.random.random(4), torch.rand(4)
        second = drive.train_model(torch.device("cpu"), epochs=2, seed=17)
        assert np.array_equal(np_first, np.random.random(4))
        assert torch.equal(torch_first, torch.rand(4))
        assert all(torch.equal(value, second.state_dict()[key]) for key, value in first.state_dict().items())
        third = drive.train_model(torch.device("cpu"), epochs=2, seed=29)
        assert any(not torch.equal(value, third.state_dict()[key]) for key, value in first.state_dict().items())
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
    finally:
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


@pytest.mark.parametrize("seed", [None, 29])
def test_drive_output_is_portable_numerical_only_and_cached_seed_not_inferred(
    tmp_path, monkeypatch, capsys, fixed_frozen, seed,
):
    import json
    import torch
    import scripts.drive_transfer as drive

    output = tmp_path / "custom_output"
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps(fixed_frozen), encoding="utf-8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(drive, "SmallUNet", lambda: torch.nn.Conv2d(1, 1, 1))
    image = np.zeros((8, 8), dtype=np.float32)
    monkeypatch.setattr(drive, "load_split", lambda split: [("01_test", image, image.astype(np.uint8))])
    calls = []

    def fake_train(device, epochs, seed):
        calls.append(seed)
        return drive.SmallUNet().to(device)

    monkeypatch.setattr(drive, "train_model", fake_train)
    argv = ["drive_transfer", "--frozen", str(frozen), "--output", str(output), "--epochs", "1"]
    if seed is not None:
        argv += ["--seed", str(seed)]
    monkeypatch.setattr("sys.argv", argv)
    drive.main()
    assert calls == [17 if seed is None else seed]
    expected_files = {"unet_drive.pt", "params.json", "per_image.csv", "summary.csv"}
    assert {path.name for path in output.iterdir()} == expected_files
    assert set(pd.read_csv(output / "per_image.csv").arm) == {"raw", "sagf_r4", "sagf_r8", "sagf_r16"}
    checkpoint = (output / "unet_drive.pt").read_bytes()
    capsys.readouterr()
    monkeypatch.setattr("sys.argv", argv + ["--seed", "43"])
    drive.main()
    assert calls == [17 if seed is None else seed]
    assert (output / "unet_drive.pt").read_bytes() == checkpoint
    assert {path.name for path in output.iterdir()} == expected_files
    assert "initialization seed is not inferred from --seed" in capsys.readouterr().out
