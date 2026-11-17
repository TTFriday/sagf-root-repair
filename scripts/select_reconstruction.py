"""Rank reconstruction candidates on validation results with the pre-declared selection criterion.

Score: mean relative error reduction (MRER) against the raw argmax decoding of the same model over
five topology endpoints, each converted to an error (1 - clDice, 1 - endpoint F1, 1 - branch-point F1,
component-count error, root HD95). Means are event-balanced (images -> event -> split). Candidates whose
complete-root Dice or macro-Dice falls more than 0.005 below raw are ineligible. A distance that is
undefined because exactly one of prediction and target is empty is charged the image diagonal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from toposeg.dataset import resolve_paths

ERRORS = {
    "complete_root_cldice": lambda v: 1 - v,
    "endpoint_f1": lambda v: 1 - v,
    "branchpoint_f1": lambda v: 1 - v,
    "component_count_error": lambda v: v,
    "root_hd95_px": lambda v: v,
}
GUARDS = ("complete_root_dice", "macro_positive_dice")
MARGIN = 0.005
REPORT = ["complete_root_dice", "macro_positive_dice", "complete_root_cldice", "endpoint_f1", "branchpoint_f1",
          "component_count_error", "root_hd95_px", "skeleton_length_error_px", "skeleton_length_absolute_error_px"]
DIAGONAL = float(np.hypot(2464, 3280))
MANIFEST = Path("manifests/splits/main.csv")
AMENDMENT = Path("outputs/reconstruction/stage2_freeze_amendment.json")
EVALUATION_IMPLEMENTATIONS = {
    "src/toposeg/reconstruction.py",
    "src/toposeg/metrics.py",
    "src/toposeg/dataset.py",
    "src/toposeg/constants.py",
    "src/toposeg/phenotyping.py",
    "scripts/tune_reconstruction.py",
}
STAGE2_FAMILY_COUNTS = {
    "raw": 1,
    "union": 5,
    "hyst": 3,
    "close": 4,
    "size": 16,
    "lcc": 5,
    "bridge": 20,
    "anchor": 12,
    "size+anchor": 20,
    "sagf": 62,
    "sagffilter": 9,
    "sagfgeom": 8,
    "sagfadd": 8,
    "sagflargest": 7,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_run(run: str) -> str:
    return str(Path(run.replace("\\", "/"))).replace("\\", "/").rstrip("/").casefold()


def protocol_path(result: Path) -> Path:
    return result.with_suffix(".protocol.json")


def load_protocols(paths: list[Path], expected_split: str = "val") -> list[dict]:
    protocols = []
    for result in paths:
        record = protocol_path(result)
        if not record.is_file():
            raise RuntimeError(f"Missing evaluation protocol for {result}")
        item = json.loads(record.read_text(encoding="utf-8"))
        if item.get("schema_version") != 1 or item.get("split") != expected_split:
            raise RuntimeError(f"Expected a {expected_split} evaluation protocol: {record}")
        protocols.append(item)
    return protocols


def load_results(paths: list[Path], expected_split: str = "val") -> pd.DataFrame:
    frames = []
    protocols = load_protocols(paths, expected_split)
    for path, item in zip(paths, protocols):
        frame = pd.read_csv(path)
        normalized_protocol_runs = [normalize_run(run) for run in item["runs"]]
        if len(set(normalized_protocol_runs)) != len(normalized_protocol_runs):
            raise RuntimeError(f"Duplicate normalized runs in evaluation protocol: {path}")
        runs = set(normalized_protocol_runs)
        frame["run"] = frame.run.map(normalize_run)
        image_entries = item["images"]
        image_events = {entry["image_id"]: entry["event_id"] for entry in image_entries}
        if len(image_events) != len(image_entries):
            raise RuntimeError(f"Duplicate image IDs in evaluation protocol: {path}")
        images = set(image_events)
        candidates = set(item["grid"])
        normalized_runs = frame.run
        frame_images = frame.image_id.astype(str)
        expected_events = frame_images.map(image_events)
        if frame.duplicated(["run", "image_id", "candidate"]).any():
            raise RuntimeError(f"Duplicate candidate rows in {path}")
        if (set(normalized_runs) != runs or set(frame_images) != images or set(frame.candidate) != candidates
                or expected_events.isna().any() or not expected_events.equals(frame.event_id.astype(str))
                or len(frame) != len(runs) * len(images) * len(candidates)):
            raise RuntimeError(f"Incomplete results or protocol mismatch in {path}")
        if item.get("manifest_sha256") != sha256(MANIFEST):
            raise RuntimeError(f"Manifest differs from evaluation protocol: {path}")
        manifest = resolve_paths(pd.read_csv(MANIFEST), None).set_index("image_id")
        labels = item.get("label_sha256", {})
        if set(labels) != images or not images.issubset(set(manifest.index)):
            raise RuntimeError(f"Evaluation protocol has incomplete label hashes: {path}")
        for image_id, expected in labels.items():
            row = manifest.loc[image_id]
            if str(row.label_sha256) != expected or sha256(Path(row.label_path)) != expected:
                raise RuntimeError(f"Label changed after evaluation: {image_id}")
        probability_entries = item.get("probability_sha256", {})
        probability_runs = {normalize_run(run): values for run, values in probability_entries.items()}
        if len(probability_runs) != len(probability_entries) or set(probability_runs) != runs:
            raise RuntimeError(f"Evaluation protocol has incomplete probability hashes: {path}")
        for run, values in probability_runs.items():
            if set(values) != images:
                raise RuntimeError(f"Evaluation protocol has incomplete probabilities for {run}")
            for image_id, expected in values.items():
                probability = Path(run) / f"probs_{item['split']}" / f"{image_id}.npz"
                if not probability.is_file() or sha256(probability) != expected:
                    raise RuntimeError(f"Probability map changed after evaluation: {probability}")
        hashes = {name.replace("\\", "/"): value for name, value in item.get("implementation_sha256", {}).items()}
        if set(hashes) != EVALUATION_IMPLEMENTATIONS:
            raise RuntimeError(f"Evaluation protocol has an incomplete implementation record: {path}")
        for name, expected in hashes.items():
            source = Path(name)
            if not source.is_file() or sha256(source) != expected:
                raise RuntimeError(f"Implementation differs from evaluation protocol: {name}")
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    if data.duplicated(["run", "image_id", "candidate"]).any():
        raise RuntimeError("Result files overlap in run, image and candidate")
    data[["root_hd95_px", "root_asd_px"]] = data[["root_hd95_px", "root_asd_px"]].fillna(DIAGONAL)
    return data


def family(name: str) -> str:
    if name.startswith("forest_"):
        return "sagffilter" if "_w0_" in name else "sagf"
    return "size+anchor" if "+" in name else name.split("_")[0]


def event_balanced(data: pd.DataFrame) -> pd.DataFrame:
    by_event = data.groupby(["run", "candidate", "event_id"])[REPORT].mean()
    return by_event.groupby(["run", "candidate"]).mean().reset_index()


def score(means: pd.DataFrame) -> pd.DataFrame:
    raw = means[means.candidate == "raw"].set_index("run")
    rows = []
    for row in means.itertuples(index=False):
        base = raw.loc[row.run]
        denominators = {metric: transform(base[metric]) for metric, transform in ERRORS.items()}
        if any(not np.isfinite(value) or value <= 0 for value in denominators.values()):
            raise RuntimeError(f"MRER is undefined because a raw error is non-positive for {row.run}")
        gains = {metric: (denominators[metric] - transform(getattr(row, metric))) / denominators[metric]
                 for metric, transform in ERRORS.items()}
        if not np.isfinite(list(gains.values())).all():
            raise RuntimeError(f"MRER contains a non-finite value for {row.run}, {row.candidate}")
        rows.append({"run": row.run, "candidate": row.candidate, "mrer": float(np.mean(list(gains.values()))),
                     **{f"gain_{m}": g for m, g in gains.items()},
                     **{f"delta_{g}": getattr(row, g) - base[g] for g in GUARDS}})
    return means.merge(pd.DataFrame(rows), on=["run", "candidate"])


def freeze_selection(summary: pd.DataFrame, data: pd.DataFrame, grid_path: Path, results: list[Path], output: Path,
                     per_event: int) -> None:
    from toposeg.dataset import SeedlingDataset
    from tune_reconstruction import load_grid

    grid = load_grid(str(grid_path))
    protocols = load_protocols(results)
    expected_grid = {name: asdict(params) for name, params in grid.items()}
    fingerprints = {json.dumps(params, sort_keys=True) for params in expected_grid.values()}
    family_counts = pd.Series([family(name) for name in grid]).value_counts().to_dict()
    grid_record_path = grid_path.with_suffix(".protocol.json")
    if not grid_record_path.is_file():
        raise RuntimeError("Stage-2 grid has no generation protocol")
    grid_record = json.loads(grid_record_path.read_text(encoding="utf-8"))
    generator_hashes = {name.replace("\\", "/"): value
                        for name, value in grid_record.get("implementation_sha256", {}).items()}
    amendment = None
    if AMENDMENT.is_file():
        amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
        amended = "scripts/select_reconstruction.py"
        if (amendment.get("schema_version") != 1 or amendment.get("file") != amended
                or amendment.get("previous_sha256") != generator_hashes.get(amended)
                or amendment.get("test_reconstruction_evaluated") is not False):
            raise RuntimeError("The freeze amendment does not extend the Stage-2 grid protocol")
        generator_hashes[amended] = amendment["amended_sha256"]
        amendment = {"path": str(AMENDMENT).replace("\\", "/"), "sha256": sha256(AMENDMENT), **amendment}
    required_generators = {"scripts/make_stage2_grid.py", "scripts/select_reconstruction.py"}
    sources_valid = all(Path(grid_record[key]).is_file()
                        and sha256(Path(grid_record[key])) == grid_record[hash_key]
                        for key, hash_key in (("ranking", "ranking_sha256"),
                                              ("ranking_protocol", "ranking_protocol_sha256")))
    if (per_event != 6 or len(grid) != 180 or len(fingerprints) != 180
            or family_counts != STAGE2_FAMILY_COUNTS or grid_record.get("schema_version") != 1
            or grid_record.get("top") != 2 or grid_record.get("grid_sha256") != sha256(grid_path)
            or grid_record.get("family_counts") != STAGE2_FAMILY_COUNTS
            or set(generator_hashes) != required_generators or not sources_valid
            or any(not Path(name).is_file() or sha256(Path(name)) != expected
                   for name, expected in generator_hashes.items())
            or any(item.get("per_event") != 6 or item.get("grid") != expected_grid for item in protocols)):
        raise RuntimeError("Stage-2 freezing requires the pre-declared unique 180-candidate protocol")
    validation = SeedlingDataset("manifests/splits/main.csv", "val").frame.sort_values(["event_id", "image_id"])
    if per_event:
        picked = []
        for _, group in validation.groupby("event_id"):
            indices = np.unique(np.linspace(0, len(group) - 1, min(per_event, len(group))).round().astype(int))
            picked.append(group.iloc[indices])
        validation = pd.concat(picked)
    if len(validation) != 59 or validation.event_id.nunique() != 11:
        raise RuntimeError("Stage-2 freezing requires the pre-declared 59 images from 11 validation events")
    expected = validation.set_index("image_id").event_id
    observed = data[["image_id", "event_id"]].drop_duplicates().set_index("image_id").event_id
    if not observed.index.is_unique or not observed.index.isin(expected.index).all() or not observed.equals(expected.reindex(observed.index)):
        raise RuntimeError("Only validation images with their original event assignments may be frozen")
    runs = {Path(run.replace("\\", "/")) for run in data.run.unique()}
    if len(runs) != 3 or {run.name for run in runs} != {"seed_17", "seed_29", "seed_43"} or any(run.parent.name != "baseline_region" for run in runs):
        raise RuntimeError("Stage-2 freezing requires all three Region validation seeds")
    if set(summary.index) != set(grid) or data.duplicated(["run", "image_id", "candidate"]).any():
        raise RuntimeError("The result candidates must match the grid exactly, without duplicate rows")
    counts = data.groupby(["run", "image_id"]).candidate.nunique()
    if not (counts == len(grid)).all():
        raise RuntimeError("Incomplete candidate evaluation")
    image_sets = data.groupby("run").image_id.agg(set).tolist()
    if any(images != image_sets[0] for images in image_sets[1:]) or set(observed.index) != set(expected.index):
        raise RuntimeError("The three runs must cover identical images from every validation event")
    eligible = summary[summary.eligible]
    ineligible = {}

    def best(fam: str) -> dict | None:
        rows = eligible[eligible.family == fam]
        if rows.empty:
            top = summary[summary.family == fam].iloc[0]
            ineligible[fam] = {"name": top.name, "params": asdict(grid[top.name]), "validation_mrer": float(top.mrer),
                               **{f"delta_{g}": float(top[f"delta_{g}"]) for g in GUARDS}}
            return None
        name = rows.index[0]
        return {"name": name, "params": asdict(grid[name]), "validation_mrer": float(rows.iloc[0].mrer)}

    selected = best("sagf")
    if selected is None:
        raise RuntimeError("No eligible SAGF configuration")
    comparators = {fam: entry for fam in ("union", "hyst", "close", "size", "lcc", "bridge", "anchor", "size+anchor")
                   if (entry := best(fam)) is not None}
    ablations = {fam: entry for fam in ("sagffilter", "sagfgeom", "sagfadd", "sagflargest")
                 if (entry := best(fam)) is not None}
    if ineligible and amendment is None:
        raise RuntimeError(f"No eligible configuration in required families {sorted(ineligible)}")
    overall = eligible.index[0]
    frozen = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selected": selected,
        "overall_best": {"name": overall, "family": family(overall), "params": asdict(grid[overall]),
                         "validation_mrer": float(eligible.iloc[0].mrer)},
        "comparators": comparators,
        "ablations": ablations,
        "ineligible_families": ineligible,
        "amendment": amendment,
        "criterion": {"score": "event-balanced mean relative error reduction, averaged over three seeds",
                      "errors": list(ERRORS), "guards": list(GUARDS), "guard_margin": MARGIN,
                      "selection_scope": "best eligible within each family; families without an eligible "
                                         "configuration are reported but not carried to test; global winner "
                                         "reported separately"},
        "validation": {"runs": sorted(str(run).replace("\\", "/") for run in runs), "images": len(observed),
                       "events": int(observed.nunique()), "images_per_event_cap": per_event, "candidates": len(grid),
                       "grid_sha256": hashlib.sha256(grid_path.read_bytes()).hexdigest(),
                       "grid_protocol_sha256": hashlib.sha256(grid_record_path.read_bytes()).hexdigest(),
                       "results_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in results},
                       "implementation_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                                                 (Path("src/toposeg/reconstruction.py"), Path("src/toposeg/metrics.py"),
                                                  Path("src/toposeg/dataset.py"), Path("src/toposeg/constants.py"),
                                                  Path("src/toposeg/phenotyping.py"), Path("scripts/tune_reconstruction.py"),
                                                  Path("scripts/select_reconstruction.py"))},
                       "test_used_for_reconstruction_selection": False, "prior_network_test_evaluation": True},
    }
    region_entries = [frozen["selected"], frozen["overall_best"], *frozen["comparators"].values(), *frozen["ablations"].values()]
    region_grid = {"raw": asdict(grid["raw"]), **{entry["name"]: entry["params"] for entry in region_entries}}
    transfer_grid = {"raw": asdict(grid["raw"]), frozen["selected"]["name"]: frozen["selected"]["params"]}
    region_path = output.with_name("test_grid_region.json")
    transfer_path = output.with_name("test_grid_transfer.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if any(path.exists() for path in (output, region_path, transfer_path)):
        raise FileExistsError("Frozen selection or a derived test grid already exists")
    for path, payload in ((output, frozen), (region_path, region_grid), (transfer_path, transfer_grid)):
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    print(f"Frozen SAGF: {frozen['selected']['name']}; global validation winner: {overall}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--freeze", type=Path, help="Write the immutable stage-2 validation selection")
    parser.add_argument("--grid", type=Path, help="Stage-2 grid JSON, required with --freeze")
    parser.add_argument("--per-event", type=int, default=0, help="Image cap used in the stage-2 evaluation")
    args = parser.parse_args()
    if args.freeze and not args.grid:
        parser.error("--freeze requires --grid")
    data = load_results(args.results)
    scored = score(event_balanced(data))
    numeric = [c for c in scored.columns if c not in ("run", "candidate")]
    summary = scored.groupby("candidate")[numeric].mean()
    summary["worst_run_mrer"] = scored.groupby("candidate")["mrer"].min()
    summary["eligible"] = (summary[[f"delta_{g}" for g in GUARDS]] >= -MARGIN).all(axis=1)
    summary["family"] = [family(n) for n in summary.index]
    summary = summary.sort_index().sort_values("mrer", ascending=False, kind="stable")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output)
    ranking_protocol = {
        "schema_version": 1,
        "split": "val",
        "ranking_sha256": sha256(args.output),
        "implementation_sha256": {"scripts/select_reconstruction.py": sha256(Path("scripts/select_reconstruction.py"))},
        "sources": [{"results": str(path).replace("\\", "/"), "results_sha256": sha256(path),
                     "protocol": str(protocol_path(path)).replace("\\", "/"),
                     "protocol_sha256": sha256(protocol_path(path)),
                     "evaluation": item}
                    for path, item in zip(args.results, load_protocols(args.results))],
    }
    args.output.with_suffix(".protocol.json").write_text(json.dumps(ranking_protocol, indent=2), encoding="utf-8")
    pd.set_option("display.width", 250)
    cols = ["family", "eligible", "mrer", "worst_run_mrer", "delta_complete_root_dice", "delta_macro_positive_dice"] + REPORT[2:]
    print(f"runs: {scored.run.nunique()}, images: {data.image_id.nunique()}, candidates: {len(summary)}")
    print("\nTop eligible:")
    print(summary[summary.eligible][cols].head(args.top).round(4).to_string())
    print("\nBest eligible per family:")
    best = summary[summary.eligible].groupby("family", sort=False).head(1)
    print(best[cols].round(4).to_string())
    print("\nRaw:")
    print(summary.loc[["raw"], cols].round(4).to_string())
    if args.freeze:
        freeze_selection(summary, data, args.grid, args.results, args.freeze, args.per_event)


if __name__ == "__main__":
    main()
