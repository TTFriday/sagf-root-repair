"""Build the stage-2 validation grid from the stage-1 ranking.

Every post-processing family contributes its two best eligible stage-1 candidates and a local refinement
around its best one, so that no family is compared at a single setting. SAGF additionally contributes its
ablation families, each with its own budget grid, and one-at-a-time sensitivity variants of the design
parameters. Families are encoded in the candidate name prefix (see select_reconstruction.family).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from select_reconstruction import STAGE2_FAMILY_COUNTS, family, load_results
from toposeg.reconstruction import ReconstructionParams as P
from tune_reconstruction import stage1_grid

BUDGETS = (2, 3, 4, 6, 8, 12, 16)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_stage1_ranking(path: Path) -> None:
    record_path = path.with_suffix(".protocol.json")
    if not record_path.is_file():
        raise RuntimeError("Stage-1 ranking has no protocol record")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    sources = record.get("sources", [])
    if (record.get("schema_version") != 1 or record.get("split") != "val"
            or record.get("ranking_sha256") != sha256(path) or len(sources) != 1):
        raise RuntimeError("Stage-1 ranking protocol is invalid")
    ranking_hashes = {name.replace("\\", "/"): value
                      for name, value in record.get("implementation_sha256", {}).items()}
    if (set(ranking_hashes) != {"scripts/select_reconstruction.py"}
            or any(not Path(name).is_file() or sha256(Path(name)) != expected
                   for name, expected in ranking_hashes.items())):
        raise RuntimeError("Stage-1 ranking implementation has changed")
    source = sources[0]
    evaluation = source.get("evaluation", {})
    runs = [run.replace("\\", "/") for run in evaluation.get("runs", [])]
    expected_grid = {name: asdict(params) for name, params in stage1_grid().items()}
    if (evaluation.get("split") != "val" or evaluation.get("per_event") != 4
            or runs != ["outputs/baseline_region/seed_17"]
            or len(evaluation.get("images", [])) != 42 or evaluation.get("grid") != expected_grid):
        raise RuntimeError("Stage 2 requires the pre-declared Stage-1 validation protocol")
    for key, hash_key in (("results", "results_sha256"), ("protocol", "protocol_sha256")):
        source_path = Path(source[key])
        if not source_path.is_file() or sha256(source_path) != source[hash_key]:
            raise RuntimeError(f"Stage-1 source changed after ranking: {source_path}")
    load_results([Path(source["results"])])


def sagf_name(prefix: str, p: P) -> str:
    parts = [prefix, p.decode, f"a{p.min_area}", f"s{p.forest_step}", f"l{p.forest_lambda}", f"r{p.forest_radius}",
             f"as{p.anchor_min_area}", f"b{p.forest_budget:g}", f"w{p.forest_width}", f"k{p.forest_keep_area}"]
    return "_".join(parts)


def comparator_refinements(best: dict[str, P]) -> dict[str, P]:
    grid = {}
    for t in (0.5, 0.45, 0.35, 0.25):
        grid[f"union_t{t}"] = P(decode="union", root_threshold=t)
    for low in (0.1, 0.15, 0.5):
        grid[f"hyst_{low}"] = P(hysteresis_low=low)
    for r in (4, 7):
        grid[f"close_{r}"] = P(closing_radius=r)
    b = best["size"]
    size_areas = sorted({max(5, round(b.min_area * f)) for f in (0.5, 0.75, 1, 1.5, 2)})
    for area, peak in itertools.product(size_areas, (0.0, b.min_peak, 0.95)):
        grid[f"size_a{area}_p{peak}"] = P(min_area=area, min_peak=peak)
    for area in sorted(size_areas, key=lambda value: (abs(value - b.min_area), value)):
        grid[f"size_a{area}_p0.85"] = P(min_area=area, min_peak=0.85)
    for k in (3, 5, 10, 20, 50):
        grid[f"lcc_k{k}"] = P(keep_largest=k)
    b = best["bridge"]
    for radius, prob in itertools.product((b.bridge_radius, 60), (0.02, b.bridge_min_prob, 0.1)):
        for angle in (30.0, 45.0, 90.0):
            grid[f"bridge_r{radius}_p{prob}_g{angle:g}"] = P(bridge_radius=radius, bridge_min_prob=prob, bridge_max_angle=angle)
    for anchor, chain in itertools.product((3, 10, 20), (0, 20, 40, 60)):
        grid[f"anchor_d{anchor}_c{chain}"] = P(anchor_distance=anchor, chain_distance=chain)
    for area, anchor, chain in itertools.product((10, 30, 50), (5, 10), (10, 20, 40)):
        grid[f"size{area}+anchor_d{anchor}_c{chain}"] = P(min_area=area, anchor_distance=anchor, chain_distance=chain)
    b = best["size+anchor"]
    fallback = sorted(itertools.product((5, 10), (10, 20, 40)),
                      key=lambda values: (abs(values[0] - b.anchor_distance), abs(values[1] - b.chain_distance), values))
    for anchor, chain in fallback:
        grid[f"size20+anchor_d{anchor}_c{chain}"] = P(min_area=20, anchor_distance=anchor, chain_distance=chain)
    return grid


def sagf_variants(base: P) -> dict[str, P]:
    grid = {}

    def add(prefix: str, p: P) -> None:
        grid[sagf_name(prefix, p)] = p

    for area, budget in itertools.product((0, 20, 30, 50), BUDGETS):
        add("sagf", replace(base, min_area=area, forest_budget=float(budget)))
    for lam, budget in itertools.product((0.25, 0.5), (2, 4, 8, 12, 16, 24, 32)):
        add("sagf", replace(base, forest_lambda=lam, forest_budget=float(budget)))
    for keep in (300, 1000, 3000):
        add("sagf", replace(base, forest_keep_area=keep))
    for width in (1, 3):
        add("sagf", replace(base, forest_width=width))
    for step in (0.1, 0.3):
        add("sagf", replace(base, forest_step=step))
    for budget in (0.5 * base.forest_budget, base.forest_budget, 2 * base.forest_budget):
        for change in (dict(forest_radius=20), dict(forest_radius=80), dict(anchor_min_area=50), dict(anchor_min_area=200),
                       dict(decode="union")):
            add("sagf", replace(base, forest_budget=float(budget), **change))
    for budget in (*BUDGETS, 24):
        add("sagffilter", replace(base, forest_budget=float(budget), forest_width=0))
    for budget in (3, 5, 10, 15, 20, 30, 40, 60):
        add("sagfgeom", replace(base, forest_lambda=0.0, forest_step=1.0, forest_budget=float(budget)))
    for budget in (2, 4, 8, 12, 16, 24, 32, 48):
        add("sagfadd", replace(base, forest_mode="additive", forest_budget=float(budget)))
    for budget in BUDGETS:
        add("sagflargest", replace(base, forest_anchor="largest", forest_budget=float(budget)))
    return grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ranking", type=Path, help="Output of select_reconstruction.py on stage 1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=2)
    args = parser.parse_args()
    if args.top != 2:
        parser.error("Stage 2 is pre-declared to carry forward exactly two candidates per family")
    verify_stage1_ranking(args.ranking)
    ranking = pd.read_csv(args.ranking, index_col=0)
    stage1 = stage1_grid()
    eligible = ranking[ranking.eligible]
    grid = {}
    fingerprints = set()
    counts_dict = {name: 0 for name in STAGE2_FAMILY_COUNTS}

    def add_candidates(candidates: dict[str, P]) -> None:
        for name, params in candidates.items():
            fam = family(name)
            if counts_dict[fam] >= STAGE2_FAMILY_COUNTS[fam]:
                continue
            fingerprint = json.dumps(asdict(params), sort_keys=True)
            if fingerprint in fingerprints:
                continue
            grid[name] = params
            fingerprints.add(fingerprint)
            counts_dict[fam] += 1

    add_candidates({"raw": P()})
    best = {}
    carried = {}
    for fam, rows in eligible.groupby("family", sort=False):
        names = rows.sort_values("mrer", ascending=False, kind="stable").index[:args.top]
        best[fam] = stage1[names[0]]
        carried.update({name: stage1[name] for name in names})
    add_candidates(carried)
    add_candidates(comparator_refinements(best))
    sagf_best = eligible[eligible.family == "sagf"].index[0]
    add_candidates(sagf_variants(stage1[sagf_best]))
    payload = {name: asdict(params) for name, params in grid.items()}
    counts = pd.Series([family(name) for name in grid]).value_counts()
    if len(grid) != 180 or counts_dict != STAGE2_FAMILY_COUNTS:
        raise RuntimeError(f"Unexpected Stage-2 design: {len(grid)} candidates, {counts_dict}")
    record_path = args.output.with_suffix(".protocol.json")
    if args.output.exists() or record_path.exists():
        raise FileExistsError("Stage-2 grid or its protocol already exists")
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
    ranking_protocol = args.ranking.with_suffix(".protocol.json")
    grid_protocol = {
        "schema_version": 1,
        "top": args.top,
        "candidates": len(grid),
        "family_counts": counts_dict,
        "grid_sha256": sha256(args.output),
        "ranking": str(args.ranking).replace("\\", "/"),
        "ranking_sha256": sha256(args.ranking),
        "ranking_protocol": str(ranking_protocol).replace("\\", "/"),
        "ranking_protocol_sha256": sha256(ranking_protocol),
        "implementation_sha256": {
            "scripts/make_stage2_grid.py": sha256(Path("scripts/make_stage2_grid.py")),
            "scripts/select_reconstruction.py": sha256(Path("scripts/select_reconstruction.py")),
        },
    }
    with record_path.open("x", encoding="utf-8") as handle:
        json.dump(grid_protocol, handle, indent=2)
    print(f"SAGF base: {sagf_best}")
    print(f"{len(grid)} candidates")
    print(counts.to_string())


if __name__ == "__main__":
    main()
