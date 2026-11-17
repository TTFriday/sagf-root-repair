"""Split the over-complete test sweep into the pre-declared confirmatory candidate sets.

The sweep evaluated all 11 frozen configurations on every run. The pre-declared analysis
(analyze_reconstruction.py) requires 11 candidates on baseline_region and only {raw, selected}
on the six transfer experiments. Dropping surplus arms (not choosing among them) reproduces
the declared design exactly; surplus arms remain in test.csv for descriptive reporting.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REC = Path("outputs/reconstruction")


def main():
    frame = pd.read_csv(REC / "test.csv")
    protocol = json.loads((REC / "test.protocol.json").read_text(encoding="utf-8"))
    frozen = json.loads((REC / "frozen.json").read_text(encoding="utf-8"))
    selected = frozen["selected"]["name"]
    region_runs = [r for r in protocol["runs"] if "/baseline_region/" in r.replace("\\", "/")]
    transfer_runs = [r for r in protocol["runs"] if r not in region_runs]
    sets = {"region": (region_runs, set(protocol["grid"])),
            "transfer": (transfer_runs, {"raw", selected})}
    for name, (runs, candidates) in sets.items():
        sub = frame[frame.run.isin(runs) & frame.candidate.isin(candidates)]
        assert len(sub) == len(runs) * 201 * len(candidates), (name, len(sub))
        sub.to_csv(REC / f"test_{name}.csv", index=False)
        record = dict(protocol)
        record["runs"] = runs
        record["grid"] = {k: v for k, v in protocol["grid"].items() if k in candidates}
        record["probability_sha256"] = {r: protocol["probability_sha256"][r] for r in runs}
        (REC / f"test_{name}.protocol.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(name, len(sub), "rows,", len(runs), "runs,", len(candidates), "candidates")


if __name__ == "__main__":
    main()