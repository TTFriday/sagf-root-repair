import csv
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_training_budget.py"
spec = importlib.util.spec_from_file_location("training_budget_audit", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_training_audit_selects_first_maximum_and_retains_actual_stop(tmp_path):
    for experiment in module.EXPERIMENTS.values():
        for seed in module.SEEDS:
            folder = tmp_path / "runs" / experiment / f"seed_{seed}"
            folder.mkdir(parents=True)
            with (folder / "training.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["epoch", "step", module.CRITERION])
                writer.writerows([(0, 275, 0.8), (1, 550, 0.8), (2, 825, 0.7)])
    records = module.audit(tmp_path / "runs", tmp_path / "audit")
    assert len(records) == 21
    assert all(row["selected_step"] == 275 and row["stopping_step"] == 825 for row in records)
    assert all(len(row["history_sha256"]) == 64 for row in records)
    summary = json.loads((tmp_path / "audit" / "summary.json").read_text())
    assert summary["stopped_early"] == 21
    assert not summary["checkpoint_hash_binding_checked"]


def test_training_audit_will_not_overwrite_existing_output(tmp_path):
    output = tmp_path / "audit"
    output.mkdir()
    with pytest.raises(FileExistsError):
        module.audit(tmp_path / "runs", output)
