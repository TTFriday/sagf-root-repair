@echo off
rem Full three-seed training queue for the leakage-free topology study.
rem Run from the toposeg directory. A finished run (test summary present) is skipped,
rem and train.py resumes from outputs\<config>\seed_<seed>\last.pt when it exists.
setlocal enabledelayedexpansion
set PY=python
set PYTHONPATH=src

for %%C in (baseline_region baseline_cldice full ablation_uniform ablation_no_joint_root ablation_no_skeleton_head) do (
  for %%S in (17 29 43) do (
    if exist outputs\%%C\seed_%%S\evaluation_test\summary.json (
      echo --- skip %%C seed %%S, test evaluation already present
    ) else (
      echo === %%C seed %%S  %time%
      %PY% scripts\train.py --config configs\%%C.yaml --seed %%S --final --resume outputs\%%C\seed_%%S\last.pt
      %PY% scripts\evaluate.py --config configs\%%C.yaml --seed %%S --split test
    )
  )
)
%PY% scripts\analyze_results.py --input outputs --output outputs\analysis --split-csv manifests\splits\main.csv
%PY% scripts\analyze_phenotypes.py --input outputs --output outputs\analysis
rem Protocol diagnostics (published split, nominal and unmapped images). Needs the
rem extra evaluation_* trees; each analyse step skips itself and prints PENDING when
rem a protocol is not present, so this is safe on a bare leak-free run.
%PY% scripts\analyze_protocol.py --input outputs --splits manifests\splits --output outputs\analysis_protocol
endlocal
