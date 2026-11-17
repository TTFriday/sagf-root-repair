#!/usr/bin/env bash
set -e
PY=python
export PYTHONPATH=src
SPLIT=${1:-val}
for exp in baseline_region baseline_cldice full ablation_uniform ablation_no_joint_root ablation_no_skeleton_head baseline_skelrecall; do
  for seed in 17 29 43; do
    $PY scripts/dump_probabilities.py --config configs/$exp.yaml --seed $seed --split $SPLIT
  done
done
echo DONE_$SPLIT
