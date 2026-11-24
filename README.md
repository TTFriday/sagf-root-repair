# sagf-root-repair

Standalone code package for **Shoot-Anchored Geodesic Forests (SAGF)**: training-free,
geometry-aware repair of fragmented root segmentations. The accompanying SAGF paper uses
an acquisition-event-grouped benchmark from the ChronoRoot 2.0 release of *Arabidopsis thaliana*.
The Python import namespace and Conda environment remain `toposeg`; the repository is
`sagf-root-repair`.

SAGF uses root probabilities, a fragment adjacency graph and a shoot-anchored bottleneck
forest to reject unattached fragments and reconnect accepted gaps. The implementation is
`src/toposeg/reconstruction.py`. It does not require retraining the segmentation network.
A compact dual-head U-Net, topology losses and the original evaluation pipeline are included
so that probabilities and numerical results can be regenerated.

## Contents and boundaries

```text
src/toposeg/       data transforms, networks, losses, inference, metrics, SAGF and statistics
scripts/           audit, training, evaluation, reconstruction and numerical analysis helpers
configs/           baseline, topology-loss and ablation configurations
manifests/         frozen event-disjoint splits and derived audit metadata
results/           copied numerical results and frozen reconstruction protocol artifacts
tests/             CPU unit and synthetic integration tests
```

Article figures, their generation/styling code, manuscript and typesetting assets, images,
model weights, probability caches and machine-local data paths are excluded from this release.
Do not overwrite frozen results or protocol records when running examples: write new work
under `outputs/`.

Aggregate analysis scripts write numerical CSV/statistical results, not article plots or
LaTeX tables. Core image loading, transforms and prediction-mask export in `evaluate.py`
remain. Checkpoints and probability caches may be generated locally; neither is redistributed.

## Dataset and split

Download the ChronoRoot 2.0 dataset from GigaScience Database,
DOI [10.5524/102804](https://doi.org/10.5524/102804), and arrange the seedling segmentation
subset as:

```text
<data root>/
  imagesTr/
  labelsTr/
  imagesTs/
```

Set your own data root; released manifests use paths relative to it:

```bash
export TOPOSEG_DATA_ROOT=/absolute/path/to/08_image_seedling_segmentation
```

Use `manifests/splits/main.csv`: train has 550 images / 24 events, validation 160 / 11,
and test 201 / 12. The 46 unmapped images are a descriptive stress set, not selection data.
The published nominal test assignment contains training-image duplicates and is retained
only for leakage diagnostics, not independent confirmation.

## Installation and tests

Run commands from the `sagf-root-repair` repository root. Python **3.10** is required by
`pyproject.toml`; the supplied environment pins PyTorch 2.5.1.

```bash
conda env create -f environment.yml
conda activate toposeg
# Alternatively, in a compatible existing environment:
python -m pip install -e '.[test]'

PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest
```

The suite covers data geometry/ignore handling, losses, inference, reconstruction,
statistics, released-path handling, maximum-cardinality point matching, GT-positive count
estimands, fixed diagnostic arms and the DRIVE helper. No real dataset or archived weights
are needed for these tests. Single-thread settings avoid CPU oversubscription.

## Train and evaluate

```bash
python scripts/train.py --config configs/baseline_region.yaml --seed 17 --final
python scripts/evaluate.py --config configs/baseline_region.yaml --seed 17 --split test
python scripts/dump_probabilities.py --config configs/baseline_region.yaml --seed 17 --split val
```

Repeat training/evaluation for seeds **17, 29, 43** and the desired configurations before
aggregation; a single example run is not sufficient. The six-arm queue is
`scripts/run_all.bat`; `scripts/run_skelrecall.bat` adds the separate Skeleton Recall
comparator. These are optional, expensive regeneration workflows.

```bash
python scripts/analyze_results.py --input outputs --output outputs/analysis \
  --split-csv manifests/splits/main.csv
python scripts/analyze_phenotypes.py --input outputs --output outputs/analysis
python scripts/analyze_comparator.py --input outputs --output outputs/analysis_comparator
```

The comparator is a separate statistical family. Named `evaluation_test`,
`evaluation_nominal` and `evaluation_ood` directories must not be pooled. Diagnostic
queues and `scripts/analyze_protocol.py` are provided for the latter protocols.

## Frozen SAGF selection and confirmation

`results/reconstruction/frozen.json` holds the original validation-selected configuration.
Test data did not select this configuration. `tune_reconstruction.py`,
`make_stage2_grid.py` and `select_reconstruction.py` retain the guarded two-stage selection
workflow. For a *new* regeneration, after creating the required validation probability
caches, the declared stages are:

```bash
python scripts/tune_reconstruction.py --runs outputs/baseline_region/seed_17 \
  --split val --grid stage1 --per-event 4 --output outputs/reconstruction/stage1_val.csv
python scripts/select_reconstruction.py outputs/reconstruction/stage1_val.csv \
  --output outputs/reconstruction/stage1_ranking.csv
python scripts/make_stage2_grid.py outputs/reconstruction/stage1_ranking.csv \
  --output outputs/reconstruction/stage2_grid.json
python scripts/tune_reconstruction.py \
  --runs outputs/baseline_region/seed_17 outputs/baseline_region/seed_29 outputs/baseline_region/seed_43 \
  --split val --grid outputs/reconstruction/stage2_grid.json --per-event 6 \
  --output outputs/reconstruction/stage2_val.csv
python scripts/select_reconstruction.py outputs/reconstruction/stage2_val.csv \
  --output outputs/reconstruction/stage2_ranking.csv --per-event 6 \
  --freeze outputs/reconstruction/frozen.json --grid outputs/reconstruction/stage2_grid.json
```

Test reconstruction requires `--allow-test`, an exact frozen candidate grid and `--frozen`.
`analyze_reconstruction.py` validates the complete pre-declared seven-experiment,
three-seed design and write-once inference record. It is not an unconstrained analysis of
arbitrary CSVs. The archived confirmatory records remain unchanged; do not reuse their
output locations for exploratory analyses. The released selector resolves relative label
paths through `TOPOSEG_DATA_ROOT` before hash verification; this portability correction
does not alter MRER, overlap guards or archived frozen decisions.

### Post-hoc diagnostics: no model selection

`supplementary_validation.py` reports all fixed diagnostic arms on validation images
previously used for tuning. These are **post-hoc exploratory** measurements, not independent
confirmation; they neither select a model nor amend the original frozen parameters.
`analyze_repair_counts.py` reports GT-positive count MAE/surplus/deficit, event-balanced
contrasts and exact sign flips. Surplus/deficit are signed aggregate count errors, not
spatially matched false-positive/false-negative points. `point_matching_sensitivity.py`
compares the original distance-assignment-then-rejection metric with maximum-cardinality
matching; frozen confirmatory metrics are not changed.

```bash
python scripts/supplementary_validation.py --runs-root outputs \
  --frozen results/reconstruction/frozen.json --output outputs/validation_diagnostics
python scripts/analyze_repair_counts.py --input results/reconstruction \
  --output outputs/count_diagnostics
python scripts/point_matching_sensitivity.py --runs-root outputs \
  --frozen results/reconstruction/frozen.json --output outputs/matching_diagnostics
```

Validation/matching evaluations require the corresponding local probability caches and
labels. Count analysis requires the copied count/comparator CSV inputs. Their output
directories must be new. Do not edit experiment scripts while their processes are running.

## DRIVE: illustrative full-frame transfer only

The supported mirror layout is **not** the original DRIVE `training/` / `test/` layout:

```text
<drive root>/
  train/input/    # IDs 21–40: official training images
  train/label/
  val/input/      # IDs 01–20: official test images, despite the mirror's "val" name
  val/label/
```

```bash
python scripts/drive_transfer.py --data /absolute/path/to/drive-mirror \
  --frozen results/reconstruction/frozen.json --output outputs/drive_new --seed 17 --epochs 120
python scripts/supplementary_drive.py --data-root /absolute/path/to/drive-mirror \
  --checkpoint outputs/drive_new/unet_drive.pt --frozen results/reconstruction/frozen.json \
  --output outputs/drive_comparators --split val
```

`drive_transfer.py` deterministically seeds NumPy/PyTorch for **new training**, defaults to
seed 17, and writes a local checkpoint, parameters and numerical CSVs, **not PNGs**. An
existing checkpoint in `--output` is loaded rather than retrained; `--seed` does not change
or identify its initialization. **The archived checkpoint's initialization seed was not
recorded.** Newly seeded training is not claimed to reproduce that checkpoint. Determinism
is conditional on compatible software/hardware, not guaranteed across platforms.

These are single-checkpoint, **full-frame illustrative** analyses: official field-of-view
masks are unavailable in this mirror. Largest-component anchoring and scale-adapted radii
are transfer adaptations, not unchanged shoot anchoring. All fixed size/closing/SAGF arms
are reported without selection. Results are not clinical evidence or a standard DRIVE
benchmark ranking; the mirror's official test images must not be used to tune arms.

## Independent graph verification and training-history audit

This 1.3.0 source package adds verification helpers and numerical records without changing
the frozen reconstruction, selection or confirmatory analysis code. The archived public core
corresponds to commit `d113c0f`; the additional files accompany the revised manuscript.

`verify_fragment_graph.py` constructs an independent eight-neighbour pixel CSR network,
including zero-weight edges, and obtains the complete distance network by per-node
multi-boundary Dijkstra propagation. Floyd min-max gives the reference bottlenecks.
It checks bottleneck values, unreachable status, fixed source owners and retention under
all supplied budgets, not equality of every additive distance or drawn path. The fixed
protocol is written before execution; no failed or skipped case is replaced.

Run from the repository root. The graph output must be a new directory outside this source
tree, with an existing parent. Omitting `--prob-dir` runs only the synthetic cases.

```bash
python scripts/verify_fragment_graph.py --output ../graph_check_new
python scripts/verify_fragment_graph.py --output ../graph_check_with_rois \
  --prob-dir /absolute/path/to/baseline_region/seed_17/probs_val
python scripts/audit_training_budget.py --runs-root outputs \
  --output outputs/training_audit_new
```

The supplied graph run plans 227 cases: 226 valid cases pass, and one fixed validation ROI
has no root after prefiltering and is recorded as skipped. The 200 random cases are all
retained. Case inputs are identified by relative names, hashes and ROI coordinates.
Synthetic grid timings compare production graph construction with a complete reference
that additionally constructs the pixel matrix and computes bottlenecks; they are not
full-image decoder speedups or a benchmark ranking. Cropped validation graphs are not
full-image performance tests and do not use ground truth.

`audit_training_budget.py` records the actual last training step and the first maximum of
validation complete-root clDice for all seven objectives and three seeds. The 30,000-step
limit is a cap: eleven runs reach it and ten stop earlier. The helper audits histories,
not checkpoint-to-evaluation hash binding. Numerical records are under
`results/verification/`; all new outputs must remain separate from those records.
The extended CPU tests cover analytic pixel costs, zero-cost ties, disconnected domains,
filtered source IDs, fixed ownership, no-anchor cases, broad drawing without repainting,
training-selection ties and write-once output handling.

## Citation and license

Please cite the accompanying SAGF article and repository; metadata is in `CITATION.cff`.
Code is MIT licensed; see `LICENSE`. Dataset licensing remains with its source.
