# FastFlow x64 Main

This folder contains the curated FastFlow experiment notebook for the shared `x64` anomaly benchmark.

## What This Notebook Covers

- a FastFlow-style anomaly model with a frozen `wide_resnet50_2` backbone
- three backbone and flow-step ablations:
  - `wrn50_l23_s6`
  - `wrn50_l2_s6`
  - `wrn50_l23_s4`
- artifact-first evaluation on the processed `50k / 5%` benchmark split

## Default Behavior

- the notebook loads saved CSV artifacts first
- it does not retrain missing variants unless you explicitly opt in
- if a saved checkpoint exists for a variant, later cells can load it for additional qualitative analysis
- future training runs now save `*_best_model.pt` and `*_latest_checkpoint.pt`

## Inputs

- dataset notebook: `data/dataset/x64/benchmark_50k_5pct/notebook.ipynb`
- training config: `train_config.toml`
- processed metadata: `data/processed/x64/wm811k/metadata_50k_5pct.csv`

## Outputs

- artifact root: `artifacts/fastflow_variant_sweep/`
- branch-level summaries under `artifacts/fastflow_variant_sweep/results/`
- branch-level figures under `artifacts/fastflow_variant_sweep/plots/`
- one subfolder per FastFlow variant, each with:
  - `checkpoints/` for best, latest, and periodic checkpoints
  - `results/` for history, score sweeps, defect breakdowns, and best-row summaries
