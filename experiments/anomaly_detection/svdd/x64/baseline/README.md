# SVDD x64 Baseline

This folder contains the curated Deep SVDD baseline for the shared `x64` anomaly benchmark.

## What This Notebook Covers

- a one-class Deep SVDD baseline trained only on normal wafers
- artifact-first reuse of the saved `best_model.pt`, `history.json`, and evaluation CSVs
- notebook-generated figures for the training curve, score distributions, threshold sweep, and top-ranked examples

## Inputs

- dataset notebook: `data/dataset/x64/benchmark_50k_5pct/notebook.ipynb`
- training config: `train_config.toml`
- training script: `scripts/train_svdd.py`

## Outputs

- artifact root: `artifacts/svdd_baseline/`
- evaluation cache: `artifacts/svdd_baseline/evaluation_notebook/`
- plots: `artifacts/svdd_baseline/plots/`
