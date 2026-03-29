# VAE x64 Baseline

This folder contains the curated VAE baseline for the shared `x64` anomaly benchmark.

## What This Notebook Covers

- the main single-run convolutional VAE baseline
- artifact-first reuse of the saved checkpoint, history, and evaluation outputs
- notebook-generated figures for training curves, score distributions, threshold sweep, and reconstruction examples

## Inputs

- dataset notebook: `data/dataset/x64/benchmark_50k_5pct/notebook.ipynb`
- training config: `train_config.toml`
- training script: `scripts/train_vae.py`

## Outputs

- artifact root: `artifacts/vae_baseline/`
- evaluation cache: `artifacts/vae_baseline/evaluation_notebook/`
- plots: `artifacts/vae_baseline/plots/`
