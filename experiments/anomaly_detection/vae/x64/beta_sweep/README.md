# VAE x64 Beta Sweep

This folder contains the curated beta-sweep notebook for the VAE family.

## What This Notebook Covers

- comparison of multiple `beta` values on the same `x64` benchmark split
- artifact-first reuse of the saved per-beta checkpoints and evaluation summaries
- notebook-generated sweep figures and per-beta training/evaluation comparisons

## Inputs

- dataset notebook: `data/dataset/x64/benchmark_50k_5pct/notebook.ipynb`
- sweep config: `train_config.toml`
- training script: `scripts/train_vae.py`

## Outputs

- artifact root: `artifacts/vae_beta_sweep/`
- per-beta run folders under `artifacts/vae_beta_sweep/beta_*/`
- plots: `artifacts/vae_beta_sweep/plots/`
