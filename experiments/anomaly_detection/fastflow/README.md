# FastFlow Family

This family contains the flow-based anomaly modeling branch.

FastFlow models the distribution of normal features rather than reconstructing the wafer or comparing against a memory bank.

## Why This Family Exists

This branch adds a density-modeling perspective to the comparison set.

It is useful as a contrast against:

- reconstruction-based methods such as the autoencoder and VAE
- feature-mismatch methods such as teacher-student
- nearest-neighbor methods such as PatchCore

## Branches

- `x64/main/`
  Main FastFlow benchmark branch.

## Common Files In A Branch

- `README.md`
- `notebook.ipynb`
- `train_config.toml`
- `artifacts/checkpoints/`
- `artifacts/plots/`
- `artifacts/results/`
