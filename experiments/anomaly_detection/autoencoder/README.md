# Autoencoder Family

This family contains the reconstruction-based anomaly baselines.

The core idea is simple: train on normal wafers only, reconstruct the input wafer map, and convert reconstruction error into an anomaly score at test time.

## Why This Family Exists

The autoencoder branch is the reference reconstruction baseline for the project.

It provides:

- the simplest anomaly pipeline in the repo
- a clean benchmark against more complex methods
- a useful place to study how architectural choices change reconstruction quality and anomaly scores

## Branches

- `x64/baseline/`
  Plain convolutional autoencoder baseline.
- `x64/batchnorm/`
  Autoencoder variant with BatchNorm.
- `x64/batchnorm_dropout/`
  BatchNorm plus dropout follow-up branch.
- `x64/residual/`
  Residual autoencoder variant.
- `x128/baseline/`
  Higher-resolution baseline used to study whether finer spatial detail improves anomaly sensitivity.

## Common Files In A Branch

- `README.md`
  Method and branch description.
- `notebook.ipynb`
  Canonical notebook for training, evaluation, and plot generation.
- `train_config.toml`
  Local training configuration when the branch uses a config file.
- `artifacts/`
  Saved checkpoints, plots, metrics, threshold sweeps, and analysis tables.
