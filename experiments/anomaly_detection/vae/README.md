# VAE Family

This family contains the variational autoencoder anomaly experiments.

The VAE branches keep the reconstruction-based structure of the autoencoder family while adding latent regularization through a KL-divergence term.

## Why This Family Exists

The VAE family tests whether a probabilistic latent model produces a more useful anomaly signal than a deterministic autoencoder.

This matters because latent regularization can improve structure in representation space, but it can also smooth away the small local defects that matter in wafer screening.

## Branches

- `x64/baseline/`
  Main VAE benchmark branch.
- `x64/beta_sweep/`
  Regularization study that varies `beta` to examine the trade-off between reconstruction fidelity and latent compression.

## Common Files In A Branch

- `README.md`
- `notebook.ipynb`
- `train_config.toml`
- `artifacts/`
