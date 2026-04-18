# Anomaly Detection Experiments

This folder groups the anomaly-detection work by method family.

The families are separated so each methodological idea can be documented, trained, and evaluated without relying on the older sequential notebook naming scheme.

## Families

- `autoencoder/`
  Reconstruction-based baselines and architectural variants.
- `vae/`
  Variational autoencoder experiments focused on latent regularization.
- `svdd/`
  Deep SVDD one-class baseline.
- `backbone_embedding/`
  Frozen feature-backbone baselines.
- `teacher_student/`
  Feature-distillation anomaly detectors.
- `patchcore/`
  Nearest-neighbor memory-bank experiments across several backbones.
- `fastflow/`
  Flow-based anomaly modeling.
- `ensemble/`
  Post-hoc score-combination analysis.

## Resolution Convention

Within each family, branches are grouped by input resolution such as `x64`, `x128`, `x224`, or `x240`.

The resolution folders do not always mean the same scientific question. In some families, resolution is a direct study variable. In others, it reflects the natural input scale of a pretrained backbone.

## Common Branch Files

Most experiment branches contain:

- `README.md`
  Method and branch description.
- `notebook.ipynb`
  Canonical notebook for the branch. It should support a train-or-reuse workflow so saved local checkpoints and result files can be reused when retraining is skipped.
- `train_config.toml` and/or `data_config.toml`
  Local config snapshots when the workflow uses them.
- `artifacts/`
  Saved checkpoints, plots, and result files.
