# EfficientNet-B1 PatchCore (`x240`)

This resolution folder groups the EfficientNet-B1 PatchCore branches that share the `x240` input pipeline.

## Why `x240`

EfficientNet-B1 is closest to its pretrained operating scale at `x240`, so these branches test whether a larger input resolution improves feature-memory anomaly detection on wafer maps.

## Branches

- `main/`
  Local one-layer benchmark run with checkpoint, benchmark metrics, holdout evaluation, and UMAP exports.
- `main_120k/`
  Script-driven one-layer follow-up that keeps the same EfficientNet-B1 main recipe but scales the normal-only split to `120k`.
- `one_layer/`
  Imported single-layer source notebook awaiting a full local training run.
- `layer3_5/`
  Imported two-stage follow-up notebook awaiting a full local training run.
- `layer3_5_no_defect_tuning/`
  Imported two-stage follow-up without defect-aware tuning.
- `umap_followup/`
  Review-oriented UMAP branch that builds on the saved artifacts from `main/`.
