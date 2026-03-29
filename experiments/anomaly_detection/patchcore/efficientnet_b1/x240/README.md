# EfficientNet-B1 PatchCore (`x240`)

This resolution folder groups the EfficientNet-B1 PatchCore branches that share the `x240` input pipeline.

## Why `x240`

EfficientNet-B1 is closer to its native pretrained scale at `x240`, so this folder collects the runs that study whether that larger input resolution improves feature-memory anomaly detection on wafer maps.

## Branches

- `main/`
  Existing reference branch for this backbone-resolution pair.
- `one_layer/`
  Single-layer feature extraction branch.
- `layer3_5/`
  Multi-layer feature extraction branch using two EfficientNet stages.
- `layer3_5_no_defect_tuning/`
  Multi-layer branch without defect-aware threshold tuning.
- `umap_followup/`
  UMAP-oriented follow-up analysis branch.
