# EfficientNet-B1 PatchCore

This subfamily groups the PatchCore experiments built on top of a pretrained EfficientNet-B1 backbone.

## Why This Subfamily Exists

EfficientNet-B1 is naturally associated with a larger input scale than the benchmark `x64` branches, so it serves as a useful high-resolution PatchCore follow-up.

The subfamily focuses on:

- single-layer versus multi-layer feature extraction
- the effect of defect-aware threshold tuning
- higher-resolution feature-memory behavior at `x240`

## Branches

- `x240/main/`
- `x240/main_120k/`
- `x240/one_layer/`
- `x240/layer3_5/`
- `x240/layer3_5_no_defect_tuning/`
- `x240/umap_followup/`

## Common Files In A Branch

- `README.md`
- `notebook.ipynb`
- local config files when used by the branch
- `artifacts/checkpoints/`
- `artifacts/plots/`
- `artifacts/results/`
