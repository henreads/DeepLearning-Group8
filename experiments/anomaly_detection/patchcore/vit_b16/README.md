# ViT-B/16 PatchCore

This subfamily groups the PatchCore experiments built on top of a pretrained ViT-B/16 backbone.

## Why This Subfamily Exists

The ViT-B/16 branches test whether transformer features offer different anomaly behavior from the CNN backbones used elsewhere in the repo.

The subfamily focuses on:

- one-layer versus two-block feature extraction
- defect-tuned versus normal-only threshold selection
- transformer-based high-resolution PatchCore behavior at `x224`

## Branches

- `x224/main/`
- `x224/one_layer_defect_tuning/`
- `x224/one_layer_no_defect_tuning/`
- `x224/two_block/`
- `x224/two_block_no_defect_tuning/`

## Common Files In A Branch

- `README.md`
- `notebook.ipynb`
- local config files when used by the branch
- `artifacts/checkpoints/`
- `artifacts/plots/`
- `artifacts/results/`
