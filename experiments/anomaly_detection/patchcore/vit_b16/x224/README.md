# ViT-B/16 PatchCore (`x224`)

This resolution folder groups the ViT-B/16 PatchCore branches that share the `x224` input pipeline.

## Why `x224`

ViT-B/16 is naturally used at a larger image scale than the benchmark `x64` branches, so this folder collects the transformer follow-ups built around the `x224` processing pipeline.

## Branches

- `main/`
  Main ViT-B/16 PatchCore reference branch.
- `one_layer_defect_tuning/`
  One-layer branch with defect-aware threshold tuning.
- `one_layer_no_defect_tuning/`
  One-layer branch with normal-only threshold selection.
- `two_block/`
  Two-block transformer feature branch.
- `two_block_no_defect_tuning/`
  Two-block branch with normal-only threshold selection.
