# PatchCore ViT-B/16 Fine-Tuning Experiments (`x224`)

This folder groups experiments that apply self-supervised fine-tuning to the ViT-B/16 backbone before building a PatchCore memory bank. Each subfolder is a standalone experiment with its own notebook, config, and artifacts.

## Branches

- `MAE/`
  Masked patch reconstruction (MAE-style) fine-tuning. Randomly masks 75% of patch tokens and trains the backbone to reconstruct missing patch content. Best AUPRC in the ViT family (0.717 vs 0.671 frozen).
