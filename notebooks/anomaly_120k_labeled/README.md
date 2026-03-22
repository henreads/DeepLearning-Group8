# Anomaly 120k Labeled Notebooks

This folder groups the larger labeled anomaly-detection workflows by model family.

Current subfolders:

- `patchcore_wrn50/`
  Dedicated WideResNet50-2 PatchCore workflow for the labeled `120k / 10k / 20k` split.

Why this folder exists:

- to make the larger labeled workflows visually separate from the original `50k` benchmark notebooks
- to keep model-family-specific notebook groups under one larger-dataset namespace
- to leave room for future `120k` workflows beyond PatchCore without crowding the top-level `notebooks/` directory
