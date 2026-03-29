# `scripts/` Guide

This folder contains command-line helpers that support the notebook workflows.

The project is still notebook-first, so `scripts/` should be read as supporting infrastructure rather than the main submission surface.

## Top-Level Scripts

These are the stable scripts that are still directly useful to the curated workflows:

- `prepare_wm811k.py`
  Builds processed dataset artifacts from the raw WM-811K pickle using a config file.
- `build_secondary_holdout_metadata.py`
  Builds the secondary holdout metadata variants used by some follow-up evaluations.
- `evaluate_reconstruction_model.py`
  Shared evaluator for reconstruction-based checkpoints.
- `evaluate_autoencoder_scores.py`
  Autoencoder score-ablation helper.
- `generate_umap_analysis.py`
  UMAP export helper used by some follow-up analyses.
- `batch_evaluate_holdout.py`
  Batch evaluator for holdout-style result bundles.
- `train_svdd.py`
  CLI runner used by the curated SVDD notebook.
- `train_vae.py`
  CLI runner used by the curated VAE notebooks.
- `train_ts_distillation.py`
  CLI runner used by the curated teacher-student notebooks.
- `run_fastflow_x64_notebook.py`
  Helper for running the FastFlow x64 notebook flow from the command line.

## Subfolders

- `classifier/`
  Supervised multiclass classifier data prep, training, inference, and ensembling utilities.
- `dev/`
  Repo-maintenance, curation, migration, and artifact-normalization tools used during submission cleanup.

## Submission Note

Graders do not need to start here.

The normal entry point is:

1. `data/dataset/.../notebook.ipynb` to create processed datasets
2. `experiments/.../notebook.ipynb` to run or review the actual experiments

Use the scripts only when a notebook explicitly calls one of them or when you want a CLI alternative.
