# `src/` Package Guide

The project is notebook-first, but the reusable implementation lives under `src/wafer_defect/`.

Use this folder when you want to understand the shared code that the curated notebooks and helper scripts import instead of duplicating logic cell by cell.

## What Lives Here

- `wafer_defect/data/`
  Dataset loading utilities, processed-array readers, and the legacy WM-811K pickle compatibility loader.
- `wafer_defect/models/`
  Model definitions for anomaly detection and backbone feature extraction.
- `wafer_defect/training/`
  Reusable training loops and fitting helpers used by notebooks and CLI scripts.
- `wafer_defect/evaluation/`
  Evaluation helpers for reconstruction metrics and UMAP-style analysis.
- `wafer_defect/classification/`
  Supervised multiclass classifier code kept separate from the anomaly-detection pipeline.
- `wafer_defect/scoring.py`
  Shared scoring helpers that turn model outputs into anomaly scores.
- `wafer_defect/config.py`
  Lightweight TOML config loading utilities.

## Recommended Reading Order

If you are following one of the curated notebooks, the usual flow is:

1. `wafer_defect/data/` loads metadata and wafer arrays.
2. `wafer_defect/models/` defines the model or feature extractor.
3. `wafer_defect/training/` provides the reusable training or memory-bank logic.
4. `wafer_defect/scoring.py` computes anomaly scores.
5. `wafer_defect/evaluation/` summarizes those scores into metrics or visual analysis.

## Important Boundary

`src/` is not a second experiment tree.

The entry points for the submission remain the curated notebooks under `data/` and `experiments/`. The code in `src/` exists so those notebooks can stay readable and reproducible.

## More Detail

See [wafer_defect package guide](c:\Users\User\Desktop\Term 8\Deep Learning\Project\DeepLearning-Group8\src\wafer_defect\README.md) for the package-level map.
