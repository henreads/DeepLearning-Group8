# `wafer_defect` Package

This package contains the shared implementation behind the curated notebooks in `experiments/` and `data/`.

## Package Map

- `data/`
  Loads processed dataset artifacts and handles the legacy `LSWMD.pkl` format.
- `models/`
  Defines the anomaly-detection models and backbone feature extractors.
- `training/`
  Reusable training loops, memory-bank construction code, and model-specific fit helpers.
- `evaluation/`
  Shared evaluation helpers such as reconstruction metrics and UMAP reference utilities.
- `classification/`
  Supervised multiclass classification helpers that are intentionally separate from the anomaly workflows.
- `scoring.py`
  Shared utilities for turning raw outputs into anomaly scores.
- `config.py`
  Small config helpers used by scripts and notebooks.

## How This Connects To The Submission

The notebooks are still the primary entry points for graders.

This package matters because it keeps those notebooks short and reproducible:

- dataset notebooks call `wafer_defect.data.*`
- anomaly notebooks import from `wafer_defect.models.*`, `wafer_defect.training.*`, and `wafer_defect.scoring`
- classifier notebooks and scripts use `wafer_defect.classification.*`

## Stable Vs Local Utilities

Most files in this package are stable shared code.

The only file that intentionally exists as a compatibility shim is:

- `data/legacy_pickle.py`
  This patches older pandas pickle module paths so the raw WM-811K pickle can still be loaded on a modern environment.
