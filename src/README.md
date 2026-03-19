# `src/` Package Guide

The notebooks are the main way to run this project.

The code under `src/wafer_defect/` is the reusable library those notebooks import so logic does not have to be copied across notebooks.

## Folder Map

- `data/`
  Loads processed wafer arrays and handles the legacy raw pickle format.
- `classification/`
  Groups the supervised multiclass classifier code so it stays separate from the anomaly-detection pipeline.
- `models/`
  Defines the model architectures used in experiments such as the autoencoder, VAE, SVDD encoder, PatchCore model, EfficientAD model, and ResNet feature extractor.
- `training/`
  Contains small training helpers such as one-epoch loops and PatchCore memory-bank utilities.
- `evaluation/`
  Converts anomaly scores into metrics like precision, recall, F1, AUROC, and threshold sweeps.
- `scoring.py`
  Computes anomaly scores from model outputs.
- `config.py`
  Loads TOML config files.

## How To Read It

If you are following a notebook, the usual flow is:

1. `data/` loads the metadata CSV and wafer arrays.
2. `models/` builds the model.
3. `training/` runs one epoch or prepares fitting helpers.
4. `scoring.py` turns outputs into anomaly scores.
5. `evaluation/` summarizes those scores into metrics.

## Why `training/` Exists

`src/training/` is not a second set of runnable scripts.
It only stores reusable functions that the notebooks or helper scripts call.

Example:

- a notebook cell is the main entry point in this project
- `src/wafer_defect/training/autoencoder.py` contains the reusable `run_autoencoder_epoch(...)` function that the script or notebook calls
