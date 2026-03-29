# x64 Holdout 70k 3p5k

This folder contains the dataset notebook and config for the secondary `70k` normal and `3.5k` defect holdout evaluation split.

## Purpose

- keep the original `50k 5%` train and validation rows unchanged
- replace the test split with a larger disjoint holdout sampled from unused WM-811K rows
- regenerate the exported metadata and arrays used by holdout evaluation runs

## Contents

- `notebook.ipynb`
  Holdout generation and validation notebook for this branch.
- `data_config.toml`
  Config snapshot for the `70k / 3.5k` holdout variant.
