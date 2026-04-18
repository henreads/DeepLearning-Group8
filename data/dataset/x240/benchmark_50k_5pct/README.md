# x240 Benchmark 50k 5pct

This folder contains the dataset notebook and config for the `240 x 240` version of the main `50k` benchmark split.

## Purpose

- regenerate the same benchmark protocol as the `x64` branch
- export arrays at `240 x 240` for the EfficientNet-oriented experiments
- verify that the written metadata and arrays under `data/processed/x240/wm811k/` are consistent

## Contents

- `notebook.ipynb`
  Dataset generation and validation notebook for this branch.
- `data_config.toml`
  Config snapshot for the `240 x 240` benchmark variant.
