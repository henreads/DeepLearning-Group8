# x64 Benchmark 50k 5pct

This folder contains the dataset notebook and config for the curated `64 x 64` benchmark split used across the submission.

## Purpose

- prepare the processed dataset files for the `50k` benchmark setup
- verify that the split and metadata are correct before running model notebooks
- keep the exact dataset configuration next to the notebook that uses it

## Contents

- `notebook.ipynb`
  Dataset preparation and validation notebook for this benchmark branch.
- `data_config.toml`
  Config snapshot used by the notebook.

Future dataset variants should follow the same pattern in their own split-specific folders.
