# x64 Dataset Variants

This resolution folder contains the dataset-preparation branches for `64 x 64` wafer maps.

## Why We Keep x64 Separate

- many baseline models were first developed on `x64`
- it is the lightest resolution for fast debugging and ablation work
- it lets us verify split logic before scaling to heavier backbones and larger inputs

## Current Variants

- `benchmark_50k_5pct/`
  The main curated benchmark branch for the standard `50k` setup with the `5%` defect evaluation mix.
- `holdout70k_3p5k/`
  A secondary evaluation branch that keeps the original `50k 5%` train and validation rows but swaps in a larger disjoint test holdout.
