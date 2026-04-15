# Dataset Variants

This folder groups the dataset-generation and dataset-check notebooks used by the project.

## What Belongs Here

- notebooks that build a processed dataset variant
- notebooks that sanity-check split sizes, labels, and metadata
- config snapshots for a specific dataset protocol

## Organization Rule

Dataset notebooks are grouped in two stages:

1. by image resolution such as `x64`, `x128`, `x224`, or `x240`
2. by split protocol such as `benchmark_50k_5pct` or `holdout70k_3p5k`

That means each concrete dataset branch should look like:

```text
data/dataset/<resolution>/<split_variant>/
  README.md
  notebook.ipynb
  data_config.toml
```

## Why There Are Multiple Dataset Variants

Different parts of the project use different data setups:

- different image resolutions change the visual detail available to the model
- different benchmark sizes change how much normal training data is available
- different evaluation protocols change how we stress-test the final pipeline

Keeping these variants together under `data/dataset/` makes the preparation logic easier to audit and easier for graders to navigate.

## Current State

- `x64/benchmark_50k_5pct/` is the curated baseline benchmark branch used by most experiments
- `x128/benchmark_50k_5pct/` is the higher-resolution reconstruction branch used by the `x128` autoencoder baseline
- `x224/benchmark_50k_5pct/` is the higher-resolution benchmark branch for larger pretrained backbones
- `x240/benchmark_50k_5pct/` is the EfficientNet-oriented benchmark branch
- `x240/benchmark_120k_5pct/` is the larger EfficientNet-oriented follow-up that keeps the same `5%` anomaly test rule while scaling the normal-only pool to `120k`
- `x64/holdout70k_3p5k/` is the secondary holdout branch that preserves the base train and validation rows while replacing the test split
