# Data

This folder contains dataset infrastructure for the whole project.

## Layout

- `raw/`
  Local raw source files such as the WM-811K pickle.
- `processed/`
  Generated arrays and metadata used by training and evaluation notebooks.
- `dataset/`
  Dataset-preparation and dataset-validation notebooks grouped by resolution and split variant.

## Why `dataset/` Lives Here

The notebooks under `data/dataset/` are not model experiments. They exist to:

- verify that dataset preparation is working
- document how each benchmark split is created
- keep different dataset variants organized in one place

That makes `data/` the right home for them.

## Variant Convention

Dataset variants should be grouped by:

1. resolution, such as `x64`, `x128`, `x224`, or `x240`
2. split protocol, such as `benchmark_50k_5pct` or `holdout70k_3p5k`

Example structure:

```text
data/dataset/
  x64/
    benchmark_50k_5pct/
    holdout70k_3p5k/
  x128/
    benchmark_50k_5pct/
  x224/
    benchmark_50k_5pct/
  x240/
    benchmark_50k_5pct/
```

Not every branch needs to exist immediately, but this is the intended organization as we clean up the submission tree.
