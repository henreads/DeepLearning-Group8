# x240 Dataset Variants

This folder contains dataset-preparation notebooks that generate `240 x 240` inputs.

## Why This Resolution Exists

- some EfficientNet branches in this project were tested at `240 x 240`
- the slightly larger input size follows the expected preprocessing for those backbones more closely
- separating it here keeps the preprocessing story reproducible for graders

## Current Variant

- `benchmark_50k_5pct/`
  The curated `240 x 240` version of the main benchmark split used for EfficientNet-style runs.
