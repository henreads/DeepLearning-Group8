# x224 Dataset Variants

This folder contains dataset-preparation notebooks that generate `224 x 224` inputs.

## Why This Resolution Exists

- several pretrained vision backbones are naturally configured around `224 x 224`
- PatchCore and transformer-style backbones often benefit from the extra spatial detail
- keeping `x224` separate avoids mixing preprocessing assumptions with the lighter `x64` benchmark

## Current Variant

- `benchmark_50k_5pct/`
  The curated `224 x 224` version of the main benchmark split.
