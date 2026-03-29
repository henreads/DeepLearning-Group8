# WRN50-2 PatchCore (`x64`, local all-in-one run)

This branch keeps the original `64x64` WideResNet50-2 PatchCore benchmark as a self-contained local notebook.

The notebook rebuilds the benchmark split from the raw `LSWMD.pkl`, trains the multilayer PatchCore model with `layer2 + layer3`, runs the small scoring sweep used in the report follow-up, and writes artifacts into the local `artifacts/` folder.

## Files

- `notebook.ipynb`
  Canonical local training and evaluation workflow for the `x64` WRN50 PatchCore run.
- `train_config.toml`
  Snapshot of the branch configuration.
- `data_config.toml`
  Snapshot of the dataset settings used by the run.
- `artifacts/`
  Local output root for checkpoints, score files, plots, and summaries created by the notebook.

## Relationship To The `x224` Branches

The higher-resolution `x224` branches remain the stronger curated WRN PatchCore follow-ups. This `x64` branch is the direct local benchmark counterpart that matches the main `50k / 5%` experiment setting.
