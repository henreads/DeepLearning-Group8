# WRN50-2 PatchCore (`x64`, local all-in-one run)

This branch keeps the original `64x64` WideResNet50-2 PatchCore benchmark as a self-contained local notebook.

The notebook rebuilds the benchmark split from the raw `LSWMD.pkl`, trains the multilayer PatchCore model with `layer2 + layer3`, runs the baseline and follow-up scoring sweeps, and writes the saved local artifacts into the branch `artifacts/` folder.

## Files

- `notebook.ipynb`
  Canonical local training and evaluation workflow for the `x64` WRN50 PatchCore run.
- `train_config.toml`
  Snapshot of the branch configuration.
- `data_config.toml`
  Snapshot of the dataset settings used by the run.
- `artifacts/patchcore_wideresnet50_multilayer/`
  Local output root for the WRN50 x64 sweep.
  It contains per-variant score CSVs and summaries, combined sweep tables, the processed benchmark metadata snapshot, and generated plots.

## Saved Outputs

- `patchcore_sweep_results.csv`
  Baseline sweep results for the main variants.
- `patchcore_follow_up_sweep_results.csv`
  Follow-up sweep results around the best configuration.
- `patchcore_combined_sweep_results.csv`
  Combined benchmark table across the baseline and follow-up variants.
- `plots/`
  Sweep-level comparison figures plus selected-variant review plots.
- `<variant>/plots/`
  Per-variant score-distribution, confusion-matrix, and defect-breakdown plots generated from the saved local score files.

## Relationship To The `x224` Branches

The higher-resolution `x224` branches remain the stronger WRN PatchCore follow-ups. This `x64` branch is the direct local benchmark counterpart that matches the main `50k / 5%` experiment setting.
