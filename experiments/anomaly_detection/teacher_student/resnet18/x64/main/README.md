# TS-ResNet18 Main

This is the main local teacher-student baseline for the `x64` benchmark split.

## What This Notebook Does

- loads the local experiment config and dataset config
- reuses the saved TS-ResNet18 checkpoint by default
- reuses the saved default evaluation outputs by default
- includes optional layer and top-k ablation sweeps, but they are disabled by default for submission use
- includes a score-sweep analysis on top of the saved checkpoint

## Default Behavior

The notebook is configured to reuse saved artifacts unless you explicitly enable the rerun flags.

- checkpoint root: `artifacts/ts_resnet18/checkpoints/`
- results root: `artifacts/ts_resnet18/results/`
- plots root: `artifacts/ts_resnet18/plots/`

## Main Files

- `train_config.toml`
- `data_config.toml`
- `notebook.ipynb`

## Notes

The ablation sweep writes generated configs under `artifacts/generated_configs/` and per-variant runs under `artifacts/ablation_variants/` when it is explicitly enabled.
