# TS-WideResNet50-2 Multilayer Self-Contained (x224)

This branch is the x224 multilayer WideResNet50-2 teacher-student experiment. It uses teacher layers `layer2` and `layer3` on the x224 benchmark split.

## What This Notebook Does

- loads the local x224 config and dataset setup
- defines the multilayer experiment inline so the notebook is self-contained
- supports retraining and score-sweep analysis when explicitly enabled

## Default Behavior

The notebook is configured to avoid retraining by default. When retraining is skipped it looks for a saved local checkpoint, reuses any saved training history and score artifacts, and regenerates plots and summaries into the local artifact folders.

- checkpoint root: `artifacts/ts_wideresnet50_multilayer_x224/checkpoints/`
- results root: `artifacts/ts_wideresnet50_multilayer_x224/results/`
- plots root: `artifacts/ts_wideresnet50_multilayer_x224/plots/`

## Files

- `notebook.ipynb`
  Canonical train-or-reuse notebook for this branch.
- `train_config.toml`
  Local configuration for the x224 run.
- `data_config.toml`
  Dataset metadata paths for the x224 processed wafer maps.
- `artifacts/`
  Output root for checkpoints, results, and plots after a local run.
