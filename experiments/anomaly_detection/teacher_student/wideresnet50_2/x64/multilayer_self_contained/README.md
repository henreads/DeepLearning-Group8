# TS-WideResNet50-2 Multilayer Self-Contained

This is the self-contained multilayer WideResNet50-2 teacher-student notebook. It uses teacher layers `layer2` and `layer3`.

## What This Notebook Does

- loads the local config and dataset setup
- defines the multilayer experiment inline so the notebook is self-contained
- supports retraining and score-sweep analysis when you explicitly enable them

## Default Behavior

The notebook is configured to avoid retraining by default. When retraining is skipped it loads the saved local multilayer checkpoint, reuses the saved training history and score artifacts, and regenerates plots and summaries into the local artifact folders.

- checkpoint root: `artifacts/ts_wideresnet50_multilayer/checkpoints/`
- results root: `artifacts/ts_wideresnet50_multilayer/results/`
- plots root: `artifacts/ts_wideresnet50_multilayer/plots/`

## Notes

This branch now keeps a single canonical artifact layout. The notebook is the source of truth for generating the checkpoint-backed results, summaries, score-sweep files, and plots stored under `artifacts/`.

The sibling `layer2_self_contained/` branch is the checked-in single-layer WideResNet50-2 path.
