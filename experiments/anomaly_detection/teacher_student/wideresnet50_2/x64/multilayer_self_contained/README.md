# TS-WideResNet50-2 Multilayer Self-Contained

This is the self-contained multilayer WideResNet50-2 teacher-student notebook. It uses teacher layers `layer2` and `layer3`.

## What This Notebook Does

- loads the local config and dataset setup
- defines the multilayer experiment inline so the notebook is self-contained
- supports retraining and score-sweep analysis when you explicitly enable them

## Default Behavior

The notebook is a self-contained multilayer branch. It is configured to avoid retraining by default. The repo now includes the artifact folder structure plus results and plot snapshots extracted from the saved notebook output, but it still does not ship with a checked-in multilayer checkpoint.

- checkpoint root: `artifacts/ts_wideresnet50_multilayer/checkpoints/`
- results root: `artifacts/ts_wideresnet50_multilayer/results/`
- plots root: `artifacts/ts_wideresnet50_multilayer/plots/`

## Notes

The sibling `layer2_self_contained/` branch is the saved artifact-backed WideResNet50-2 path. Use that folder for the checked-in single-layer run.
