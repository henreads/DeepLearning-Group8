# TS-ResNet50 Teacher-Student Distillation

This is the main local `TS-ResNet50` experiment notebook for the curated `x64` benchmark split.

## What This Notebook Does

- reuses the saved local checkpoint and evaluation outputs by default
- can retrain the model locally when you explicitly enable the training flag
- regenerates evaluation plots in the notebook and saves them under `artifacts/ts_resnet50/plots/`
- includes an optional score-sweep follow-up, but it is disabled by default

## Default Behavior

The notebook is configured to avoid retraining, reevaluation, and score sweeps unless you explicitly enable those flags.

- checkpoints: `artifacts/ts_resnet50/checkpoints/`
- results: `artifacts/ts_resnet50/results/`
- plots: `artifacts/ts_resnet50/plots/`

## Main Result Branches

- `results/evaluation/`: default evaluation summary, score CSVs, threshold sweep, and score-sweep outputs
- `results/evaluation_holdout70k_3p5k/`: holdout evaluation outputs when available
