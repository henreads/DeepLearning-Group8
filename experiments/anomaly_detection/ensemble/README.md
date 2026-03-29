# Ensemble Family

This family contains post-hoc score-combination experiments.

The ensemble branches do not train a new base anomaly model. Instead, they combine saved scores from stronger anomaly branches to study whether fusion improves robustness or threshold behavior.

## Why This Family Exists

Different anomaly models can be strong in different ways:

- one model may rank defects well
- another may achieve a better operating threshold
- another may recover a specific defect type more reliably

The ensemble family studies whether combining those saved outputs yields a more stable final detector.

## Branches

- `x64/score_ensemble/`
  Main score-combination notebook.

## Common Files In A Branch

- `README.md`
- `notebook.ipynb`
- `artifacts/results/`
- `artifacts/plots/`
