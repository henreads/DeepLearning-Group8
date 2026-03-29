# Classifier Experiments

This folder contains the supervised multiclass wafer-defect classification work.

Unlike the anomaly-detection families, these branches are trained with explicit defect labels and are evaluated as direct class-prediction models.

## Main Family

- `multiclass/`
  Residual CNN classifier training, showcase analysis, pseudo-label generation, and ensemble evaluation.

## Common Branch Files

Most classifier branches contain:

- `README.md`
- `notebook.ipynb`
- local config files when the branch uses them
- `artifacts/` or saved external outputs associated with the run
