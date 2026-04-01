# Multiclass Classifier Family

This family contains the supervised wafer-defect classification workflow.

Unlike the anomaly branches, these experiments are trained directly on labeled defect classes and are evaluated as multiclass predictors.

## Why This Family Exists

The classifier family supports three goals:

- reporting supervised classification performance on known defect types
- generating pseudo-label candidates for unlabeled wafers
- exporting seed-specific follow-up analysis such as pseudo-label review and UMAP views

## Branches

- `x64/training/`
  Main classifier training workflow.
- `x64/showcase/`
  Visualization and presentation-oriented analysis.
- `x64/final_labeling/`
  Final pseudo-label generation workflow.
- `x64/seed07/`
  Seed07 full-labeled checkpoint review and pseudo-label workflow.
- `x64/umap/`
  UMAP follow-up analysis built on the seed07 artifacts.

## Common Files In A Branch

- `README.md`
- `notebook.ipynb`
- local config files when the branch uses them
- saved artifacts or exported result files associated with the classifier workflow
