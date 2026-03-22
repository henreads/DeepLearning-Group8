# Notebook Layout

The notebook folders are split by experiment family and dataset setup:

- `anomaly_50k/`
  Main anomaly-detection experiment sequence for the original `50k`-normal benchmark split used across notebooks `01-20`.
- `anomaly_120k_labeled/`
  Larger labeled anomaly-detection workflows grouped by model family.
  Current contents include the root-level WRN50 PatchCore workflow and `ts_resnet50/` for the single-test teacher-student ResNet50 benchmark.
- `classifier/`
  Multiclass wafer-defect classification and pseudo-labeling workflow.

Use the repo root README for dataset-prep commands and the experiment-specific folder README files for run details.
