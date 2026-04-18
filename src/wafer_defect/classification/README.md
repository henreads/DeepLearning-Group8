# Classification Package

This package contains the supervised multiclass wafer-defect classification code that is separate from the anomaly-detection pipeline.

Modules:

- `data.py` for labeled and unlabeled supervised dataset preparation
- `models.py` for classifier architectures
- `ensemble.py` for averaged and stacked ensemble helpers

This code is used by the classifier notebooks and by the CLI utilities in `scripts/classifier/`.
