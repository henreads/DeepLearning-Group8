# SVDD Family

This family contains the Deep SVDD one-class baseline.

Instead of reconstructing an image, Deep SVDD learns a compact representation of normal wafers and treats distance from that normal region as the anomaly signal.

## Why This Family Exists

The SVDD branch provides a non-reconstruction baseline for the project.

It helps separate two questions:

- whether anomaly detection benefits mainly from reconstruction error
- whether a one-class feature-space objective can already separate normal and defective wafers effectively

## Branches

- `x64/baseline/`
  Main Deep SVDD benchmark branch.

## Common Files In A Branch

- `README.md`
- `notebook.ipynb`
- `train_config.toml`
- `artifacts/`
