# Experiments

This tree contains the submission-facing experiment notebooks.

The experiment folders are grouped by task family rather than by the older numbered notebook sequence.

## Top-Level Structure

- `anomaly_detection/`
  Unsupervised and one-class anomaly-detection experiments.
- `classifier/`
  Supervised multiclass wafer-defect classification experiments.
- `archive/`
  Secondary or exploratory branches kept for reference rather than as the primary submission path.

## Branch Convention

Most experiment branches follow the same structure:

```text
README.md
notebook.ipynb
train_config.toml
data_config.toml
artifacts/
```

Not every branch uses both config files, because some older workflows kept part of the setup directly in the notebook. When config files are present, they are stored beside the notebook so the run settings stay local to the branch.

## Artifact Convention

Where possible, branch artifacts are organized into:

- `artifacts/checkpoints/`
- `artifacts/plots/`
- `artifacts/results/`

This keeps weights, figures, and tabular outputs separate and makes it easier to inspect a branch without reopening the notebook.

## Relationship To `data/dataset/`

Dataset construction and dataset validation are kept outside this tree in `data/dataset/`.

That separation makes `experiments/` the model-facing part of the repo, while `data/dataset/` remains the place for dataset generation and split validation.
