## Classifier Notebooks

This folder contains the supervised multiclass wafer-defect classification notebooks that were moved out of the main anomaly-detection notebook sequence to reduce top-level clutter.

Files here cover:

- methodology notes
- classifier training
- classifier showcase / presentation views
- final pseudo-label generation
- ensemble workflow
- Kaggle retraining on the full labeled `80 / 10 / 10` split
- seed07-based pseudo-label export on the unlabeled WM-811K rows

Use `scripts/classifier/sync_notebook6_kaggle_outputs.py` after downloading notebook `6` Kaggle outputs into `outputs/kaggle_notebook6010fb082e` to sync the pseudo-label and UMAP summaries into the local classifier docs.
