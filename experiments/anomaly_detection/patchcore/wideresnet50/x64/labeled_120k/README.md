# WRN50-2 PatchCore (`x64`, labeled 120k follow-up)

This branch studies the same WideResNet50-2 PatchCore model on a larger labeled split:

- `120,000` training wafers with `6,000` anomalies
- `10,000` validation wafers with `500` anomalies
- `20,000` test wafers with `1,000` anomalies

The purpose of the branch is to examine how the strongest `x64` WRN PatchCore setting behaves when more labeled data is available and when threshold policy becomes a larger part of deployment quality.

## Files

- `notebook.ipynb`
  Main local training and evaluation workflow for the labeled `120k` split.
- `dataset_helper.ipynb`
  Split-construction and dataset validation notebook for the labeled branch.
- `results_review.ipynb`
  Lightweight local review notebook for saved results from the main run.
- `threshold_policies.ipynb`
  Post-hoc threshold and review-band analysis for the selected saved variant.
- `helpers/`
  Shared local helper modules used by the notebooks in this folder.
- `artifacts/`
  Local output root populated by the training notebook and consumed by the review notebooks.

## Why This Branch Exists

The main benchmark emphasizes comparability across families. This follow-up isolates a different question: whether the strongest WideResNet50-2 PatchCore recipe remains stable when the split is enlarged and when threshold policy is tuned more explicitly for downstream handling.
