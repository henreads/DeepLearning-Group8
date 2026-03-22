PatchCore WRN50 120k Labeled Notebooks

This folder keeps the dedicated WideResNet50-2 PatchCore work for the `120k / 10k / 20k` labeled split inside the larger `notebooks/anomaly_120k_labeled/` namespace.

Contents

- `1_patchcore_wideresnet50_dataset_helper.ipynb`
- `1_patchcore_wideresnet50_multilayer_training.ipynb`
- `2_patchcore_wideresnet50_modal_results.ipynb`
- `3_patchcore_threshold_policies.ipynb`
- `helpers/patchcore_wrn50_modal.py`
- `helpers/patchcore_wrn50_kaggle.py`
- `helpers/patchcore_threshold_tools.py`

Why this folder exists

- to avoid colliding with the existing top-level PatchCore notebook naming
- to keep this labeled `120k / 10k / 20k` workflow grouped under one notebook namespace
- to make the local notebook copies runnable without depending on the Kaggle upload folder
- to keep the working notebook copies aligned with the newer Modal bundle under `modal_upload/`
- to keep the Modal training review notebook next to the helper and training notebooks

Latest Modal run

- run date: `2026-03-20`
- local artifact folder: `outputs/modal_runs/patchcore_wrn50_120k_20260320/patchcore_wrn50_multilayer_120k_5pct`
- review notebook: `2_patchcore_wideresnet50_modal_results.ipynb`
- selected variant: `topk_mb50k_r005`
- default validation-normal quantile threshold: `0.559844`
- validation-F1 sweep threshold: `0.571337`

Held-out test results

- test split size: `20,000` with `1,000` anomalies and `19,000` normal wafers
- AUROC: `0.9273`
- AUPRC: `0.5382`
- precision: `0.4206`
- recall: `0.7120`
- F1: `0.5288`
- accuracy: `0.9366`
- balanced accuracy: `0.8302`
- confusion matrix: `TN=18,019`, `FP=981`, `FN=288`, `TP=712`
- predicted anomalies at deployment threshold: `1,693`

Validation F1 sweep update

- review csv: `outputs/modal_runs/patchcore_wrn50_120k_20260320/patchcore_wrn50_multilayer_120k_5pct/validation_f1_threshold_review.csv`
- per-variant validation sweep: `val_threshold_sweep.csv`
- recommended threshold for `topk_mb50k_r005`: `0.571337`
- held-out test precision / recall / F1 at that threshold: `0.5147 / 0.6120 / 0.5592`
- held-out test confusion matrix at that threshold: `TN=18,423`, `FP=577`, `FN=388`, `TP=612`
- held-out predicted anomalies at that threshold: `1,189`
- delta versus the default quantile threshold: `+0.0304` F1, `-404` false positives, `-0.1000` recall

Interpretation

- The model separates normal and anomalous wafers well overall. The `0.9273` AUROC shows that the score ranking is strong even before choosing a threshold.
- The current threshold is recall-leaning. It catches `712 / 1,000` defects, but it also flags `981` normal wafers as anomalous, so precision stays around `42%`.
- The validation-F1 threshold is a cleaner operating point for this WRN50 run. It lifts held-out F1 from `0.5288` to `0.5592` and cuts false positives from `981` to `577`, but it gives up `100` true positives on the test split.
- Relative to the actual anomaly rate of `5%`, the model predicts about `8.5%` of the test set as anomalous (`1,693 / 20,000`). That makes it useful as a screening model, but still noisy if used as a direct pass/fail gate.
- With the validation-F1 threshold, the predicted anomaly rate drops to about `5.9%` (`1,189 / 20,000`), which is much closer to the true anomaly prevalence while still preserving a little over `61%` recall.
- The top-k aggregation variants are very close, but `topk_mb50k_r005` narrowly beat the report-style `topk_mb50k_r010` on this larger labeled split in F1, AUROC, and AUPRC.
- The plain mean-reduction baseline underperformed the top-k variants by a visible margin, so the PatchCore top-k scoring choice still matters here.
- The saved test errors show that many misses come from `Center`, `Edge-Loc`, and `Loc` defects, which suggests the remaining false negatives are concentrated in more subtle or spatially compact defect patterns.

Practical takeaway

- This is a solid anomaly-ranking model for the current setup, and it remains one of the stronger candidates for a recall-oriented defect triage stage.
- If we want the best single-threshold F1 from the saved WRN50 run, `0.571337` is the better recommendation than the old `0.559844` quantile threshold.
- If we want a more recall-heavy triage rule instead, we should keep the lower threshold and treat the extra false positives as part of the screening cost.
