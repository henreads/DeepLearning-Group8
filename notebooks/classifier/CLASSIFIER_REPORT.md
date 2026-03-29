# Multiclass Classifier Development Report

**Date:** March 18, 2026

## Scope

This report summarizes the current state of the supervised multiclass wafer-defect classifier in this repository. It focuses on labeled-data preparation, model development, test-set results, ensemble behavior, and pseudo-label readiness.

This is a separate report from the anomaly-detection summary in `REPORT.md`.

## Development Snapshot

- The classifier currently targets `9` classes: `none`, `Center`, `Donut`, `Edge-Loc`, `Edge-Ring`, `Loc`, `Near-full`, `Random`, and `Scratch`.
- The main supervised dataset builder and training pipeline are already in place.
- The strongest validated artifacts today are the three seed-trained residual CNN checkpoints released under `kaggle_upload/Outputs/model_a`, `model_b`, and `model_c`.
- The best current deployment candidate for class-balanced performance is the simple averaged ensemble of those three checkpoints.
- A newer enhanced-classifier branch is implemented in code, but the saved artifact under `artifacts/multiclass_classifier_50k` appears to be a limited smoke run rather than a full benchmark, so it should not yet be used as the main reported result.
- Unlabeled inference, ensemble stacking, and Kaggle packaging utilities are already implemented.

## Dataset Status

From `data/processed/x64/wm811k_multiclass_50k/dataset_summary.txt`:

- total labeled rows available in WM-811K: `172,950`
- labeled rows sampled for the main classifier subset: `50,000`
- unlabeled rows still available for pseudo-labeling or inference: `638,507`
- split sizes: `35,000` train, `7,500` validation, `7,500` test

Class balance in the `50k` labeled subset:

| Class | Count | Share | Test Support |
| --- | ---: | ---: | ---: |
| `none` | 24,481 | 48.96% | 3,672 |
| `Edge-Ring` | 9,680 | 19.36% | 1,452 |
| `Edge-Loc` | 5,189 | 10.38% | 779 |
| `Center` | 4,294 | 8.59% | 644 |
| `Loc` | 3,593 | 7.19% | 539 |
| `Scratch` | 1,193 | 2.39% | 179 |
| `Random` | 866 | 1.73% | 130 |
| `Donut` | 555 | 1.11% | 83 |
| `Near-full` | 149 | 0.30% | 22 |

The dataset is still strongly imbalanced, especially for `Near-full`, `Donut`, `Random`, and `Scratch`, so balanced accuracy and per-class recall remain more informative than raw accuracy alone.

One important consequence of the current `50k` construction is that it already uses all `25,519` labeled defect wafers available in the raw WM-811K labels. The labeled remainder outside the `50k` subset is therefore `122,950` rows of `none` only. That means any secondary validation outside the `50k` set is a useful unseen-normal check, but it is not a full external multiclass holdout.

## Current Pipeline

### Validated baseline branch

The strongest saved results come from the baseline residual CNN path:

- `WaferClassifier` baseline variant from `src/wafer_defect/models/classifier.py`
- `base_channels = 48`
- `hidden_dim = 512`
- `dropout = 0.15`
- weighted sampling enabled
- label smoothing enabled at `0.05`
- AdamW optimizer with learning-rate decay and early stopping
- checkpoint selection by validation balanced accuracy

This branch is represented by:

- `configs/training/train_multiclass_classifier_50k_seed07.toml`
- `configs/training/train_multiclass_classifier_50k_seed13.toml`
- `configs/training/train_multiclass_classifier_50k_seed21.toml`
- `kaggle_upload/Outputs/model_a`
- `kaggle_upload/Outputs/model_b`
- `kaggle_upload/Outputs/model_c`

### Enhanced experimental branch

The repo also contains a more ambitious classifier branch:

- enhanced residual blocks with squeeze-excitation
- dual average/max pooled classifier head
- block dropout
- mixup
- gradient clipping

This branch is configured in `configs/training/train_multiclass_classifier_50k.toml`.

However, the saved artifact under `artifacts/multiclass_classifier_50k` is not a full `50k` benchmark run. Its saved reports only cover `256` train, `64` validation, and `64` test examples, which strongly suggests it was used as a smoke test. Because of that, it is not yet a fair comparison against the released full-data baseline checkpoints.

## Result Summary

### Single checkpoints

| Model | Best Epoch | Validation Balanced Accuracy | Test Accuracy | Test Balanced Accuracy | Test Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `model_a` | 24 | 0.9290 | 0.9320 | 0.9328 | 0.8916 |
| `model_b` | 60 | 0.9347 | 0.9433 | 0.9320 | 0.9105 |
| `model_c` | 44 | 0.9399 | 0.9432 | 0.9411 | 0.9148 |

Key takeaways:

- `model_c` is the strongest single checkpoint overall.
- `model_b` and `model_c` are close on accuracy, but `model_c` is better balanced across classes.
- Seed variation matters, but all three runs are strong enough to justify ensemble use.

### Ensemble results

| Ensemble Mode | Test Accuracy | Test Balanced Accuracy | Test Macro F1 | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Average probabilities | 0.9463 | 0.9456 | 0.9194 | Best overall class-balanced result |
| Stacking, selected by validation balanced accuracy | 0.9553 | 0.9344 | 0.9285 | Best raw accuracy, but weaker macro recall balance |
| Stacking, selected by validation accuracy | 0.9548 | 0.9272 | 0.9273 | Most accuracy-biased option |

Interpretation:

- Averaging improves on every single model in both accuracy and balanced accuracy.
- If the project priority is balanced performance across defect classes, the averaged ensemble is the safest headline result.
- Stacking increases raw test accuracy and macro F1, but it shifts the operating point toward higher precision and lower balanced recall. That tradeoff is visible in the test balanced-accuracy drop.

### Secondary external validation outside the `50k`

A secondary validation was run on a random `10,000`-row sample from the labeled rows not included in the multiclass `50k` dataset. Because the `50k` subset already consumed every labeled defect row, this outside-of-`50k` validation set contains only `none` wafers. It should therefore be interpreted as an unseen-normal false-positive check rather than a full multiclass generalization benchmark.

| Holdout | Sample Size | Label Composition | Average-Ensemble Accuracy | Stacked-Ensemble Accuracy | Interpretation |
| --- | ---: | --- | ---: | ---: | --- |
| Unseen labeled remainder outside `50k` | 10,000 | 10,000 `none` | 0.9352 | 0.9607 | Stacking reduces false defect predictions on unseen normals |

Additional notes:

- full unseen labeled remainder outside the `50k`: `122,950` rows
- full unseen label distribution outside the `50k`: all `none`
- average ensemble false defect predictions on the sampled external holdout: `648`
- stacked ensemble false defect predictions on the sampled external holdout: `393`

Interpretation:

- the external labeled remainder confirms that the main out-of-sample risk is false positives on normal wafers
- on this normal-only holdout, stacking is clearly more conservative than averaging and produces fewer incorrect defect flags
- these numbers should not replace the main multiclass test-set result because the outside-of-`50k` holdout contains no unseen defect classes

## Per-Class Behavior

Using the averaged ensemble:

- strongest large-support classes are `none`, `Edge-Ring`, `Center`, and `Random`
- `Donut` also performs well, but the class is small enough that its estimate is less stable
- `Near-full` reaches perfect recall on test, but the support is only `22`, so that result should be treated cautiously
- the hardest practical classes remain `Scratch` and `Loc`

Average-ensemble test performance for the weaker classes:

| Class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| `Scratch` | 0.6923 | 0.9050 | 0.7845 |
| `Loc` | 0.8099 | 0.8776 | 0.8424 |
| `Edge-Loc` | 0.8920 | 0.9114 | 0.9016 |

This suggests the model is already strong, but local defect families still overlap in feature space and produce the most meaningful errors.

## Error Patterns

Most common averaged-ensemble confusions on the test set:

- `none -> Loc`: `60`
- `none -> Edge-Loc`: `50`
- `none -> Center`: `47`
- `none -> Scratch`: `43`
- `Edge-Loc -> Loc`: `30`
- `Loc -> Scratch`: `25`

Main implication:

- the largest remaining error source is false positives on normal wafers
- the next biggest issue is confusion among the local defect families, especially `Edge-Loc`, `Loc`, and `Scratch`

Compared with averaging, stacking reduces some normal false positives and improves precision on difficult classes, but it also increases recall tradeoffs between `none`, `Edge-Loc`, and `Loc`, which is why its balanced accuracy is lower.

## Pseudo-Labeling Readiness

The repo is already set up for pseudo-label generation:

- `scripts/predict_unlabeled_multiclass.py`
- `scripts/predict_unlabeled_multiclass_ensemble.py`
- `scripts/ensemble_multiclass_classifier.py`

Current evidence from the saved smoke run in `artifacts/multiclass_classifier_50k_stacking_eval`:

- unlabeled rows evaluated: `1,000`
- acceptance threshold: `0.98`
- accepted pseudo-labels: `306`
- acceptance rate: `30.6%`
- all accepted rows were predicted as `none`

Interpretation:

- the current threshold is conservative and appears safe for harvesting high-confidence normal wafers
- it is not yet showing strong confident capture of defect pseudo-labels
- if the goal is to expand rare defect classes, the acceptance rule will likely need class-aware tuning or manual review support

<!-- BEGIN: NOTEBOOK6_KAGGLE_SYNC -->
### All-Labeled `seed07` Kaggle Pseudo-Label Run

Synced from `jikutopepega/notebook6010fb082e` into `outputs\seed07_pseudolabel_bundle_kaggle_outputs`.

- pseudo-label summary: `outputs\seed07_pseudolabel_bundle_kaggle_outputs\unlabeled_predictions.seed07.symmary.json`
- rows scored: `638,507`
- configured confidence threshold: `0.90`
- accepted pseudo-labels: `417,831` (65.44%)
- predicted defect fraction across all scored unlabeled rows: `36.12%`
- predicted defect fraction inside the accepted subset: `30.61%`
- mean confidence across all pseudo labels: `0.8637`
- mean confidence inside the accepted subset: `0.9475`

Confidence bucket review from the saved pseudo-label CSV:

| Threshold | Accepted Rows | Accepted Fraction | Defect Rows | `none` Rows |
| --- | --- | --- | --- | --- |
| 50% | 609,690 | 95.49% | 211,426 | 398,264 |
| 75% | 525,337 | 82.28% | 168,629 | 356,708 |
| 90% | 417,831 | 65.44% | 127,918 | 289,913 |

Per-label coverage across the saved pseudo-label export and both UMAP summaries:

| Label | All Scored | Accepted | Std UMAP | 10A UMAP |
| --- | --- | --- | --- | --- |
| `none` | 407,882 | 289,913 | 800 | 5,405 |
| `Center` | 21,294 | 13,825 | 800 | 285 |
| `Donut` | 613 | 364 | 465 | 12 |
| `Edge-Loc` | 33,053 | 10,958 | 800 | 304 |
| `Edge-Ring` | 15,074 | 6,254 | 800 | 147 |
| `Loc` | 25,583 | 7,248 | 800 | 233 |
| `Near-full` | 90,485 | 72,101 | 800 | 1,164 |
| `Random` | 13,223 | 9,486 | 800 | 182 |
| `Scratch` | 31,300 | 7,682 | 800 | 268 |

Standard classifier embedding UMAP:

- summary file: `outputs\seed07_pseudolabel_bundle_kaggle_outputs\umap_visualization\seed07_pseudolabel_umap.summary.json`
- labeled reference points: `3,349`
- pseudo-labeled points plotted: `6,865`
- mean plotted pseudo confidence: `0.9139`
- UMAP settings: `n_neighbors = 30`, `min_dist = 0.1`, `metric = cosine`

10A-style classifier UMAP:

- summary file: `outputs\seed07_pseudolabel_bundle_kaggle_outputs\umap_10a_style\umap_10a_style.summary.json`
- labeled normal points: `4,000`
- labeled defect points: `4,000`
- pseudo-labeled points plotted: `8,000`
- PCA dimension before UMAP: `50`
- UMAP settings: `n_neighbors = 15`, `min_dist = 0.1`, `metric = euclidean`

Interpretation:

- this notebook `6` export is the main artifact path for reviewing the unlabeled WM-811K pseudo-label distribution from the all-labeled `seed07` classifier
- the threshold table shows how much unlabeled coverage remains available at `50%`, `75%`, and the configured acceptance threshold
- the two UMAP summaries make it easier to compare broad classifier-space coverage against the stricter 10A-style PCA -> UMAP view
<!-- END: NOTEBOOK6_KAGGLE_SYNC -->

## Current Recommendation

Based on the saved development evidence in the repository:

1. Report the averaged three-model ensemble as the main classifier result.
2. Keep stacking as an optional accuracy-oriented variant rather than the primary balanced-performance result.
3. Treat the enhanced classifier branch as implemented but not yet fully benchmarked.
4. Use the all-labeled `seed07` pseudo-label export as the working unlabeled annotation source for downstream anomaly-threshold tuning, with the `0.90` accepted subset as the safer starting slice.
5. Review accepted defect-heavy classes such as `Near-full`, `Center`, `Edge-Loc`, and `Scratch` before promoting them to automatic downstream supervision, because the accepted subset now contains substantial non-`none` volume rather than only normal wafers.
6. If a true external multiclass validation is required, rebuild the training subset so some labeled defect rows are intentionally held out instead of using all labeled defects inside the `50k`.

## Relevant Files

- `scripts/prepare_wm811k_multiclass.py`
- `scripts/train_multiclass_classifier.py`
- `scripts/ensemble_multiclass_classifier.py`
- `scripts/predict_unlabeled_multiclass_ensemble.py`
- `scripts/classifier/sync_notebook6_kaggle_outputs.py`
- `configs/data/data_multiclass_50k.toml`
- `configs/training/train_multiclass_classifier_50k.toml`
- `configs/training/train_multiclass_classifier_50k_seed07.toml`
- `configs/training/train_multiclass_classifier_50k_seed13.toml`
- `configs/training/train_multiclass_classifier_50k_seed21.toml`
- `src/wafer_defect/data/supervised.py`
- `src/wafer_defect/models/classifier.py`
- `kaggle_upload/Outputs/model_a/metrics.json`
- `kaggle_upload/Outputs/model_b/metrics.json`
- `kaggle_upload/Outputs/model_c/metrics.json`
- `kaggle_upload/Outputs/ensemble_abc/metrics.json`
- `artifacts/multiclass_classifier_50k_eval_20260318/metrics.json`
- `artifacts/multiclass_classifier_50k_stacking_accuracy_eval/metrics.json`
- `artifacts/external_unseen_labeled_normal_sample10k_eval_20260318/metrics.json`
- `artifacts/multiclass_classifier_50k_stacking_eval/smoke_unlabeled_predictions.summary.json`
- `outputs/seed07_pseudolabel_bundle_kaggle_outputs/unlabeled_predictions.seed07.symmary.json`
- `outputs/seed07_pseudolabel_bundle_kaggle_outputs/unlabeled_predictions.seed07.pseudolabels.csv`
- `outputs/seed07_pseudolabel_bundle_kaggle_outputs/umap_visualization/seed07_pseudolabel_umap.summary.json`
- `outputs/seed07_pseudolabel_bundle_kaggle_outputs/umap_10a_style/umap_10a_style.summary.json`
