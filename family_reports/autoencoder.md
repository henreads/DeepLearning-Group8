# Autoencoder Family Report

## Overview

This report covers all autoencoder-based anomaly detection experiments on the WM-811K wafer map dataset.

The family progresses from a plain reconstruction baseline through BatchNorm, dropout, and residual architectural variants, and across two image resolutions. Every branch evaluates the same checkpoint across multiple scoring rules to separate model quality from score design.

Notebooks and artifacts live under [`experiments/anomaly_detection/autoencoder/`](../experiments/anomaly_detection/autoencoder/).

---

## Dataset and Evaluation Protocol

- **Train:** 40,000 normal wafers
- **Validation:** 5,000 normal wafers
- **Test:** 5,000 normal + 250 anomaly wafers
- **Anomaly cap:** 5% of test-normal count
- **Threshold rule:** 95th percentile of validation-normal scores (deployment-style)
- **Primary metric:** val-threshold F1

Defect types in the test set: Edge-Ring (84), Center (50), Edge-Loc (53), Loc (34), Scratch (15), Donut (7), Random (5), Near-full (2).

---

## Family Summary

![Autoencoder family comparison](../artifacts/report_plots/autoencoder_family_comparison.png)

| experiment | model | score | image size | precision | recall | F1 | AUROC | AUPRC | best sweep F1 |
|---|---|---|---|---|---|---|---|---|---|
| AE-64-BN-max | AE + BatchNorm | `max_abs` | 64×64 | 0.401442 | 0.668000 | 0.501502 | 0.834023 | 0.568039 | 0.629808 |
| AE-64-BN-DO0.00 | AE + BN + Dropout 0.00 | `max_abs` | 64×64 | 0.393120 | 0.640000 | 0.487062 | 0.850790 | 0.616946 | 0.656642 |
| AE-64-BN-DO0.10 | AE + BN + Dropout 0.10 | `max_abs` | 64×64 | 0.385343 | 0.652000 | 0.484398 | 0.844670 | 0.570245 | 0.634615 |
| AE-64-BN-DO0.05 | AE + BN + Dropout 0.05 | `max_abs` | 64×64 | 0.377828 | 0.668000 | 0.482659 | 0.835035 | 0.551700 | 0.609959 |
| AE-64-BN-DO0.20 | AE + BN + Dropout 0.20 | `max_abs` | 64×64 | 0.370115 | 0.644000 | 0.470073 | 0.841431 | 0.574973 | 0.633929 |
| AE-64-topk | AE baseline | `topk_abs_mean` | 64×64 | 0.390374 | 0.584000 | 0.467949 | 0.839282 | 0.522171 | 0.509091 |
| AE-64-Res-max | Residual AE | `max_abs` | 64×64 | 0.374419 | 0.644000 | 0.473529 | 0.843360 | 0.588907 | 0.625592 |
| AE-64-topk-43ep | AE baseline (longer run) | `topk_abs_mean` | 64×64 | 0.381579 | 0.580000 | 0.460317 | 0.834819 | 0.525162 | 0.520661 |
| AE-64-Res-topk | Residual AE | `topk_abs_mean` | 64×64 | 0.356974 | 0.604000 | 0.448737 | 0.804607 | 0.626014 | 0.678133 |
| AE-64-BN-topk | AE + BatchNorm | `topk_abs_mean` | 64×64 | 0.346247 | 0.572000 | 0.431373 | 0.790020 | 0.603447 | 0.655172 |
| AE-128-topk | AE baseline | `topk_abs_mean` | 128×128 | 0.374631 | 0.508000 | 0.431239 | 0.814856 | 0.455031 | 0.479821 |
| AE-64-mse | AE baseline | `mse_mean` | 64×64 | 0.346154 | 0.504000 | 0.410423 | 0.809694 | 0.447970 | 0.473318 |

**Key finding:** Score choice matters as much as architecture. The same BatchNorm checkpoint ranks from F1=0.431 (`topk_abs_mean`) to F1=0.502 (`max_abs`). Always ablate scoring rules before concluding a model is weak.

---

## Experiment 1: Baseline Autoencoder `64×64`

**Notebook:** [`experiments/anomaly_detection/autoencoder/x64/baseline/notebook.ipynb`](../experiments/anomaly_detection/autoencoder/x64/baseline/notebook.ipynb)
**Artifact dir:** `experiments/anomaly_detection/autoencoder/x64/baseline/artifacts/autoencoder_baseline/`

### Configuration

| parameter | value |
|---|---|
| latent dimension | 128 |
| optimizer | Adam |
| learning rate | 0.001 |
| weight decay | 0.0001 |
| max epochs | 50 |
| early stopping patience | 5 |

### Training

| epoch | train loss | val loss |
|---|---|---|
| 1 | 0.026390 | 0.024768 |
| 10 | 0.024169 | 0.024185 |
| 20 | 0.020241 | 0.020260 |
| 25 | 0.019691 | 0.019755 |

- Best epoch: **38**, best val loss: **0.019262**, epochs ran: **43**

![Training curves](../experiments/anomaly_detection/autoencoder/x64/baseline/artifacts/autoencoder_baseline/plots/training_curves.png)

### Score Ablation

The same checkpoint was evaluated under seven scoring rules to find the strongest deployed signal.

| score | val-threshold F1 | AUROC | AUPRC | best sweep F1 |
|---|---|---|---|---|
| `topk_abs_mean` | **0.467949** | **0.839282** | **0.522171** | **0.509091** |
| `mse_mean` | 0.410423 | 0.809694 | 0.447970 | 0.473318 |
| `max_abs` | 0.323944 | 0.778442 | 0.239525 | 0.330341 |
| `foreground_mse` | 0.317263 | 0.760905 | 0.354126 | 0.377358 |
| `mae_mean` | 0.299835 | 0.762006 | 0.326166 | 0.352645 |
| `pooled_mae_mean` | 0.292763 | 0.754371 | 0.315182 | 0.342342 |
| `foreground_mae` | 0.239203 | 0.727066 | 0.278870 | 0.301370 |

![Score ablation summary](../experiments/anomaly_detection/autoencoder/x64/baseline/artifacts/autoencoder_baseline/plots/score_ablation_summary.png)

**Selected score:** `topk_abs_mean`

### Evaluation (selected score)

| metric | value |
|---|---|
| precision | 0.381579 |
| recall | 0.580000 |
| F1 | 0.460317 |
| AUROC | 0.834819 |
| AUPRC | 0.525162 |
| threshold | 0.630677 |
| best sweep F1 | 0.520661 |

Confusion matrix: `[[4765, 235], [105, 145]]`

![Score distribution](../experiments/anomaly_detection/autoencoder/x64/baseline/artifacts/autoencoder_baseline/plots/score_distribution.png)

![Threshold sweep](../experiments/anomaly_detection/autoencoder/x64/baseline/artifacts/autoencoder_baseline/plots/threshold_sweep.png)

![Confusion matrix](../experiments/anomaly_detection/autoencoder/x64/baseline/artifacts/autoencoder_baseline/plots/confusion_matrix.png)

### Failure Analysis

| error type | count | mean score |
|---|---|---|
| true positive | 145 | 0.758859 |
| false negative | 105 | 0.540089 |
| false positive | 235 | 0.671553 |
| true negative | 4765 | 0.512201 |

![Failure examples — false positives](../experiments/anomaly_detection/autoencoder/x64/baseline/artifacts/autoencoder_baseline/plots/failure_examples_fp.png)

![Failure examples — false negatives](../experiments/anomaly_detection/autoencoder/x64/baseline/artifacts/autoencoder_baseline/plots/failure_examples_fn.png)

### Per-Defect Recall

| defect type | count | detected | recall |
|---|---|---|---|
| Edge-Ring | 84 | 68 | 0.809524 |
| Center | 50 | 36 | 0.720000 |
| Edge-Loc | 53 | 23 | 0.433962 |
| Loc | 34 | 7 | 0.205882 |
| Donut | 7 | 4 | 0.571429 |
| Scratch | 15 | 3 | 0.200000 |
| Random | 5 | 3 | 0.600000 |
| Near-full | 2 | 1 | 0.500000 |

### Reconstruction Examples

![Reconstruction examples](../experiments/anomaly_detection/autoencoder/x64/baseline/artifacts/autoencoder_baseline/plots/reconstruction_examples.png)

### Interpretation

- `topk_abs_mean` clearly outperformed `mse_mean` — local high-error regions carry more signal than full-image averages
- The model is strong on broad defects (Edge-Ring, Center) and weak on small local defects (Loc, Scratch)
- Longer training (38 vs 25 epochs) produced only marginal change — epoch count is not the main bottleneck
- `max_abs` was unstable — a single extreme pixel is noisier than a small cluster of high-error pixels

---

## Experiment 2: Autoencoder `64×64` with BatchNorm

**Notebook:** [`experiments/anomaly_detection/autoencoder/x64/batchnorm/notebook.ipynb`](../experiments/anomaly_detection/autoencoder/x64/batchnorm/notebook.ipynb)
**Artifact dir:** `experiments/anomaly_detection/autoencoder/x64/batchnorm/artifacts/autoencoder_batchnorm/`

### Configuration

Same as Experiment 1 except:

| parameter | value |
|---|---|
| BatchNorm | enabled in encoder and decoder |
| max epochs | 50 |

### Training

- Early stopped at epoch **13**, best epoch **8**, best val loss **0.014935**

![Training curves](../experiments/anomaly_detection/autoencoder/x64/batchnorm/artifacts/autoencoder_batchnorm/plots/training_curves.png)

### Score Ablation

BatchNorm changed the error distribution enough that `max_abs` (weak on the plain baseline) became the best score here.

| score | val-threshold F1 | AUROC | AUPRC | best sweep F1 |
|---|---|---|---|---|
| `max_abs` | **0.501502** | **0.834023** | **0.568039** | **0.629808** |
| `topk_abs_mean` | 0.431373 | 0.790020 | 0.603447 | 0.655172 |
| `mse_mean` | 0.326733 | 0.779451 | 0.345216 | 0.385000 |
| `foreground_mse` | 0.278317 | 0.738702 | 0.278022 | 0.329114 |
| `mae_mean` | 0.259567 | 0.728685 | 0.284041 | 0.330969 |
| `pooled_mae_mean` | 0.257095 | 0.722959 | 0.278594 | 0.323760 |
| `foreground_mae` | 0.242017 | 0.700971 | 0.244630 | 0.296512 |

![Score ablation summary](../experiments/anomaly_detection/autoencoder/x64/batchnorm/artifacts/autoencoder_batchnorm/plots/score_ablation_summary.png)

**Selected score:** `max_abs`

### Evaluation (selected score)

| metric | value |
|---|---|
| precision | 0.401442 |
| recall | 0.668000 |
| F1 | 0.501502 |
| AUROC | 0.834023 |
| AUPRC | 0.568039 |
| threshold | 0.759817 |
| best sweep F1 | 0.629808 |

Confusion matrix: `[[4730, 270], [83, 167]]`

![Score distribution](../experiments/anomaly_detection/autoencoder/x64/batchnorm/artifacts/autoencoder_batchnorm/plots/score_distribution.png)

![Threshold sweep](../experiments/anomaly_detection/autoencoder/x64/batchnorm/artifacts/autoencoder_batchnorm/plots/threshold_sweep.png)

![Confusion matrix](../experiments/anomaly_detection/autoencoder/x64/batchnorm/artifacts/autoencoder_batchnorm/plots/confusion_matrix.png)

### Failure Analysis

| error type | count | mean score |
|---|---|---|
| true positive | 143 | 0.740250 |
| false negative | 107 | 0.478619 |
| false positive | 270 | 0.562311 |
| true negative | 4730 | 0.475182 |

![Failure examples — false positives](../experiments/anomaly_detection/autoencoder/x64/batchnorm/artifacts/autoencoder_batchnorm/plots/failure_examples_fp.png)

![Failure examples — false negatives](../experiments/anomaly_detection/autoencoder/x64/batchnorm/artifacts/autoencoder_batchnorm/plots/failure_examples_fn.png)

### Per-Defect Recall (under `topk_abs_mean` for cross-family comparison)

| defect type | count | detected | recall |
|---|---|---|---|
| Edge-Ring | 84 | 70 | 0.833333 |
| Center | 50 | 35 | 0.700000 |
| Edge-Loc | 53 | 21 | 0.396226 |
| Loc | 34 | 6 | 0.176471 |
| Scratch | 15 | 3 | 0.200000 |
| Donut | 7 | 3 | 0.428571 |
| Random | 5 | 3 | 0.600000 |
| Near-full | 2 | 2 | 1.000000 |

### Reconstruction Examples

![Reconstruction examples](../experiments/anomaly_detection/autoencoder/x64/batchnorm/artifacts/autoencoder_batchnorm/plots/reconstruction_examples.png)

### Interpretation

- BatchNorm converged faster (8 epochs) and to a lower val loss (0.014935 vs 0.019262)
- The key result: **architecture and scoring interact strongly** — the same checkpoint went from F1=0.431 (`topk_abs_mean`) to F1=0.502 (`max_abs`)
- BatchNorm + `max_abs` is the best reconstruction baseline in the family
- Weak classes (Loc, Scratch, Edge-Loc) remain unchanged — BatchNorm alone does not fix small local defects

---

## Experiment 3: Autoencoder `64×64` with BatchNorm + Dropout Sweep

**Notebook:** [`experiments/anomaly_detection/autoencoder/x64/batchnorm_dropout/notebook.ipynb`](../experiments/anomaly_detection/autoencoder/x64/batchnorm_dropout/notebook.ipynb)
**Artifact dir:** `experiments/anomaly_detection/autoencoder/x64/batchnorm_dropout/artifacts/autoencoder_batchnorm_dropout/`

### Configuration

Same as Experiment 2 with a dropout sweep over the latent layer: `0.00`, `0.05`, `0.10`, `0.20`.

### Dropout Sweep — Training Summary

| dropout | best epoch | best val loss | epochs ran |
|---|---|---|---|
| 0.00 | 11 | 0.014824 | 16 |
| 0.05 | 16 | 0.015063 | 21 |
| 0.10 | 25 | 0.014978 | 30 |
| 0.20 | 20 | 0.015660 | 25 |

Selected: **dropout = 0.00** (lowest val loss).

![Dropout sweep summary](../experiments/anomaly_detection/autoencoder/x64/batchnorm_dropout/artifacts/autoencoder_batchnorm_dropout/plots/dropout_sweep_summary.png)

### Dropout Sweep — Best Score Per Variant

Note: full score ablation artifacts are only available for the selected `dropout=0.00` run. The values below for dropout 0.05/0.10/0.20 are from the original run and cannot be reverified from the current artifact layout.

| dropout | best score | precision | recall | F1 | AUROC | AUPRC | best sweep F1 |
|---|---|---|---|---|---|---|---|
| 0.00 | `max_abs` | 0.393120 | 0.640000 | **0.487062** | 0.850790 | 0.616946 | 0.656642 |
| 0.05 | `max_abs` | 0.377828 | 0.668000 | 0.482659 | 0.835035 | 0.551700 | 0.609959 |
| 0.10 | `max_abs` | 0.385343 | 0.652000 | 0.484398 | 0.844670 | 0.570245 | 0.634615 |
| 0.20 | `max_abs` | 0.370115 | 0.644000 | 0.470073 | 0.841431 | 0.574973 | 0.633929 |

### Score Ablation — Selected Run (dropout = 0.00)

| score | val-threshold F1 | AUROC | AUPRC | best sweep F1 |
|---|---|---|---|---|
| `max_abs` | **0.487062** | 0.850790 | 0.616946 | 0.656642 |
| `topk_abs_mean` | 0.435703 | 0.799805 | 0.602296 | 0.668380 |
| `mse_mean` | 0.336601 | 0.784481 | 0.343180 | 0.394432 |
| `foreground_mse` | 0.272131 | 0.734939 | 0.263042 | 0.302083 |
| `mae_mean` | 0.256494 | 0.727365 | 0.262883 | 0.318408 |
| `pooled_mae_mean` | 0.251634 | 0.721275 | 0.256669 | 0.310502 |
| `foreground_mae` | 0.236887 | 0.694310 | 0.228523 | 0.276029 |

![Score ablation summary (dropout=0.00)](../experiments/anomaly_detection/autoencoder/x64/batchnorm_dropout/artifacts/autoencoder_batchnorm_dropout/dropout_0p00/plots/score_ablation_summary.png)

### Evaluation — Selected Run (dropout = 0.00, `max_abs`)

| metric | value |
|---|---|
| precision | 0.393120 |
| recall | 0.640000 |
| F1 | 0.487062 |
| AUROC | 0.850790 |
| AUPRC | 0.616946 |
| threshold | 0.753411 |
| best sweep F1 | 0.656642 |

Confusion matrix: `[[4752, 248], [90, 160]]`

![Score distribution (dropout=0.00)](../experiments/anomaly_detection/autoencoder/x64/batchnorm_dropout/artifacts/autoencoder_batchnorm_dropout/dropout_0p00/plots/score_distribution.png)

![Threshold sweep (dropout=0.00)](../experiments/anomaly_detection/autoencoder/x64/batchnorm_dropout/artifacts/autoencoder_batchnorm_dropout/dropout_0p00/plots/threshold_sweep.png)

![Confusion matrix (dropout=0.00)](../experiments/anomaly_detection/autoencoder/x64/batchnorm_dropout/artifacts/autoencoder_batchnorm_dropout/dropout_0p00/plots/confusion_matrix.png)

![Failure examples — false positives](../experiments/anomaly_detection/autoencoder/x64/batchnorm_dropout/artifacts/autoencoder_batchnorm_dropout/dropout_0p00/plots/failure_examples_fp.png)

![Failure examples — false negatives](../experiments/anomaly_detection/autoencoder/x64/batchnorm_dropout/artifacts/autoencoder_batchnorm_dropout/dropout_0p00/plots/failure_examples_fn.png)

### Interpretation

- Dropout did not help — the best result was `dropout=0.00`
- Higher dropout (`0.20`) hurt training stability (higher val loss) and produced the weakest F1
- Even the selected run (F1=0.487) falls below BatchNorm-only best (F1=0.502), confirming dropout is not a useful lever here
- Negative result: latent dropout is not worth pursuing further in this AE family

---

## Variant: Residual Autoencoder `64×64`

**Notebook:** [`experiments/anomaly_detection/autoencoder/x64/residual/notebook.ipynb`](../experiments/anomaly_detection/autoencoder/x64/residual/notebook.ipynb)
**Artifact dir:** `experiments/anomaly_detection/autoencoder/x64/residual/artifacts/autoencoder_residual/`

### Configuration

Same as Experiment 2 except:

| parameter | value |
|---|---|
| architecture | residual down/up blocks |
| BatchNorm | enabled |

### Training

- Early stopped at epoch **20**, best epoch **15**, best val loss **0.014504** (lower than BatchNorm-only 0.014935)

![Training curves](../experiments/anomaly_detection/autoencoder/x64/residual/artifacts/autoencoder_residual/plots/training_curves.png)

### Score Ablation

| score | val-threshold F1 | AUROC | AUPRC | best sweep F1 |
|---|---|---|---|---|
| `max_abs` | **0.473529** | **0.843360** | **0.588907** | **0.625592** |
| `topk_abs_mean` | 0.448737 | 0.804607 | 0.626014 | 0.678133 |
| `mse_mean` | 0.392220 | 0.806132 | 0.426133 | 0.463415 |
| `foreground_mse` | 0.322581 | 0.778402 | 0.353748 | 0.410758 |
| `mae_mean` | 0.269360 | 0.734534 | 0.263283 | 0.315789 |
| `pooled_mae_mean` | 0.260000 | 0.728333 | 0.255770 | 0.308824 |
| `foreground_mae` | 0.245791 | 0.710838 | 0.230483 | 0.270784 |

![Score ablation summary](../experiments/anomaly_detection/autoencoder/x64/residual/artifacts/autoencoder_residual/plots/score_ablation_summary.png)

**Selected score:** `max_abs`

### Evaluation (selected score)

| metric | value |
|---|---|
| precision | 0.374419 |
| recall | 0.644000 |
| F1 | 0.473529 |
| AUROC | 0.843360 |
| AUPRC | 0.588907 |
| threshold | 0.921727 |
| best sweep F1 | 0.625592 |

Confusion matrix: `[[4733, 267], [89, 161]]`

![Score distribution](../experiments/anomaly_detection/autoencoder/x64/residual/artifacts/autoencoder_residual/plots/score_distribution.png)

![Threshold sweep](../experiments/anomaly_detection/autoencoder/x64/residual/artifacts/autoencoder_residual/plots/threshold_sweep.png)

![Confusion matrix](../experiments/anomaly_detection/autoencoder/x64/residual/artifacts/autoencoder_residual/plots/confusion_matrix.png)

### Per-Defect Recall (under `topk_abs_mean`)

| defect type | count | detected | recall |
|---|---|---|---|
| Edge-Ring | 84 | 72 | 0.857143 |
| Center | 50 | 36 | 0.720000 |
| Edge-Loc | 53 | 23 | 0.433962 |
| Loc | 34 | 8 | 0.235294 |
| Scratch | 15 | 2 | 0.133333 |
| Donut | 7 | 4 | 0.571429 |
| Random | 5 | 4 | 0.800000 |
| Near-full | 2 | 2 | 1.000000 |

### Reconstruction Examples

![Reconstruction examples](../experiments/anomaly_detection/autoencoder/x64/residual/artifacts/autoencoder_residual/plots/reconstruction_examples.png)

### Failure Examples

![Failure examples — false positives](../experiments/anomaly_detection/autoencoder/x64/residual/artifacts/autoencoder_residual/plots/failure_examples_fp.png)

![Failure examples — false negatives](../experiments/anomaly_detection/autoencoder/x64/residual/artifacts/autoencoder_residual/plots/failure_examples_fn.png)

### Interpretation

- Residual architecture achieved the lowest val loss in the family (0.014504) and strong AUPRC under `topk_abs_mean` (0.626)
- Best thresholded F1 (0.474 with `max_abs`) still trails BatchNorm-only (0.502) — architecture depth is not the bottleneck
- Residual + `topk_abs_mean` gives the highest best-sweep F1 in the family (0.678), suggesting better ranking quality when the threshold is allowed to move
- Small local defects (Scratch, Loc) remain the hard failure mode

---

## Variant: Autoencoder `128×128`

**Notebook:** [`experiments/anomaly_detection/autoencoder/x128/baseline/notebook.ipynb`](../experiments/anomaly_detection/autoencoder/x128/baseline/notebook.ipynb)
**Artifact dir:** `experiments/anomaly_detection/autoencoder/x128/baseline/artifacts/autoencoder_baseline/`

### Configuration

Same as Experiment 1 except:

| parameter | value |
|---|---|
| image size | 128×128 |
| batch size | 32 |
| metadata | `data/processed/x128/wm811k/metadata_50k_5pct.csv` |

### Training

- Early stopped at epoch **22**, best epoch **17**, best val loss **0.020438**

![Training curves](../experiments/anomaly_detection/autoencoder/x128/baseline/artifacts/autoencoder_baseline/plots/training_curves.png)

### Score Ablation

| score | val-threshold F1 | AUROC | AUPRC | best sweep F1 |
|---|---|---|---|---|
| `topk_abs_mean` | **0.431239** | **0.814856** | **0.455031** | **0.479821** |
| `mse_mean` | 0.370370 | 0.795673 | 0.393266 | 0.426724 |
| `max_abs` | 0.308426 | 0.751154 | 0.287075 | 0.341121 |
| `foreground_mse` | 0.284375 | 0.753384 | 0.337506 | 0.352941 |
| `mae_mean` | 0.263333 | 0.749153 | 0.305733 | 0.336508 |
| `pooled_mae_mean` | 0.257525 | 0.745732 | 0.302292 | 0.336283 |
| `foreground_mae` | 0.234694 | 0.723569 | 0.276116 | 0.302857 |

> **Note:** The original run used `mse_mean` as the default score. The notebook was later rerun with `topk_abs_mean` to match the rest of the family. The old numbers (`mse_mean`: precision=0.309973, recall=0.460, F1=0.370370, AUROC=0.795673, confusion `[[4744, 256], [135, 115]]`) are preserved here for reference.

![Score ablation summary](../experiments/anomaly_detection/autoencoder/x128/baseline/artifacts/autoencoder_baseline/plots/score_ablation_summary.png)

**Selected score:** `topk_abs_mean`

### Evaluation (selected score)

| metric | value |
|---|---|
| precision | 0.374631 |
| recall | 0.508000 |
| F1 | 0.431239 |
| AUROC | 0.814856 |
| AUPRC | 0.455031 |
| threshold | 0.671781 |
| best sweep F1 | 0.479821 |

Confusion matrix: `[[4788, 212], [123, 127]]`

![Score distribution](../experiments/anomaly_detection/autoencoder/x128/baseline/artifacts/autoencoder_baseline/plots/score_distribution.png)

![Threshold sweep](../experiments/anomaly_detection/autoencoder/x128/baseline/artifacts/autoencoder_baseline/plots/threshold_sweep.png)

![Confusion matrix](../experiments/anomaly_detection/autoencoder/x128/baseline/artifacts/autoencoder_baseline/plots/confusion_matrix.png)

### Per-Defect Recall (`topk_abs_mean`)

| defect type | count | detected | recall |
|---|---|---|---|
| Edge-Ring | 84 | 68 | 0.809524 |
| Center | 50 | 31 | 0.620000 |
| Edge-Loc | 53 | 17 | 0.320755 |
| Loc | 34 | 3 | 0.088235 |
| Random | 5 | 3 | 0.600000 |
| Scratch | 15 | 2 | 0.133333 |
| Near-full | 2 | 2 | 1.000000 |
| Donut | 7 | 1 | 0.142857 |

### Resolution Comparison

![Resolution comparison](../artifacts/report_plots/autoencoder_resolution_comparison.png)

### Interpretation

- Even with the updated score, `128×128` (F1=0.431) is clearly below `64×64` (F1=0.468 score ablation best)
- Higher resolution did not help and was slower/more expensive to train
- Notably worse on Center (0.620 vs 0.720), Edge-Loc (0.321 vs 0.434), Loc (0.088 vs 0.206), Donut (0.143 vs 0.571)
- The autoencoder evidence firmly favors `64×64`

---

## Cross-Branch Summary

![Autoencoder family comparison](../artifacts/report_plots/autoencoder_family_comparison.png)

### Best Score Per Branch

| branch | best score | F1 | AUROC | AUPRC |
|---|---|---|---|---|
| x64/batchnorm | `max_abs` | **0.501502** | 0.834023 | 0.568039 |
| x64/batchnorm_dropout (d=0.00) | `max_abs` | 0.487062 | 0.850790 | **0.616946** |
| x64/residual | `max_abs` | 0.473529 | 0.843360 | 0.588907 |
| x64/baseline | `topk_abs_mean` | 0.467949 | **0.839282** | 0.522171 |
| x128/baseline | `topk_abs_mean` | 0.431239 | 0.814856 | 0.455031 |

### Key Takeaways

1. **Score design matters as much as architecture.** BatchNorm changed which score worked best (`topk_abs_mean` → `max_abs`). Always run a score ablation before declaring a checkpoint weak.
2. **BatchNorm + `max_abs` is the family winner** on deployed F1 (0.502). It is also simpler and trains faster than the residual variant.
3. **Dropout does not help.** The sweep confirmed `dropout=0.00` as best — latent dropout is not a useful regulariser for this AE.
4. **Residual architecture has the best ranking quality** (highest best-sweep F1 = 0.678 under `topk_abs_mean`) but trails on deployed threshold F1.
5. **Higher resolution (128×128) hurts.** Worse on nearly every defect type at the same score. Do not pursue larger AE resolutions without a fundamentally different architecture.
6. **Persistent failure pattern.** Across all variants: Scratch, Loc, and Edge-Loc recall stays low. The bottleneck is local anomaly sensitivity, not reconstruction quality. This motivated the move to PatchCore and teacher-student methods.
