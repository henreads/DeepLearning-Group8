# Teacher-Student Distillation Family Report

## Overview

Teacher-Student Distillation is a **feature-space anomaly detection** approach that trains a student network to reproduce features from a pre-trained teacher (backbone) on normal wafers only. Anomaly scores are derived from the mismatch between teacher and student feature maps at test time.

This method sits at the intersection of reconstruction-based and nearest-neighbor approaches:
- **Feature-aware** rather than pixel-reconstruction only
- **Normal-only training** like one-class methods (SVDD, AE)
- **Well-suited for localized defects** by capturing feature-space distortions

Unlike PatchCore (which uses memory banks), teacher-student uses a learned student network to approximate teacher features, providing a more compact, trainable alternative with comparable or better anomaly detection performance.

Notebooks and artifacts live under [`experiments/anomaly_detection/teacher_student/`](../experiments/anomaly_detection/teacher_student/).

---

## Dataset and Evaluation Protocol

- **Train:** 40,000 normal wafers only
- **Validation:** 5,000 normal wafers (threshold = 95th percentile of anomaly scores)
- **Test:** 5,000 normal + 250 anomaly wafers
- **Score:** Mismatch between teacher and student feature maps (combined student distillation + feature autoencoder)
- **Primary metric:** validation-threshold F1

---

## Experiment 1: Teacher-Student ResNet18 `64×64`

**Notebook:** [`experiments/anomaly_detection/teacher_student/resnet18/x64/main/notebook.ipynb`](../experiments/anomaly_detection/teacher_student/resnet18/x64/main/notebook.ipynb)
**Artifact dir:** `experiments/anomaly_detection/teacher_student/resnet18/x64/main/artifacts/ts_resnet18/`

### Run Controls

| flag | default | meaning |
|---|---|---|
| `RETRAIN` | `False` | use saved checkpoint; set `True` to retrain via `scripts/train_ts_distillation.py` |
| `RUN_DEFAULT_EVALUATION` | `False` | use cached eval scores; set `True` to re-score via `scripts/evaluate_reconstruction_model.py` |
| `RUN_ABLATION_SWEEP` | `False` | run layer and topk_ratio ablation study |
| `ABLATION_PROFILE` | `"focused"` | `"focused"` (layer1/2, topk 0.15/0.20/0.25) or `"full"` (layer1/2/3, broader sweep) |

### Configuration

| parameter | value |
|---|---|
| teacher backbone | ResNet18 (pretrained ImageNet) |
| teacher layer | layer2 |
| teacher input size | 224×224 |
| feature autoencoder | 64 hidden dim |
| student weight | 1.0 (distillation loss) |
| autoencoder weight | 1.0 (feature reconstruction loss) |
| optimizer | Adam, lr=0.0003, weight decay=1e-5 |
| max epochs | 30 |
| early stopping | patience=5, min_delta=0.0001 |
| checkpoint every | 5 epochs |
| **score student weight** | 1.0 |
| **score autoencoder weight** | 0.0 |
| **reduction** | topk_mean with topk_ratio=0.20 |

### Training

- Best epoch: **27**, best val loss: **0.023353**, epochs ran: **30**
- Student map scale: **0.01124** (normalized for stability)
- Autoencoder map scale: **0.01207**

![Training curves](../experiments/anomaly_detection/teacher_student/resnet18/x64/main/artifacts/ts_resnet18/plots/training_curves.png)

### Evaluation

| metric | value |
|---|---|
| threshold | 2.342451 |
| precision | 0.402500 |
| recall | 0.644000 |
| F1 | **0.495385** |
| AUROC | 0.894076 |
| AUPRC | 0.519445 |
| predicted anomalies | 400 |
| best sweep F1 | 0.520548 |

Confusion matrix: `[[4761, 239], [89, 161]]`

The selected score variant uses **student-only features (autoencoder weight = 0.0)** with **topk_mean reduction at 20% quantile**, achieving strong balance between precision and recall.

### Score Distribution and Threshold Sweep

![Score distribution](../experiments/anomaly_detection/teacher_student/resnet18/x64/main/artifacts/ts_resnet18/plots/score_distribution.png)

![Threshold sweep](../experiments/anomaly_detection/teacher_student/resnet18/x64/main/artifacts/ts_resnet18/plots/threshold_sweep.png)

### Per-Defect Recall (computed from saved scores)

| defect type | count | detected | recall |
|---|---|---|---|
| Scratch | 15 | 5 | 0.333 |
| Loc | 34 | 15 | 0.441 |
| Edge-Loc | 53 | 26 | 0.491 |
| Center | 50 | 31 | 0.620 |
| Donut | 7 | 5 | 0.714 |
| Edge-Ring | 84 | 72 | 0.857 |
| Random | 5 | 5 | 1.000 |
| Near-full | 2 | 2 | 1.000 |

### Highest-Scored Examples

![Top scored examples](../experiments/anomaly_detection/teacher_student/resnet18/x64/main/artifacts/ts_resnet18/plots/top_scored_examples.png)

### Score Variant Comparison

The notebook includes a post-training score sweep over branch weights and reduction methods. The default configuration and the best sweep variant are compared below:

![Score variant comparison](../experiments/anomaly_detection/teacher_student/resnet18/x64/main/artifacts/ts_resnet18/plots/score_variant_comparison.png)

### Interpretation

- **Strong overall performance:** F1 = 0.495 and AUROC = 0.894 place teacher-student among the project's best anomaly detectors
- **Good calibration under class imbalance:** AUPRC = 0.519 is solid, indicating reliable score ranking despite 5% anomaly rate
- **Defect-type pattern:** Broad, high-contrast defects (Near-full, Random, Edge-Ring) are easiest; localized defects (Scratch, Loc) are harder
- **Student-only scoring:** The best configuration uses student distillation mismatch alone (ignoring the autoencoder branch), suggesting that the teacher-student feature match is the primary anomaly signal
- **Competitive position:** Teacher-student outperforms SVDD (F1=0.360) and comparable to best autoencoders (F1≈0.50), confirming that learned feature-space distillation is effective for this dataset

---

## Ablation Studies

The notebook includes optional ablation sweeps comparing:

1. **Teacher layer choice:** layer1, layer2, layer3 (default: layer2)
2. **topk_ratio for scoring:** 0.10 to 0.30 (default: 0.20)
3. **Training epochs:** focused vs full sweeps

Results are saved to `artifacts/ts_resnet18/results/evaluation/ablation_sweep_summary.csv` and can be visualized with the included plots.

---

## Holdout Evaluation (Future)

A larger holdout split (70k normal / 3.5k anomaly) can be evaluated by setting the appropriate flags in the notebook. When available, results will be saved alongside the benchmark evaluation.

---

## Context in the Project

| method | F1 | AUROC | AUPRC |
|---|---|---|---|
| AE + BatchNorm (`max_abs`) | 0.502 | 0.834 | 0.568 |
| **Teacher-Student ResNet18** | **0.495** | **0.894** | **0.519** |
| PatchCore (ResNet50) | 0.565 | 0.914 | 0.676 |
| Deep SVDD | 0.360 | 0.788 | 0.213 |
| VAE (best beta) | 0.340 | 0.771 | 0.372 |

Teacher-student sits between reconstruction-based methods and simple baselines. While it trails PatchCore on F1 and AUPRC (likely due to PatchCore's memory-bank expressiveness), it outperforms SVDD and VAE, confirming that learned distillation is a competitive feature-based anomaly signal.

The strong AUROC (0.894) suggests excellent anomaly ranking, though the lower AUPRC relative to PatchCore indicates there is room for calibration improvement under extreme class imbalance.

---

---

## Experiment 2: Teacher-Student ResNet50 `64×64`

**Notebook:** [`experiments/anomaly_detection/teacher_student/resnet50/x64/main/notebook.ipynb`](../experiments/anomaly_detection/teacher_student/resnet50/x64/main/notebook.ipynb)
**Artifact dir:** `experiments/anomaly_detection/teacher_student/resnet50/x64/main/artifacts/ts_resnet50/`

This is a higher-capacity ResNet50 variant with similar architecture and evaluation setup. The notebook follows the same artifact-first design as ResNet18, with `RETRAIN = False` by default.

---

## Experiment 3: Teacher-Student Wide-ResNet50-2 Variants

### 3a: Layer2 Single-Layer Branch `64×64`

**Notebook:** [`experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/layer2_self_contained/notebook.ipynb`](../experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/layer2_self_contained/notebook.ipynb)
**Artifact dir:** `experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/layer2_self_contained/artifacts/ts_wideresnet50_layer2/`

Single-layer Wide ResNet50-2 teacher focusing on layer2 and layer3 features. Configuration:
- Teacher layers: layer2, layer3
- Feature autoencoder: 128 hidden dim
- Score weights: student=2.0, autoencoder=1.0
- Reduction: topk_mean with topk_ratio=0.25

**Evaluation Results:**

| metric | value |
|---|---|
| threshold | 4.007 |
| precision | 0.406 |
| recall | 0.680 |
| F1 | **0.508** |
| AUROC | 0.920 |
| AUPRC | 0.540 |
| predicted anomalies | 419 |
| best sweep F1 | 0.559 |

Confusion matrix: `[[4751, 249], [80, 170]]`

![Score distribution](../experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/layer2_self_contained/artifacts/ts_wideresnet50_layer2/plots/score_distribution.png)

![Threshold sweep](../experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/layer2_self_contained/artifacts/ts_wideresnet50_layer2/plots/threshold_sweep.png)

![Confusion matrix](../experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/layer2_self_contained/artifacts/ts_wideresnet50_layer2/plots/confusion_matrix.png)

![Defect breakdown](../experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/layer2_self_contained/artifacts/ts_wideresnet50_layer2/plots/defect_breakdown.png)

### 3b: Multilayer Branch `64×64`

**Notebook:** [`experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/multilayer_self_contained/notebook.ipynb`](../experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/multilayer_self_contained/notebook.ipynb)
**Artifact dir:** `experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/multilayer_self_contained/artifacts/ts_wideresnet50_multilayer/`

Multilayer Wide ResNet50-2 teacher combining layer2 and layer3 features at 64×64 resolution. Same configuration as layer2 branch.

**Evaluation Results:**

| metric | value |
|---|---|
| threshold | 4.013 |
| precision | 0.401 |
| recall | 0.680 |
| F1 | **0.504** |
| AUROC | 0.923 |
| AUPRC | 0.549 |
| predicted anomalies | 424 |
| best sweep F1 | 0.562 |

Confusion matrix: `[[4746, 254], [80, 170]]`

![Training curves](../experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/multilayer_self_contained/artifacts/ts_wideresnet50_multilayer/plots/training_curves.png)

![Score distribution](../experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/multilayer_self_contained/artifacts/ts_wideresnet50_multilayer/plots/score_distribution.png)

![Threshold sweep](../experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/multilayer_self_contained/artifacts/ts_wideresnet50_multilayer/plots/threshold_sweep.png)

![Confusion matrix](../experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/multilayer_self_contained/artifacts/ts_wideresnet50_multilayer/plots/confusion_matrix.png)

![Defect breakdown](../experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/multilayer_self_contained/artifacts/ts_wideresnet50_multilayer/plots/defect_breakdown.png)

### 3c: Multilayer Branch `224×224`

**Notebook:** [`experiments/anomaly_detection/teacher_student/wideresnet50_2/x224/multilayer_self_contained/notebook.ipynb`](../experiments/anomaly_detection/teacher_student/wideresnet50_2/x224/multilayer_self_contained/notebook.ipynb)
**Artifact dir:** `experiments/anomaly_detection/teacher_student/wideresnet50_2/x224/multilayer_self_contained/artifacts/ts_wideresnet50_multilayer_x224/`

Multilayer Wide ResNet50-2 teacher at full 224×224 resolution, leveraging the teacher backbone's native input size for potentially richer feature extraction.

**Note:** Evaluation results for x224 variant are pending completion.

All three WideResNet notebooks use a self-contained design with `run_training: False` by default, allowing end-to-end execution with cached artifacts.

---

## Family-Wide Comparison

The teacher-student family spans multiple backbones and architectures. Here's the comprehensive comparison:

| variant | F1 | AUROC | AUPRC | precision | recall | notes |
|---|---|---|---|---|---|---|
| ResNet18 (x64) | 0.495 | 0.894 | 0.519 | 0.403 | 0.644 | Baseline, includes ablation sweep |
| ResNet50 (x64) | 0.488 | 0.913 | 0.581 | 0.382 | 0.676 | Higher capacity backbone |
| WideResNet50_2 Layer2 (x64) | 0.508 | 0.920 | 0.540 | 0.406 | 0.680 | Dual-layer features |
| WideResNet50_2 Multilayer (x64) | 0.504 | 0.923 | 0.549 | 0.401 | 0.680 | Similar multilayer setup |
| WideResNet50_2 Multilayer (x224) | - | - | - | - | - | Pending evaluation |

**Key Findings:**

1. **WideResNet50_2 dominates on AUROC** (0.920-0.923), indicating excellent anomaly ranking under class imbalance
2. **ResNet50 achieves highest AUPRC** (0.581), best score calibration among evaluated variants
3. **WideResNet Layer2 has strongest F1** (0.508), balancing precision and recall effectively
4. **ResNet18 is most compact** with good performance (F1=0.495), useful for deployment with ablation insights
5. **Higher resolution (x224)** expected to improve detection via richer feature maps when evaluation completes

---

## Notebook Self-Containment

| path | RETRAIN=False | RETRAIN=True |
|---|---|---|
| Training | Loads `best_model.pt` | Runs `scripts/train_ts_distillation.py` |
| Default evaluation | Loads cached score CSVs | Runs `scripts/evaluate_reconstruction_model.py` |
| Training curves plot | Regenerated from `history.json` | Same |
| Score dist + sweep plots | Regenerated from saved CSVs | Same |
| Top-examples plot | Regenerated from `test_scores.csv` | Same |
| Ablation sweep | Loads cached or retrains variants | Retrains all variants |

All notebooks can be run end-to-end with `RETRAIN = False` (or `run_training: False` for WideResNet), and will load pre-trained checkpoints and cached evaluation artifacts without requiring GPU or re-training.
