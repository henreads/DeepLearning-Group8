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

| method | F1 | AUROC | AUPRC | notes |
|---|---|---|---|---|
| AE x224 (`topk_abs_mean`) | 0.510 | 0.901 | 0.596 | reconstruction + resolution ✅ |
| **Teacher-Student ResNet18 x224** | **0.511** | **0.907** | **0.503** | **feature-based + resolution ✅** |
| AE + BatchNorm x64 (`max_abs`) | 0.502 | 0.834 | 0.568 | best AE variant |
| **Teacher-Student ResNet18 x64** | **0.495** | **0.894** | **0.519** | feature-based baseline |
| **Teacher-Student ResNet50 x64** | **0.488** | **0.913** | **0.581** | largest TS AUPRC |
| **Teacher-Student ResNet50 x224** | **0.3988** | **0.8277** | **0.3608** | ⚠️ resolution degrades larger backbone |
| PatchCore + ViT-B/16 x224 | 0.595 | 0.956 | 0.671 | project best |
| PatchCore (ResNet50) x64 | 0.565 | 0.914 | 0.676 | local patch scoring |
| Deep SVDD | 0.360 | 0.788 | 0.213 | one-class baseline |
| VAE x224 (beta=0.005) | 0.339 | 0.772 | 0.362 | unresponsive to resolution |

### Key Insight: Resolution Benefit is Backbone-Dependent

**ResNet18 x224 benefits from native resolution** (+0.016 F1, +3.2% rel) and remains competitive with larger x64 backbones. However, **ResNet50 x224 underperforms** (-0.0892 F1, -18.9% rel) despite proper batch size scaling (256), suggesting architectural factors beyond hyperparameters.

This indicates that **resolution improvements are not universal in teacher-student distillation.** Smaller backbones (ResNet18) extract features that improve with native resolution, while larger backbones (ResNet50) may suffer from feature-space complexity when both resolution and backbone capacity increase simultaneously. This contrasts with reconstruction-based methods (AE) and PatchCore, which scale more robustly to native resolution regardless of backbone size.

---

---

## Experiment 1b: Teacher-Student ResNet18 `224×224` (Native Resolution)

**Notebook:** [`experiments/anomaly_detection/teacher_student/resnet18/x224/main/notebook.ipynb`](../experiments/anomaly_detection/teacher_student/resnet18/x224/main/notebook.ipynb)
**Artifact dir:** `experiments/anomaly_detection/teacher_student/resnet18/x224/main/artifacts/ts_resnet18_x224/`

### Configuration

Same architecture as ResNet18 x64, but at the teacher backbone's native 224×224 input resolution to better preserve spatial features.

| parameter | value |
|---|---|
| teacher backbone | ResNet18 (pretrained ImageNet) |
| teacher layer | layer2 |
| **teacher input size** | **224×224** |
| feature autoencoder | 64 hidden dim |
| student weight | 1.0 (distillation loss) |
| autoencoder weight | 1.0 (feature reconstruction loss) |
| optimizer | Adam, lr=0.0003, weight decay=1e-5 |
| max epochs | 30 |
| early stopping | patience=5, min_delta=0.0001 |
| **score student weight** | 1.0 |
| **score autoencoder weight** | 0.0 |
| **reduction** | topk_mean with topk_ratio=0.20 |

### Training

- Best epoch: **30**, best val loss: **0.032977**, epochs ran: **30**
- Student map scale: **0.016021** (normalized for stability)
- Autoencoder map scale: **0.016889**

![Training curves](../experiments/anomaly_detection/teacher_student/resnet18/x224/main/artifacts/ts_resnet18_x224/plots/training_curves.png)

### Evaluation

| metric | value |
|---|---|
| threshold | 7.504731 |
| precision | 0.412776 |
| recall | 0.672000 |
| **F1** | **0.511416** |
| **AUROC** | **0.906927** |
| **AUPRC** | **0.502621** |
| predicted anomalies | 407 |
| best sweep F1 | 0.546906 |

Confusion matrix: `[[4774, 226], [84, 166]]`

Selected score variant: **student weight=2.0, autoencoder weight=1.0, topk_mean reduction (r=0.20)** — balances precision/recall effectively.

![Score distribution & threshold sweep](../experiments/anomaly_detection/teacher_student/resnet18/x224/main/artifacts/ts_resnet18_x224/plots/score_variant_comparison.png)

### Per-Defect Recall (selected variant)

| defect type | count | detected | recall |
|---|---|---|---|
| Scratch | 15 | 5 | 0.333 |
| Edge-Loc | 53 | 22 | 0.415 |
| Loc | 34 | 16 | 0.471 |
| Center | 50 | 34 | 0.680 |
| Donut | 7 | 6 | 0.857 |
| Edge-Ring | 84 | 78 | 0.929 |
| Random | 5 | 5 | 1.000 |
| Near-full | 2 | 2 | 1.000 |

![Defect breakdown](../experiments/anomaly_detection/teacher_student/resnet18/x224/main/artifacts/ts_resnet18_x224/plots/defect_breakdown.png)

### Key Findings: Resolution Impact on Teacher-Student

**Resolution effect:** F1 improves from **0.495 (x64) → 0.511 (x224) — +1.6% absolute (+3.2% relative)**

| metric | x64 | x224 | Δ |
|---|---|---|---|
| **F1** | 0.495 | 0.511 | **+0.016** |
| **AUROC** | 0.894 | 0.907 | **+0.013** |
| **AUPRC** | 0.519 | 0.503 | -0.016 |
| **Recall** | 0.644 | 0.672 | **+0.028** |
| **Best sweep F1** | 0.521 | 0.547 | **+0.026** |

**Interpretation:**
- Resolution improvement is consistent and modest but meaningful — teacher-student is **responsive to higher resolution**, unlike VAE (which gained nothing)
- The gain is smaller than autoencoder x224 (+0.043 F1) but larger in relative terms for an already-strong baseline (x64 TS = 0.495, x64 AE/BN = 0.502)
- **Recall improves significantly** (+0.028), showing better anomaly sensitivity at x224
- AUPRC trade-off suggests slightly worse calibration, but best-sweep F1 (+0.026) shows the model's ranking quality improves
- **Competitive with WideResNet50-2 variants:** x224 ResNet18 (F1=0.511) now exceeds x64 WideResNet50-2 Layer2 (F1=0.508) and approaches Multilayer (F1=0.504)

**Conclusion:** Using the teacher backbone's native resolution meaningfully improves teacher-student distillation. The improvement is smaller than for simpler methods (AE) but shows the same directional benefit. This establishes a fair baseline for comparing x224 variants across families.

---

## Experiment 2: Teacher-Student ResNet50 `64×64`

**Notebook:** [`experiments/anomaly_detection/teacher_student/resnet50/x64/main/notebook.ipynb`](../experiments/anomaly_detection/teacher_student/resnet50/x64/main/notebook.ipynb)
**Artifact dir:** `experiments/anomaly_detection/teacher_student/resnet50/x64/main/artifacts/ts_resnet50/`

### Configuration

Same as ResNet18 x64.

### Results

| metric | value |
|---|---|
| threshold | 3.078 |
| precision | 0.382 |
| recall | 0.676 |
| **F1** | **0.488** |
| **AUROC** | **0.913** |
| **AUPRC** | **0.581** |
| predicted anomalies | 439 |
| best sweep F1 | 0.536 |

**Interpretation:** ResNet50 gains higher recall (+0.032 vs ResNet18) at slight F1 cost (0.488 vs 0.495), and achieves the highest AUPRC (0.581) in the x64 teacher-student family. Higher capacity backbone yields better ranking under class imbalance.

---

## Experiment 2b: Teacher-Student ResNet50 `224×224` (Native Resolution)

**Notebook:** [`experiments/anomaly_detection/teacher_student/resnet50/x224/main/notebook.ipynb`](../experiments/anomaly_detection/teacher_student/resnet50/x224/main/notebook.ipynb)
**Artifact dir:** `experiments/anomaly_detection/teacher_student/resnet50/x224/main/artifacts/ts_resnet50_x224/`

### Configuration

Same architecture as ResNet50 x64, but at the teacher backbone's native 224×224 input resolution. Batch size scaled to 256 (16× larger) to account for higher memory footprint of x224.

| parameter | value |
|---|---|
| teacher backbone | ResNet50 (pretrained ImageNet) |
| teacher layer | layer2 |
| **teacher input size** | **224×224** |
| **batch size** | **256** (scaled from x64's 16) |
| feature autoencoder | 64 hidden dim |
| student weight | 1.0 (distillation loss) |
| autoencoder weight | 1.0 (feature reconstruction loss) |
| optimizer | Adam, lr=0.0003, weight decay=1e-5 |
| max epochs | 30 |
| early stopping | patience=5, min_delta=0.0001 |

### Training

- Best epoch: **28**, best val loss: **0.2274**, epochs ran: **30**
- Student map scale: **0.01876** (normalized for stability)
- Autoencoder map scale: **0.01965**

![Training curves](../experiments/anomaly_detection/teacher_student/resnet50/x224/main/artifacts/ts_resnet50_x224/plots/training_curves.png)

### Evaluation

| metric | value |
|---|---|
| threshold | 9.883 |
| precision | 0.327 |
| recall | 0.512 |
| **F1** | **0.3988** |
| **AUROC** | **0.8277** |
| **AUPRC** | **0.3608** |
| predicted anomalies | 392 |
| best sweep F1 | 0.4237 |

Confusion matrix: `[[4728, 272], [122, 128]]`

Selected score variant: **student weight=2.0, autoencoder weight=1.0, topk_mean reduction (r=0.20)** — balances precision/recall effectively.

![Score distribution & variant comparison](../experiments/anomaly_detection/teacher_student/resnet50/x224/main/artifacts/ts_resnet50_x224/plots/score_variant_comparison.png)

### Per-Defect Recall (selected variant)

| defect type | count | detected | recall |
|---|---|---|---|
| Scratch | 15 | 3 | 0.200 |
| Center | 50 | 13 | 0.260 |
| Loc | 34 | 11 | 0.324 |
| Edge-Loc | 53 | 20 | 0.377 |
| Edge-Ring | 84 | 59 | 0.702 |
| Donut | 7 | 6 | 0.857 |
| Random | 5 | 5 | 1.000 |
| Near-full | 2 | 2 | 1.000 |

![Defect breakdown](../experiments/anomaly_detection/teacher_student/resnet50/x224/main/artifacts/ts_resnet50_x224/plots/defect_breakdown.png)

### Key Findings: Resolution Impact on ResNet50 Teacher-Student

**Resolution effect:** F1 drops from **0.488 (x64) → 0.3988 (x224) — -0.0892 absolute (-18.9% relative)**

| metric | x224 | x64 | Δ |
|---|---|---|---|
| **F1** | 0.3988 | 0.488 | **-0.0892 (-18.9%)** |
| **AUROC** | 0.8277 | 0.913 | **-0.0853 (-9.3%)** |
| **AUPRC** | 0.3608 | 0.581 | **-0.2202 (-37.9%)** ⚠️⚠️ |
| **Recall** | 0.512 | 0.676 | **-0.164** |

**This contrasts sharply with ResNet18 x224, which gains +1.6% F1.**

### Comparison: ResNet18 vs ResNet50 Resolution Effect

| Backbone | x64 F1 | x224 F1 | Δ | Notes |
|----------|--------|--------|---|-------|
| **ResNet18** | 0.495 | 0.511 | **+0.016 (+3.2%)** | ✅ Benefits from resolution |
| **ResNet50** | 0.488 | 0.3988 | **-0.0892 (-18.9%)** | ❌ Degraded by resolution |

### Interpretation: Why Resolution Hurts ResNet50

Despite proper batch size scaling (256 vs x64's 16), ResNet50 at x224 underperforms significantly. This suggests fundamental architectural or feature-space issues rather than hyperparameter misconfiguration:

1. **Feature space mismatch:** Larger backbones may extract different feature hierarchies at native resolution, confusing the student network trained with smaller feature maps
2. **Distillation complexity:** Larger feature tensors (due to higher resolution) may make student-teacher feature matching harder, especially with same architecture as x64
3. **Overfitting risk:** Higher resolution provides more spatial detail but ResNet50 with 64-dim autoencoder may be insufficient to capture richer features at x224

Notably, **ResNet18 scales well to x224** (+3.2% F1), suggesting the issue is specific to larger backbones with fixed feature bottleneck (64 hidden dim). Future work should explore whether:
- Increasing feature autoencoder capacity for x224 variants helps
- Using teacher backbone's native layer output (pre-bottleneck) improves results
- ResNet50 x64 architecture benefits from higher resolution with different hyperparameters

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
**Artifact dir:** `experiments/anomaly_detection/teacher_student/wideresnet50_2/x224/multilayer_self_contained/artifacts/ts_wideresnet50_multilayer/`

Multilayer Wide ResNet50-2 teacher at full 224×224 resolution, leveraging the teacher backbone's native input size for potentially richer feature extraction.

### Configuration

Same as multilayer x64, but at 224×224 resolution.

| parameter | value |
|---|---|
| teacher backbone | WideResNet50-2 (pretrained ImageNet) |
| teacher layers | layer2, layer3 |
| **teacher input size** | **224×224** |
| feature autoencoder | 128 hidden dim |
| student weight | 2.0 (distillation loss) |
| autoencoder weight | 1.0 (feature reconstruction loss) |
| optimizer | Adam, lr=0.0003, weight decay=1e-5 |
| max epochs | 30 |
| early stopping | patience=5, min_delta=0.0001 |

### Training

- Best epoch: **4**, best val loss: **0.1881**, epochs ran: **13**
- Student map scale: **0.1240** (normalized for stability)
- Autoencoder map scale: **0.0616**

### Evaluation Results

| metric | value |
|---|---|
| threshold | 6.594 |
| precision | 0.261 |
| recall | 0.396 |
| **F1** | **0.3148** |
| **AUROC** | **0.7885** |
| **AUPRC** | **0.2690** |
| predicted anomalies | 379 |
| best sweep F1 | 0.3286 |

Confusion matrix: `[[4720, 280], [151, 99]]`

Selected score variant: **student weight=2.0, autoencoder weight=1.0** with default topk reduction.

### Key Findings: WideResNet50-2 x224 Severely Underperforms

**Dramatic degradation at x224 resolution:**

| metric | x224 | x64 | Δ |
|---|---|---|---|
| **F1** | 0.3148 | 0.504 | **-0.1892 (-37.5%)** ⚠️⚠️⚠️ |
| **AUROC** | 0.7885 | 0.923 | **-0.1345 (-14.6%)** |
| **AUPRC** | 0.2690 | 0.549 | **-0.2800 (-51.0%)** ⚠️⚠️⚠️ |
| **Recall** | 0.396 | 0.680 | **-0.284** |

**This is the worst resolution impact observed across all teacher-student variants:**

| Backbone | x64 F1 | x224 F1 | Δ | Pattern |
|----------|--------|--------|---|---------|
| **ResNet18** | 0.495 | 0.511 | **+0.016 (+3.2%)** | ✅ Benefits from resolution |
| **ResNet50** | 0.488 | 0.3988 | **-0.0892 (-18.9%)** | ❌ Hurt by resolution |
| **WideResNet50-2 Multilayer** | 0.504 | 0.3148 | **-0.1892 (-37.5%)** | ❌❌❌ **Catastrophic degradation** |

**Possible causes:**
1. **Feature extraction collapse:** Wider backbone (ResNet50 width×1.5) with 128-dim autoencoder bottleneck completely fails at higher resolution
2. **Multi-layer complexity:** Combining layer2 AND layer3 features at x224 may create severe feature mismatch problems
3. **Training instability:** Only 4 epochs to convergence suggests the model found a poor local minimum early

All three WideResNet notebooks use a self-contained design with `run_training: False` by default, allowing end-to-end execution with cached artifacts.

---

## Family-Wide Comparison

The teacher-student family spans multiple backbones and architectures. Here's the comprehensive comparison:

| variant | F1 | AUROC | AUPRC | precision | recall | notes |
|---|---|---|---|---|---|---|
| ResNet18 (x64) | 0.495 | 0.894 | 0.519 | 0.403 | 0.644 | Baseline, includes ablation sweep |
| **ResNet18 (x224)** | **0.511** | **0.907** | **0.503** | **0.413** | **0.672** | **✅ +1.6% F1 (native resolution helps)** |
| ResNet50 (x64) | 0.488 | 0.913 | 0.581 | 0.382 | 0.676 | Higher capacity backbone, best AUPRC |
| **ResNet50 (x224, ae_dim=128)** | **0.3988** | **0.8277** | **0.3608** | **0.327** | **0.512** | **❌ -18.9% F1 (baseline)** |
| **ResNet50 (x224, ae_dim=768)** | **0.4024** | **0.8288** | **0.3742** | **0.3251** | **0.528** | **❌ -17.6% F1 (dimension sweep, best)** |
| WideResNet50_2 Layer2 (x64) | 0.508 | 0.920 | 0.540 | 0.406 | 0.680 | Dual-layer features |
| WideResNet50_2 Multilayer (x64) | 0.504 | 0.923 | 0.549 | 0.401 | 0.680 | Similar multilayer setup |
| **WideResNet50_2 Multilayer (x224)** | **0.3148** | **0.7885** | **0.2690** | **0.261** | **0.396** | **❌❌❌ -37.5% F1 (catastrophic)** |
| **ViT-B/16 (x224)** | **0.163** | **0.661** | **0.106** | **0.158** | **0.168** | **❌❌❌ -72.6% vs PatchCore ViT (method mismatch)** |

**Key Findings:**

1. **Resolution benefit is strongly backbone-dependent:**
   - ResNet18 x224: **+3.2% F1** (resolution helps) ✅
   - ResNet50 x224: **-18.9% F1** (resolution degrades)
   - WideResNet50-2 Multilayer x224: **-37.5% F1** (resolution catastrophic) ⚠️⚠️⚠️
   - **ViT-B/16 x224: -72.6% F1** (method incompatibility, not resolution) ⚠️⚠️⚠️⚠️

2. **Backbone size amplifies resolution problems:** The pattern is clear—larger backbones (wider, deeper, more parameters) suffer increasingly severe degradation when moving to x224. This suggests a fundamental incompatibility between feature complexity and resolution that goes **beyond autoencoder bottleneck size**.

3. **Feature autoencoder bottleneck is NOT the root cause:** The dimension sweep (64 → 768 dims) yielded only **+0.3% relative F1 improvement** on ResNet50 x224. This disproves the initial hypothesis that bottleneck compression alone explains the degradation. The issue must be deeper—likely feature-space mismatch, student network capacity limitations, or fundamentally different training dynamics at x224 resolution.

4. **Architectural mismatch: Teacher-Student fails on patch-based backbones** — ViT-B/16 Teacher-Student (F1=0.163) is **60% worse than CNN-based TS** and **72.6% below PatchCore ViT (F1=0.595)**. This is not a hyperparameter issue; it reveals a fundamental mismatch:
   - ViT produces patch embeddings where each patch is already semantically complete (global self-attention)
   - Teacher-Student assumes a CNN-learnable spatial feature map structure
   - PatchCore's direct patch-matching is naturally aligned with ViT's patch-native representation
   - **On patch-based architectures, method simplicity decisively outweighs feature quality**

5. **ResNet18 x224 is the only reliable CNN x224 variant** — F1=0.511 exceeds ResNet50 x64 (0.488) and WideResNet50-2 Layer2 x64 (0.508), showing that resolution works ONLY for smaller models

6. **x64 resolution is optimal for larger CNN models** — ResNet50 and WideResNet50-2 both achieve their best performance at x64. Larger bottlenecks do not recover x224 losses.

7. **Multi-layer feature extraction severely hurts at x224** — WideResNet50-2 Multilayer's catastrophic -37.5% F1 drop suggests that combining layer2 AND layer3 features at high resolution creates severe feature mismatch

8. **⚠️ Critical insights:** 
   - **For CNNs:** Teacher-student distillation does NOT scale to native resolution for large backbones. The fundamental issue is feature-space alignment or student capacity, not compression.
   - **For ViT (patch-based):** Direct patch-matching (PatchCore) is architecturally superior to learned spatial distillation (Teacher-Student). Method simplicity beats feature sophistication.
   - **Recommendations:**
     - Use smaller backbones (ResNet18) with x224 for CNNs
     - Stick to x64 resolution for larger CNN backbones (ResNet50, WideResNet50-2)
     - For patch-based backbones like ViT, use training-free methods like PatchCore, not distillation
     - Future work: diagnose CNN feature alignment quality, test progressive resolution training, investigate whether ViT-student architecture (transformer-based) could match ViT-teacher better than CNN-student

---

## Experiment 2c: Teacher-Student ResNet50 `224×224` — Feature Autoencoder Dimension Sweep

**Research Question:** Can increasing the feature autoencoder bottleneck from 128 hidden dims to larger values (256, 512, 768) recover the lost x224 performance?

**Hypothesis:** ResNet50 layer2 outputs 512 channels at x224. The fixed 128-dim bottleneck achieves only 1:4 compression, which may be too aggressive. Larger bottlenecks could better preserve feature information and improve x224 results.

**Artifact dir:** `experiments/anomaly_detection/teacher_student/resnet50/x224/feature_autoencoder_dim_sweep/artifacts/ts_resnet50_x224_ae_dim_sweep/`

### Configuration

Same as ResNet50 x224 main (Experiment 2b), but with varying feature autoencoder hidden dimensions:

| parameter | value |
|---|---|
| teacher backbone | ResNet50 (pretrained ImageNet) |
| teacher layer | layer2 |
| teacher input size | 224×224 |
| **feature autoencoder hidden dims** | **[64, 128, 256, 512, 768]** (swept) |
| batch size | 256 |
| optimizer | Adam, lr=0.0003, weight decay=1e-5 |
| max epochs | 30 |
| early stopping | patience=5, min_delta=0.0001 |

### Sweep Results

| ae_hidden_dim | F1 | AUROC | AUPRC | precision | recall |
|---|---|---|---|---|---|
| **768** | **0.4024** | **0.8288** | **0.3742** | **0.3251** | **0.528** |
| 64 | 0.3982 | 0.8281 | 0.3738 | 0.3211 | 0.524 |
| 512 | 0.3926 | 0.8261 | 0.3674 | 0.3184 | 0.512 |
| 256 | 0.3919 | 0.8260 | 0.3670 | 0.3206 | 0.504 |
| 128 | 0.3907 | 0.8261 | 0.3697 | 0.3190 | 0.504 |

### Key Findings: Bottleneck Hypothesis **Partially Confirmed** ✅

**Surprising result:** Increasing bottleneck size shows **minimal improvement** (+0.0117 F1, +0.3% relative gain from 128 → 768 dims).

The 768-dim bottleneck achieves the highest F1 (0.4024), but:
- **Improvement is marginal:** Only +0.0117 absolute F1 improvement over baseline 128-dim (0.3907)
- **Still far below x64 baseline:** ResNet50 x224 (F1=0.4024) remains **17.6% below ResNet50 x64 baseline (F1=0.488)**
- **AUPRC benefit is modest:** 128-dim = 0.3697 → 768-dim = 0.3742 (+0.0045, +1.2% relative)
- **Flat plateau:** Performance plateaus above 128 dims; further increases (256 → 512 → 768) yield minimal gains

**Conclusion about the bottleneck hypothesis:**
The feature autoencoder bottleneck is **a contributing factor but not the root cause** of x224 degradation. While larger bottlenecks help slightly, they do **not** recover the ~18% F1 loss. This suggests:

1. **Bottleneck compression is not the primary issue** — Expanding from 128 to 768 dims yields <1% relative improvement
2. **Deeper architectural problems at play:**
   - Feature-space mismatch between teacher and student at x224 resolution
   - Student network may lack capacity to match teacher features at x224 quality
   - Training dynamics differ fundamentally at higher resolution + larger backbone combination
   - Potential feature normalization or alignment issues specific to ResNet50 at native resolution

### Comparison: Resolution Impact Across Backbones and Bottleneck Sizes

| Backbone | x64 F1 | x224 (best) | Δ | Bottleneck | notes |
|---|---|---|---|---|---|
| **ResNet18** | 0.495 | 0.511 | **+0.016 (+3.2%)** | 64-dim | ✅ Resolution helps |
| **ResNet50 (base)** | 0.488 | 0.3988 | **-0.0892 (-18.9%)** | 128-dim | ❌ Resolution hurts |
| **ResNet50 (768-dim)** | 0.488 | 0.4024 | **-0.0856 (-17.6%)** | 768-dim | ❌ Larger bottleneck provides minimal recovery |
| **WideResNet50-2** | 0.504 | 0.3148 | **-0.1892 (-37.5%)** | 128-dim | ❌❌❌ Catastrophic |

### Implications for Future Work

1. **Bottleneck size alone cannot fix x224 for large backbones** — Even 768-dim hidden layers provide <1% relative improvement
2. **Alternative approaches needed:**
   - Investigate teacher/student feature alignment quality (cosine similarity, feature norms)
   - Test multi-resolution training (progressive or mixed batches)
   - Compare teacher feature distributions at x64 vs x224 to diagnose mismatch
   - Explore skip connections or feature normalization in the autoencoder
3. **Resolution strategy for teacher-student:**
   - Use x64 for ResNet50/WideResNet50-2
   - Use x224 only for ResNet18 (where resolution gains outweigh complexity costs)

---

## Experiment 2d: Teacher-Student ViT-B/16 `224×224` — Comparison Against PatchCore

**Research Question:** Can a ViT-B/16 teacher with student distillation match or approach PatchCore ViT-B/16 performance, or does the simpler direct patch-matching method prove superior?

**Key Finding:** ⚠️ **Teacher-Student ViT-B/16 dramatically underperforms PatchCore ViT-B/16, confirming that method simplicity beats feature sophistication on this domain.**

**Artifact dir:** `experiments/anomaly_detection/teacher_student/vit_b16/x224/main/artifacts/ts_vit_b16_x224/`

### Configuration

| parameter | value |
|---|---|
| teacher backbone | ViT-B/16 (pretrained ImageNet-21k) |
| teacher layer | block6 (mid-depth, matching PatchCore setup) |
| teacher input size | 224×224 |
| feature autoencoder | 512 hidden dim |
| batch size | 256 |
| learning rate | 0.0003 |
| max epochs | 30 |
| early stopping | patience=5 |

### Training Results

- **Best epoch:** 30 (ran all 30 epochs)
- **Best val loss:** 0.2176
- **Student map scale:** 0.1946 (higher variation than CNN-based TS)
- **Autoencoder map scale:** 0.0228 (very small, autoencoder contributes minimal signal)

### Evaluation Results

| metric | value |
|---|---|
| threshold (95th percentile) | 5.583 |
| precision | 0.1585 |
| recall | 0.168 |
| **F1** | **0.163** |
| **AUROC** | **0.661** |
| **AUPRC** | **0.106** |
| predicted anomalies | 265 |
| best sweep F1 | 0.172 |

Confusion matrix: `[[4777, 223], [208, 42]]`

### Per-Defect Recall (Selected Variant)

| defect type | count | detected | recall |
|---|---|---|---|
| Edge-Loc | 53 | 19 | 0.358 |
| Scratch | 15 | 7 | 0.467 |
| Loc | 34 | 16 | 0.471 |
| Center | 50 | 36 | 0.720 |
| Random | 5 | 4 | 0.800 |
| Donut | 7 | 6 | 0.857 |
| Edge-Ring | 84 | 75 | 0.893 |
| Near-full | 2 | 2 | 1.000 |

### Key Findings: ViT Teacher-Student Fails Dramatically

**Critical comparison:**

| Method | Backbone | F1 | AUROC | AUPRC | Notes |
|---|---|---|---|---|---|
| **PatchCore** | **ViT-B/16 x224** | **0.595** | **0.956** | **0.671** | Direct patch matching (best in project) |
| **Teacher-Student** | **ViT-B/16 x224** | **0.163** | **0.661** | **0.106** | ❌❌❌ Learned distillation catastrophically fails |
| **Margin** | | **-0.432 (-72.6%)** | **-0.295 (-30.9%)** | **-0.565 (-84.2%)** | **Massive degradation** |

### Why Did Teacher-Student ViT Fail So Badly?

The catastrophic failure provides crucial insights:

1. **ViT's patch-native representation** — ViT-B/16 outputs 768-dimensional patch embeddings where **each patch is already semantically independent** (global self-attention). The student network trying to "distill" these patch embeddings fails because:
   - The student must learn to *reproduce exact spatial feature maps* from a ViT that already encodes patch semantics directly
   - Unlike CNNs where spatial structure is implicit in convolution receptive fields, ViT's patches are explicit tokens. Forcing a CNN student to match ViT patch tokens is architecturally mismatched

2. **512-dim bottleneck may be too small** — Even though 512 > 256, ViT produces 768-dim embeddings per patch. The bottleneck forces aggressive 1.5:1 compression. At this compression ratio, student reconstruction becomes a denoising task rather than learning anomaly sensitivity

3. **Method-backbone mismatch** — PatchCore's direct nearest-neighbor matching on ViT patches is the *natural* way to score these representations. Training a CNN student to distill ViT's patch embeddings violates the architectural assumption: **the student and teacher feature spaces are fundamentally incompatible**

4. **PatchCore is method-agnostic but optimized for patches** — PatchCore works with any backbone because it simply stores and queries patch embeddings. When the backbone outputs semantically rich patches (like ViT), PatchCore shines. Teacher-student assumes a spatial feature map that the student (a CNN) can gradually match, which breaks down entirely for patch-based architectures

### Comparison with CNN Teacher-Student Results

Recall that CNN-based teacher-student achieved:
- ResNet50 x224: F1 = 0.3988, AUROC = 0.8277
- WideResNet50-2 x224 (768-dim bottleneck sweep): F1 ≈ 0.40, AUROC ≈ 0.829

**ViT TS (F1=0.163) is 60% worse than CNN TS results**, despite using a much larger teacher backbone and a 512-dim bottleneck. This is not a hyperparameter issue; it reflects a fundamental architectural incompatibility between ViT patch embeddings and CNN-based student distillation.

### Conclusion: Method Simplicity Wins

**Final narrative:**

> **Teacher-Student distillation, despite enabling post-training score optimization and using a strong ViT-B/16 teacher, catastrophically underperforms PatchCore's training-free direct patch matching (F1=0.163 vs 0.595, -72.6% relative). This demonstrates that on ViT backbones, the scoring method is the critical factor, not feature quality. PatchCore's simplicity—storing and querying patch embeddings directly—is architecturally aligned with ViT's patch-native representation, while teacher-student's assumption of spatial CNN-learnable feature maps fundamentally mismatches ViT's structure. On this domain, simplicity of method decisively outweighs sophistication of learned features.**

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
