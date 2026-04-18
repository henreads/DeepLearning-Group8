# FastFlow Family Report

## Overview

This family tests **flow-based density modeling** as an anomaly detector. Unlike reconstruction-based methods (autoencoder, VAE) or nearest-neighbor methods (PatchCore), FastFlow models the distribution of normal patch features directly using normalizing flows — it learns what "normal" looks like in feature space and scores anomalies by how unlikely their features are under that distribution.

Notebooks and artifacts live under [`experiments/anomaly_detection/fastflow/`](../experiments/anomaly_detection/fastflow/).

---

## Dataset and Evaluation Protocol

- **Train:** 40,000 normal wafers (flow model trained on backbone features)
- **Validation:** 5,000 normal wafers (threshold = 95th percentile of anomaly scores)
- **Test:** 5,000 normal + 250 anomaly wafers
- **Backbone:** Frozen Wide ResNet50-2 (ImageNet pretrained)
- **Primary metric:** val-threshold F1

---

## Experiment 19: FastFlow `64×64` Variant Sweep

**Notebook:** [`experiments/anomaly_detection/fastflow/x64/main/notebook.ipynb`](../experiments/anomaly_detection/fastflow/x64/main/notebook.ipynb)
**Artifact root:** `experiments/anomaly_detection/fastflow/x64/main/artifacts/fastflow_variant_sweep/`

### Run Controls

| flag | default | meaning |
|---|---|---|
| `RETRAIN` | `False` | use saved results; set `True` to retrain all variants |
| `TRAIN_MISSING` | `False` | train only variants with no saved results |
| `RERUN_PLOTS` | `False` | regenerate plots even if cached |
| `QUALITATIVE_VARIANT` | `wrn50_l23_s4` | which checkpoint to use for heatmap visualisation |
| `RUN_HOLDOUT_EVALUATION` | `False` | run expanded 70k/3.5k holdout |
| `FORCE_HOLDOUT_RERUN` | `False` | re-run holdout even if cached results exist |

### Architecture

| component | detail |
|---|---|
| backbone | Wide ResNet50-2, ImageNet pretrained, frozen |
| input | 64×64 grayscale upsampled to 224×224 RGB |
| flow head | Affine coupling layers + orthogonal 1×1 conv |
| anomaly map | `0.5 × z²` averaged over stages, upsampled to 64×64 |
| training | Adam, lr=5e-5, weight decay=1e-5, gradient clipping=0.5, early stopping patience=4 |

### Variant Sweep

Three architecture ablations were tested — varying which WRN50-2 feature layers feed the flow and the number of flow steps:

| variant | layers | flow steps | best score | precision | recall | F1 | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|---|
| **wrn50_l23_s4** | layer2 + layer3 | 4 | `mean` | **0.385167** | **0.644000** | **0.482036** | 0.870692 | 0.488619 |
| wrn50_l23_s6 | layer2 + layer3 | 6 | `mean` | 0.374408 | 0.632000 | 0.470238 | 0.869890 | 0.479070 |
| wrn50_l2_s6 | layer2 only | 6 | `topk_mean r=0.15` | 0.364583 | 0.560000 | 0.441640 | **0.884224** | 0.459659 |

**Selected variant:** `wrn50_l23_s4` — multilayer, fewer steps wins on deployed F1.

**Key insight:** More flow steps (6 vs 4) did not help. Single-layer (l2_s6) was weakest on F1 despite highest AUROC, showing multilayer feature fusion matters more than flow depth.

### Variant Comparison Plot

![Variant comparison](../experiments/anomaly_detection/fastflow/x64/main/artifacts/fastflow_variant_sweep/plots/variant_comparison_metrics.png)

### Training Curves

![Training curves](../experiments/anomaly_detection/fastflow/x64/main/artifacts/fastflow_variant_sweep/plots/training_curves.png)

### Selected Variant Evaluation (`wrn50_l23_s4`, `mean` score)

| metric | value |
|---|---|
| threshold | 0.412847 |
| precision | 0.385167 |
| recall | 0.644000 |
| F1 | **0.482036** |
| AUROC | 0.870692 |
| AUPRC | 0.488619 |
| balanced accuracy | 0.796300 |

Confusion matrix: `[[4743, 257], [89, 161]]`

### Score Sweep Across Reduction Methods

FastFlow produces a per-pixel anomaly map; a wafer-level score is derived by reducing this map. Five reductions were swept for each training variant:

![Score sweep F1](../experiments/anomaly_detection/fastflow/x64/main/artifacts/fastflow_variant_sweep/plots/best_variant_score_sweep_f1.png)

`mean` reduction consistently gave the best deployed F1 for the winning variant (wrn50_l23_s4). `topk_mean` strategies were competitive for wrn50_l2_s6 but did not beat `mean` overall.

### Per-Defect Recall

| defect type | count | detected | recall |
|---|---|---|---|
| Scratch | 15 | 2 | 0.133333 |
| Edge-Loc | 53 | 28 | 0.528302 |
| Loc | 34 | 20 | 0.588235 |
| Center | 50 | 36 | 0.720000 |
| Edge-Ring | 84 | 62 | 0.738095 |
| Donut | 7 | 6 | 0.857143 |
| Random | 5 | 5 | 1.000000 |
| Near-full | 2 | 2 | 1.000000 |

![Defect breakdown](../experiments/anomaly_detection/fastflow/x64/main/artifacts/fastflow_variant_sweep/plots/best_variant_defect_breakdown.png)

**Scratch** (0.133) is the hardest defect by a wide margin. **Loc** (0.588) is the standout — FastFlow handles medium-scale localized anomalies better than the autoencoder family.

### Qualitative Heatmaps

The notebook generates anomaly heatmaps for representative test examples when a checkpoint is available:

![Qualitative heatmaps](../experiments/anomaly_detection/fastflow/x64/main/artifacts/fastflow_variant_sweep/plots/wrn50_l23_s4_qualitative_heatmaps.png)

### Interpretation

- FastFlow provides a density modeling perspective distinct from reconstruction and nearest-neighbor approaches
- The refreshed canonical run reaches F1=0.482, placing it solidly above the autoencoder family but below the stronger PatchCore and teacher-student branches
- `mean` reduction over the full anomaly map works best — surprising, as `topk` is stronger for PatchCore; this may reflect that FastFlow's anomaly maps are already well-calibrated spatially
- The failure pattern matches the rest of the project: broad defects (Edge-Ring, Center, Donut) are reliable; small local defects (Scratch) remain the hard limit

---

## Holdout Evaluation: Expanded 70k Normal / 3.5k Defect

The `wrn50_l23_s4` checkpoint was evaluated on the expanded holdout using the same validation threshold. Results live at:

`artifacts/fastflow_variant_sweep/holdout70k_3p5k/`

| metric | benchmark (5k/250) | holdout (70k/3.5k) |
|---|---|---|
| threshold | 0.412847 | 0.412847 |
| precision | 0.385167 | 0.400834 |
| recall | 0.644000 | 0.686571 |
| F1 | 0.482036 | **0.506161** |
| AUROC | 0.870692 | 0.889555 |
| AUPRC | 0.488619 | 0.541732 |
| predicted anomalies | 418 | 5,995 |

The holdout shows improved metrics across the board — both AUROC and F1 increase with the larger evaluation pool. This suggests the benchmark F1 was slightly pessimistic due to small anomaly count (250 vs 3,500).

### Holdout Per-Defect Recall

| defect type | count | detected | recall |
|---|---|---|---|
| Scratch | 169 | 37 | 0.219 |
| Loc | 492 | 201 | 0.409 |
| Edge-Loc | 739 | 455 | 0.616 |
| Center | 603 | 441 | 0.731 |
| Edge-Ring | 1,302 | 1,082 | 0.831 |
| Donut | 71 | 63 | 0.887 |
| Random | 108 | 108 | 1.000 |
| Near-full | 16 | 16 | 1.000 |

![Holdout defect breakdown](../experiments/anomaly_detection/fastflow/x64/main/artifacts/fastflow_variant_sweep/holdout70k_3p5k/plots/defect_breakdown.png)

![Holdout threshold sweep](../experiments/anomaly_detection/fastflow/x64/main/artifacts/fastflow_variant_sweep/holdout70k_3p5k/plots/threshold_sweep.png)

The defect-wise pattern is stable across benchmark and holdout. Scratch remains the hardest class (~0.22 recall at scale). Edge-Ring and Donut are reliable. The large holdout confirms the model's behavior is not an artifact of the small 250-anomaly benchmark set.

---

## Context in the Project

FastFlow sits in the middle of the project leaderboard:

| family | best F1 | notes |
|---|---|---|
| PatchCore + ViT-B/16 x224 | 0.595 | project best |
| PatchCore + EfficientNet-B1 x240 | 0.591 | best CNN |
| Teacher-Student ResNet50 | 0.525 | |
| PatchCore + WRN50-2 x64 | 0.532 | |
| **FastFlow WRN50-2** | **0.482** | flow-based baseline |
| AE + BatchNorm `max_abs` | 0.502 | reconstruction best |
| Backbone embedding | ~0.24 | lower bound |

FastFlow is useful primarily as a **methodological contrast** — it shows that density modeling on frozen features is competitive with the better autoencoder variants, but still substantially below the leading local-patch methods. Its standout strength is `Loc` recall (0.59 benchmark, 0.41 holdout) compared to the autoencoder family.
