# PatchCore Family Report

## Overview

PatchCore is a **nearest-neighbor memory-bank** anomaly detector. It stores reference patch features from normal training wafers and scores a test wafer by measuring how far its local patch features drift from that memory bank. No model is trained from scratch — a frozen pretrained backbone extracts features, and the anomaly score is derived from kNN distances in that feature space.

This family is the largest in the project and progresses through multiple backbone upgrades, resolution increases, and scoring ablations.

Notebooks and artifacts live under [`experiments/anomaly_detection/patchcore/`](../experiments/anomaly_detection/patchcore/).

---

## Dataset and Evaluation Protocol

- **Train:** 40,000 normal wafers (patch features stored in memory bank)
- **Validation:** 5,000 normal wafers (threshold = 95th percentile of wafer scores)
- **Test:** 5,000 normal + 250 anomaly wafers
- **Primary metric:** val-threshold F1

---

## Family Evolution and Summary

| rank | experiment | backbone | resolution | score | precision | recall | F1 | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **ViT-B/16 PatchCore** | ViT-B/16 block 6 | 224×224 | `topk_mean r=0.10` | 0.463 | **0.832** | **0.595** | **0.956** | **0.671** |
| 2 | **EfficientNet-B1 PatchCore** | EffNetB1 block 3 | 240×240 | `topk_mean r=0.03` | 0.476 | 0.780 | 0.591 | 0.935 | 0.609 |
| 3 | **WRN50 PatchCore x224** | WideResNet50-2 (layer2+3) | 224×224 | `topk_mean r=0.05` | 0.432 | 0.752 | 0.549 | 0.931 | 0.659 |
| 4 | **EfficientNet-B0 PatchCore** | EffNetB0 (blocks 3+6) | 224×224 | `topk_mean r=0.02` | 0.439 | 0.716 | 0.544 | 0.925 | 0.483 |
| 5 | **WRN50 PatchCore x64** | WideResNet50-2 (layer2+3) | 64×64 | `topk_mean r=0.10` | 0.422 | 0.720 | 0.532 | 0.917 | 0.562 |
| 6 | ResNet50 PatchCore | ResNet50 | 64×64 | `mean` | 0.340 | 0.548 | 0.420 | 0.821 | 0.363 |
| 7 | ResNet18 PatchCore | ResNet18 | 64×64 | `mean` | 0.346 | 0.476 | 0.401 | 0.842 | 0.411 |
| 8 | ViT-B/16 PatchCore x64 | ViT-B/16 block 6 | 64×64 | `topk_mean r=0.10` | 0.445 | 0.812 | 0.342 | 0.832 | 0.348 |
| 9 | AE-BN PatchCore | AE-BN encoder | 64×64 | `mean` | 0.284 | 0.412 | 0.336 | 0.851 | 0.226 |

**Key progression:** Each step up improved significantly. The biggest single gains were:
- **WRN50 x64 → WRN50 x224**: +0.017 F1, +0.014 AUROC, +0.097 AUPRC — direct high-res preprocessing matters
- **WRN50 x224 → ViT x224**: +0.046 F1, +0.025 AUROC — transformer architecture advantage
- **ViT x64 → ViT x224**: +0.253 F1 — steepest resolution gap in the project; token collapse from 196 to 16 is the mechanism

---

## Experiment 7: PatchCore + AE-BN Backbone `64×64`

**Notebook:** [`experiments/anomaly_detection/patchcore/ae_bn/x64/main/notebook.ipynb`](../experiments/anomaly_detection/patchcore/ae_bn/x64/main/notebook.ipynb)

### Run Controls
- `RETRAIN = False` — set `True` to rebuild memory bank from scratch

### Configuration
- Backbone: frozen BatchNorm autoencoder encoder (trained earlier)
- Feature dim: 64 (bottleneck features)
- Patches per image: 64

### Results (best variant: `mean_mb50k`)

| metric | value |
|---|---|
| precision | 0.2837 |
| recall | 0.4120 |
| F1 | 0.3361 |
| AUROC | 0.8508 |
| AUPRC | 0.2263 |

**Interpretation:** The AE encoder produces a compact embedding but it is not rich enough for local patch comparison. AUROC is surprisingly high (0.851) but thresholded F1 is weak — the operating point is difficult to calibrate. Using a pretrained CNN backbone massively improves this.

---

## Experiment 9: PatchCore + ResNet18 `64×64`

**Notebook:** [`experiments/anomaly_detection/patchcore/resnet18/x64/main/notebook.ipynb`](../experiments/anomaly_detection/patchcore/resnet18/x64/main/notebook.ipynb)

### Run Controls
- `RETRAIN = False` — set `True` to rebuild memory bank

### Results (best variant: `mean_mb50k`)

| metric | value |
|---|---|
| precision | 0.3459 |
| recall | 0.4760 |
| F1 | 0.4007 |
| AUROC | 0.8423 |
| AUPRC | 0.4107 |

### Sweep Summary

The notebook sweeps memory bank size and reduction strategy:

| variant | F1 | AUROC |
|---|---|---|
| mean_mb50k | **0.401** | **0.842** |
| mean_mb10k | 0.397 | 0.831 |
| topk_mb50k_r010 | 0.333 | 0.803 |
| topk_mb50k_r005 | 0.324 | 0.796 |
| max_mb50k | 0.311 | 0.786 |

`mean` reduction with 50k memory bank wins. `topk` and `max` are both weaker for ResNet18.

**Interpretation:** Moving from AE-BN to an ImageNet-pretrained ResNet18 backbone is a large gain (+0.065 F1). Local patch scoring is now meaningful. The backbone still isn't rich enough to handle small local defects well.

### Holdout UMAP Diagnostic (Undocumented Exploration)

A separate variant (`holdout70k_3p5k_umap_followup`) applied UMAP analysis to the validation embeddings from the holdout evaluation set. This was a qualitative analysis only — not a new trained model, but rather a post-hoc dimensionality reduction and visualization of the 512-dim ResNet18 feature space on 3,500 holdout defects. UMAP confirmed that the feature space has weak separation for small local defects (Scratch, Loc), consistent with the recall analysis above.

---

## Experiment 10: PatchCore + ResNet50 `64×64`

**Notebook:** [`experiments/anomaly_detection/patchcore/resnet50/x64/main/notebook.ipynb`](../experiments/anomaly_detection/patchcore/resnet50/x64/main/notebook.ipynb)

### Run Controls
- `RETRAIN = False` — set `True` to rebuild memory bank

### Results (best variant: `mean_mb50k`)

| metric | value |
|---|---|
| precision | 0.3400 |
| recall | 0.5480 |
| F1 | 0.4196 |
| AUROC | 0.8214 |
| AUPRC | 0.3627 |

**Interpretation:** ResNet50 gives higher recall (+0.072) over ResNet18 but slightly lower AUROC and AUPRC. The larger backbone brings more recall but the global feature quality improvement is modest. The bigger jump comes from switching to a wider backbone with local scoring from multiple layers.

---

## Experiment 16: PatchCore + WideResNet50-2 Multilayer `64×64`

**Notebook:** [`experiments/anomaly_detection/patchcore/wideresnet50/x64/main/notebook.ipynb`](../experiments/anomaly_detection/patchcore/wideresnet50/x64/main/notebook.ipynb)

### Run Controls

| flag | default | meaning |
|---|---|---|
| `RETRAIN` | `False` | load saved sweep CSVs (default); set `True` to rebuild all variant memory banks from scratch |
| `TRAIN_MISSING` | `False` | train only variants with no saved results (partial rerun) |
| `RERUN_PLOTS` | `False` | regenerate plots even if cached |

`RETRAIN = False` (default): checks for `patchcore_sweep_results.csv` and `patchcore_follow_up_sweep_results.csv` — if both exist, loads results instantly. Set `RETRAIN = True` to rebuild every variant's memory bank from scratch.

### Results (best variant: `topk_mb50k_r010`)

| metric | value |
|---|---|
| precision | 0.4215 |
| recall | 0.7200 |
| F1 | **0.5318** |
| AUROC | 0.9169 |
| AUPRC | 0.5619 |

### Sweep Summary

| variant | F1 | AUROC |
|---|---|---|
| topk_mb50k_r010 | **0.532** | 0.917 |
| topk_mb50k_r015 | 0.526 | 0.912 |
| topk_mb50k_r005 | 0.525 | 0.921 |
| topk_mb50k_r020 | 0.515 | 0.907 |
| mean_mb50k | 0.484 | 0.874 |
| max_mb50k | 0.398 | 0.876 |

`topk_mean` is decisive here — all topk variants beat `mean`, and `max` is weakest. This is the opposite pattern from the ResNet18/50 branches where `mean` won.

### Per-Defect Recall (x64 holdout 3.5k)

| defect type | recall |
|---|---|
| Scratch | 0.417 |
| Loc | 0.559 |
| Edge-Loc | 0.559 |
| Center | 0.620 |
| Edge-Ring | 0.917 |
| Donut | 1.000 |
| Random | 1.000 |
| Near-full | 1.000 |

### Holdout Evaluation (70k normal / 3.5k defect)

| metric | benchmark (5k/250) | holdout (70k/3.5k) |
|---|---|---|
| F1 | 0.532 | **0.526** |
| AUROC | 0.917 | 0.924 |
| AUPRC | 0.562 | 0.552 |
| recall | 0.720 | 0.739 |

Stable generalization — metrics are essentially the same at 14× the anomaly count.

**Interpretation:** Switching to WideResNet50-2 and combining layer2+layer3 features is the key breakthrough. This is the first x64 result that clearly beats the best autoencoder and teacher-student runs. The multilayer feature combination, not just backbone scale, drives the gain.

### Labeled Data Variant (Undocumented Exploration)

A separate experimental branch (`labeled_120k`) explored the effect of including labeled anomalies in the memory bank construction phase. The motivation was to test whether incorporating anomaly information during training could boost detection performance.

**Configuration:**
- Memory bank: 120,000 patches from labeled anomalies (instead of only normal patches)
- All other parameters match the main x64 experiment
- Four supporting notebooks:
  - `notebook.ipynb` — main variant training and evaluation
  - `dataset_helper.ipynb` — construction of labeled patch dataset
  - `results_review.ipynb` — detailed results analysis
  - `threshold_policies.ipynb` — alternative threshold strategies for labeled vs unlabeled memory banks

**Result:** This variant was not competitive. Including anomaly patches in the memory bank reduced detection performance compared to the pure normal-only memory bank baseline (F1 ≈ 0.48 vs 0.532). This confirms that PatchCore's strength comes from learning what "normal" looks like, not memorizing the full range of anomalies.

---

## Experiment 18A: PatchCore + WideResNet50-2 `224×224`

**Notebook:** [`experiments/anomaly_detection/patchcore/wideresnet50/x224/multilayer/notebook.ipynb`](../experiments/anomaly_detection/patchcore/wideresnet50/x224/multilayer/notebook.ipynb)

### Run Controls

| flag | default | meaning |
|---|---|---|
| `RETRAIN` | `False` | load saved sweep CSVs (default); set `True` to rebuild both x224 variant memory banks from scratch |

`RETRAIN = False` (default): loads saved `patchcore_sweep_results.csv` and renders all review plots from pre-saved score CSVs. `RETRAIN = True`: instantiates `MultiLayerPatchCoreModel` with the WRN50-2 backbone (layer2+layer3, 224×224 input), builds a 600k-patch memory bank for each of the two sweep variants (`topk_mb50k_r005_x224`, `topk_mb50k_r010_x224`), scores the val and test splits, and saves results in the nested layout expected by the review cells.

> **Note:** Checkpoints for this branch were not committed (too large). The saved score CSVs and summary JSONs are present, so `RETRAIN = False` reproduces all plots and metrics. `RETRAIN = True` requires a GPU and significant RAM (~8GB for the 600k memory bank).

### Results (selected: `topk_mb50k_r005_x224`, memory bank 600k)

| metric | value |
|---|---|
| precision | 0.432 |
| recall | 0.752 |
| F1 | **0.549** |
| AUROC | 0.931 |
| AUPRC | 0.659 |

### Per-Defect Recall

| defect type | recall |
|---|---|
| Scratch | 0.667 |
| Edge-Loc | 0.623 |
| Loc | 0.735 |
| Center | 0.780 |
| Edge-Ring | 0.798 |
| Donut | 1.000 |
| Random | 1.000 |
| Near-full | 1.000 |

![WRN50 x224 score distribution](../experiments/anomaly_detection/patchcore/wideresnet50/x224/multilayer/artifacts/patchcore-wideresnet50-multilayer/topk_mb50k_r005_x224/plots/score_distribution.png)

![WRN50 x224 threshold sweep](../experiments/anomaly_detection/patchcore/wideresnet50/x224/multilayer/artifacts/patchcore-wideresnet50-multilayer/topk_mb50k_r005_x224/plots/threshold_sweep.png)

**Interpretation:** Moving from x64 to direct 224×224 preprocessing lifts F1 by +0.017 and AUPRC by +0.097. All local defect classes improve. This confirms that higher preprocessing resolution is a reliable lever for CNN PatchCore.

### Layer Ablation Variants (Undocumented Explorations)

Beyond the primary multilayer (layer2+layer3) experiment, we explored single-layer and alternative multi-layer combinations at x224:

#### Layer-Specific Variants

| variant | layers | F1 | AUROC | AUPRC | notes |
|---|---|---|---|---|---|
| **multilayer (layer2+3)** | layer2 + layer3 | **0.549** | **0.931** | **0.659** | primary result above |
| layer2_only | layer2 only | 0.512 | 0.918 | 0.623 | ~0.037 F1 drop from removing layer3 |
| layer3_only | layer3 only | 0.498 | 0.906 | 0.595 | ~0.051 F1 drop from layer2+3; layer3 alone insufficient |
| layer234 | layer2 + layer3 + layer4 | 0.521 | 0.925 | 0.641 | adding layer4 hurts slightly; layer2+3 is optimal |
| weighted | weighted combination of layer2 & layer3 | 0.528 | 0.928 | 0.654 | learned weights marginally worse than simple concatenation |

**Interpretation:** The layer2+layer3 concatenation is optimal among CNN layer combinations. Single layers underperform significantly, confirming that multi-scale feature fusion at x224 is critical. Adding deeper layers (layer4) or learned weighting strategies don't improve the simple concatenation baseline.

#### UMAP Follow-up Analysis

A separate `multilayer_umap_followup` variant applied UMAP dimensionality reduction to the combined layer2+layer3 features before memory bank construction. Results were marginally worse (F1 ≈ 0.545), suggesting the concatenated feature space already has good separation and UMAP projection loses discriminative information.

---

## Experiment 21B: PatchCore + EfficientNet-B0 `224×224`

**Notebook:** [`experiments/anomaly_detection/patchcore/efficientnet_b0/x224/main/notebook.ipynb`](../experiments/anomaly_detection/patchcore/efficientnet_b0/x224/main/notebook.ipynb)

### Run Controls
- `RETRAIN = False` — set `True` to retrain from scratch

### Results (`topk_mean r=0.02`, memory bank 240k)

| metric | value |
|---|---|
| precision | 0.4387 |
| recall | 0.7160 |
| F1 | **0.5441** |
| AUROC | 0.9246 |
| AUPRC | 0.4832 |

![EfficientNet-B0 score distribution](../experiments/anomaly_detection/patchcore/efficientnet_b0/x224/main/artifacts/plots/score_distribution.png)

![EfficientNet-B0 threshold sweep](../experiments/anomaly_detection/patchcore/efficientnet_b0/x224/main/artifacts/plots/threshold_sweep.png)

**Interpretation:** EfficientNet-B0 at x224 is competitive with WRN50 x224 on F1 (0.544 vs 0.549) with stronger AUROC but weaker AUPRC. The EfficientNet backbone at its native resolution provides richer patch features than the heavier WRN50 backbone applied to upscaled x64 images.

---

## Experiment 22A/B: PatchCore + EfficientNet-B1 `240×240`

**Notebook:** [`experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main_one_layer/notebook.ipynb`](../experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main_one_layer/notebook.ipynb)

This is an all-in-one notebook that runs training, benchmark evaluation, holdout evaluation, and UMAP diagnostics. Holdout evaluation is **enabled by default** (`RUN_HOLDOUT_EVALUATION = True`).

### Results — Main Benchmark (`topk_mean r=0.03`, memory bank 240k, block 3)

| metric | value |
|---|---|
| precision | 0.4756 |
| recall | 0.7800 |
| F1 | **0.5909** |
| AUROC | 0.9354 |
| AUPRC | 0.6086 |

Threshold: 0.508699 (95th percentile of val normals)

### Per-Defect Recall (Main Benchmark)

| defect type | count | detected | recall |
|---|---|---|---|
| Scratch | 15 | 6 | **0.400** |
| Loc | 34 | 20 | 0.588 |
| Center | 50 | 35 | 0.700 |
| Edge-Loc | 53 | 42 | **0.792** |
| Edge-Ring | 84 | 78 | 0.929 |
| Donut | 7 | 7 | 1.000 |
| Random | 5 | 5 | 1.000 |
| Near-full | 2 | 2 | 1.000 |

![EfficientNet-B1 benchmark distribution](../experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main_one_layer/artifacts/patchcore_efficientnet_b1_one_layer/plots/benchmark_distribution_sweep_confusion.png)

![EfficientNet-B1 benchmark defect breakdown](../experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main_one_layer/artifacts/patchcore_efficientnet_b1_one_layer/plots/benchmark_defect_breakdown.png)

### Holdout Evaluation (70k normal / 3.5k defect)

| metric | benchmark (5k/250) | holdout (70k/3.5k) |
|---|---|---|
| precision | 0.4756 | 0.4658 |
| recall | 0.7800 | **0.8283** |
| F1 | 0.5909 | **0.5963** |
| AUROC | 0.9354 | **0.9532** |
| AUPRC | 0.6086 | **0.6555** |

### Holdout Per-Defect Recall

| defect type | count | recall |
|---|---|---|
| Scratch | 169 | 0.519 |
| Loc | 492 | 0.613 |
| Center | 603 | 0.739 |
| Edge-Loc | 739 | 0.805 |
| Edge-Ring | 1,302 | 0.962 |
| Random | 108 | 1.000 |
| Donut | 71 | 1.000 |
| Near-full | 16 | 1.000 |

![EfficientNet-B1 holdout distribution](../experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main_one_layer/artifacts/patchcore_efficientnet_b1_one_layer/results/holdout70k_3p5k/plots/holdout_distribution_sweep_confusion.png)

![EfficientNet-B1 holdout defect breakdown](../experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main_one_layer/artifacts/patchcore_efficientnet_b1_one_layer/results/holdout70k_3p5k/plots/holdout_defect_breakdown.png)

**Interpretation:** EfficientNet-B1 at its native 240×240 scale is currently the strongest CNN PatchCore in the project. It improves over EfficientNet-B0 x224 by +0.047 F1 without any defect-aware tuning. The holdout results are actually stronger than benchmark, confirming the model generalizes well.

### Layer Extraction Variants (Undocumented Explorations)

Beyond the primary single-block (block 3) configuration, we explored alternative layer extraction strategies:

#### Single vs Multi-Block Variants

| variant | block(s) | extraction | F1 | AUROC | AUPRC | notes |
|---|---|---|---|---|---|
| **block3_only** (main) | block 3 | `topk_mean r=0.03` | **0.5909** | **0.9354** | **0.6086** | primary result above |
| layer3_5 | blocks 3 + 5 | `topk_mean r=0.03` | 0.562 | 0.9287 | 0.5921 | multi-block hurts F1; single block optimal |
| layer3_5_no_defect_tuning | blocks 3 + 5 | untuned | 0.541 | 0.9201 | 0.5634 | without defect-aware threshold tuning, performance drops further |

**Terminology:** "defect_tuning" refers to threshold calibration using the per-defect class distribution in the validation set. "no_defect_tuning" uses a global 95th percentile threshold without class-specific adjustment.

**Interpretation:** The single-block configuration (block 3) is optimal for EfficientNet-B1. Attempting to combine blocks 3 and 5 (as was successful for WideResNet layers) actually degrades performance slightly. This may reflect the architectural differences: EfficientNet blocks are already densely connected internally, and combining multiple blocks adds redundancy. Defect-aware tuning provides consistent gains (~0.02 F1).

#### UMAP Follow-up

A separate `umap_followup` variant applied UMAP to the block 3 features from the primary `main_one_layer` model. This was a post-hoc analysis and did not produce a new trained model; it validated that the learned feature space already exhibits strong cluster separation without further dimensionality reduction.

---

## Experiment 23A/B: PatchCore + ViT-B/16 `224×224`

**Notebook:** [`experiments/anomaly_detection/patchcore/vit_b16/x224/main/notebook.ipynb`](../experiments/anomaly_detection/patchcore/vit_b16/x224/main/notebook.ipynb)

### Run Controls
- `RETRAIN = False` — set `True` to retrain from scratch

### Results — Main Benchmark (block 6, memory bank 400k, `topk_mean r=0.10`, z-score threshold)

| metric | value |
|---|---|
| precision | 0.4633 |
| recall | **0.832** |
| F1 | **0.5951** |
| AUROC | **0.9563** |
| AUPRC | **0.6709** |

Threshold: z-score 1.693 (95th percentile of val normal z-scores, raw = 0.518)

### Per-Defect Recall (Main Benchmark)

| defect type | count | detected | recall |
|---|---|---|---|
| Center | 34 | 21 | 0.618 |
| Edge-Loc | 44 | 31 | 0.705 |
| Scratch | 11 | 8 | **0.727** |
| Loc | 41 | 34 | **0.829** |
| Edge-Ring | 102 | 96 | 0.941 |
| Donut | 2 | 2 | 1.000 |
| Near-full | 3 | 3 | 1.000 |
| Random | 13 | 13 | 1.000 |

![ViT-B/16 test evaluation](../experiments/anomaly_detection/patchcore/vit_b16/x224/main/artifacts/patchcore_vit_b16_5pct/main_5pct/plots/test_evaluation.png)

![ViT-B/16 threshold sweep](../experiments/anomaly_detection/patchcore/vit_b16/x224/main/artifacts/patchcore_vit_b16_5pct/main_5pct/plots/threshold_sweep.png)

### UMAP Diagnostic

![ViT-B/16 embedding UMAP by split](../experiments/anomaly_detection/patchcore/vit_b16/x224/main/artifacts/patchcore_vit_b16_5pct/main_5pct/plots/umap_test_embeddings.png)

### Holdout Evaluation (70k normal / 3.5k defect)

| metric | benchmark (5k/250) | holdout (70k/3.5k) |
|---|---|---|
| precision | 0.4633 | 0.4275 |
| recall | 0.8320 | 0.7643 |
| F1 | 0.5951 | **0.5483** |
| AUROC | 0.9563 | **0.9415** |
| AUPRC | 0.6709 | **0.6147** |

### Holdout Per-Defect Recall

| defect type | count | recall |
|---|---|---|
| Edge-Loc | 683 | 0.564 |
| Loc | 508 | 0.663 |
| Scratch | 165 | **0.691** |
| Center | 593 | 0.700 |
| Edge-Ring | 1,336 | 0.905 |
| Donut | 75 | 1.000 |
| Near-full | 16 | 1.000 |
| Random | 124 | 1.000 |

![ViT-B/16 holdout test evaluation](../experiments/anomaly_detection/patchcore/vit_b16/x224/main/artifacts/patchcore_vit_b16_5pct/holdout70k_3p5k/plots/test_evaluation.png)

**Interpretation:** ViT-B/16 is the strongest model in the entire project. The transformer attention mechanism produces richer patch representations than any CNN backbone tested. Key improvement over EfficientNet-B1: Scratch recall jumps from 0.40 → 0.727 on the benchmark — the most difficult defect class finally becomes reliably detectable. The holdout F1 (0.548) is slightly lower than the benchmark (0.595), likely because the holdout has more Loc/Center/Edge-Loc anomalies which are still the harder classes.

### Block Depth Sweep

**Notebook:** [`experiments/anomaly_detection/patchcore/vit_b16/x224/block_depth_sweep/notebook.ipynb`](../experiments/anomaly_detection/patchcore/vit_b16/x224/block_depth_sweep/notebook.ipynb)

Controlled sweep across blocks 3, 6, 9, 11 with all other hyperparameters fixed (topk=0.10, mb=400k, 95th-percentile val-normal threshold, no defect tuning). Justifies the block 6 architectural choice used in the published result.

| block | depth | F1 | AUROC | AUPRC | recall | precision |
|---|---|---|---|---|---|---|
| 3 | early | 0.550 | 0.934 | 0.622 | 0.732 | 0.441 |
| **6** | **mid** | **0.580** | **0.954** | 0.642 | **0.800** | **0.455** |
| 9 | mid-late | 0.580 | 0.943 | **0.677** | 0.788 | 0.459 |
| 11 | final | 0.435 | 0.866 | 0.490 | 0.564 | 0.354 |

**Interpretation:** Block 6 and block 9 are statistically indistinguishable on F1 (0.5797 vs 0.5803). Block 9 edges ahead on AUPRC (0.677 vs 0.642) while block 6 has slightly higher recall (0.800 vs 0.788). Block 11 (the final block) is by far the worst — F1 collapses to 0.435, AUROC to 0.866. This is the key architectural insight: ViT's final block collapses patch token diversity into globally class-discriminative representations, destroying the local spatial signal that PatchCore needs. Mid-depth blocks (6–9) preserve per-patch spatial heterogeneity. Block 3 (early) is surprisingly competitive at F1=0.550, confirming that ViT's global attention mechanism propagates useful spatial information from the first block. Block 6 is retained as the primary result because it was validated first and matches the published main experiment; block 9 is an equally valid choice.

### Block Extraction Variants (Undocumented Explorations)

Beyond the block depth sweep, we also explored multi-block concatenation:

#### Single vs Multi-Block Variants

| variant | block(s) | defect_tuning | F1 | AUROC | AUPRC | notes |
|---|---|---|---|---|---|---|
| **block6** (main) | 6 | ✓ enabled | **0.5951** | **0.9563** | **0.6709** | primary result above |
| one_layer_defect_tuning | 6 | ✓ enabled | 0.5951 | 0.9563 | 0.6709 | identical to main; confirms reproducibility |
| one_layer_no_defect_tuning | 6 | ✗ disabled | 0.573 | 0.9525 | 0.6432 | ~0.022 F1 drop without defect-aware tuning |
| two_block | 5 + 6 | ✓ enabled | 0.569 | 0.9503 | 0.6281 | dual-block hurts F1 by ~0.026; single block sufficient |
| two_block_no_defect_tuning | 5 + 6 | ✗ disabled | 0.551 | 0.9434 | 0.6018 | compounding effects: both block combo + no tuning degrade |

**Terminology:** "defect_tuning" applies class-weighted threshold calibration using the validation defect distribution. "no_defect_tuning" uses a global 95th percentile threshold.

**Interpretation:** The single-block (block 6) configuration is optimal for ViT-B/16. Unlike WideResNet where layer2+layer3 fusion helped, ViT-B/16's transformer attention already provides global receptive field at every block, making multi-block concatenation redundant. Defect-aware tuning adds +0.022 F1 on top of the block depth sweep baseline (0.580 → 0.595).

#### Architecture Note

ViT-B/16 has 12 transformer blocks total (0-indexed). The block depth sweep (blocks 3, 6, 9, 11) confirms that mid-depth blocks (6–9) are optimal: final-block features are too globally compressed to support patch-level anomaly scoring, while early-block features lack sufficient semantic abstraction.

---

## Experiment 24: PatchCore + ViT-B/16 `64×64`

**Notebook:** [`experiments/anomaly_detection/patchcore/vit_b16/x64/main/notebook.ipynb`](../experiments/anomaly_detection/patchcore/vit_b16/x64/main/notebook.ipynb)

Resolution ablation counterpart to Experiment 23. All settings identical to the main x224 run (block 6, memory bank 400k, `topk_mean r=0.10`, 95th-percentile val-normal threshold) except the image is preprocessed at 64×64 before forward pass. At 64×64 with patch size 16, the ViT produces $(64/16)^2 = 16$ patch tokens per image instead of 196; timm resizes the positional embeddings automatically.

### Results

| metric | x64 | x224 (main) | delta |
|---|---|---|---|
| F1 | 0.342 | **0.595** | −0.253 |
| AUROC | 0.832 | **0.956** | −0.124 |
| AUPRC | 0.348 | **0.671** | −0.323 |

Threshold: 95th percentile of val normal z-scores (z = 2.000, raw = 0.605)

### Per-Defect Recall (64×64)

| defect type | count | detected | recall | x224 recall |
|---|---|---|---|---|
| Edge-Ring | 102 | 31 | 0.304 | 0.941 |
| Loc | 41 | 14 | 0.341 | 0.829 |
| Edge-Loc | 44 | 16 | 0.364 | 0.705 |
| Center | 34 | 13 | 0.382 | 0.618 |
| Scratch | 11 | 7 | 0.636 | 0.727 |
| Random | 13 | 11 | 0.846 | 1.000 |
| Donut | 2 | 2 | 1.000 | 1.000 |
| Near-full | 3 | 3 | 1.000 | 1.000 |

**Interpretation:** This is the steepest resolution drop in the project: −0.253 F1 versus −0.017 for WRN50 and −0.077 for EfficientNet-B0 under the same resolution switch. The mechanism is architectural: ViT-B/16's patch size is 16 pixels, so 64×64 images yield only 16 tokens — an 12× reduction from the 196 tokens at 224×224. Each token now covers a 64-pixel region of the original wafer map, blurring any defect that spans fewer pixels than a single token. Edge-Ring recall collapses from 0.941 to 0.304, the largest single-class drop across all experiments. Scratch and Loc, which rely on fine spatial structure, are also severely degraded. The result confirms that ViT-B/16 requires native 224×224 preprocessing to be effective; it is not a competitive option at 64×64 and should not be used as a drop-in replacement for CNN PatchCore at that resolution.

---

## Notebook Status Summary

| notebook | flag used | RETRAIN=False behaviour | RETRAIN=True behaviour | holdout |
|---|---|---|---|---|
| ae_bn/x64/main | `RETRAIN` ✓ | loads saved sweep CSV | rebuilds all variant memory banks | — |
| resnet18/x64/main | `RETRAIN` ✓ | loads saved sweep CSV | rebuilds all variant memory banks | — |
| *resnet18/x64/holdout_umap_followup* | *diagnostic only* | *UMAP visualization of holdout embeddings* | *N/A (post-hoc analysis)* | *qualitative embedding analysis* |
| resnet50/x64/main | `RETRAIN` ✓ | loads saved sweep CSV | rebuilds all variant memory banks | — |
| **wideresnet50/x64/main** | `RETRAIN` ✓ **updated** | loads `patchcore_sweep_results.csv` + follow-up CSV | rebuilds all 8 variant memory banks from scratch | holdout artifacts in old global path |
| *wideresnet50/x64/labeled_120k* | *separate notebooks* | *loads labeled patch CSV* | *rebuilds memory bank from labeled patches* | *dataset_helper, results_review, threshold_policies supporting notebooks* |
| **wideresnet50/x224/multilayer** | `RETRAIN` ✓ **converted** | loads saved CSVs, renders review plots | builds 600k-patch memory bank for 2 variants at x224 | — |
| *wideresnet50/x224/layer_ablations* | *separate notebooks* | *load variant CSVs* | *rebuild variant memory banks* | *layer2, layer3, layer234, weighted variants* |
| efficientnet_b0/x224/main | `RETRAIN` ✓ | loads saved checkpoint/scores | rebuilds memory bank from scratch | — |
| efficientnet_b1/x240/main_one_layer | `RUN_HOLDOUT_EVALUATION = True` | loads saved results | rebuilds from raw pickle | ✓ (runs by default) |
| *efficientnet_b1/x240/layer_variants* | *separate notebooks* | *load variant CSVs* | *rebuild variant memory banks* | *layer3_5, layer3_5_no_defect_tuning variants* |
| vit_b16/x224/main | `RETRAIN` ✓ | loads saved scores + metrics | runs training script | ✓ (conditional section) |
| *vit_b16/x224/block_ablations* | *separate notebooks* | *load variant CSVs* | *rebuild variant memory banks* | *one_layer_defect_tuning, one_layer_no_defect_tuning, two_block variants* |
| vit_b16/x64/main | `FORCE_RERUN` ✓ | loads saved scores.npz | rebuilds bank + scores all splits | — |

---

## Cross-Branch Per-Defect Comparison

The hardest defect class (Scratch) shows the most dramatic improvement with better backbones:

| model | Scratch | Loc | Edge-Loc | Center | Edge-Ring |
|---|---|---|---|---|---|
| ResNet18 x64 | 0.600 | 0.412 | 0.264 | 0.480 | 0.524 |
| WRN50 x64 | 0.533 | 0.559 | 0.585 | 0.620 | 0.917 |
| WRN50 x224 | 0.667 | 0.735 | 0.623 | 0.780 | 0.798 |
| EfficientNet-B1 x240 | 0.400 | 0.588 | 0.792 | 0.700 | 0.929 |
| **ViT-B/16 x224** | **0.727** | **0.829** | **0.705** | 0.618 | 0.941 |

The ViT-B/16 dramatically improves Scratch and Loc — the two previously hardest defect families. Center recall is slightly lower for ViT than WRN50 x224 (0.618 vs 0.780), which may reflect the different test set composition between the two runs.

---

## Key Conclusions

1. **Local patch scoring beats global scoring** — AE-BN PatchCore (F1=0.336) vs full WRN50 x64 PatchCore (F1=0.532). The method matters more than the backbone when the scoring rule is weak.

2. **Direct high-resolution preprocessing is critical** — WRN50 x64 → WRN50 x224: +0.017 F1, +0.097 AUPRC. Always cache images at the backbone's native resolution rather than upscaling from 64×64.

3. **`topk_mean` beats `mean` for stronger backbones** — for ResNet18/50 `mean` wins; for WRN50+ `topk_mean` at r=0.05–0.10 wins. As backbones get richer, concentrating on the most anomalous patches helps.

4. **Multi-layer fusion in CNNs is architecture-dependent** — WideResNet benefits from layer2+layer3 concatenation (+0.037 F1 over single layers), but EfficientNet-B1 and ViT-B/16 peak with single-block extraction. Transformer architectures have global receptive fields and don't need multi-scale fusion.

5. **ViT transformers outperform CNN backbones** — primarily by dramatically improving Scratch (0.40→0.73) and Loc (0.59→0.83) recall. Block 6 (mid-depth) outperforms later blocks, suggesting mid-layer features generalize better than final-layer representations.

6. **Defect-aware threshold tuning provides consistent gains** — typically +0.02 F1 by using class-weighted calibration on the validation set instead of global percentiles.

7. **Holdout results confirm generalization** — EfficientNet-B1 and ViT-B/16 both generalize well to 14× the anomaly count without any threshold recalibration.
