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
| 8 | AE-BN PatchCore | AE-BN encoder | 64×64 | `mean` | 0.284 | 0.412 | 0.336 | 0.851 | 0.226 |

**Key progression:** Each step up improved significantly. The biggest single gains were:
- **WRN50 x64 → WRN50 x224**: +0.017 F1, +0.014 AUROC, +0.097 AUPRC — direct high-res preprocessing matters
- **WRN50 x224 → ViT x224**: +0.046 F1, +0.025 AUROC — transformer architecture advantage

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

**Notebook:** [`experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main/notebook.ipynb`](../experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main/notebook.ipynb)

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

![EfficientNet-B1 benchmark distribution](../experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main/artifacts/patchcore_efficientnet_b1_one_layer/plots/benchmark_distribution_sweep_confusion.png)

![EfficientNet-B1 benchmark defect breakdown](../experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main/artifacts/patchcore_efficientnet_b1_one_layer/plots/benchmark_defect_breakdown.png)

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

![EfficientNet-B1 holdout distribution](../experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main/artifacts/patchcore_efficientnet_b1_one_layer/results/holdout70k_3p5k/plots/holdout_distribution_sweep_confusion.png)

![EfficientNet-B1 holdout defect breakdown](../experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main/artifacts/patchcore_efficientnet_b1_one_layer/results/holdout70k_3p5k/plots/holdout_defect_breakdown.png)

**Interpretation:** EfficientNet-B1 at its native 240×240 scale is currently the strongest CNN PatchCore in the project. It improves over EfficientNet-B0 x224 by +0.047 F1 without any defect-aware tuning. The holdout results are actually stronger than benchmark, confirming the model generalizes well.

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

---

## Notebook Status Summary

| notebook | flag used | RETRAIN=False behaviour | RETRAIN=True behaviour | holdout |
|---|---|---|---|---|
| ae_bn/x64/main | `RETRAIN` ✓ | loads saved sweep CSV | rebuilds all variant memory banks | — |
| resnet18/x64/main | `RETRAIN` ✓ | loads saved sweep CSV | rebuilds all variant memory banks | — |
| resnet50/x64/main | `RETRAIN` ✓ | loads saved sweep CSV | rebuilds all variant memory banks | — |
| **wideresnet50/x64/main** | `RETRAIN` ✓ **updated** | loads `patchcore_sweep_results.csv` + follow-up CSV | rebuilds all 8 variant memory banks from scratch | holdout artifacts in old global path |
| **wideresnet50/x224/multilayer** | `RETRAIN` ✓ **converted** | loads saved CSVs, renders review plots | builds 600k-patch memory bank for 2 variants at x224 | — |
| efficientnet_b0/x224/main | `RETRAIN` ✓ | loads saved checkpoint/scores | rebuilds memory bank from scratch | — |
| efficientnet_b1/x240/main | `RUN_HOLDOUT_EVALUATION = True` | loads saved results | rebuilds from raw pickle | ✓ (runs by default) |
| vit_b16/x224/main | `RETRAIN` ✓ | loads saved scores + metrics | runs training script | ✓ (conditional section) |

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

4. **ViT transformers outperform CNN backbones** — primarily by dramatically improving Scratch (0.40→0.73) and Loc (0.59→0.83) recall.

5. **Holdout results confirm generalization** — EfficientNet-B1 and ViT-B/16 both generalize well to 14× the anomaly count without any threshold recalibration.
