# Backbone Embedding Family Report

## Overview

This family tests frozen pretrained visual backbones as anomaly detectors using the simplest possible scoring rule: **L2 distance from the training-normal feature center**.

The purpose is to isolate how much anomaly signal comes from generic ImageNet features alone, without any wafer-specific training or local patch scoring. Both branches are intentional baselines — they are expected to be weak, and their weakness motivates the more sophisticated methods that follow.

Notebooks and artifacts live under [`experiments/anomaly_detection/backbone_embedding/`](../experiments/anomaly_detection/backbone_embedding/).

---

## Dataset and Evaluation Protocol

- **Train:** 40,000 normal wafers (embeddings extracted, center computed)
- **Validation:** 5,000 normal wafers (threshold = 95th percentile of L2 distances)
- **Test:** 5,000 normal + 250 anomaly wafers
- **Score:** L2 distance from train-normal feature center (`center_l2`)
- **Primary metric:** val-threshold F1

---

## Family Summary

| experiment       | backbone        | embedding dim | precision | recall   | F1       | AUROC    | AUPRC    | best sweep F1 |
| ---------------- | --------------- | ------------- | --------- | -------- | -------- | -------- | -------- | ------------- |
| ResNet18-center  | ResNet18        | 512           | 0.201705  | 0.284000 | 0.235880 | 0.684746 | 0.194977 | 0.259740      |
| WideRes50-center | Wide ResNet50-2 | 2048          | 0.221854  | 0.268000 | 0.242754 | 0.677274 | 0.142323 | 0.269504      |

**Key finding:** Both models score below 0.25 F1 and below 0.69 AUROC. The larger backbone (WRN50-2, 4× more dimensions) does not improve over ResNet18 — in fact AUROC is slightly lower. Global center-distance scoring is insufficient regardless of backbone capacity.

---

## Experiment 8: Pretrained ResNet18 Backbone Baseline `64×64`

**Notebook:** [`experiments/anomaly_detection/backbone_embedding/resnet18/x64/baseline/notebook.ipynb`](../experiments/anomaly_detection/backbone_embedding/resnet18/x64/baseline/notebook.ipynb)
**Artifact dir:** `experiments/anomaly_detection/backbone_embedding/resnet18/x64/baseline/artifacts/resnet18_embedding_baseline/`

### Configuration

| parameter          | value                                             |
| ------------------ | ------------------------------------------------- |
| backbone           | ResNet18 (ImageNet pretrained, frozen)            |
| embedding dim      | 512                                               |
| input size         | 224×224 (internal resize)                         |
| scoring rule       | L2 distance from train-normal center              |
| threshold quantile | 0.95                                              |
| metadata           | `data/processed/x64/wm811k/metadata_50k_5pct.csv` |

### Evaluation

| metric              | value     |
| ------------------- | --------- |
| threshold           | 12.720178 |
| precision           | 0.201705  |
| recall              | 0.284000  |
| F1                  | 0.235880  |
| AUROC               | 0.684746  |
| AUPRC               | 0.194977  |
| predicted anomalies | 352       |
| best sweep F1       | 0.259740  |

Confusion matrix: `[[4719, 281], [179, 71]]`

### Score Distribution and Threshold Sweep

![Score histograms](../experiments/anomaly_detection/backbone_embedding/resnet18/x64/baseline/artifacts/resnet18_embedding_baseline/evaluation/plots/score_histograms.png)

![Threshold sweep](../experiments/anomaly_detection/backbone_embedding/resnet18/x64/baseline/artifacts/resnet18_embedding_baseline/evaluation/plots/threshold_sweep_metrics.png)

### Failure Analysis

| error type     | count | mean score |
| -------------- | ----- | ---------- |
| true positive  | 71    | 15.933     |
| false negative | 179   | 9.166      |
| false positive | 281   | 14.271     |
| true negative  | 4719  | 8.471      |

Normal and anomaly score distributions overlap heavily — the mean FP score (14.27) is nearly as high as the mean TP score (15.93), making the threshold almost arbitrary.

![Failure examples — false positives](../experiments/anomaly_detection/backbone_embedding/resnet18/x64/baseline/artifacts/resnet18_embedding_baseline/evaluation/plots/failure_examples_fp.png)

![Failure examples — false negatives](../experiments/anomaly_detection/backbone_embedding/resnet18/x64/baseline/artifacts/resnet18_embedding_baseline/evaluation/plots/failure_examples_fn.png)

### Per-Defect Recall

| defect type | count | detected | recall   |
| ----------- | ----- | -------- | -------- |
| Random      | 5     | 4        | 0.800000 |
| Near-full   | 2     | 1        | 0.500000 |
| Edge-Ring   | 84    | 47       | 0.559524 |
| Donut       | 7     | 1        | 0.142857 |
| Scratch     | 15    | 2        | 0.133333 |
| Edge-Loc    | 53    | 8        | 0.150943 |
| Loc         | 34    | 4        | 0.117647 |
| Center      | 50    | 4        | 0.080000 |

Only broad, high-contrast defects (Random, Edge-Ring) are detectable. All local or center-focused defects (Center, Loc, Scratch, Edge-Loc) have essentially random recall.

### Interpretation

- The ResNet18 embedding space groups normal wafers into a roughly compact cluster, but many anomalous wafers land inside or near that cluster
- Global center-distance scoring is too coarse — it cannot detect localized defects that leave most of the wafer intact
- The strong `Edge-Ring` recall (0.56) makes sense: ring defects dramatically alter the global feature distribution
- `Center` recall is only 0.08 despite being a broad structural defect — ImageNet features are clearly not aligned with what makes wafer center defects distinctive
- This establishes the lower bound: adding PatchCore on the same backbone (Experiment 9) substantially improves every metric by switching to local patch-level scoring

---

## Holdout Evaluation: ResNet18 on Expanded 70k Normal / 3.5k Defect

This holdout was run via a Modal script using the same checkpoint and threshold policy.

**Holdout artifact dir:** `artifacts/resnet18_embedding_baseline/holdout70k_3p5k/`

| metric              | benchmark (5k/250) | holdout (70k/3.5k) |
| ------------------- | ------------------ | ------------------ |
| precision           | 0.201705           | 0.226513           |
| recall              | 0.284000           | 0.336857           |
| F1                  | 0.235880           | 0.270879           |
| AUROC               | 0.684746           | 0.706993           |
| AUPRC               | 0.194977           | 0.236019           |
| predicted anomalies | 352                | 5,205              |

The holdout shows marginally better metrics. The precision improvement is a statistical artifact — 3,500 true anomalies give a stronger signal than 250. The model is still fundamentally weak.

### Holdout Per-Defect Recall

| defect type | count | detected | recall |
| ----------- | ----- | -------- | ------ |
| Center      | 603   | 40       | 0.066  |
| Donut       | 71    | 8        | 0.113  |
| Edge-Loc    | 739   | 106      | 0.143  |
| Loc         | 492   | 91       | 0.185  |
| Scratch     | 169   | 40       | 0.237  |
| Edge-Ring   | 1,302 | 803      | 0.617  |
| Random      | 108   | 79       | 0.731  |
| Near-full   | 16    | 12       | 0.750  |

The same pattern holds at scale: broad high-contrast defects are detectable, small local defects are essentially missed.

![Holdout threshold sweep](../experiments/anomaly_detection/backbone_embedding/resnet18/x64/baseline/artifacts/resnet18_embedding_baseline/holdout70k_3p5k/plots/plots/threshold_sweep.png)

![Holdout defect breakdown](../experiments/anomaly_detection/backbone_embedding/resnet18/x64/baseline/artifacts/resnet18_embedding_baseline/holdout70k_3p5k/plots/plots/defect_breakdown.png)

---

## Experiment 13: Pretrained Wide ResNet50-2 Backbone Baseline `64×64`

**Notebook:** [`experiments/anomaly_detection/backbone_embedding/wide_resnet50_2/x64/baseline/notebook.ipynb`](../experiments/anomaly_detection/backbone_embedding/wide_resnet50_2/x64/baseline/notebook.ipynb)
**Artifact dir:** `experiments/anomaly_detection/backbone_embedding/wide_resnet50_2/x64/baseline/artifacts/wide_resnet50_2_embedding_baseline/`

### Configuration

| parameter          | value                                             |
| ------------------ | ------------------------------------------------- |
| backbone           | Wide ResNet50-2 (ImageNet pretrained, frozen)     |
| embedding dim      | 2048                                              |
| input size         | 224×224 (internal resize)                         |
| scoring rule       | L2 distance from train-normal center              |
| threshold quantile | 0.95                                              |
| metadata           | `data/processed/x64/wm811k/metadata_50k_5pct.csv` |

### Evaluation

| metric              | value    |
| ------------------- | -------- |
| threshold           | 8.660283 |
| precision           | 0.221854 |
| recall              | 0.268000 |
| F1                  | 0.242754 |
| AUROC               | 0.677274 |
| AUPRC               | 0.142323 |
| predicted anomalies | 302      |
| best sweep F1       | 0.269504 |

Confusion matrix: `[[4765, 235], [183, 67]]`

### Score Distribution and Threshold Sweep

![Score histograms](../experiments/anomaly_detection/backbone_embedding/wide_resnet50_2/x64/baseline/artifacts/wide_resnet50_2_embedding_baseline/evaluation/plots/score_histograms.png)

![Threshold sweep](../experiments/anomaly_detection/backbone_embedding/wide_resnet50_2/x64/baseline/artifacts/wide_resnet50_2_embedding_baseline/evaluation/plots/threshold_sweep_metrics.png)

### Failure Analysis

| error type     | count | mean score |
| -------------- | ----- | ---------- |
| true positive  | 67    | 10.635     |
| false negative | 183   | 6.274      |
| false positive | 235   | 10.077     |
| true negative  | 4765  | 5.816      |

Same pattern as ResNet18: FP and TP mean scores are nearly identical (10.08 vs 10.64), making the threshold almost useless.

![Failure examples — false positives](../experiments/anomaly_detection/backbone_embedding/wide_resnet50_2/x64/baseline/artifacts/wide_resnet50_2_embedding_baseline/evaluation/plots/failure_examples_fp.png)

![Failure examples — false negatives](../experiments/anomaly_detection/backbone_embedding/wide_resnet50_2/x64/baseline/artifacts/wide_resnet50_2_embedding_baseline/evaluation/plots/failure_examples_fn.png)

### Per-Defect Recall

| defect type | count | detected | recall   |
| ----------- | ----- | -------- | -------- |
| Random      | 5     | 3        | 0.600000 |
| Donut       | 7     | 2        | 0.285714 |
| Scratch     | 15    | 3        | 0.200000 |
| Loc         | 34    | 6        | 0.176471 |
| Center      | 50    | 7        | 0.140000 |
| Edge-Loc    | 53    | 7        | 0.132075 |
| Edge-Ring   | 84    | 39       | 0.464286 |
| Near-full   | 2     | 0        | 0.000000 |

Note: WRN50-2 is actually **worse** than ResNet18 on `Edge-Ring` (0.46 vs 0.56) and misses `Near-full` entirely.

### UMAP Embedding Visualization

The UMAP below shows why global center-distance scoring fails: anomaly wafers (red) are scattered throughout the same manifold as normal wafers rather than forming a separate cluster.

![WRN50-2 embedding UMAP](../experiments/anomaly_detection/backbone_embedding/wide_resnet50_2/x64/baseline/artifacts/umaps/wideresnet50A_embedding_baseline/evaluation/plots/embedding_umap.png)

---

## Cross-Branch Comparison

### Main Benchmark

|                   | ResNet18 | Wide ResNet50-2 |
| ----------------- | -------- | --------------- |
| embedding dim     | 512      | 2048            |
| train center norm | 24.46    | 14.33           |
| F1                | 0.236    | 0.243           |
| AUROC             | 0.685    | 0.677           |
| AUPRC             | 0.195    | **0.142**       |
| best sweep F1     | 0.260    | 0.270           |

### Per-Defect Recall Comparison

| defect type | ResNet18  | Wide ResNet50-2 |
| ----------- | --------- | --------------- |
| Edge-Ring   | **0.560** | 0.464           |
| Random      | **0.800** | 0.600           |
| Near-full   | 0.500     | 0.000           |
| Scratch     | 0.133     | **0.200**       |
| Donut       | 0.143     | **0.286**       |
| Loc         | 0.118     | **0.176**       |
| Edge-Loc    | **0.151** | 0.132           |
| Center      | 0.080     | **0.140**       |

### Key Takeaways

1. **Backbone scale alone does not help.** WRN50-2 (2048-dim) is not better than ResNet18 (512-dim) on the primary F1 metric, and actually has lower AUROC and AUPRC. More capacity without a better scoring rule is wasted.

2. **Global center-distance is the bottleneck.** Both backbones produce embeddings where anomalies are deeply interleaved with normals — the UMAP confirms this directly. The feature space is not cleanly separable at a global level.

3. **Only catastrophic defects are detectable.** Random, Edge-Ring, and Near-full have recall above 0.5 for ResNet18. Every local defect (Loc, Scratch, Edge-Loc, Center) is near-random.

4. **These are lower bounds, not competitors.** The entire value of these experiments is establishing that `F1 ≈ 0.24` and `AUROC ≈ 0.68` is the floor when you use ImageNet features + global scoring. Every subsequent method should exceed this comfortably.

5. **The fix is local scoring, not a bigger backbone.** PatchCore on ResNet18 (Experiment 9) reaches `F1 = 0.401` — a 70% relative improvement — purely by switching from global center-distance to local patch nearest-neighbor scoring on the same backbone.
