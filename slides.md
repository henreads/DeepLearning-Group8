---
marp: true
theme: default
size: 16:9
paginate: true
style: |
  section {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #ffffff;
    color: #1f2937;
    padding: 30px 50px;
  }
  h1 {
    color: #1e3a8a;
    border-bottom: 3px solid #bfdbfe;
    padding-bottom: 6px;
    margin-bottom: 16px;
    font-size: 1.6em;
  }
  h2 { color: #2563eb; }
  strong { color: #1e3a8a; }
  em { font-style: normal; color: #6b7280; }
  table { font-size: 0.78em; width: 100%; border-collapse: collapse; }
  th { background: #eff6ff; color: #1e3a8a; padding: 6px 10px; }
  td { padding: 6px 10px; border-bottom: 1px solid #e5e7eb; }
  blockquote {
    background: #f0f9ff;
    border-left: 4px solid #2563eb;
    padding: 8px 14px;
    border-radius: 0 8px 8px 0;
    font-style: normal;
    color: #1f2937;
    margin: 12px 0;
  }
  section.lead { display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; }
  section.lead h1 { border-bottom: none; font-size: 2.2em; color: #1e3a8a; }
  section.lead h2 { color: #6b7280; font-weight: normal; font-size: 1.1em; }
  section.figure { display: flex; flex-direction: column; align-items: center; justify-content: flex-start; text-align: center; padding: 20px 40px; }
  section.figure h1 { width: 100%; text-align: left; }
  img[alt~="center"] { display: block; margin: 0 auto; }
---

<!-- _class: lead -->

# Wafer Map Anomaly Detection
## A Comparative Study of Deep Learning Approaches

<br>

**50.039 Deep Learning · Y2026**
Group 8

---

# The Problem

Semiconductor wafers must be inspected for defects before chips are shipped.

- **Defects cause yield loss** — each faulty chip costs money and delays production
- **Manual inspection is slow** — 811k wafer maps cannot be reviewed by hand
- **Labelled defect data is scarce** — in production, defects are rare and often unlabelled
- **9 distinct defect patterns** — scratch, edge-ring, donut, center, loc, edge-loc, random, near-full, and none

> **Goal:** Build a system that detects defective wafers automatically, using only normal wafers for training.

---

# The Dataset: WM-811K

**811,457 wafer maps** from the LSWMD dataset

| Split | Count | Share |
|---|---|---|
| Normal (labelled) | 147,431 | 18.2% |
| Defective (labelled) | 25,118 | 3.1% |
| Unlabelled | 638,908 | 78.7% |

**Key challenge:** Class imbalance — a model that predicts every wafer as normal achieves 96%+ accuracy while catching zero defects.

The 9 defect pattern types differ in shape, size, and sparsity — making detection non-trivial.

---

<!-- _class: figure -->

# Defect Type Gallery

![w:900px center](artifacts/report_plots/defect_type_gallery.png)

---

# Our Plan: Normal-Only Training

We train **exclusively on normal wafer maps**. At inference, each wafer receives an anomaly score — higher means more deviant from normal.

**Why not supervised classification?**

- Requires balanced, labelled defect data — scarce in production
- Cannot detect defect types unseen at training time
- Anomaly detection generalises to new defect patterns automatically

**Training regime:**
- **Train:** normal wafers only
- **Validation:** normal wafers only (threshold calibration)
- **Test:** normal + all labelled defect types (held out completely)

---

# How We Evaluate

Three metrics — because **accuracy alone is useless** for imbalanced data.

| Metric | What it measures | Why it matters here |
|---|---|---|
| **AUROC** | Ranking quality across all thresholds | Threshold-free; our primary metric |
| **AUPRC** | Precision-recall area under curve | Sensitive to imbalanced class performance |
| **F1** | Harmonic mean at the deployed threshold | Reflects real production performance |

A model that predicts *all wafers as normal* achieves **96%+ accuracy** and **zero defect recall**. AUROC and AUPRC expose this immediately.

---

# Thresholding Strategy

We set the anomaly threshold **without using any defect labels** — simulating deployment.

**Strategy: 95th percentile of validation anomaly scores**

1. Run the trained model on the normal-only validation set
2. Record all anomaly scores
3. Set the threshold at the 95th percentile — 5% of normals will be flagged as false positives
4. Apply this threshold unchanged to the held-out test set

> This is a realistic constraint: in production, you calibrate on what you have — normal wafers — and accept a fixed false positive rate as the operating point.

---

<!-- _style: "table { font-size: 0.64em; }" -->

# Top 10 Models — Where We Ended Up

| Rank | Model | F1 | AUROC | AUPRC |
|---|---|---|---|---|
| 1 | **Ensemble: ViT-B/16 + DINOv2 (max-fusion)** | **0.623** | **0.967** | 0.716 |
| 2 | PatchCore + ViT-B/16, MAE fine-tuned (25% mask) | 0.594 | 0.962 | **0.762** |
| 3 | PatchCore + ViT-B/16, MAE fine-tuned (75% mask) | 0.595 | 0.959 | 0.717 |
| 4 | PatchCore + ViT-B/16, frozen, 224×224 | 0.595 | 0.956 | 0.671 |
| 5 | PatchCore + EfficientNet-B1, 240×240 | 0.591 | 0.935 | 0.609 |
| 6 | PatchCore + WideResNet50-2, 224×224 | 0.549 | 0.931 | 0.659 |
| 7 | Teacher-Student + ResNet50 | 0.525 | 0.909 | 0.599 |
| 8 | Teacher-Student + WideResNet50-2 | 0.524 | 0.923 | 0.546 |
| 9 | Autoencoder (plain, 224×224) | 0.510 | 0.901 | 0.596 |
| 10 | WideResNet50-2 center-distance | 0.243 | 0.677 | 0.142 |

The journey from row 10 to row 1 is what the next section explains.

---

<!-- _class: figure -->

# Overall Model Comparison

![w:950px center](artifacts/report_plots/overall_experiment_comparison.png)

---

<!-- _style: "table { font-size: 0.80em; }" -->

# Method Journey: Six Hypotheses

We tested six questions in sequence — each answer motivated the next question.

| | Hypothesis | Key result |
|---|---|---|
| **H1** | Can reconstruction error detect defects? | AE works but hits a ceiling |
| **H2** | Do pretrained global features do better? | Global pooling discards spatial info |
| **H3** | Do local patch features matter? | **Large jump** — local scoring is the key |
| **H4** | Does backbone architecture matter? | ViT-B/16 at 224×224 wins |
| **H5** | Does MAE fine-tuning the ViT help? | AUPRC improves; F1 stays stable |
| **H6** | Can ensembles push further? | +0.011 AUROC with ViT + DINOv2 |

---

# H1: Can Reconstruction Error Detect Defects?

**Setup:** Train an Autoencoder and Beta-VAE on normal wafers. Score each test wafer by its reconstruction error — defective wafers should reconstruct poorly.

**Architecture:** Convolutional encoder-decoder, MSE loss, 64×64 input.

**Hypothesis:** Defective wafers are out-of-distribution — the AE will fail to reconstruct them, producing high reconstruction error.

---

# H1 Result: Works, But Has a Ceiling

**Best result: Autoencoder — F1 = 0.510 · AUROC = 0.901**

**What worked:** The AE does assign higher scores to defectives on average. A viable baseline.

**Key limitation:** The AE also learns to reconstruct defects — they are not out-of-distribution enough for a pixel-level model.

- Edge-ring and near-full defects score high ✓
- Scratch and loc defects are nearly invisible — sparse linear defects do not raise reconstruction error meaningfully ✗

> **Conclusion:** Reconstruction is a weak anomaly signal. We need features that represent *what normal looks like* with more precision than pixel values.

---

<!-- _class: figure -->

# H1: Autoencoder Family Score Distributions

![w:950px center](artifacts/report_plots/autoencoder_family_comparison.png)

---

# H2: Do Pretrained Global Features Do Better?

**Setup:** Freeze a pretrained backbone (ResNet18, WideResNet50-2). At inference, compute the cosine distance from each wafer's global embedding to the centroid of normal embeddings.

**Hypothesis:** ImageNet features encode richer visual semantics than pixel-level reconstruction error, and will better separate normal from defective.

---

# H2 Result: Global Pooling Discards Too Much

**WideResNet50-2 centroid distance: F1 = 0.243 · AUROC = 0.677**
Worse than the AE. Global average pooling collapses all spatial structure into a single vector — a defect occupying 5% of the wafer disappears.

**Teacher-Student distillation (ResNet50): F1 = 0.525 · AUROC = 0.909**
Better — multi-layer distillation preserves more spatial structure than raw centroid distance, improving over the AE.

> **Key insight:** The failure is not the pretrained features — it is the pooling. Spatial structure matters. We need to score at the **patch level**, not the image level.

---

<!-- _class: figure -->

# H2: Global Feature Baseline Comparison

![w:950px center](artifacts/report_plots/compact_baseline_comparison.png)

---

# H3: Do Local Patch Features Matter?

**Setup:** PatchCore — extract patch-level features from a frozen pretrained backbone, build a memory bank of normal patches, score each test wafer by its maximum nearest-neighbour distance to the memory bank.

**Key idea:** Instead of comparing entire images, compare *individual patches*. A defect occupying 5% of the wafer will be caught if even one patch is anomalous.

**Hypothesis:** Patch-level scoring will dramatically outperform global image scoring.

---

# H3 Result: Local Features Are the Key Insight

**PatchCore + WideResNet50-2: F1 = 0.549 · AUROC = 0.931**

A large jump — from AUROC 0.909 (Teacher-Student) to 0.931 in one step.

**What changed:** The granularity of comparison, not the backbone. The same WideResNet50 features that failed at the image level succeed at the patch level.

- Scratch patterns — nearly invisible to the AE — are now detectable ✓
- Edge-ring patterns, already well-detected, improve further ✓

> **Conclusion:** PatchCore is the right framework. The next question: which backbone produces the best patch features?

---

<!-- _class: figure -->

# H3: PatchCore Family Comparison

![w:950px center](artifacts/report_plots/patchcore_family_comparison.png)

---

# H4: Does the Backbone Architecture Matter?

**Setup:** Fix the PatchCore framework. Sweep backbone architectures and input resolutions with all other hyperparameters held constant.

**Backbones:** ResNet18 · ResNet50 · WideResNet50-2 · EfficientNet-B0/B1 · ViT-B/16 · DINOv2 ViT-B/14

**Resolutions:** 64 × 64 · 128 × 128 · 224 × 224 · 240 × 240

**Hypothesis:** Stronger backbone features and higher resolution will improve anomaly scores.

---

# H4 Result: ViT-B/16 at 224×224 Wins

| Backbone | Resolution | F1 | AUROC |
|---|---|---|---|
| WideResNet50-2 | 64×64 | 0.532 | 0.917 |
| WideResNet50-2 | 224×224 | 0.549 | 0.931 |
| EfficientNet-B1 | 240×240 | 0.591 | 0.935 |
| **ViT-B/16 (frozen)** | **224×224** | **0.595** | **0.956** |

**Why ViT?** Transformer self-attention captures long-range spatial relationships across the entire wafer — context that CNN local receptive fields cannot see. Combined with PatchCore's local scoring, ViT provides both global context and local precision simultaneously.

---

# H5: Does MAE Fine-tuning the ViT Improve Results?

**Setup:** Apply Masked Autoencoding (MAE) pre-training to ViT-B/16 using normal wafer maps before running PatchCore. The ViT learns to reconstruct randomly masked patches — adapting ImageNet features to the wafer domain.

**Two masking ratios tested:** 75% (aggressive masking) and 25% (light masking)

All PatchCore hyperparameters are identical to the frozen ViT-B/16 baseline.

**Hypothesis:** Domain-adapted features will improve anomaly scoring above ImageNet-pretrained features alone.

---

<!-- _style: "font-size: 0.88em;" -->

# H5 Result: AUPRC Improves Substantially; Deployed F1 Stays Stable

| Backbone | F1 | AUROC | AUPRC | Recall |
|---|---|---|---|---|
| Frozen ViT-B/16 (ImageNet) | 0.595 | 0.956 | 0.671 | ~80% |
| MAE fine-tuned (mask 75%) | 0.595 | 0.959 | 0.717 | 82.4% |
| **MAE fine-tuned (mask 25%)** | **0.594** | **0.962** | **0.762** | **84.4%** |

**Why 25% masking wins:** Wafer maps are sparse binary grids. At 75% masking, only 49 of 196 patches remain visible — too little context. At 25%, 147 patches remain, giving the model enough local structure to learn meaningful wafer representations.

**Why F1 is stable:** The threshold is fixed at 5% FPR on validation normals. AUPRC (+0.091) and recall (+4.4%) reveal real improvements that the deployed F1 cannot capture.

---

# H6: Can Ensembles Push Further?

**Setup:** Combine anomaly scores from two complementary backbones — ViT-B/16 (ImageNet supervised) and DINOv2 ViT-B/14 (self-supervised, contrastive). Fusion strategy: take the **maximum score** per wafer across both models.

**Hypothesis:** Two backbones with different pretraining objectives will catch complementary defect patterns, yielding better coverage than either alone.

---

# H6 Result: Ensemble Reaches Best Overall Performance

| Model | F1 | AUROC | AUPRC |
|---|---|---|---|
| PatchCore + ViT-B/16 (frozen) | 0.595 | 0.956 | 0.671 |
| PatchCore + DINOv2 ViT-B/14 | — | ~0.950 | — |
| **Ensemble: ViT-B/16 + DINOv2 (max-fusion)** | **0.623** | **0.967** | **0.716** |

**Gain over single best model:** +0.011 AUROC · +0.028 F1 · +0.045 AUPRC

**Observation:** The gain is consistent but modest. Backbone diversity (different pretraining objectives) matters more than simply doubling model count. The remaining gap reflects domain difficulty — sparse wafer patterns — not model capacity.

---

<!-- _style: "font-size: 0.88em;" -->

# What If: What If We Had Used Defect Samples in Training?

Our entire approach assumes **normal-only training**. But what if that assumption costs us performance?

**1. Defect-tuning PatchCore:** Tune selected ViT-B/16 backbone layers using contrastive loss with defect examples alongside normals.

**2. Supervised classifier:** Train a classification model using 1,016 labelled defect samples across all nine defect classes.

> **The question:** Does knowing what defects look like actually help?

---

<!-- _style: "font-size: 0.88em;" -->

# What If Result: Normal-Only Is the Right Choice

**Defect-tuning results (PatchCore ViT-B/16):**

| Variant | F1 | AUROC |
|---|---|---|
| Normal-only — baseline | 0.595 | 0.956 |
| One-layer defect tuning | 0.607 | 0.957 |
| Two-block defect tuning | **0.383** | 0.932 |

One-layer barely gains (+0.012 F1). Two-block **catastrophically collapses** — defect supervision disrupts the learned normal distribution.

**Supervised classifier:** Matched AUROC = 0.956, but **Scratch recall = 0.167** vs **0.727** for normal-only PatchCore. It cannot detect defect types underrepresented at training time.

> **Conclusion:** Normal-only anomaly detection is not just a practical constraint — it is the more robust choice.

---

# Final Results vs Published Baselines

| Method | F1 | AUROC |
|---|---|---|
| Jo & Lee (2021) — best published on WM-811K | < 0.50 | — |
| Our AE baseline | 0.510 | 0.901 |
| Our best single model (PatchCore + ViT-B/16) | 0.595 | 0.956 |
| **Our best ensemble (ViT-B/16 + DINOv2)** | **0.623** | **0.967** |
| Roth et al. (2022) PatchCore on MVTec AD | — | ~0.990 |

Our results substantially exceed prior WM-811K work (Jo & Lee 2021). The remaining gap vs MVTec AD reflects domain difficulty — sparse wafer patterns and strict deployment-style thresholding — not implementation quality.

---

<!-- _class: figure -->

# What PatchCore Learned: Feature Space Separation

![w:820px center](artifacts/umaps/patchcore_efficientnet_b1_one_layer_all-in-one_224/evaluation/plots/umap_by_split.png)

Normal wafers form a tight cluster. Defective wafers scatter to the periphery — detectable without ever seeing a defect during training.

---

# Key Takeaways

**What the experiments showed:**

- **Local > global:** Patch-level PatchCore outperforms global reconstruction and global embedding distance
- **Pretrained > from scratch:** Frozen ImageNet ViT features are strong; MAE domain adaptation further improves precision-recall (+0.091 AUPRC)
- **ViT > CNN:** Transformer attention captures long-range wafer context that CNN receptive fields miss
- **Ensembles add consistent gains:** Diverse pretraining objectives (supervised + self-supervised) complement each other
- **Normal-only is the robust choice:** Defect supervision either barely helps or catastrophically hurts; supervised models fail on unseen defect types

**Limitation:** Threshold calibration without defect labels is imprecise — F1 understates true model quality. AUPRC is the more informative signal.

---

<!-- _class: lead -->

# Thank You

**Group 8 · 50.039 Deep Learning · Y2026**

<br>

| Member | Contribution |
|---|---|
| Member 1 | [Area] |
| Member 2 | [Area] |
| Member 3 | [Area] |

<br>

**Questions?**
