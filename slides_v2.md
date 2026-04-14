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
    padding: 96px 50px 30px;
    font-size: 16pt;
    position: relative;
  }
  h1 {
    position: absolute;
    top: 30px;
    left: 50px;
    right: 50px;
    color: #1e3a8a;
    border-bottom: 3px solid #bfdbfe;
    padding-bottom: 6px;
    margin: 0;
    font-size: 1.6em;
  }
  h2 { color: #2563eb; }
  strong { color: #1e3a8a; }
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
  section.figure { display: flex; flex-direction: column; align-items: center; justify-content: flex-start; text-align: center; }
  section.figure h1 { width: 100%; text-align: left; }
  img[alt~="center"] { display: block; margin: 0 auto; }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 28px; font-size: 0.78em; }
  .two-col > div { min-width: 0; }
---

<!-- _class: lead -->

# Wafer Map Anomaly Detection on WM-811K

## From Reconstruction Baselines to Pretrained Local Feature Modelling

<br>

**50.039 Deep Learning · Y2026 · Group 08**

Henry Lee Jun (1004219) &nbsp;·&nbsp; Chia Tang (1007200) &nbsp;·&nbsp; Genson Low (1005931)

---

# The Problem

Semiconductor fabs generate hundreds of thousands of wafer maps. Most are normal. A few are not.

- **Defects are rare and costly:** each bad wafer drives up yield loss
- **Defect types are unpredictable:** 9 known patterns, but new ones can emerge
- **Labelled defect data is scarce:** 3.1% of the dataset is labelled defective; the rest is unlabelled
- **Classification does not scale:** you cannot enumerate every failure mode at training time

> **Goal:** Learn what a normal wafer looks like. Flag anything that deviates.

---

# The Dataset: WM-811K

**811,457 wafer maps** from the LSWMD dataset

| Split                | Count   | Share |
| -------------------- | ------- | ----- |
| Normal (labelled)    | 147,431 | 18.2% |
| Defective (labelled) | 25,118  | 3.1%  |
| Unlabelled           | 638,908 | 78.7% |

**Key challenge:** Class imbalance: a model predicting every wafer as normal achieves 96%+ accuracy while catching zero defects.

The 9 defect pattern types differ in shape, size, and sparsity, making detection non-trivial.

---

<!-- _class: figure -->

# Defect Type Gallery

![w:900px center](artifacts/report_plots/defect_type_gallery.png)

---

# Our Plan: Normal-Only Training

We train **exclusively on normal wafer maps**. At inference, each wafer receives an anomaly score (higher means more deviant from normal).

**Why not supervised classification?**

- Requires balanced, labelled defect data: scarce in production
- Cannot detect defect types unseen at training time
- Anomaly detection generalises to new defect patterns automatically

**Training plan:**

- **Train:** normal wafers only
- **Validation:** normal wafers only (threshold calibration)
- **Test:** normal + all labelled defect types (held out completely)

---

# How We Evaluate

Three metrics, because **accuracy alone is useless** for imbalanced data.

| Metric    | What it measures                        | Why it matters here                       |
| --------- | --------------------------------------- | ----------------------------------------- |
| **AUROC** | Ranking quality across all thresholds   | Threshold-free; strong ranking signal     |
| **AUPRC** | Precision-recall area under curve       | **Our primary metric** -- most informative at 5% prevalence |
| **F1**    | Harmonic mean at the deployed threshold | Reflects real production performance      |

A model predicting all wafers as normal achieves **96%+ accuracy** and **zero defect recall**. AUROC and AUPRC expose this immediately.

**Threshold:** 95th percentile of normal-only validation scores, no defect labels required.

---

# Method Journey

<!-- _style: "table { font-size: 0.72em; }" -->

Each method family was selected to address the limitations of its predecessor.

|        | Family                | Hypothesis                                 | Finding                                            |
| ------ | --------------------- | ------------------------------------------ | -------------------------------------------------- |
| **H1** | Reconstruction        | Can pixel reconstruction detect anomalies? | Pixel errors too diffuse; sparse defects invisible |
| **H2** | Global Features       | Do pretrained global features do better?   | Pooling collapses spatial structure                |
| **H3** | **PatchCore + CNN**   | **Do local patch features matter?**        | **Large jump: patch-level scoring is the key**     |
| **H4** | PatchCore + ViT       | Does backbone architecture matter?         | ViT-B/16 at 224×224 wins                           |
| **H5** | PatchCore + ViT (MAE) | Does domain adaptation improve features?   | AUPRC improves; F1 stays stable                    |
| **H6** | **Ensemble**          | **Can diverse pretraining push further?**  | **Best overall: AUROC 0.967**                      |

---

<!-- _class: figure -->

# PatchCore: Local Patch Scoring in Action

![w:900px center](experiments/anomaly_detection/patchcore/vit_b16/x224/main/artifacts/patchcore_vit_b16_5pct/main_5pct/plots/patchcore_heatmap_examples.png)

<span style="font-size: 0.7em; color: #6b7280;">Top row: original wafer binary maps (red = failed dies). Bottom row: per-patch anomaly score heatmap at the ViT-B/16 patch grid resolution. Patch-level scoring localises small defects that pixel-averaged methods miss entirely.</span>

---

<!-- _style: "table { font-size: 0.60em; }" -->

# Top 10 Models

<span style="font-size: 0.5em">**64 training runs:** Reconstruction — AE (11) · VAE (9) · SVDD (1) &nbsp;·&nbsp; Center-distance (2) &nbsp;·&nbsp; Teacher-Student (5) &nbsp;·&nbsp; FastFlow (3) &nbsp;·&nbsp; RD4AD (1) &nbsp;·&nbsp; PatchCore — CNN (14) · ViT (8) · DINOv2 (3) &nbsp;·&nbsp; Ensemble (5) &nbsp;·&nbsp; Supervised (1)</span>

| Rank | Family          | Model                               | Sweep F1  | AUROC     | AUPRC     |
| ---- | --------------- | ----------------------------------- | --------- | --------- | --------- |
| 1    | Ensemble        | **ViT-B/16 + DINOv2 (max-fusion)**  | **0.676** | **0.967** | 0.716     |
| 2    | PatchCore + ViT | ViT-B/16, MAE fine-tuned (25% mask) | 0.692     | 0.962     | **0.762** |
| 3    | PatchCore + ViT | ViT-B/16, MAE fine-tuned (75% mask) | 0.668     | 0.959     | 0.717     |
| 4    | PatchCore + ViT | ViT-B/16, frozen, 224×224           | 0.651     | 0.956     | 0.671     |
| 5    | PatchCore + CNN | EfficientNet-B1, 240×240            | 0.653     | 0.935     | 0.609     |
| 6    | PatchCore + CNN | WideResNet50-2, 224×224             | 0.634     | 0.931     | 0.659     |
| 7    | Teacher-Student | ResNet50                            | 0.606     | 0.909     | 0.599     |
| 8    | Teacher-Student | WideResNet50-2                      | 0.561     | 0.923     | 0.546     |
| 9    | PatchCore + ViT | DINOv2 ViT-B/14, block-9            | 0.570     | 0.915     | 0.561     |
| 10   | Reconstruction  | Autoencoder, 224×224                | 0.587     | 0.901     | 0.596     |

---

# ViT-B/16 vs DINOv2: Two Different Views of the Same Image

| | ViT-B/16 | DINOv2 ViT-B/14 |
|---|---|---|
| **Pretraining** | Supervised ImageNet classification | Self-supervised self-distillation (no labels) |
| **Objective** | Predict the correct object class | Make semantically similar images cluster together |
| **What it learns** | Discriminative features that separate categories | Semantic features that capture visual similarity |
| **Sensitivity** | Structured, boundary-anchored defects | Texture and sparse local pattern deviations |
| **Patch size** | 16×16 px (196 tokens per image) | 14×14 px (256 tokens, finer granularity) |
| **Scratch recall** | 0.67 (weaker on thin linear defects) | **0.93** (stronger on sparse, low-coverage patterns) |
| **Loc recall** | **0.79** | 0.71 |

**Because the two models fail on different defect types, taking the maximum anomaly score per wafer recovers detections that either model alone would miss.**

---

<!-- _class: figure -->

# Complementary Signals in Score Space

![w:860px center](images/ensemble_umap_score_space.png)

<span style="font-size: 0.7em; color: #6b7280;">UMAP of [ViT score, DINOv2 score] per test wafer. Left: normal vs defect separation. Right: per-class clusters. Scratch concentrates where DINOv2 is high; Edge-Ring and Edge-Loc concentrate where ViT is high. Max-fusion captures both regions.</span>

---

<!-- _class: figure -->

# Mahalanobis Fusion: Tighter Decision Boundary in Score Space

![w:860px center](images/mahal_boundary.png)

<span style="font-size: 0.7em; color: #6b7280;">Left: Mahalanobis ellipse (orange) vs max-fusion L-shape (red dashed) in normalised score space. Mahalanobis requires joint signal from both models. Right: score distributions show clear separation. Mahalanobis achieves AUROC 0.968, AUPRC 0.762, Sweep F1 0.722 — all above max-fusion.</span>

---

# Best Model: ViT-B/16 + DINOv2 Ensemble

<div class="two-col" style="margin-top: 8px;">
<div>
<strong>Per-defect recall</strong>
<table>
<tr><th>Defect</th><th>ViT-B/16</th><th>DINOv2</th><th>Ensemble</th></tr>
<tr><td>Scratch</td><td>0.67</td><td><strong>0.93</strong></td><td><strong>0.80</strong></td></tr>
<tr><td>Loc</td><td><strong>0.79</strong></td><td>0.71</td><td><strong>0.91</strong></td></tr>
<tr><td>Edge-Loc</td><td><strong>0.79</strong></td><td>0.55</td><td><strong>0.85</strong></td></tr>
<tr><td>Center</td><td><strong>0.82</strong></td><td>0.56</td><td><strong>0.88</strong></td></tr>
<tr><td>Edge-Ring</td><td><strong>0.89</strong></td><td>0.74</td><td><strong>0.89</strong></td></tr>
<tr><td>Donut</td><td>1.00</td><td>1.00</td><td><strong>1.00</strong></td></tr>
</table>
<p style="margin-top: 8px; color: #6b7280;">Where one model is weak, the other is strong.</p>
</div>
<div>
<strong>Overall metrics</strong>
<table>
<tr><th>Model</th><th>AUROC</th><th>AUPRC</th><th>F1</th></tr>
<tr><td>ViT-B/16</td><td>0.956</td><td>0.671</td><td>0.595</td></tr>
<tr><td>DINOv2</td><td>0.915</td><td>0.561</td><td>0.492</td></tr>
<tr><td><strong>Ensemble (max-fusion)</strong></td><td>0.967</td><td>0.716</td><td><strong>0.623</strong></td></tr>
<tr><td><strong>Ensemble (Mahalanobis)</strong></td><td><strong>0.968</strong></td><td><strong>0.762</strong></td><td>0.592</td></tr>
</table>
<blockquote style="margin-top: 12px;">Mahalanobis requires joint signal from both models. Better ranking (AUROC, AUPRC) but stricter threshold lowers deployed F1.</blockquote>
</div>
</div>

---

# What If We Had Used Defect Labels?

Our entire approach assumes **normal-only training**. But what if that assumption costs us performance?

**Two experiments to test this:**

- **Defect-tuning PatchCore:** Fine-tune ViT-B/16 backbone layers with contrastive loss on defect examples
- **Supervised linear probe:** Train a classifier on up to 1,016 labelled defect samples across all defect classes — with **Scratch and Loc withheld** to test generalisation

> **The question:** Does knowing what defects look like actually help?

---

# Result: Normal-Only Is the More Robust Choice

<!-- _style: "table { font-size: 0.74em; }" -->

**Supervised linear probe sweep** (Scratch + Loc withheld from training):

| Defect % | Train defects | AUROC | F1 | Sweep F1 | Scratch recall | Loc recall |
|---|---|---|---|---|---|---|
| 0.1% | 20 | 0.895 | 0.499 | 0.640 | 0.083 | 0.400 |
| 1.0% | 203 | 0.941 | 0.554 | 0.721 | 0.333 | 0.657 |
| 5.0% | 1,016 | 0.956 | 0.581 | 0.760 | 0.167 | 0.686 |
| **PatchCore (no labels)** | **0** | **0.956** | **0.595** | **0.651** | **0.727** | **0.829** |

With 1,016 labels, supervised AUROC matches PatchCore. But Scratch recall is **0.167** vs **0.727**: the classifier never saw Scratch during training. More labels only help for defect types you already have.

> **Conclusion:** Anomaly detection has no blind spots. It flags anything that looks unusual, whether or not that defect type existed at training time.
