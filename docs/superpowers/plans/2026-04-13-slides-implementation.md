# Slide Deck Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write all 26 slides of the Marp presentation into `slides.md`, section by section, using the approved design spec.

**Architecture:** Single `slides.md` file in the project root. All images referenced as relative paths from the root (e.g. `artifacts/report_plots/defect_type_gallery.png`). Each task writes one section of slides and commits. Verification is visual — open Marp preview in VS Code (click the Marp icon top-right when `slides.md` is open).

**Tech Stack:** Marp for VS Code extension · Marp CLI (`npm install -g @marp-team/marp-cli`) · Git

---

### Task 1: Frontmatter, Theme, and Title Slides (Slides 1–2)

**Files:**
- Modify: `slides.md`

- [ ] **Step 1: Replace current placeholder content with frontmatter and slides 1–2**

Replace the entire contents of `slides.md` with:

```markdown
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
    padding: 40px 60px;
  }
  h1 {
    color: #1e3a8a;
    border-bottom: 3px solid #bfdbfe;
    padding-bottom: 8px;
    margin-bottom: 20px;
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
    padding: 10px 16px;
    border-radius: 0 8px 8px 0;
    font-style: normal;
    color: #1f2937;
    margin: 16px 0;
  }
  section.lead { text-align: center; justify-content: center; }
  section.lead h1 { border-bottom: none; font-size: 2.2em; color: #1e3a8a; }
  section.lead h2 { color: #6b7280; font-weight: normal; font-size: 1.1em; }
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
```

- [ ] **Step 2: Verify in Marp preview**

Open `slides.md` in VS Code. Click the Marp icon (top-right of the editor) to open the preview panel. Confirm:
- Slide 1 is centred, blue title, grey subtitle, bold course line
- Slide 2 has blue `h1`, bullet list, and blue-bordered blockquote
- Pagination shows `1` and `2`

- [ ] **Step 3: Commit**

```bash
git add slides.md
git commit -m "feat: add marp frontmatter, theme, and slides 1-2 (title + problem)"
```

---

### Task 2: Dataset and Training Plan (Slides 3–4)

**Files:**
- Modify: `slides.md`

- [ ] **Step 1: Append slides 3–4 to `slides.md`**

Append to the end of `slides.md`:

```markdown
---

# The Dataset: WM-811K

![bg right:42%](artifacts/report_plots/defect_type_gallery.png)

**811,457 wafer maps** from the LSWMD dataset

| Split | Count | Share |
|---|---|---|
| Normal (labelled) | 147,431 | 18.2% |
| Defective (labelled) | 25,118 | 3.1% |
| Unlabelled | 638,908 | 78.7% |

**Key challenge:** Class imbalance — a model that predicts every wafer as normal achieves 96%+ accuracy while catching zero defects.

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
```

- [ ] **Step 2: Verify in Marp preview**

Confirm:
- Slide 3 shows the defect gallery image on the right, table on the left, no overflow
- Slide 4 has two clear sections with bold headers

- [ ] **Step 3: Commit**

```bash
git add slides.md
git commit -m "feat: add slides 3-4 (dataset and normal-only training plan)"
```

---

### Task 3: Evaluation Setup (Slides 5–6)

**Files:**
- Modify: `slides.md`

- [ ] **Step 1: Append slides 5–6 to `slides.md`**

```markdown
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
```

- [ ] **Step 2: Verify in Marp preview**

Confirm:
- Slide 5 table has three rows, blue header row, no overflow
- Slide 6 numbered list is readable and blockquote renders correctly

- [ ] **Step 3: Commit**

```bash
git add slides.md
git commit -m "feat: add slides 5-6 (metrics and thresholding strategy)"
```

---

### Task 4: Results Scoreboard (Slide 7)

**Files:**
- Modify: `slides.md`

- [ ] **Step 1: Append slide 7 to `slides.md`**

```markdown
---

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
```

- [ ] **Step 2: Verify in Marp preview**

Confirm the table fits within the slide bounds. If it overflows vertically, reduce the `font-size` in the global style to `0.72em` or add `<!-- _style: "table { font-size: 0.68em; }" -->` above this slide.

- [ ] **Step 3: Commit**

```bash
git add slides.md
git commit -m "feat: add slide 7 (top 10 results scoreboard)"
```

---

### Task 5: Method Roadmap and H1 — Reconstruction (Slides 8–10)

**Files:**
- Modify: `slides.md`

- [ ] **Step 1: Append slides 8–10 to `slides.md`**

```markdown
---

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

![bg right:40%](artifacts/report_plots/autoencoder_family_comparison.png)

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
```

- [ ] **Step 2: Verify in Marp preview**

Confirm:
- Slide 8 table has 6 rows, readable at default font size
- Slide 9 image appears on the right with text on the left
- Slide 10 bullet list with checkmarks renders correctly

- [ ] **Step 3: Commit**

```bash
git add slides.md
git commit -m "feat: add slides 8-10 (method roadmap and H1 reconstruction)"
```

---

### Task 6: H2 — Global Pretrained Features (Slides 11–12)

**Files:**
- Modify: `slides.md`

- [ ] **Step 1: Append slides 11–12 to `slides.md`**

```markdown
---

# H2: Do Pretrained Global Features Do Better?

![bg right:40%](artifacts/report_plots/compact_baseline_comparison.png)

**Setup:** Freeze a pretrained backbone (ResNet18, WideResNet50-2). At inference, compute the cosine distance from each wafer's global embedding to the centroid of normal embeddings.

**Hypothesis:** ImageNet features encode richer visual semantics than pixel-level reconstruction, and will better separate normal from defective.

---

# H2 Result: Global Pooling Discards Too Much

**WideResNet50-2 centroid distance: F1 = 0.243 · AUROC = 0.677**
Worse than the AE. Global average pooling collapses all spatial structure into a single vector — a defect occupying 5% of the wafer disappears.

**Teacher-Student distillation (ResNet50): F1 = 0.525 · AUROC = 0.909**
Better — multi-layer distillation preserves more spatial structure than raw centroid distance, improving over the AE.

> **Key insight:** The failure is not the pretrained features — it is the pooling. Spatial structure matters. We need to score at the **patch level**, not the image level.
```

- [ ] **Step 2: Verify in Marp preview**

Confirm slide 12 blockquote and bold text render without overflow.

- [ ] **Step 3: Commit**

```bash
git add slides.md
git commit -m "feat: add slides 11-12 (H2 global pretrained features)"
```

---

### Task 7: H3 — PatchCore Local Scoring (Slides 13–14)

**Files:**
- Modify: `slides.md`

- [ ] **Step 1: Append slides 13–14 to `slides.md`**

```markdown
---

# H3: Do Local Patch Features Matter?

![bg right:40%](artifacts/report_plots/patchcore_family_comparison.png)

**Setup:** PatchCore — extract patch-level features from a frozen pretrained backbone, build a memory bank of normal patches, score each test wafer by its maximum nearest-neighbour distance to the memory bank.

**Key idea:** Instead of comparing entire images, compare *individual patches*. A defect occupying 5% of the wafer will be caught if even one patch is anomalous.

**Hypothesis:** Patch-level scoring will dramatically outperform global image scoring.

---

# H3 Result: Local Features Are the Key Insight

**PatchCore + WideResNet50-2: F1 = 0.549 · AUROC = 0.931**

A large jump — from AUROC 0.909 (Teacher-Student) to 0.931 in one step.

**What changed:** The granularity of comparison, not the backbone. The same WideResNet50 features that failed at the image level succeed at the patch level.

- Scratch patterns — nearly invisible to the AE — are now detectable: their anomalous patches stand out against the normal memory bank ✓
- Edge-ring patterns, already well-detected, improve further ✓

> **Conclusion:** PatchCore is the right framework. The next question: which backbone produces the best patch features?
```

- [ ] **Step 2: Verify in Marp preview**

Confirm the image renders right of the text on slide 13.

- [ ] **Step 3: Commit**

```bash
git add slides.md
git commit -m "feat: add slides 13-14 (H3 PatchCore local scoring)"
```

---

### Task 8: H4 — Backbone and Resolution Sweep (Slides 15–16)

**Files:**
- Modify: `slides.md`

- [ ] **Step 1: Append slides 15–16 to `slides.md`**

```markdown
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
```

- [ ] **Step 2: Verify in Marp preview**

Confirm the table on slide 16 renders cleanly with the bold last row visible.

- [ ] **Step 3: Commit**

```bash
git add slides.md
git commit -m "feat: add slides 15-16 (H4 backbone and resolution sweep)"
```

---

### Task 9: H5 — MAE Fine-tuning (Slides 17–18)

**Files:**
- Modify: `slides.md`

- [ ] **Step 1: Append slides 17–18 to `slides.md`**

```markdown
---

# H5: Does MAE Fine-tuning the ViT Improve Results?

**Setup:** Apply Masked Autoencoding (MAE) pre-training to ViT-B/16 using normal wafer maps before running PatchCore. The ViT learns to reconstruct randomly masked patches — adapting ImageNet features to the wafer domain.

**Two masking ratios tested:** 75% (aggressive masking) and 25% (light masking)

All PatchCore hyperparameters are identical to the frozen ViT-B/16 baseline.

**Hypothesis:** Domain-adapted features will improve anomaly scoring above ImageNet-pretrained features alone.

---

# H5 Result: AUPRC Improves Substantially; Deployed F1 Stays Stable

| Backbone | F1 | AUROC | AUPRC | Recall |
|---|---|---|---|---|
| Frozen ViT-B/16 (ImageNet) | 0.595 | 0.956 | 0.671 | ~80% |
| MAE fine-tuned (mask 75%) | 0.595 | 0.959 | 0.717 | 82.4% |
| **MAE fine-tuned (mask 25%)** | **0.594** | **0.962** | **0.762** | **84.4%** |

**Why 25% masking wins:** Wafer maps are sparse binary grids. At 75% masking, only 49 of 196 patches remain visible — too little context. At 25%, 147 patches remain, giving the model enough local structure to learn meaningful wafer representations.

**Why F1 is stable:** The threshold is fixed at 5% FPR on validation normals. AUPRC (+0.091) and recall (+4.4%) reveal real improvements that the deployed F1 cannot capture.
```

- [ ] **Step 2: Verify in Marp preview**

Confirm the table on slide 18 shows 3 rows with the last row bolded, no overflow.

- [ ] **Step 3: Commit**

```bash
git add slides.md
git commit -m "feat: add slides 17-18 (H5 MAE fine-tuning results)"
```

---

### Task 10: H6 — Ensembles (Slides 19–20)

**Files:**
- Modify: `slides.md`

- [ ] **Step 1: Append slides 19–20 to `slides.md`**

```markdown
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

**Observation:** The gain is consistent but modest. Backbone diversity (different pretraining objectives) matters more than simply doubling model count. The remaining gap to near-perfect performance reflects domain difficulty — sparse wafer patterns — not model capacity.
```

- [ ] **Step 2: Verify in Marp preview**

Confirm table is readable and bold row stands out.

- [ ] **Step 3: Commit**

```bash
git add slides.md
git commit -m "feat: add slides 19-20 (H6 ensemble results)"
```

---

### Task 11: What If? Counterfactual (Slides 21–22)

**Files:**
- Modify: `slides.md`

- [ ] **Step 1: Append slides 21–22 to `slides.md`**

```markdown
---

# What If: What If We Had Used Defect Samples in Training?

Our entire approach assumes **normal-only training**. But what if that assumption costs us performance?

**Two experiments tested this assumption:**

**1. Defect-tuning PatchCore:** Inject defect supervision into the ViT-B/16 backbone — tune selected layers using contrastive loss with defect examples alongside normals.

**2. Supervised classifier:** Train a classification model using 1,016 labelled defect samples across all nine defect classes.

> **The question:** Does knowing what defects look like actually help?

---

# What If Result: Normal-Only Is the Right Choice

**Defect-tuning results (PatchCore ViT-B/16):**

| Variant | F1 | AUROC |
|---|---|---|
| Normal-only — baseline | 0.595 | 0.956 |
| One-layer defect tuning | 0.607 | 0.957 |
| Two-block defect tuning | **0.383** | 0.932 |

One-layer barely gains (+0.012 F1). Two-block **catastrophically collapses** — defect supervision disrupts the learned normal distribution.

**Supervised classifier:** Matched AUROC = 0.956, but **Scratch recall = 0.167** vs **0.727** for normal-only PatchCore. It cannot detect defect types underrepresented at training time.

> **Conclusion:** Normal-only anomaly detection is not just a practical constraint — it is the more robust choice. Generalisation to rare or unseen defect types is only possible without defect supervision.
```

- [ ] **Step 2: Verify in Marp preview**

Confirm the table has 3 rows and the two-block result renders in bold. Confirm blockquote at bottom doesn't overflow.

- [ ] **Step 3: Commit**

```bash
git add slides.md
git commit -m "feat: add slides 21-22 (what-if counterfactual: normal-only vs defect training)"
```

---

### Task 12: Conclusion — Final Results, UMAP, Takeaways, Q&A (Slides 23–26)

**Files:**
- Modify: `slides.md`

- [ ] **Step 1: Append slides 23–26 to `slides.md`**

```markdown
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

# What PatchCore Learned: Feature Space Separation

![bg right:50%](artifacts/umaps/patchcore_efficientnet_b1_one_layer_all-in-one_224/evaluation/plots/umap_by_split.png)

UMAP of patch-level embeddings from PatchCore.

**Normal wafers** form a tight, compact cluster.

**Defective wafers** scatter to the periphery — their anomalous patches are measurably distant from the normal memory bank.

This separation is what enables anomaly detection without ever seeing a defective wafer during training.

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
```

- [ ] **Step 2: Verify in Marp preview — full deck check**

Scroll through all 26 slides and confirm:
- No slide overflows its bounds (text clipped, tables too wide, images too large)
- All images load (no broken image icons) — check slides 3, 9, 11, 13, 24
- Pagination shows correctly on every slide (1–26)
- Title slide (1) and Q&A slide (26) are centred with the `lead` class
- Bold text in tables renders as `#1e3a8a` blue (check slide 7 and 16)

- [ ] **Step 3: Fill in group member names and contributions**

Replace the placeholder rows in slide 26 with actual group member names and their contributions. Use the group contributions documented in `Report_Latex_v2.tex` Appendix A.

- [ ] **Step 4: Commit**

```bash
git add slides.md
git commit -m "feat: add slides 23-26 (conclusion, UMAP, takeaways, Q&A) — complete first draft"
```

---

### Task 13: Export to PDF and PPTX

**Files:**
- Create: `slides.pdf`
- Create: `slides.pptx` (optional)

- [ ] **Step 1: Export to PDF via Marp CLI**

```bash
marp slides.md --pdf --output slides.pdf
```

Expected: `slides.pdf` created in project root, 26 pages.

- [ ] **Step 2: Open PDF and verify all slides**

Open `slides.pdf`. Check:
- Images are visible on slides 3, 9, 11, 13, 24
- No text is cut off
- Tables are legible
- Title and Q&A slides are centred

- [ ] **Step 3: Export to PPTX (optional, for editable version)**

```bash
marp slides.md --pptx --output slides.pptx
```

Note: standard PPTX renders slides as images. For editable text, LibreOffice with `--pptx-editable` is required (see article). This is optional — the PDF is the primary submission artifact.

- [ ] **Step 4: Commit exported files**

```bash
git add slides.pdf
git commit -m "feat: export slides to PDF (first draft)"
```

---

## Self-Review Against Spec

**Spec coverage check:**

| Spec requirement | Task that covers it |
|---|---|
| Slide 1: Title | Task 1 |
| Slide 2: Problem | Task 1 |
| Slide 3: Dataset (WM-811K, class distribution, gallery) | Task 2 |
| Slide 4: Normal-only training plan | Task 2 |
| Slide 5: Metrics (AUROC, AUPRC, F1) | Task 3 |
| Slide 6: 95th percentile thresholding | Task 3 |
| Slide 7: Top 10 scoreboard table | Task 4 |
| Slide 8: Method roadmap (6 hypotheses) | Task 5 |
| Slides 9–10: H1 Reconstruction | Task 5 |
| Slides 11–12: H2 Global features | Task 6 |
| Slides 13–14: H3 PatchCore local scoring | Task 7 |
| Slides 15–16: H4 Backbone sweep | Task 8 |
| Slides 17–18: H5 MAE fine-tuning | Task 9 |
| Slides 19–20: H6 Ensembles | Task 10 |
| Slides 21–22: What If? counterfactual | Task 11 |
| Slide 23: Final results vs baselines | Task 12 |
| Slide 24: UMAP diagnostic | Task 12 |
| Slide 25: Key takeaways + limitations | Task 12 |
| Slide 26: Q&A + group members | Task 12 |
| Export to PDF | Task 13 |

All 26 slides and all spec requirements are covered.
