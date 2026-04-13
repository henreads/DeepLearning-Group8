# Slide Deck Design — Wafer Anomaly Detection
**Date:** 2026-04-13
**Course:** 50.039 Deep Learning, Y2026 (SUTD)
**Group:** Group 8
**File:** slides.md (Marp)

---

## Presentation Context

- **Audience:** Course final presentation — professors (Matthieu De Mari, Dileepa Fernando) and classmates. Academic setting; audience knows deep learning but has not seen the project internals.
- **Goal:** Show the journey through method families, each motivated by the last, landing on a rigorous and well-supported conclusion. Both narrative (the story of what we tried) and scientific (the numbers are trustworthy).
- **Length:** 15–20 minutes, 26 slides.
- **No demo.** Slides only — results speak through charts and figures.

---

## Visual Style

- **Marp theme:** `default`
- **Background:** White
- **Accent colors:** Blue (#2563eb) for headers and highlights, green (#15803d) for positive results, red (#dc2626) for limitations
- **Size:** 16:9
- **Paginate:** true
- **Font:** Default (sans-serif body, clean hierarchy)

Frontmatter:
```yaml
---
marp: true
theme: default
size: 16:9
paginate: true
style: |
  section { font-family: sans-serif; }
  h1 { color: #1e3a8a; }
  h2 { color: #2563eb; border-bottom: 2px solid #bfdbfe; padding-bottom: 4px; }
  strong { color: #1e3a8a; }
---
```

---

## Narrative Approach

**Hypothesis-driven.** Each method family is introduced as a testable question. The answer motivates the next question. This makes the progression feel deliberate rather than accidental, and demonstrates rigorous scientific thinking to the professors.

The top-10 scoreboard appears early (slide 7) as a destination map so the audience knows where things land before the detail begins.

A "What If?" counterfactual section after the method journey challenges the core assumption (normal-only training) and shows the comparison result.

---

## Slide-by-Slide Breakdown

### Section 1 — Problem Setup (Slides 1–4)

**Slide 1 — Title**
- Title: "Wafer Map Anomaly Detection"
- Subtitle: "A Comparative Study of Deep Learning Approaches"
- Group 8 · 50.039 Deep Learning · Y2026

**Slide 2 — The Problem**
- Semiconductor wafer manufacturing context
- Defective wafers cause yield loss — catching defects early matters
- The challenge: defects are rare; labelled defect data is scarce in production
- Key image: example of a defective wafer map vs a normal one

**Slide 3 — The Dataset**
- WM-811K (LSWMD) dataset
- 811k wafer maps total; 18.2% normal, 3.1% labelled defective, 78.7% unlabelled
- 9 defect pattern types (scratch, edge-ring, donut, etc.)
- Show class distribution chart and sample wafer map images per defect type
- Highlight the imbalance — this is the core challenge

**Slide 4 — Our Plan: Normal-Only Training**
- We train only on normal wafers and score anomalies at test time
- Why: labelled defect data is rare in practice; anomaly detection is the realistic setting
- Contrast with supervised classification (which would require balanced defect labels)
- This is the assumption the "What If?" section later challenges

---

### Section 2 — Evaluation Setup (Slides 5–6)

**Slide 5 — Metrics**
- AUROC: rank-ordering quality, threshold-free — primary metric
- AUPRC: precision-recall area — better for imbalanced classes
- F1: harmonic mean at chosen threshold — practical deployment measure
- Brief visual showing why accuracy is misleading here (a model predicting all-normal gets 96%+ accuracy)

**Slide 6 — Thresholding Strategy**
- We set the threshold at the 95th percentile of validation anomaly scores on normal-only validation data
- This simulates a deployment scenario where no defect labels are available at threshold-selection time
- Show a score distribution diagram: normal score distribution with the 95th percentile cutoff marked

---

### Section 3 — Results Scoreboard (Slide 7)

**Slide 7 — Top 10 Models**
- Table with top 10 models ranked by AUROC
- Columns: Model Family | Backbone | Resolution | AUROC | AUPRC | F1
- Highlight the winner row (PatchCore ViT-B/16 ensemble)
- Note: this is the destination — the method journey explains how we got here

---

### Section 4 — Method Journey: Hypothesis-Driven (Slides 8–20)

**Slide 8 — Roadmap**
- Overview of the 6 hypotheses tested, shown as a numbered list or flow diagram
- H1: Reconstruction · H2: Global features · H3: Local scoring · H4: Backbone architecture · H5: Fine-tuning · H6: Ensembles
- Sets audience expectations for the journey

**Slides 9–10 — H1: Can reconstruction error detect defects?**
- Slide 9: AE and Beta-VAE architecture overview. Train on normals; score by reconstruction error. Simple, interpretable baseline.
- Slide 10: Results — AUROC in low-to-mid range. Key limitation: AE learns to reconstruct defects too (they are not out-of-distribution enough). This motivates moving to pretrained features.

**Slides 11–12 — H2: Do pretrained global features do better?**
- Slide 11: Backbone embedding distance approach. Freeze a pretrained CNN (ResNet, EfficientNet) or ViT; embed images; score by distance to normal cluster centroid.
- Slide 12: Results — meaningful improvement over AE. But global pooling loses spatial structure. This motivates local patch-level scoring.

**Slides 13–14 — H3: Do local patch features matter?**
- Slide 13: PatchCore — extract patch-level features from frozen backbone; build a memory bank of normal patches; score by nearest-neighbour distance in patch space.
- Slide 14: Results — large jump in AUROC. Local spatial scoring is the key insight. Defects are localised patterns; global pooling discards exactly this information.

**Slides 15–16 — H4: Does backbone architecture matter?**
- Slide 15: Sweep across backbones (ResNet18, ResNet50, WideResNet50, EfficientNet-B0/B1, ViT-B/16, DINOv2) and resolutions (64, 128, 224, 240px) with PatchCore.
- Slide 16: Results — ViT-B/16 at 224×224 wins. Transformer attention captures global context while PatchCore preserves local structure. Show bar chart of AUROC per backbone.

**Slides 17–18 — H5: Does fine-tuning the ViT backbone improve results?**
- Slide 17: Setup — take the best frozen ViT-B/16 PatchCore model and fine-tune the backbone on normal wafers. Two variants: full fine-tune vs. partial (last N layers).
- Slide 18: Results and trade-offs — compare fine-tuned vs frozen on AUROC/F1. Discuss whether the added compute cost is justified and any overfitting observed.
- **Note:** Confirm fine-tuning experiments have been run and results are available before writing this slide's content.

**Slides 19–20 — H6: Can ensembles push further?**
- Slide 19: Ensemble approach — combine ViT-B/16 and DINOv2 anomaly scores using max-fusion.
- Slide 20: Results — ViT+DINOv2 max-fusion reaches AUROC 0.967, F1 0.623. Small but consistent gain over single model. Diminishing returns suggest architectural diversity matters more than model count.

---

### Section 5 — What If? Counterfactual (Slides 21–22)

**Slide 21 — The Question**
- "We assumed normal-only training. What if we were wrong?"
- Setup: take best model (PatchCore ViT-B/16); retrain with defect samples included
- This tests whether the normal-only assumption actually costs us performance

**Slide 22 — The Comparison**
- Side-by-side: normal-only vs defect-included training
- Metrics table: AUROC, AUPRC, F1 for both regimes
- Interpretation: discuss what the result means for the anomaly detection assumption and its practical implications

---

### Section 6 — Conclusion (Slides 23–26)

**Slide 23 — Final Results vs Published Baselines**
- Best result: PatchCore ViT-B/16+DINOv2 ensemble — AUROC 0.967, F1 0.623
- Comparison table: our results vs Jo and Lee (2021) and Roth et al. (2022)
- We are not expected to beat state-of-the-art, but context shows where we sit

**Slide 24 — UMAP Diagnostic**
- UMAP of patch-level embeddings coloured by normal vs defect
- Shows the feature space separation that PatchCore exploits
- Visual evidence for why local features work

**Slide 25 — Key Takeaways**
- Local patch scoring beats global reconstruction and global embeddings
- Pretrained frozen features beat training from scratch
- ViT architecture outperforms CNNs for this task
- Ensembles provide a small consistent gain
- Normal-only training: the counterfactual result goes here once the comparison experiment is run
- Limitation: thresholding without defect labels is inherently imprecise

**Slide 26 — Thank You / Q&A**
- Group member names and contributions
- GitHub repo link (or QR code)
- Open for questions

---

## Slide Count and Pacing

| Section | Slides | Estimated Time |
|---|---|---|
| Problem Setup | 1–4 | ~4 min |
| Evaluation Setup | 5–6 | ~2 min |
| Results Scoreboard | 7 | ~1 min |
| Method Journey | 8–20 | ~11 min |
| What If? | 21–22 | ~2 min |
| Conclusion | 23–26 | ~3 min |
| **Total** | **26** | **~23 min** |

If timing is tight during rehearsal, the easiest cuts are slide 6 (Thresholding, fold into slide 5) or slide 24 (UMAP, fold into slide 25). This brings the deck to 24 slides and shaves ~1–2 minutes.

---

## Key Content Assets Needed

- Wafer map images per defect type (from dataset)
- Class distribution bar chart (normal/defective/unlabelled)
- Score distribution histogram with 95th percentile threshold marked
- Top-10 results table (AUROC, AUPRC, F1)
- Per-backbone AUROC bar chart (for H4 slide)
- Fine-tuning comparison table (frozen vs fine-tuned)
- Normal-only vs defect-included comparison table (What If?)
- UMAP plot of patch embeddings (already in artifacts/)
- Published baseline comparison table
