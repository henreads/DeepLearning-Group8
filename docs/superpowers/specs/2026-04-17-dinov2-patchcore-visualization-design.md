# Design: DINOv2 PatchCore Feature Visualization

**Date:** 2026-04-17
**Model:** PatchCore with DINOv2 ViT-B/14 backbone (frozen)
**Target file:** `experiments/anomaly_detection/patchcore/dinov2_vit_b14/x224/notebook.ipynb`
**Goals:** Generate spatial heatmap figures for the course report and diagnose per-class failure modes

---

## Context

The DINOv2 ViT-B/14 PatchCore model is the best-performing model in the repo (ROC-AUC 0.9665, F1 0.6234). The model is fully frozen — anomaly detection is performed by comparing patch embeddings from block 9 to a memory bank of normal patch embeddings. The anomaly score per image is the mean distance of the top-10% most anomalous patches to their nearest neighbours in the memory bank.

No classification head exists. All three visualization methods operate on the anomaly score (a distance), not a softmax probability.

---

## Scope

Add a new **Visualization** section to `notebook.ipynb` between the evaluation plots cell and the UMAP cell. The section consists of one markdown cell and three code cells.

No existing cells are modified except `score_loader`: add an optional `return_patch_dists=False` keyword argument so it can return per-patch distances alongside scalar scores. Default is `False`, so existing behaviour is unchanged.

---

## Components

### 1. PatchCore Native Heatmap

**What:** The existing `score_loader` already computes `patch_dists` — a `[B, 256]` tensor of per-patch nearest-neighbour distances — before aggregating to a scalar. With `return_patch_dists=True`, these are accumulated per image.

**How:**
- Modify `score_loader` to accept `return_patch_dists=False`. When `True`, append `patch_dists.cpu()` to a list and return `(scores_array, patch_dists_array)` where `patch_dists_array` is `[N_images, 256]`.
- Helper `patch_dists_to_heatmap(dists_1d)`: reshape `[256]` → `[16, 16]`, bilinear upsample to `[224, 224]` via `F.interpolate`, min-max normalise to `[0, 1]`.
- Produces a spatial distance map: bright = most anomalous patches.

**Cost:** Zero extra inference. Data already computed during normal scoring.

---

### 2. DINOv2 Attention Map

**What:** DINOv2 self-attention weights from block `VIT_FEATURE_BLOCK` capture which patches the model attends to. Averaging across attention heads and taking the CLS-to-patch row gives a `[256]` spatial importance vector.

**How:**
- Register a temporary hook on `extractor.vit.blocks[VIT_FEATURE_BLOCK].attn` to capture `attn_weights: [B, num_heads, 257, 257]`.
- Average over heads: `[B, 257, 257]`. Take CLS row (index 0), drop CLS column: `[:, 0, 1:]` → `[B, 256]`.
- Reshape and upsample identically to the patch heatmap.
- Hook is registered only for the visualization pass, then removed with `hook.remove()`.

**Cost:** One additional forward pass over the visualization images only (not the full dataset).

---

### 3. GradCAM Variant

**What:** Gradient of the scalar anomaly score with respect to the block-9 patch token features. Channels weighted by their global-average gradient, then spatially projected — the standard GradCAM formula adapted to the ViT patch token space.

**How:**
- For each selected image, run a fresh single-image forward pass with `torch.enable_grad()` (temporarily remove `@torch.inference_mode()` context for this call only).
- Compute anomaly score using the pre-built `memory_bank`: `torch.cdist(emb, memory_bank)` → top-k mean over patches (same formula as `score_loader`).
- Call `.backward()` on the scalar score.
- Read gradients from the block-9 hook: `grad: [1, 256, 768]`.
- Channel weights: `alpha = grad.mean(dim=1)` → `[1, 768]`.
- Weighted activation: `(alpha.unsqueeze(1) * feat).sum(dim=2)` → `[1, 256]`.
- Apply ReLU, reshape `[16, 16]`, upsample to `[224, 224]`, normalise to `[0, 1]`.

**Cost:** ~1 second per image on CPU, <100 ms on GPU. Only runs for the small set of selected images.

**Requires:** `memory_bank` must be in scope (loaded from checkpoint or freshly built). The visualization section must run before the cleanup cell.

---

## Visualization Output

For each defect class (or the classes listed in `VIS_CLASSES`):
- Select up to `VIS_N_EXAMPLES` images, prioritising **misses** (predicted normal, true defect — images where score < threshold). If fewer misses exist than `VIS_N_EXAMPLES`, fill with correctly detected defects.
- Produce a figure with one row per image, four panels per row:
  1. Raw wafer map (one-hot colour: background / normal / defect cell)
  2. PatchCore distance heatmap overlaid on wafer
  3. DINOv2 attention map overlaid on wafer
  4. GradCAM map overlaid on wafer
- Title each row with: class name, true label, predicted label, anomaly score vs threshold.
- Save to `PLOTS_DIR/viz_<class_name>.png` at 150 dpi.

---

## Control Flags

Added at the top of the Visualization markdown cell (not in the config cell, to keep them local):

```python
RUN_VIZ        = True   # False → skip entire section
VIS_N_EXAMPLES = 3      # max images per defect class
VIS_CLASSES    = None   # None → all classes; list of strings to restrict
```

---

## Placement in Notebook

```
... existing cells ...
[cell-21-plots]   ← evaluation figures (unchanged)
[cell-viz-md]     ← NEW: ## Visualization  (markdown)
[cell-viz-setup]  ← NEW: control flags + modified score_loader call + helper functions
[cell-viz-attn]   ← NEW: attention hook + GradCAM function
[cell-viz-loop]   ← NEW: per-class visualization loop + save figures
[cell-22-umap-md] ← UMAP section (unchanged)
... existing cells ...
```

---

## Memory Bank Recovery

When `FORCE_REBUILD_SCORES=False` (the default), `memory_bank` is `None` after the scoring cell because scores were loaded from disk. The visualization setup cell must recover it before computing patch distances or GradCAM:

```python
if memory_bank is None and os.path.exists(MODEL_EXPORT_PATH):
    ckpt = torch.load(MODEL_EXPORT_PATH, map_location=DEVICE)
    memory_bank = ckpt['memory_bank'].to(DEVICE)
    print(f'Memory bank loaded from checkpoint: {len(memory_bank):,} vectors')
```

If the checkpoint does not exist (model was never saved), GradCAM and patch heatmaps are skipped; only the attention map renders.

The `extractor` model is always in scope at the visualization section because it runs before the cleanup cell.

---

## Error Handling

- If `memory_bank` cannot be recovered (no checkpoint, `FORCE_REBUILD_SCORES=False`), GradCAM and patch heatmaps are skipped with a printed warning. Attention maps still render.
- If `umap-learn` is not installed the UMAP cell already handles it — no impact here.
- If a defect class has zero images in `test_defect_df` (impossible given the dataset split, but guarded), skip that class silently.

---

## Testing / Validation

- Run the notebook end-to-end with `RUN_VIZ=True`, `VIS_N_EXAMPLES=2`, `VIS_CLASSES=['Scratch', 'Ring']` to confirm figures save without error.
- Visually confirm that the three heatmaps highlight different spatial patterns (they should differ — distance, attention, and gradient encode different signals).
- Confirm `score_loader` with `return_patch_dists=False` (default) produces identical scalar scores to the original (regression check: compare saved `scores.npz` values).
