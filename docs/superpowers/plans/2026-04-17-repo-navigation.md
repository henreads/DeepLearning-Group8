# Repo Navigation and Notebook Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit all ~50 experiment notebooks for RETRAIN=False runability, then rewrite README.md as a clean navigation document with Top 10 Models and Per Family sections.

**Architecture:** Sweep family by family — for each family, check artifact completeness via static notebook inspection and filesystem checks, record issues in NOTEBOOK_STATUS.md, then write the README in one final pass using confirmed knowledge.

**Tech Stack:** Bash (grep, ls), Read tool for notebooks, Write/Edit for README and status file.

---

## Files

- Modify: `README.md` — full rewrite
- Create: `NOTEBOOK_STATUS.md` — audit tracking file (not linked from README)

---

## Audit Methodology (apply to every notebook)

For each notebook:
1. **RETRAIN flag check** — grep the `.ipynb` file for `RETRAIN` (or `FORCE_RERUN`, `RUN_HOLDOUT`). If found → mark `(Trainable)`.
2. **Artifact path check** — grep for file loads in the RETRAIN=False branch: look for `.csv`, `.npy`, `.pth`, `.pkl`, `.json` path strings. Confirm those paths exist on disk.
3. **Record any missing artifacts** in NOTEBOOK_STATUS.md.

Grep command template:
```bash
grep -o '"RETRAIN[^"]*"\|RETRAIN[[:space:]]*=[[:space:]]*\(True\|False\)' path/to/notebook.ipynb | head -5
grep -oP '(?<=")[^"]*\.(csv|npy|pth|pkl|json)(?=")' path/to/notebook.ipynb | head -20
```

---

## Task 1: Audit Autoencoder Family

**Notebooks (6):**
- `experiments/anomaly_detection/autoencoder/x64/baseline/notebook.ipynb`
- `experiments/anomaly_detection/autoencoder/x64/batchnorm/notebook.ipynb`
- `experiments/anomaly_detection/autoencoder/x64/batchnorm_dropout/notebook.ipynb`
- `experiments/anomaly_detection/autoencoder/x64/residual/notebook.ipynb`
- `experiments/anomaly_detection/autoencoder/x128/baseline/notebook.ipynb`
- `experiments/anomaly_detection/autoencoder/x224/main/notebook.ipynb`

- [ ] **Step 1: Check RETRAIN flags and artifact paths for all 6 AE notebooks**

```bash
for nb in \
  "experiments/anomaly_detection/autoencoder/x64/baseline/notebook.ipynb" \
  "experiments/anomaly_detection/autoencoder/x64/batchnorm/notebook.ipynb" \
  "experiments/anomaly_detection/autoencoder/x64/batchnorm_dropout/notebook.ipynb" \
  "experiments/anomaly_detection/autoencoder/x64/residual/notebook.ipynb" \
  "experiments/anomaly_detection/autoencoder/x128/baseline/notebook.ipynb" \
  "experiments/anomaly_detection/autoencoder/x224/main/notebook.ipynb"; do
  echo "=== $nb ==="; \
  grep -c "RETRAIN" "$nb" && echo "Has RETRAIN flag" || echo "No RETRAIN flag"; \
  grep -oP '(?<=")[^"]*\.(csv|npy|pth|pt|pkl|json)(?=")' "$nb" | sort -u | head -10; \
done
```

- [ ] **Step 2: Verify artifact files exist for each path found above**

For each path returned, run:
```bash
ls "experiments/anomaly_detection/autoencoder/<variant>/artifacts/" 2>/dev/null || echo "MISSING"
```

- [ ] **Step 3: Record any missing artifacts in NOTEBOOK_STATUS.md (running log — append after each family)**

Start the file now with this header if not already created:
```bash
cat > NOTEBOOK_STATUS.md << 'EOF'
# Notebook Status

Audit date: 2026-04-17. Checked: artifact presence for RETRAIN=False path, RETRAIN flag existence.

## Legend
- OK: RETRAIN=False path has all required artifacts
- MISSING_ARTIFACTS: RETRAIN=False path tries to load files that do not exist on disk
- NO_RETRAIN_FLAG: notebook has training cells but no RETRAIN toggle

## Issues Found

EOF
```

---

## Task 2: Audit VAE Family

**Notebooks (4):**
- `experiments/anomaly_detection/vae/x64/baseline/notebook.ipynb`
- `experiments/anomaly_detection/vae/x64/beta_sweep/notebook.ipynb`
- `experiments/anomaly_detection/vae/x64/latent_dim_sweep/notebook.ipynb`
- `experiments/anomaly_detection/vae/x224/main/notebook.ipynb`

- [ ] **Step 1: Check RETRAIN flags and artifact paths**

```bash
for nb in \
  "experiments/anomaly_detection/vae/x64/baseline/notebook.ipynb" \
  "experiments/anomaly_detection/vae/x64/beta_sweep/notebook.ipynb" \
  "experiments/anomaly_detection/vae/x64/latent_dim_sweep/notebook.ipynb" \
  "experiments/anomaly_detection/vae/x224/main/notebook.ipynb"; do
  echo "=== $nb ==="; \
  grep -c "RETRAIN" "$nb" && echo "Has RETRAIN" || echo "No RETRAIN"; \
  grep -oP '(?<=")[^"]*\.(csv|npy|pth|pt|pkl|json)(?=")' "$nb" | sort -u | head -10; \
done
```

- [ ] **Step 2: Verify artifact files exist for each path found**

```bash
ls experiments/anomaly_detection/vae/x64/baseline/artifacts/ 2>/dev/null
ls experiments/anomaly_detection/vae/x64/beta_sweep/artifacts/ 2>/dev/null
ls experiments/anomaly_detection/vae/x64/latent_dim_sweep/artifacts/ 2>/dev/null
ls experiments/anomaly_detection/vae/x224/main/artifacts/ 2>/dev/null
```

- [ ] **Step 3: Append any issues to NOTEBOOK_STATUS.md**

---

## Task 3: Audit SVDD, Backbone Embedding, RD4AD, FastFlow

**Notebooks (5):**
- `experiments/anomaly_detection/svdd/x64/baseline/notebook.ipynb`
- `experiments/anomaly_detection/backbone_embedding/resnet18/x64/baseline/notebook.ipynb`
- `experiments/anomaly_detection/backbone_embedding/wide_resnet50_2/x64/baseline/notebook.ipynb`
- `experiments/anomaly_detection/rd4ad/wideresnet50/x224/main/notebook.ipynb`
- `experiments/anomaly_detection/fastflow/x64/main/notebook.ipynb`

- [ ] **Step 1: Check RETRAIN flags and artifact paths**

```bash
for nb in \
  "experiments/anomaly_detection/svdd/x64/baseline/notebook.ipynb" \
  "experiments/anomaly_detection/backbone_embedding/resnet18/x64/baseline/notebook.ipynb" \
  "experiments/anomaly_detection/backbone_embedding/wide_resnet50_2/x64/baseline/notebook.ipynb" \
  "experiments/anomaly_detection/rd4ad/wideresnet50/x224/main/notebook.ipynb" \
  "experiments/anomaly_detection/fastflow/x64/main/notebook.ipynb"; do
  echo "=== $nb ==="; \
  grep -c "RETRAIN" "$nb" && echo "Has RETRAIN" || echo "No RETRAIN"; \
  grep -oP '(?<=")[^"]*\.(csv|npy|pth|pt|pkl|json)(?=")' "$nb" | sort -u | head -10; \
done
```

- [ ] **Step 2: Verify artifact files exist**

```bash
ls experiments/anomaly_detection/svdd/x64/baseline/artifacts/ 2>/dev/null
ls experiments/anomaly_detection/backbone_embedding/resnet18/x64/baseline/artifacts/ 2>/dev/null
ls experiments/anomaly_detection/backbone_embedding/wide_resnet50_2/x64/baseline/artifacts/ 2>/dev/null
ls experiments/anomaly_detection/rd4ad/wideresnet50/x224/main/artifacts/ 2>/dev/null
ls experiments/anomaly_detection/fastflow/x64/main/artifacts/ 2>/dev/null
```

- [ ] **Step 3: Append any issues to NOTEBOOK_STATUS.md**

---

## Task 4: Audit Teacher-Student Family

**Notebooks (10):**
- `experiments/anomaly_detection/teacher_student/resnet18/x64/main/notebook.ipynb`
- `experiments/anomaly_detection/teacher_student/resnet18/x224/main/notebook.ipynb`
- `experiments/anomaly_detection/teacher_student/resnet50/x64/main/notebook.ipynb`
- `experiments/anomaly_detection/teacher_student/resnet50/x64/layer_ablation/notebook.ipynb`
- `experiments/anomaly_detection/teacher_student/resnet50/x224/main/notebook.ipynb`
- `experiments/anomaly_detection/teacher_student/resnet50/x224/feature_autoencoder_dim_sweep/notebook.ipynb`
- `experiments/anomaly_detection/teacher_student/vit_b16/x224/main/notebook.ipynb`
- `experiments/anomaly_detection/teacher_student/wideresnet50_2/x224/multilayer_self_contained/notebook.ipynb`
- `experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/layer2_self_contained/notebook.ipynb`
- `experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/multilayer_self_contained/notebook.ipynb`

- [ ] **Step 1: Check RETRAIN flags and artifact paths**

```bash
for nb in \
  "experiments/anomaly_detection/teacher_student/resnet18/x64/main/notebook.ipynb" \
  "experiments/anomaly_detection/teacher_student/resnet18/x224/main/notebook.ipynb" \
  "experiments/anomaly_detection/teacher_student/resnet50/x64/main/notebook.ipynb" \
  "experiments/anomaly_detection/teacher_student/resnet50/x64/layer_ablation/notebook.ipynb" \
  "experiments/anomaly_detection/teacher_student/resnet50/x224/main/notebook.ipynb" \
  "experiments/anomaly_detection/teacher_student/resnet50/x224/feature_autoencoder_dim_sweep/notebook.ipynb" \
  "experiments/anomaly_detection/teacher_student/vit_b16/x224/main/notebook.ipynb" \
  "experiments/anomaly_detection/teacher_student/wideresnet50_2/x224/multilayer_self_contained/notebook.ipynb" \
  "experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/layer2_self_contained/notebook.ipynb" \
  "experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/multilayer_self_contained/notebook.ipynb"; do
  echo "=== $nb ==="; \
  grep -c "RETRAIN" "$nb" 2>/dev/null && echo "Has RETRAIN" || echo "No RETRAIN"; \
  grep -oP '(?<=")[^"]*\.(csv|npy|pth|pt|pkl|json)(?=")' "$nb" | sort -u | head -10; \
done
```

- [ ] **Step 2: Verify artifact directories exist for each TS variant**

```bash
find experiments/anomaly_detection/teacher_student -name "artifacts" -type d | sort
```

- [ ] **Step 3: Append any issues to NOTEBOOK_STATUS.md**

---

## Task 5: Audit PatchCore Family — WideResNet50 variants

**Notebooks (6):**
- `experiments/anomaly_detection/patchcore/ae_bn/x64/main/notebook.ipynb`
- `experiments/anomaly_detection/patchcore/resnet18/x64/main/notebook.ipynb`
- `experiments/anomaly_detection/patchcore/resnet50/x64/main/notebook.ipynb`
- `experiments/anomaly_detection/patchcore/wideresnet50/x64/main/notebook.ipynb`
- `experiments/anomaly_detection/patchcore/wideresnet50/x64/labeled_120k/notebook.ipynb`
- `experiments/anomaly_detection/patchcore/wideresnet50/x224/multilayer_umap/notebook.ipynb`

- [ ] **Step 1: Check RETRAIN flags and artifact paths**

```bash
for nb in \
  "experiments/anomaly_detection/patchcore/ae_bn/x64/main/notebook.ipynb" \
  "experiments/anomaly_detection/patchcore/resnet18/x64/main/notebook.ipynb" \
  "experiments/anomaly_detection/patchcore/resnet50/x64/main/notebook.ipynb" \
  "experiments/anomaly_detection/patchcore/wideresnet50/x64/main/notebook.ipynb" \
  "experiments/anomaly_detection/patchcore/wideresnet50/x64/labeled_120k/notebook.ipynb" \
  "experiments/anomaly_detection/patchcore/wideresnet50/x224/multilayer_umap/notebook.ipynb"; do
  echo "=== $nb ==="; \
  grep -c "RETRAIN" "$nb" 2>/dev/null && echo "Has RETRAIN" || echo "No RETRAIN"; \
  grep -oP '(?<=")[^"]*\.(csv|npy|pth|pt|pkl|json)(?=")' "$nb" | sort -u | head -10; \
done
```

- [ ] **Step 2: Confirm whether `multilayer_umap` is the primary WRN50 x224 result (rank #3)**

Read the first markdown cell of the multilayer_umap notebook:
```bash
python -c "
import json
with open('experiments/anomaly_detection/patchcore/wideresnet50/x224/multilayer_umap/notebook.ipynb') as f:
    nb = json.load(f)
for cell in nb['cells'][:5]:
    if cell['cell_type'] == 'markdown':
        print(''.join(cell['source']))
        break
"
```
Also check its F1 result against rank #3 (F1=0.549, AUROC=0.931). Check the saved CSV or JSON:
```bash
grep -r "0.549\|0.931" experiments/anomaly_detection/patchcore/wideresnet50/x224/ 2>/dev/null | head -5
```
Also check the layer2, layer3, weighted notebooks — confirm none of them is the main multilayer:
```bash
for nb in \
  "experiments/anomaly_detection/patchcore/wideresnet50/x224/layer2/notebook.ipynb" \
  "experiments/anomaly_detection/patchcore/wideresnet50/x224/layer3/notebook.ipynb" \
  "experiments/anomaly_detection/patchcore/wideresnet50/x224/weighted/notebook.ipynb"; do
  echo "=== $nb ==="; \
  python -c "
import json
with open('$nb') as f:
    nb = json.load(f)
for cell in nb['cells'][:3]:
    if cell['cell_type'] == 'markdown':
        print(''.join(cell['source'])[:300])
        break
"; done
```

- [ ] **Step 3: Verify key artifacts exist**

```bash
ls experiments/anomaly_detection/patchcore/wideresnet50/x64/main/artifacts/ 2>/dev/null
ls experiments/anomaly_detection/patchcore/wideresnet50/x224/multilayer_umap/artifacts/ 2>/dev/null
```

- [ ] **Step 4: Append any issues to NOTEBOOK_STATUS.md; record which x224 folder maps to rank #3**

---

## Task 6: Audit PatchCore Family — EfficientNet and ViT variants

**Notebooks (9):**
- `experiments/anomaly_detection/patchcore/wideresnet50/x224/layer2/notebook.ipynb`
- `experiments/anomaly_detection/patchcore/wideresnet50/x224/layer3/notebook.ipynb`
- `experiments/anomaly_detection/patchcore/wideresnet50/x224/weighted/notebook.ipynb`
- `experiments/anomaly_detection/patchcore/efficientnet_b0/x224/main/notebook.ipynb`
- `experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main_one_layer/notebook.ipynb`
- `experiments/anomaly_detection/patchcore/efficientnet_b1/x240/layer3_5/notebook.ipynb`
- `experiments/anomaly_detection/patchcore/efficientnet_b1/x240/layer3_5_no_defect_tuning/notebook.ipynb`
- `experiments/anomaly_detection/patchcore/dinov2_vit_b14/x224/notebook.ipynb`
- `experiments/anomaly_detection/patchcore/vit_b16/x64/main/notebook.ipynb`

- [ ] **Step 1: Check RETRAIN flags and artifact paths**

```bash
for nb in \
  "experiments/anomaly_detection/patchcore/wideresnet50/x224/layer2/notebook.ipynb" \
  "experiments/anomaly_detection/patchcore/wideresnet50/x224/layer3/notebook.ipynb" \
  "experiments/anomaly_detection/patchcore/wideresnet50/x224/weighted/notebook.ipynb" \
  "experiments/anomaly_detection/patchcore/efficientnet_b0/x224/main/notebook.ipynb" \
  "experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main_one_layer/notebook.ipynb" \
  "experiments/anomaly_detection/patchcore/efficientnet_b1/x240/layer3_5/notebook.ipynb" \
  "experiments/anomaly_detection/patchcore/efficientnet_b1/x240/layer3_5_no_defect_tuning/notebook.ipynb" \
  "experiments/anomaly_detection/patchcore/dinov2_vit_b14/x224/notebook.ipynb" \
  "experiments/anomaly_detection/patchcore/vit_b16/x64/main/notebook.ipynb"; do
  echo "=== $nb ==="; \
  grep -c "RETRAIN\|FORCE_RERUN" "$nb" 2>/dev/null && echo "Has flag" || echo "No flag"; \
  grep -oP '(?<=")[^"]*\.(csv|npy|pth|pt|pkl|json)(?=")' "$nb" | sort -u | head -10; \
done
```

- [ ] **Step 2: Verify key artifact directories**

```bash
find experiments/anomaly_detection/patchcore/efficientnet_b1 -name "artifacts" -type d
find experiments/anomaly_detection/patchcore/efficientnet_b0 -name "artifacts" -type d
find experiments/anomaly_detection/patchcore/dinov2_vit_b14 -name "artifacts" -type d
```

- [ ] **Step 3: Append any issues to NOTEBOOK_STATUS.md**

---

## Task 7: Audit PatchCore ViT-B/16 x224 ablation variants

**Notebooks (5):**
- `experiments/anomaly_detection/patchcore/vit_b16/x224/main/notebook.ipynb`
- `experiments/anomaly_detection/patchcore/vit_b16/x224/block_depth_sweep/notebook.ipynb`
- `experiments/anomaly_detection/patchcore/vit_b16/x224/one_layer_defect_tuning/notebook.ipynb`
- `experiments/anomaly_detection/patchcore/vit_b16/x224/one_layer_no_defect_tuning/notebook.ipynb`
- `experiments/anomaly_detection/patchcore/vit_b16/x224/two_block/notebook.ipynb`
- `experiments/anomaly_detection/patchcore/vit_b16/x224/two_block_no_defect_tuning/notebook.ipynb`

- [ ] **Step 1: Check RETRAIN flags and artifact paths**

```bash
for nb in \
  "experiments/anomaly_detection/patchcore/vit_b16/x224/main/notebook.ipynb" \
  "experiments/anomaly_detection/patchcore/vit_b16/x224/block_depth_sweep/notebook.ipynb" \
  "experiments/anomaly_detection/patchcore/vit_b16/x224/one_layer_defect_tuning/notebook.ipynb" \
  "experiments/anomaly_detection/patchcore/vit_b16/x224/one_layer_no_defect_tuning/notebook.ipynb" \
  "experiments/anomaly_detection/patchcore/vit_b16/x224/two_block/notebook.ipynb" \
  "experiments/anomaly_detection/patchcore/vit_b16/x224/two_block_no_defect_tuning/notebook.ipynb"; do
  echo "=== $nb ==="; \
  grep -c "RETRAIN\|FORCE_RERUN" "$nb" 2>/dev/null && echo "Has flag" || echo "No flag"; \
  grep -oP '(?<=")[^"]*\.(csv|npy|pth|pt|pkl|json)(?=")' "$nb" | sort -u | head -10; \
done
```

- [ ] **Step 2: Verify artifact directories for ViT-B/16 x224 variants**

```bash
find experiments/anomaly_detection/patchcore/vit_b16/x224 -name "artifacts" -type d | sort
```

- [ ] **Step 3: Append any issues to NOTEBOOK_STATUS.md**

---

## Task 8: Audit Ensemble, Report Figures, Classifier, Supervised Defect

**Notebooks (8):**
- `experiments/anomaly_detection/ensemble/x64/score_ensemble/notebook.ipynb`
- `experiments/anomaly_detection/ensemble/x224/vit_effnetb1_ensemble/notebook.ipynb`
- `experiments/anomaly_detection/report_figures/notebook.ipynb`
- `experiments/classifier/multiclass/x64/training/notebook.ipynb`
- `experiments/classifier/multiclass/x64/seed07/notebook.ipynb`
- `experiments/classifier/multiclass/x64/final_labeling/notebook.ipynb`
- `experiments/classifier/multiclass/x64/showcase/notebook.ipynb`
- `experiments/anomaly_detection_defect/supervised_sweep/vit_b16/x224/main/notebook.ipynb`

- [ ] **Step 1: Check RETRAIN flags and artifact paths**

```bash
for nb in \
  "experiments/anomaly_detection/ensemble/x64/score_ensemble/notebook.ipynb" \
  "experiments/anomaly_detection/ensemble/x224/vit_effnetb1_ensemble/notebook.ipynb" \
  "experiments/anomaly_detection/report_figures/notebook.ipynb" \
  "experiments/classifier/multiclass/x64/training/notebook.ipynb" \
  "experiments/classifier/multiclass/x64/seed07/notebook.ipynb" \
  "experiments/classifier/multiclass/x64/final_labeling/notebook.ipynb" \
  "experiments/classifier/multiclass/x64/showcase/notebook.ipynb" \
  "experiments/anomaly_detection_defect/supervised_sweep/vit_b16/x224/main/notebook.ipynb"; do
  echo "=== $nb ==="; \
  grep -c "RETRAIN\|FORCE_RERUN\|RETRAIN_MODEL" "$nb" 2>/dev/null && echo "Has flag" || echo "No flag"; \
  grep -oP '(?<=")[^"]*\.(csv|npy|pth|pt|pkl|json)(?=")' "$nb" | sort -u | head -10; \
done
```

- [ ] **Step 2: Check supervised_cnn folder for any notebooks (expected: none)**

```bash
find experiments/anomaly_detection_defect/supervised_cnn -name "*.ipynb" 2>/dev/null || echo "No notebooks found"
```

- [ ] **Step 3: Verify artifact directories**

```bash
ls experiments/anomaly_detection/ensemble/x64/score_ensemble/artifacts/ 2>/dev/null
ls experiments/anomaly_detection/ensemble/x224/vit_effnetb1_ensemble/artifacts/ 2>/dev/null
ls experiments/anomaly_detection/report_figures/artifacts/ 2>/dev/null
ls experiments/classifier/multiclass/x64/training/artifacts/ 2>/dev/null
ls experiments/anomaly_detection_defect/supervised_sweep/vit_b16/x224/main/artifacts/ 2>/dev/null
```

- [ ] **Step 4: Append any issues to NOTEBOOK_STATUS.md**

---

## Task 9: Fix any path mismatches found during audit

For each notebook flagged as a path mismatch (artifact exists on disk but notebook loads from wrong path):

- [ ] **Step 1: For each mismatch, open the notebook and find the broken path cell**

Use Read tool to open the notebook JSON and locate the cell with the wrong path string.

- [ ] **Step 2: Edit the path string in the notebook JSON**

Use Edit tool. The path string will be inside a `"source"` array in the notebook JSON. Update the string to the correct relative path.

- [ ] **Step 3: Verify the corrected path exists**

```bash
ls "corrected/path/to/artifact"
```

- [ ] **Step 4: Commit path fixes**

```bash
git add <modified notebooks>
git commit -m "fix: correct artifact load paths in notebooks with path mismatches"
```

---

## Task 10: Write NOTEBOOK_STATUS.md

After all audit tasks are complete, finalize the file.

- [ ] **Step 1: If no issues found, write a clean status entry**

Append to `NOTEBOOK_STATUS.md`:
```markdown
## Summary

All notebooks checked. See issues section above for any MISSING_ARTIFACTS or NO_RETRAIN_FLAG entries.
If the issues section is empty: all RETRAIN=False paths have their required artifacts on disk.
```

- [ ] **Step 2: Commit NOTEBOOK_STATUS.md**

```bash
git add NOTEBOOK_STATUS.md
git commit -m "docs: add notebook audit status tracking file"
```

---

## Task 11: Write README.md

Replace the full content of `README.md` with the structure below. Fill in the `(Trainable)` tags and any `[TBC]` notes based on audit findings from Tasks 1–8.

> **Note on rank #3:** If Task 5 confirms `multilayer_umap` is the primary WRN50 x224 result, use that notebook path. If audit reveals a different folder, use the correct one.
>
> **Note on ranks 5/7/8:** These are different variants within the same notebook (`wideresnet50/x64/main`). The table links to the same notebook for all three but names the variant.

- [ ] **Step 1: Write the new README.md**

```markdown
# Automated IC Wafer Defect Detection — WM-811K

**Group 08** · Henry Lee Jun · Chia Tang · Genson Low

## Quick Start

1. Create environment and install: `py -3.11 -m venv .venv && .venv\Scripts\activate && pip install -e .`
2. Place the dataset at `data/raw/LSWMD.pkl` (WM-811K pickle)
3. Launch Jupyter from the repo root: `jupyter notebook`

---

## Top 10 Models

Ranked by val-threshold F1 (primary metric). All metrics from the main benchmark (5k normal / 250 anomaly test set).

| Rank | Model | F1 | AUROC | Notebook | |
|---|---|---|---|---|---|
| 1 | PatchCore + ViT-B/16 (block 6, 224×224) | 0.595 | 0.956 | [notebook](experiments/anomaly_detection/patchcore/vit_b16/x224/main/notebook.ipynb) | (Trainable) |
| 2 | PatchCore + EfficientNet-B1 (block 3, 240×240) | 0.591 | 0.935 | [notebook](experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main_one_layer/notebook.ipynb) | (Trainable) |
| 3 | PatchCore + WideResNet50-2 (layer2+3, 224×224) | 0.549 | 0.931 | [notebook](experiments/anomaly_detection/patchcore/wideresnet50/x224/multilayer_umap/notebook.ipynb) | (Trainable) |
| 4 | PatchCore + EfficientNet-B0 (blocks 3+6, 224×224) | 0.544 | 0.925 | [notebook](experiments/anomaly_detection/patchcore/efficientnet_b0/x224/main/notebook.ipynb) | (Trainable) |
| 5 | PatchCore + WideResNet50-2 (layer2+3, 64×64, r=0.10) | 0.532 | 0.917 | [notebook](experiments/anomaly_detection/patchcore/wideresnet50/x64/main/notebook.ipynb) | (Trainable) |
| 6 | Score Ensemble (PatchCore-WRN50-x64 + TS-Res50-x64) | 0.529 | 0.916 | [notebook](experiments/anomaly_detection/ensemble/x64/score_ensemble/notebook.ipynb) | (Trainable) |
| 7 | PatchCore + WideResNet50-2 (layer2+3, 64×64, r=0.15) | 0.526 | 0.912 | [notebook](experiments/anomaly_detection/patchcore/wideresnet50/x64/main/notebook.ipynb) | (Trainable) |
| 8 | PatchCore + WideResNet50-2 (layer2+3, 64×64, r=0.05) | 0.525 | 0.921 | [notebook](experiments/anomaly_detection/patchcore/wideresnet50/x64/main/notebook.ipynb) | (Trainable) |
| 9 | Teacher-Student + ResNet50 (64×64, topk_mean r=0.20) | 0.525 | 0.909 | [notebook](experiments/anomaly_detection/teacher_student/resnet50/x64/main/notebook.ipynb) | (Trainable) |
| 10 | Teacher-Student + WideResNet50-2 (layer2+3, 64×64, topk_mean r=0.15) | 0.524 | 0.923 | [notebook](experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/multilayer_self_contained/notebook.ipynb) | (Trainable) |

> Ranks 5, 7, and 8 are sweep variants within the same notebook — set `RETRAIN=False` to load saved results for all variants.

---

## Per Family

### Autoencoder

1. AE x64 Baseline — [experiments/anomaly_detection/autoencoder/x64/baseline/](experiments/anomaly_detection/autoencoder/x64/baseline/) (Trainable)
2. AE x64 + BatchNorm — [experiments/anomaly_detection/autoencoder/x64/batchnorm/](experiments/anomaly_detection/autoencoder/x64/batchnorm/) (Trainable)
3. AE x64 + BatchNorm + Dropout — [experiments/anomaly_detection/autoencoder/x64/batchnorm_dropout/](experiments/anomaly_detection/autoencoder/x64/batchnorm_dropout/) (Trainable)
4. AE x64 Residual — [experiments/anomaly_detection/autoencoder/x64/residual/](experiments/anomaly_detection/autoencoder/x64/residual/) (Trainable)
5. AE x128 Baseline — [experiments/anomaly_detection/autoencoder/x128/baseline/](experiments/anomaly_detection/autoencoder/x128/baseline/) (Trainable)
6. AE x224 Main — [experiments/anomaly_detection/autoencoder/x224/main/](experiments/anomaly_detection/autoencoder/x224/main/) (Trainable)

### VAE

1. VAE x64 Baseline — [experiments/anomaly_detection/vae/x64/baseline/](experiments/anomaly_detection/vae/x64/baseline/) (Trainable)
2. VAE x64 Beta Sweep — [experiments/anomaly_detection/vae/x64/beta_sweep/](experiments/anomaly_detection/vae/x64/beta_sweep/) (Trainable)
3. VAE x64 Latent Dim Sweep — [experiments/anomaly_detection/vae/x64/latent_dim_sweep/](experiments/anomaly_detection/vae/x64/latent_dim_sweep/) (Trainable)
4. VAE x224 Main — [experiments/anomaly_detection/vae/x224/main/](experiments/anomaly_detection/vae/x224/main/) (Trainable)

### SVDD

1. Deep SVDD x64 — [experiments/anomaly_detection/svdd/x64/baseline/](experiments/anomaly_detection/svdd/x64/baseline/) (Trainable)

### Backbone Embedding

1. Backbone Embedding ResNet18 x64 — [experiments/anomaly_detection/backbone_embedding/resnet18/x64/baseline/](experiments/anomaly_detection/backbone_embedding/resnet18/x64/baseline/) (Trainable)
2. Backbone Embedding WideResNet50-2 x64 — [experiments/anomaly_detection/backbone_embedding/wide_resnet50_2/x64/baseline/](experiments/anomaly_detection/backbone_embedding/wide_resnet50_2/x64/baseline/) (Trainable)

### Teacher-Student

1. TS ResNet18 x64 — [experiments/anomaly_detection/teacher_student/resnet18/x64/main/](experiments/anomaly_detection/teacher_student/resnet18/x64/main/) (Trainable)
2. TS ResNet18 x224 — [experiments/anomaly_detection/teacher_student/resnet18/x224/main/](experiments/anomaly_detection/teacher_student/resnet18/x224/main/) (Trainable)
3. TS ResNet50 x64 Main — [experiments/anomaly_detection/teacher_student/resnet50/x64/main/](experiments/anomaly_detection/teacher_student/resnet50/x64/main/) (Trainable)
4. TS ResNet50 x64 Layer Ablation — [experiments/anomaly_detection/teacher_student/resnet50/x64/layer_ablation/](experiments/anomaly_detection/teacher_student/resnet50/x64/layer_ablation/) (Trainable)
5. TS ResNet50 x224 Main — [experiments/anomaly_detection/teacher_student/resnet50/x224/main/](experiments/anomaly_detection/teacher_student/resnet50/x224/main/) (Trainable)
6. TS ResNet50 x224 Feature AE Dim Sweep — [experiments/anomaly_detection/teacher_student/resnet50/x224/feature_autoencoder_dim_sweep/](experiments/anomaly_detection/teacher_student/resnet50/x224/feature_autoencoder_dim_sweep/) (Trainable)
7. TS ViT-B/16 x224 — [experiments/anomaly_detection/teacher_student/vit_b16/x224/main/](experiments/anomaly_detection/teacher_student/vit_b16/x224/main/) (Trainable)
8. TS WideResNet50-2 x224 Multilayer — [experiments/anomaly_detection/teacher_student/wideresnet50_2/x224/multilayer_self_contained/](experiments/anomaly_detection/teacher_student/wideresnet50_2/x224/multilayer_self_contained/) (Trainable)
9. TS WideResNet50-2 x64 Layer2 — [experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/layer2_self_contained/](experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/layer2_self_contained/) (Trainable)
10. TS WideResNet50-2 x64 Multilayer — [experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/multilayer_self_contained/](experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/multilayer_self_contained/) (Trainable)

### PatchCore

1. PatchCore AE-BN Backbone x64 — [experiments/anomaly_detection/patchcore/ae_bn/x64/main/](experiments/anomaly_detection/patchcore/ae_bn/x64/main/) (Trainable)
2. PatchCore ResNet18 x64 — [experiments/anomaly_detection/patchcore/resnet18/x64/main/](experiments/anomaly_detection/patchcore/resnet18/x64/main/) (Trainable)
3. PatchCore ResNet50 x64 — [experiments/anomaly_detection/patchcore/resnet50/x64/main/](experiments/anomaly_detection/patchcore/resnet50/x64/main/) (Trainable)
4. PatchCore WideResNet50-2 x64 — [experiments/anomaly_detection/patchcore/wideresnet50/x64/main/](experiments/anomaly_detection/patchcore/wideresnet50/x64/main/) (Trainable)
5. PatchCore WideResNet50-2 x64 Labeled 120k — [experiments/anomaly_detection/patchcore/wideresnet50/x64/labeled_120k/](experiments/anomaly_detection/patchcore/wideresnet50/x64/labeled_120k/) (Trainable)
6. PatchCore WideResNet50-2 x224 Layer2 — [experiments/anomaly_detection/patchcore/wideresnet50/x224/layer2/](experiments/anomaly_detection/patchcore/wideresnet50/x224/layer2/) (Trainable)
7. PatchCore WideResNet50-2 x224 Layer3 — [experiments/anomaly_detection/patchcore/wideresnet50/x224/layer3/](experiments/anomaly_detection/patchcore/wideresnet50/x224/layer3/) (Trainable)
8. PatchCore WideResNet50-2 x224 Multilayer — [experiments/anomaly_detection/patchcore/wideresnet50/x224/multilayer_umap/](experiments/anomaly_detection/patchcore/wideresnet50/x224/multilayer_umap/) (Trainable)
9. PatchCore WideResNet50-2 x224 Weighted — [experiments/anomaly_detection/patchcore/wideresnet50/x224/weighted/](experiments/anomaly_detection/patchcore/wideresnet50/x224/weighted/) (Trainable)
10. PatchCore EfficientNet-B0 x224 — [experiments/anomaly_detection/patchcore/efficientnet_b0/x224/main/](experiments/anomaly_detection/patchcore/efficientnet_b0/x224/main/) (Trainable)
11. PatchCore EfficientNet-B1 x240 (one layer) — [experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main_one_layer/](experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main_one_layer/) (Trainable)
12. PatchCore EfficientNet-B1 x240 (layer 3+5) — [experiments/anomaly_detection/patchcore/efficientnet_b1/x240/layer3_5/](experiments/anomaly_detection/patchcore/efficientnet_b1/x240/layer3_5/) (Trainable)
13. PatchCore EfficientNet-B1 x240 (layer 3+5, no defect tuning) — [experiments/anomaly_detection/patchcore/efficientnet_b1/x240/layer3_5_no_defect_tuning/](experiments/anomaly_detection/patchcore/efficientnet_b1/x240/layer3_5_no_defect_tuning/) (Trainable)
14. PatchCore ViT-B/16 x224 Main — [experiments/anomaly_detection/patchcore/vit_b16/x224/main/](experiments/anomaly_detection/patchcore/vit_b16/x224/main/) (Trainable)
15. PatchCore ViT-B/16 x224 Block Depth Sweep — [experiments/anomaly_detection/patchcore/vit_b16/x224/block_depth_sweep/](experiments/anomaly_detection/patchcore/vit_b16/x224/block_depth_sweep/) (Trainable)
16. PatchCore ViT-B/16 x224 One Layer Defect Tuning — [experiments/anomaly_detection/patchcore/vit_b16/x224/one_layer_defect_tuning/](experiments/anomaly_detection/patchcore/vit_b16/x224/one_layer_defect_tuning/) (Trainable)
17. PatchCore ViT-B/16 x224 One Layer No Defect Tuning — [experiments/anomaly_detection/patchcore/vit_b16/x224/one_layer_no_defect_tuning/](experiments/anomaly_detection/patchcore/vit_b16/x224/one_layer_no_defect_tuning/) (Trainable)
18. PatchCore ViT-B/16 x224 Two Block — [experiments/anomaly_detection/patchcore/vit_b16/x224/two_block/](experiments/anomaly_detection/patchcore/vit_b16/x224/two_block/) (Trainable)
19. PatchCore ViT-B/16 x224 Two Block No Defect Tuning — [experiments/anomaly_detection/patchcore/vit_b16/x224/two_block_no_defect_tuning/](experiments/anomaly_detection/patchcore/vit_b16/x224/two_block_no_defect_tuning/) (Trainable)
20. PatchCore ViT-B/16 x64 — [experiments/anomaly_detection/patchcore/vit_b16/x64/main/](experiments/anomaly_detection/patchcore/vit_b16/x64/main/) (Trainable)
21. PatchCore DINOv2 ViT-B/14 x224 — [experiments/anomaly_detection/patchcore/dinov2_vit_b14/x224/](experiments/anomaly_detection/patchcore/dinov2_vit_b14/x224/) (Trainable)

### RD4AD

1. RD4AD WideResNet50-2 x224 — [experiments/anomaly_detection/rd4ad/wideresnet50/x224/main/](experiments/anomaly_detection/rd4ad/wideresnet50/x224/main/) (Trainable)

### FastFlow

1. FastFlow x64 — [experiments/anomaly_detection/fastflow/x64/main/](experiments/anomaly_detection/fastflow/x64/main/) (Trainable)

### Ensemble

1. Score Ensemble x64 (PatchCore-WRN50 + TS-Res50) — [experiments/anomaly_detection/ensemble/x64/score_ensemble/](experiments/anomaly_detection/ensemble/x64/score_ensemble/)
2. ViT + EfficientNet-B1 Ensemble x224 — [experiments/anomaly_detection/ensemble/x224/vit_effnetb1_ensemble/](experiments/anomaly_detection/ensemble/x224/vit_effnetb1_ensemble/)

### Report Figures

1. Report Figures — [experiments/anomaly_detection/report_figures/](experiments/anomaly_detection/report_figures/)

### Supervised Multiclass Classifier

1. Multiclass Classifier x64 Training — [experiments/classifier/multiclass/x64/training/](experiments/classifier/multiclass/x64/training/) (Trainable)
2. Multiclass Classifier x64 Seed 07 — [experiments/classifier/multiclass/x64/seed07/](experiments/classifier/multiclass/x64/seed07/) (Trainable)
3. Multiclass Classifier x64 Final Labeling — [experiments/classifier/multiclass/x64/final_labeling/](experiments/classifier/multiclass/x64/final_labeling/) (Trainable)
4. Multiclass Classifier x64 Showcase — [experiments/classifier/multiclass/x64/showcase/](experiments/classifier/multiclass/x64/showcase/)

### Supervised Defect Detection

1. Supervised Sweep ViT-B/16 x224 — [experiments/anomaly_detection_defect/supervised_sweep/vit_b16/x224/main/](experiments/anomaly_detection_defect/supervised_sweep/vit_b16/x224/main/) (Trainable)

---

## Repository Layout

| Folder | Contents |
|---|---|
| `experiments/` | All experiment notebooks, configs, and artifacts |
| `data/dataset/` | Dataset construction notebooks |
| `data/raw/` | Raw dataset files (not committed — place LSWMD.pkl here) |
| `src/wafer_defect/` | Shared package code used by all notebooks |
| `scripts/` | CLI utilities for dataset prep and evaluation |
| `family_reports/` | Per-family result summaries and analysis |
| `configs/` | Shared config snapshots |
| `artifacts/` | Global shared artifacts and report plots |
```

- [ ] **Step 2: Adjust the README based on audit findings**

- If rank #3 notebook path differs from `multilayer_umap`, update that row in the Top 10 table.
- Remove `(Trainable)` from any notebook that the audit confirmed has no RETRAIN flag.
- Add a note under any family with missing artifacts pointing to NOTEBOOK_STATUS.md.

- [ ] **Step 3: Commit README**

```bash
git add README.md
git commit -m "docs: rewrite README with Top 10 models and per-family navigation"
```

---

## Self-Review Checklist

- [x] Spec coverage: Quick Start ✓, Top 10 table ✓, Per Family ✓, Repo Layout ✓, Audit ✓, NOTEBOOK_STATUS.md ✓, path fix task ✓
- [x] Placeholder scan: no TBD/TODO — the `[TBC]` note in Task 11 is a conditional instruction, not a blank to fill in; it resolves to one of two known values
- [x] Rank #3 ambiguity flagged with resolution step in Task 5 and conditional update in Task 11
- [x] All ~50 notebook paths accounted for across Tasks 1–8
- [x] supervised_cnn check included (Task 8 Step 2)
