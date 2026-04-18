---
name: Repo Navigation and Notebook Audit
description: Design for restructuring README and auditing all notebooks for RETRAIN=False runability
type: spec
date: 2026-04-17
---

# Repo Navigation and Notebook Audit

## Goal

Make the repo immediately navigable for someone who has read the report. They should be able to find any notebook within 30 seconds and know whether it can be run as-is or requires retraining.

## README Structure

The new `README.md` has four sections in this order:

### 1. Quick Start (3 bullet points only)
- Create venv + `pip install -e .`
- Place `LSWMD.pkl` in `data/raw/`
- Launch Jupyter from repo root

### 2. Top 10 Models
Ranked table matching the report's REPORT.md ranking. Columns:

| Rank | Model | F1 | AUROC | Notebook | |
|---|---|---|---|---|---|
| 1 | ... | ... | ... | [link] | (Trainable) |

- Linked directly to the notebook file
- `(Trainable)` tag on any model with a `RETRAIN=True` flag

### 3. Per Family
One subsection per family. Each lists its notebooks as a numbered list:

```
**Autoencoder**
1. AE x64 Baseline — [experiments/anomaly_detection/autoencoder/x64/baseline/](link) (Trainable)
2. AE x64 BatchNorm — [experiments/.../batchnorm/](link) (Trainable)
...
```

Families covered (in order):
1. Autoencoder
2. VAE
3. SVDD
4. Backbone Embedding
5. Teacher-Student
6. PatchCore
7. RD4AD
8. FastFlow
9. Ensemble
10. Classifier (Supervised Multiclass)
11. Supervised Defect Detection

### 4. Repository Layout
One-liner per top-level folder (condensed from current README). No narrative.

**Removed from current README:** experiment narrative, extended notes on execution, long quick start prose.

---

## Notebook Audit

### Scope
All notebooks under `experiments/` — approximately 50 notebooks.

### Per-notebook checks (static inspection only — no execution)
1. **Artifact check**: Read the first ~50 lines of code cells. Identify what files the `RETRAIN=False` path tries to load (CSVs, `.npy`, checkpoints, result JSONs). Confirm those files exist on disk.
2. **Trainable flag check**: Does the notebook have a `RETRAIN` (or equivalent) boolean flag cell? If yes → `(Trainable)`.

### Outputs
- **`NOTEBOOK_STATUS.md`** at repo root — tracking file only, not linked from README. Lists:
  - Notebooks with missing artifacts (broken RETRAIN=False path)
  - Any notebooks with no RETRAIN flag that have a training section (flag as "needs RETRAIN toggle")
- **README.md** — reflects only working state; no broken/missing callouts inline

### Fix scope
- Path mismatches (artifact exists but notebook loads from wrong path) → fix path in notebook
- Missing artifacts entirely → logged in `NOTEBOOK_STATUS.md` only, no notebook modification
- No notebook code modified beyond path fixes

---

## Approach

Option C (parallel sweep): sweep family by family. For each family, check artifacts + flags, then immediately write that family's README section and any NOTEBOOK_STATUS entries. One continuous pass.

After the sweep: apply any path-mismatch fixes found, then commit README and NOTEBOOK_STATUS.md together.

---

## Known Issues to Investigate

- `experiments/anomaly_detection/patchcore/wideresnet50/x224/multilayer/` — referenced in REPORT.md (rank #3) but the folder does not exist. Actual folders: `layer2`, `layer3`, `multilayer_umap`, `weighted`. Likely `multilayer_umap` is the primary; needs confirmation.
- `experiments/anomaly_detection_defect/supervised_cnn/` — has subfolders `full_defect` and `half_defect` but no `notebook.ipynb` found; may be scripts-only.
