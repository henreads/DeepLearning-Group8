# Future Experiments to Run

This document tracks experiments that should be executed after repository cleanup and finalization.

---

## Status Summary

**Completed (2026-04-02)**:
- ✓ **Autoencoder x224 Main** — Resolution comparison baseline, establishes fair comparison point against 224×224 pretrained methods

**Remaining (Tier 1 Priority)**:
- [ ] VAE x224 Main — Resolution comparison for VAE family
- [ ] Teacher-Student ResNet18 x224 — TS family resolution study
- [ ] Teacher-Student ResNet50 x224 — TS family resolution study
- [ ] PatchCore EfficientNet-B0 x224 — Efficient backbone at native resolution
- [ ] PatchCore ViT-B/16 x224 — Vision Transformer at native resolution

**Remaining (Tier 2 Priority)**:
- [ ] VAE Latent Dimension Sweep — Hyperparameter study for VAE family
- [ ] Teacher-Student ResNet50 x224 Feature Autoencoder Dimension Sweep — Tests if larger bottlenecks fix x224 degradation

---

## Autoencoder Family Enhancements

### Experiment 1: Autoencoder x224 Main (Resolution Study)

**Status**: ✓ COMPLETED (2026-04-02)

**Location**: `experiments/anomaly_detection/autoencoder/x224/main/`

**Files**:
- `train_config.toml` - Configuration with image_size=224, batch_size=512 (A10G GPU)
- Modal app: `modal_apps/autoencoder_x224_main/`
- Notebook: `experiments/anomaly_detection/autoencoder/x224/main/notebook.ipynb` (with Windows encoding fix and runnable score ablation cell)
- Runner script: `scripts/run_autoencoder_x224_main_notebook.py`

**Actual Results**:
- **Training**: 20 epochs, best epoch 18, best val loss 0.0182 (early convergence by epoch 10)
- **Evaluation Metrics** (with `topk_abs_mean` scoring):
  - F1: 0.510 (vs. x64 baseline 0.467, x64/BN 0.502)
  - AUROC: 0.901 (vs. x64 baseline 0.839, x64/BN 0.834) — **+6.2 percentage points**
  - AUPRC: 0.596 (vs. x64 baseline 0.522, x64/BN 0.568) — **+7.4 percentage points**
  - Best sweep F1: 0.587
- **Score Ablation Results** (7 scoring rules tested):
  - Winner: `topk_abs_mean` (F1=0.510, AUROC=0.901, AUPRC=0.596)
  - Runner-up: `max_abs` (F1=0.485, AUROC=0.833, AUPRC=0.473)
- **Per-Defect Recall**:
  - Edge-Ring: 0.893 (vs. x64/BN 0.833)
  - Center: 0.760 (vs. x64/BN 0.700)
  - Donut: 0.857
  - Scratch: 0.200 (persistent weakness across all AE variants)

**Documentation**:
- Family Report: `family_reports/autoencoder.md` (lines 503-597, complete x224 section)
- LaTeX Report: `Report_Latex.tex` (main leaderboard, appendix, Q1/Q4 explanations)
- Artifacts: `experiments/anomaly_detection/autoencoder/x224/main/artifacts/autoencoder_x224/`
  - Plots: training curves, score distribution, threshold sweep, confusion matrix, reconstruction examples, failure examples, score ablation summary
  - CSV: metrics.csv, score_ablation.csv, threshold_sweep.csv, failure_analysis.csv, failure_defect_recall.csv

**Key Finding**: **Resolution alone dramatically improves autoencoder ranking quality.** AUROC jumped 0.839→0.901 (+7.7%) without any architectural changes. This establishes a fair baseline for comparing reconstruction-based methods against pretrained approaches, both of which use 224×224 natively.

---

## VAE Family Enhancements

### Experiment 1: VAE x224 Main (Resolution Study)

**Status**: Infrastructure ready, notebook structure prepared, waiting for training

**Location**: `experiments/anomaly_detection/vae/x224/main/`

**Files**:
- `train_config.toml` - Configuration with image_size=224, batch_size=512, num_workers=8
- Modal app: `modal_apps/vae_x224_main/`
- Runner script: `scripts/run_vae_x224_main_notebook.py`

**Run Command**:
```powershell
modal run --detach modal_apps/vae_x224_main/app.py::main --no-sync-back
```

**Download Artifacts**:
```powershell
modal run modal_apps/vae_x224_main/app.py::download_artifacts
```

**Expected Output**:
- Training checkpoints and history
- Evaluation metrics (F1, AUROC, AUPRC)
- Plots: training curves, score distribution, threshold sweep, confusion matrix, reconstruction examples
- Comparison with VAE x64 baseline (F1=0.418, AUROC=0.772, AUPRC=0.372)

**Research Question**: Does using higher resolution (224×224) help VAE preserve spatial detail and improve anomaly detection over downsampled 64×64?

---

### Experiment 2: VAE x64 Latent Dimension Sweep

**Status**: Infrastructure ready, notebook structure prepared, resumable per latent dimension

**Location**: `experiments/anomaly_detection/vae/x64/latent_dim_sweep/`

**Files**:
- `train_config.toml` - Configuration with latent_dims=[32, 64, 128, 256, 512], batch_size=512, num_workers=8
- Modal app: `modal_apps/vae_latent_dim_sweep/`
- Runner script: `scripts/run_vae_latent_dim_sweep_notebook.py`

**Run Commands**:
```powershell
modal run --detach modal_apps/vae_latent_dim_sweep/app.py::main --no-sync-back
```

Use saved runs only:
```powershell
modal run --detach modal_apps/vae_latent_dim_sweep/app.py::main --no-sync-back
```

Run only missing latent dimensions:
```powershell
modal run --detach modal_apps/vae_latent_dim_sweep/app.py::main --retrain --no-sync-back
```

Force full rerun from scratch for all latent dimensions:
```powershell
modal run --detach modal_apps/vae_latent_dim_sweep/app.py::main --full-retrain --no-sync-back
```

**Download Artifacts**:
```powershell
modal run modal_apps/vae_latent_dim_sweep/app.py::download_artifacts
```

**Expected Output**:
- Per-latent-dim checkpoints and histories under:
  - `artifacts/vae_latent_dim_sweep/latent_dim_32/`
  - `artifacts/vae_latent_dim_sweep/latent_dim_64/`
  - `artifacts/vae_latent_dim_sweep/latent_dim_128/`
  - `artifacts/vae_latent_dim_sweep/latent_dim_256/`
  - `artifacts/vae_latent_dim_sweep/latent_dim_512/`
- Evaluation metrics and sweep summaries
- Plots: latent dimension comparison, training curves across dims, best-dim confusion matrix
- Comparison with VAE baseline latent_dim=128 (F1=0.418, AUROC=0.772, AUPRC=0.372)

**Current Resume Semantics**:
- `RETRAIN = False`: use existing completed latent-dim runs only
- `RETRAIN = True`: skip completed latent dims and run only missing ones
- `FULL_RETRAIN = True`: ignore saved checkpoints and rerun every latent dim from scratch

**Research Question**: Is 128 the optimal latent dimension, or does KL regularization compress out the signal regardless? Can larger latent spaces preserve defect signals?

---

## Teacher-Student x224 Variants (Resolution Comparison Study)

These experiments are **fully set up and ready to run** via Modal. They will compare the impact of using the teacher backbone's native 224×224 resolution vs. the downsampled 64×64 resolution.

### Experiment 1: Teacher-Student ResNet18 x224 Main

**Status**: Infrastructure ready, notebook created, waiting for training

**Location**: `experiments/anomaly_detection/teacher_student/resnet18/x224/main/`

**Files**:
- `notebook.ipynb` - Created from x64 version, ready for execution
- `train_config.toml` - Configuration with image_size=224
- Modal app: `modal_apps/teacher_student_resnet18_x224_main/`

**Run Command**:
```powershell
modal run --detach modal_apps/teacher_student_resnet18_x224_main/app.py::main --no-sync-back
```

**Download Artifacts**:
```powershell
modal run modal_apps/teacher_student_resnet18_x224_main/app.py::download_artifacts
```

**Expected Output**:
- Training checkpoints
- Evaluation metrics (F1, AUROC, AUPRC, per-defect breakdown)
- Plots: training curves, score distribution, threshold sweep, confusion matrix, defect breakdown
- Comparison with ResNet18 x64 (F1=0.495, AUROC=0.894, AUPRC=0.519)

---

### Experiment 2: Teacher-Student ResNet50 x224 Main

**Status**: Infrastructure ready, script-driven Modal runner with optional fresh-train

**Location**: `experiments/anomaly_detection/teacher_student/resnet50/x224/main/`

**Files**:
- `notebook.ipynb` - Created from x64 version, ready for execution
- `train_config.toml` - Configuration with image_size=224, batch_size=256, num_workers=8
- Modal app: `modal_apps/teacher_student_resnet50_x224_main/`
- Runner script: `scripts/run_ts_resnet50_x224_main_notebook.py`

**Run Command**:
```powershell
modal run --detach modal_apps/teacher_student_resnet50_x224_main/app.py::main --no-sync-back
```

**Force Fresh Retrain**:
```powershell
modal run --detach modal_apps/teacher_student_resnet50_x224_main/app.py::main --fresh-train --no-sync-back
```

**Download Artifacts**:
```powershell
modal run modal_apps/teacher_student_resnet50_x224_main/app.py::download_artifacts
```

**Expected Output**:
- Training checkpoints
- Evaluation metrics (F1, AUROC, AUPRC, per-defect breakdown)
- Plots: training curves, score distribution, threshold sweep, confusion matrix, defect breakdown
- Comparison with ResNet50 x64 (F1=0.488, AUROC=0.913, AUPRC=0.581)

**Current Modal Behavior**:
- default run reuses saved checkpoints/artifacts when available
- `--fresh-train` clears existing experiment artifacts and retrains from scratch
- notebook intent is preserved: reuse by default, retrain only when explicitly requested

---

### Experiment 3: Teacher-Student WideResNet50-2 x224 Multilayer

**Status**: Infrastructure ready, script-driven Modal runner verified, detached run + later artifact download supported

**Location**: `experiments/anomaly_detection/teacher_student/wideresnet50_2/x224/multilayer_self_contained/`

**Files**:
- `notebook.ipynb` - multilayer teacher-student notebook with artifact-backed outputs
- `train_config.toml` - experiment-side config reference
- `data_config.toml` - experiment-side data config reference
- Modal app: `modal_apps/teacher_student_wrn50_x224_multilayer/`
- Runner script: `scripts/run_ts_wrn50_x224_multilayer_notebook.py`

**Run Command**:
```powershell
modal run --detach modal_apps/teacher_student_wrn50_x224_multilayer/app.py::main --no-sync-back
```

**Download Artifacts**:
```powershell
modal run modal_apps/teacher_student_wrn50_x224_multilayer/app.py::download_artifacts
```

**Expected Output**:
- Training checkpoints under `artifacts/ts_wideresnet50_multilayer/checkpoints/`
- Evaluation outputs under `artifacts/ts_wideresnet50_multilayer/results/`
- Default evaluation summary under `artifacts/ts_wideresnet50_multilayer/results/evaluation/summary.json`
- Plots under `artifacts/ts_wideresnet50_multilayer/plots/`
- A reproducibility snapshot under `artifacts/ts_wideresnet50_multilayer/results/config.json`

**Current Modal Behavior**:
- detached/background execution works via `modal run --detach ... --no-sync-back`
- processed x224 data is cached in the shared Modal processed-data volume and reused when valid
- artifacts can be downloaded later back into the local experiment `artifacts/` folder
- the notebook saves a JSON config snapshot into `results/config.json`
- the original `train_config.toml` and `data_config.toml` are mounted into the run, but are not currently copied into the artifact bundle as TOML snapshots

---

## Teacher-Student Feature Autoencoder Dimension Sweep

### ResNet50 x224 Feature Autoencoder Dimension Sweep

**Status**: ✓ INFRASTRUCTURE READY (awaiting run)

**Location**: `experiments/anomaly_detection/teacher_student/resnet50/x224/feature_autoencoder_dim_sweep/`

**Motivation**: ResNet50 x224 shows -18.9% F1 degradation vs x64. Hypothesis: 128-dim bottleneck too small for 512-channel features at x224. Testing if larger bottlenecks (256, 512, 768 dims) can recover performance.

**Files**:
- `train_config.toml` - Base config (feature_autoencoder_hidden_dim overridden per sweep)
- `notebook.ipynb` - Analysis notebook that loads all results and compares across dimensions
- Modal app: `modal_apps/teacher_student_resnet50_x224_ae_dim_sweep/`
- Runner script: `scripts/run_ts_resnet50_x224_ae_dim_sweep_notebook.py`

**Run Command**:
```powershell
modal run modal_apps/teacher_student_resnet50_x224_ae_dim_sweep/app.py::main --no-sync-back --detach
```

**Download Artifacts**:
```powershell
modal run modal_apps/teacher_student_resnet50_x224_ae_dim_sweep/app.py::download_artifacts
```

**Dimensions Tested**: [64, 128, 256, 512, 768]

**Expected Output**:
- Per-dimension folders under `artifacts/ts_resnet50_x224_ae_dim_sweep/ae_dim_*/`
- Each folder contains: checkpoints, training history, evaluation metrics (F1, AUROC, AUPRC), plots
- Aggregated summary: `ae_dimension_sweep_summary.csv` with all dimensions compared

**Key Metrics**:
- Can ae_dim=256/512/768 match x64 baseline (F1=0.488)?
- Which dimension has best F1/AUROC/AUPRC trade-off?
- Is this a bottleneck problem or deeper architectural issue?

**Timeline**: ~5-6 hours total compute (1 hour per dimension on A10G)

**Research Question**: Is the fixed 128-dim autoencoder bottleneck the root cause of ResNet50 x224 degradation, or is there a deeper feature-alignment issue?

---

## PatchCore x224 Variants (Resolution Comparison Study)

These experiments are **fully set up and ready to run** via Modal. They will compare high-fidelity backbone embeddings at 224×224 resolution.

### Experiment 1: PatchCore EfficientNet-B0 x224 Main

**Status**: Infrastructure ready, notebook created, waiting for training

**Location**: `experiments/anomaly_detection/patchcore/efficientnet_b0/x224/main/`

**Files**:
- `notebook.ipynb` - Artifact-backed evaluation notebook
- Modal app: `modal_apps/patchcore_effb0_x224_main/`
- Runner script: `scripts/run_patchcore_effb0_x224_main_notebook.py`

**Run Command**:
```bash
modal run --detach modal_apps/patchcore_effb0_x224_main/app.py::main --no-sync-back
```

**Download Artifacts**:
```bash
modal run modal_apps/patchcore_effb0_x224_main/app.py::download_artifacts
```

**GPU & Batch Configuration**: A10G GPU, batch_size=128 (supports efficient batch processing)

**Expected Output**:
- Training artifacts (memory bank, embeddings)
- Evaluation metrics (F1, AUROC, AUPRC, per-defect breakdown)
- Plots: score distribution, threshold sweep, confusion matrix, defect breakdown
- Comparison with EfficientNet-B0 x64 baseline

**Research Question**: Does EfficientNet-B0's efficient architecture preserve spatial detail at 224×224 compared to downsampled 64×64?

---

### Experiment 2: PatchCore ViT-B/16 x224 Main

**Status**: Infrastructure ready, notebook created, waiting for training

**Location**: `experiments/anomaly_detection/patchcore/vit_b16/x224/main/`

**Files**:
- `notebook.ipynb` - Artifact-backed evaluation notebook with UMAP analysis
- Modal app: `modal_apps/patchcore_vit_b16_x224_main/`
- Runner script: `scripts/run_patchcore_vit_b16_x224_main_notebook.py`

**Run Command**:
```bash
modal run --detach modal_apps/patchcore_vit_b16_x224_main/app.py::main --no-sync-back
```

**Download Artifacts**:
```bash
modal run modal_apps/patchcore_vit_b16_x224_main/app.py::download_artifacts
```

**GPU & Batch Configuration**: A10G GPU, batch_size=128 (native ViT resolution support)

**Expected Output**:
- Training artifacts (memory bank, embeddings)
- Evaluation metrics (F1, AUROC, AUPRC, per-defect breakdown)
- UMAP embeddings and visualization
- Plots: score distribution, threshold sweep, confusion matrix, defect breakdown
- Comparison with ViT-B/16 x64 baseline

**Research Question**: Does Vision Transformer's attention mechanism better preserve spatial detail at 224×224? Can ViT's patch embeddings improve anomaly localization?

---

## Research Questions These Experiments Address

1. **Impact of Input Resolution**: Does using the teacher's native 224×224 resolution improve anomaly detection over downsampled 64×64?

2. **Feature Extraction**: Can ResNet backbones extract more discriminative features at their intended resolution?

3. **Computational Trade-off**: What's the performance gain vs. training/inference time trade-off?

4. **Generalization**: Do x224 models better preserve ImageNet pretraining benefits?

---

## Integration with Family Report

Once both x224 experiments complete, update `family_reports/teacher_student.md`:

1. Add results section for ResNet18 x224 with metrics table
2. Add results section for ResNet50 x224 with metrics table
3. Update family-wide comparison table to include x224 variants
4. Add plots (training curves, score distribution, threshold sweep, etc.)
5. Update context section to discuss resolution impact

---

## Timeline Notes

- **Setup Complete**: 2026-03-31
- **Training Duration**: ~2-3 hours each via Modal A10G GPU
- **Total Compute**: ~4-6 hours for both experiments
- **Artifact Download**: ~10-15 minutes per experiment
- **Report Update**: ~30 minutes

---

## Recommended Execution Order

1. **Autoencoder x224 Main** (~2-2.5 hours, A10G, batch_size=128) — Quick resolution comparison for AE family, tests if reconstruction-based scoring is limited by resolution
2. **VAE x224 Main** (~2-2.5 hours, A10G, batch_size=128) — Quick resolution comparison for VAE family
3. **VAE Latent Dim Sweep** (~4-6 hours, A10G, batch_size=128, 5 models) — Answers if KL compression is the bottleneck
4. **Teacher-Student ResNet18 x224** (~2-2.5 hours, A10G, batch_size=128) — Resolution comparison for TS family
5. **Teacher-Student ResNet50 x224** (~2-2.5 hours, A10G, batch_size=128) — Completes TS x224 variants
6. **PatchCore EfficientNet-B0 x224** (~2-2.5 hours, A10G, batch_size=128) — Efficient backbone x224 resolution study
7. **PatchCore ViT-B/16 x224** (~2-2.5 hours, A10G, batch_size=128) — Vision Transformer x224 resolution study

**Total compute time**: ~17-22 hours (can run multiple experiments in parallel across GPUs)

**Priority grouping**:
- **Tier 1 (Resolution across families)**: AE x224, VAE x224, TS ResNet18/50 x224, PatchCore EfficientNet-B0/ViT-B/16 x224 — Can run in parallel
- **Tier 2 (Hyperparameter sweep)**: VAE latent_dim_sweep — Builds on VAE x224

---

## Checklist for Running

Before running these experiments:

- [ ] Repository has been cleaned up/finalized
- [ ] All x64 experiments are merged/committed
- [ ] Modal credentials are active (`modal setup` completed)
- [ ] LSWMD.pkl is in shared Modal raw-data volume
- [ ] Sufficient Modal compute credits available
- [ ] Network connection stable

---

## Related Documentation

- Modal app READMEs:
  - `modal_apps/autoencoder_x224_main/README.md`
  - `modal_apps/vae_x224_main/README.md`
  - `modal_apps/vae_latent_dim_sweep/README.md`
  - `modal_apps/teacher_student_resnet18_x224_main/README.md`
  - `modal_apps/teacher_student_resnet50_x224_main/README.md`
  - `modal_apps/teacher_student_wrn50_x224_multilayer/README.md`
  - `modal_apps/patchcore_effb0_x224_main/README.md`
  - `modal_apps/patchcore_vit_b16_x224_main/README.md`
- Runner scripts:
  - `scripts/run_autoencoder_x224_main_notebook.py`
  - `scripts/run_vae_x224_main_notebook.py`
  - `scripts/run_vae_latent_dim_sweep_notebook.py`
  - `scripts/run_ts_resnet18_x224_main_notebook.py`
  - `scripts/run_ts_resnet50_x224_main_notebook.py`
  - `scripts/run_ts_wrn50_x224_multilayer_notebook.py`
  - `scripts/run_patchcore_effb0_x224_main_notebook.py`
  - `scripts/run_patchcore_vit_b16_x224_main_notebook.py`
- Current family reports:
  - `family_reports/autoencoder.md`
  - `family_reports/vae.md`
  - `family_reports/teacher_student.md`
  - `family_reports/patchcore.md`
