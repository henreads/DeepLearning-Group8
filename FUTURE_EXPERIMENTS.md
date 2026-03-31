# Future Experiments to Run

This document tracks experiments that should be executed after repository cleanup and finalization.

## Autoencoder Family Enhancements

### Experiment 1: Autoencoder x224 Main (Resolution Study)

**Status**: Infrastructure ready, notebook structure prepared, waiting for training

**Location**: `experiments/anomaly_detection/autoencoder/x224/main/`

**Files**:
- `train_config.toml` - Configuration with image_size=224, batch_size=128 (A10G GPU)
- Modal app: `modal_apps/autoencoder_x224_main/`
- Runner script: `scripts/run_autoencoder_x224_main_notebook.py`

**Run Command**:
```bash
modal run --detach modal_apps/autoencoder_x224_main/app.py::main --no-sync-back
```

**Download Artifacts**:
```bash
modal run modal_apps/autoencoder_x224_main/app.py::download_artifacts
```

**Expected Output**:
- Training checkpoints and history
- Evaluation metrics (F1, AUROC, AUPRC)
- Plots: training curves, score distribution, threshold sweep, confusion matrix, reconstruction examples
- Failure analysis and per-defect breakdown
- Comparison with autoencoder x64 baseline (F1=0.467, AUROC=0.839, AUPRC=0.522)
- Comparison with AE x64 + BatchNorm (best AE variant: F1=0.502, AUROC=0.834, AUPRC=0.568)

**Research Question**: Does using higher resolution (224×224) help autoencoder preserve spatial detail and improve anomaly detection over downsampled 64×64? Is reconstruction-based scoring bottlenecked by information loss at low resolution?

---

## VAE Family Enhancements

### Experiment 1: VAE x224 Main (Resolution Study)

**Status**: Infrastructure ready, notebook structure prepared, waiting for training

**Location**: `experiments/anomaly_detection/vae/x224/main/`

**Files**:
- `train_config.toml` - Configuration with image_size=224, batch_size=128 (A10G GPU)
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

**Status**: Infrastructure ready, notebook structure prepared, waiting for training

**Location**: `experiments/anomaly_detection/vae/x64/latent_dim_sweep/`

**Files**:
- `train_config.toml` - Configuration with latent_dims=[32, 64, 128, 256, 512]
- Modal app: `modal_apps/vae_latent_dim_sweep/`
- Runner script: `scripts/run_vae_latent_dim_sweep_notebook.py`

**Run Command**:
```powershell
modal run --detach modal_apps/vae_latent_dim_sweep/app.py::main --no-sync-back
```

**Download Artifacts**:
```powershell
modal run modal_apps/vae_latent_dim_sweep/app.py::download_artifacts
```

**Expected Output**:
- Per-latent-dim checkpoints and histories
- Evaluation metrics and sweep summaries
- Plots: latent dimension comparison, training curves across dims, best-dim confusion matrix
- Comparison with VAE baseline latent_dim=128 (F1=0.418, AUROC=0.772, AUPRC=0.372)

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

**Status**: Infrastructure ready, notebook created, waiting for training

**Location**: `experiments/anomaly_detection/teacher_student/resnet50/x224/main/`

**Files**:
- `notebook.ipynb` - Created from x64 version, ready for execution
- `train_config.toml` - Configuration with image_size=224
- Modal app: `modal_apps/teacher_student_resnet50_x224_main/`

**Run Command**:
```powershell
modal run --detach modal_apps/teacher_student_resnet50_x224_main/app.py::main --no-sync-back
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
  - `modal_apps/patchcore_effb0_x224_main/README.md`
  - `modal_apps/patchcore_vit_b16_x224_main/README.md`
- Runner scripts:
  - `scripts/run_autoencoder_x224_main_notebook.py`
  - `scripts/run_vae_x224_main_notebook.py`
  - `scripts/run_vae_latent_dim_sweep_notebook.py`
  - `scripts/run_ts_resnet18_x224_main_notebook.py`
  - `scripts/run_ts_resnet50_x224_main_notebook.py`
  - `scripts/run_patchcore_effb0_x224_main_notebook.py`
  - `scripts/run_patchcore_vit_b16_x224_main_notebook.py`
- Current family reports:
  - `family_reports/autoencoder.md`
  - `family_reports/vae.md`
  - `family_reports/teacher_student.md`
  - `family_reports/patchcore.md`
