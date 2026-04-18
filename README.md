# Automated IC Wafer Defect Detection — WM-811K

**Group 08** · Henry Lee Jun · Chia Tang · Genson Low

## Quick Start

1. Create and activate the environment, then install the project: `py -3.11 -m venv .venv && .venv\Scripts\Activate.ps1 && pip install -e .`
2. Place the raw WM-811K pickle at `data/raw/LSWMD.pkl`.
3. Build the processed dataset by running notebook in data/dataset/ 
    For example [notebook](data/dataset/x64/benchmark_50k_5pct/notebook.ipynb).
4. Download the artifact bundle from https://sutdapac-my.sharepoint.com/:f:/g/personal/henry_lee_mymail_sutd_edu_sg/IgDoXYip5GTZS68DEXWup5lIAcflFy-6c0__vcd4uaSOZOA?e=ljJtk6 and replace the matching folder under `artifacts/` with the extracted contents.
5. Open the experiment notebook you want to inspect or rerun. For the top-10 table, start with [experiments/anomaly_detection/patchcore/dinov2_vit_b14/x224/ensemble.ipynb](experiments/anomaly_detection/patchcore/dinov2_vit_b14/x224/ensemble.ipynb).

If an experiment already has saved outputs, set `RETRAIN=False` in the notebook to load the stored artifacts instead of retraining.
---

## Top 10 Models

Ranked by val-threshold F1 (primary metric). All metrics from the main benchmark (5k normal / 250 anomaly test set).

| Rank | Model | F1 | AUROC | AUPRC | Notebook |
|---|---|---|---|---|---|
| 1 | Ensemble: ViT-B/16 + DINOv2 ViT-B/14 (max-fusion) | 0.623 | 0.967 | 0.716 | [notebook](experiments/anomaly_detection/patchcore/dinov2_vit_b14/x224/ensemble.ipynb) |
| 2 | PatchCore + ViT-B/16 (block 6, 224×224) | 0.595 | 0.956 | 0.671 | [notebook](experiments/anomaly_detection/patchcore/vit_b16/x224/main/notebook.ipynb) |
| 3 | Ensemble: ViT-B/16 + DINOv2 ViT-B/14 (Mahalanobis) | 0.592 | 0.968 | 0.762 | [notebook](experiments/anomaly_detection/patchcore/dinov2_vit_b14/x224/ensemble.ipynb) |
| 4 | PatchCore + EfficientNet-B1 (block 3, 240×240) | 0.591 | 0.935 | 0.609 | [notebook](experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main_one_layer/notebook.ipynb) |
| 5 | Ensemble: ViT-B/16 + EfficientNet-B1 (80/20) | 0.576 | 0.952 | 0.643 | [notebook](experiments/anomaly_detection/ensemble/x224/vit_effnetb1_ensemble/notebook.ipynb) |
| 6 | PatchCore + WideResNet50-2 (layer2+3, 224×224) | 0.549 | 0.931 | 0.659 | [notebook](experiments/anomaly_detection/patchcore/wideresnet50/x224/multilayer_umap/notebook.ipynb) |
| 7 | PatchCore + EfficientNet-B0 (blocks 3+6, 224×224) | 0.544 | 0.925 | 0.483 | [notebook](experiments/anomaly_detection/patchcore/efficientnet_b0/x224/main/notebook.ipynb) |
| 8 | PatchCore + WideResNet50-2 (layer2+3, 64×64, r=0.10) | 0.532 | 0.917 | 0.562 | [notebook](experiments/anomaly_detection/patchcore/wideresnet50/x64/main/notebook.ipynb) |
| 9 | Score Ensemble (PatchCore-WRN50-x64 + TS-Res50-x64) | 0.529 | 0.916 | 0.611 | [notebook](experiments/anomaly_detection/ensemble/x64/score_ensemble/notebook.ipynb) |
| 10 | PatchCore + WideResNet50-2 (layer2+3, 64×64, r=0.15) | 0.526 | 0.912 | 0.549 | [notebook](experiments/anomaly_detection/patchcore/wideresnet50/x64/main/notebook.ipynb) |

> Ranks 1 and 3 use the same ensemble notebook. Ranks 8 and 10 are sweep variants within the same notebook — set `RETRAIN=False` to load saved results for all variants.

---

## Per Family

### Autoencoder

| # | Model | F1 | AUROC | AUPRC | Notebook |
|---|---|---|---|---|---|
| 1 | AE x64 Baseline (Trainable) | 0.468 | 0.839 | 0.522 | [notebook](experiments/anomaly_detection/autoencoder/x64/baseline/notebook.ipynb) |
| 2 | AE x64 + BatchNorm (Trainable) | 0.502 | 0.834 | 0.568 | [notebook](experiments/anomaly_detection/autoencoder/x64/batchnorm/notebook.ipynb) |
| 3 | AE x64 + BatchNorm + Dropout sweep (Trainable) | 0.487 | 0.851 | 0.617 | [notebook](experiments/anomaly_detection/autoencoder/x64/batchnorm_dropout/notebook.ipynb) |
| 4 | AE x64 Residual (Trainable) | 0.474 | 0.843 | 0.589 | [notebook](experiments/anomaly_detection/autoencoder/x64/residual/notebook.ipynb) |
| 5 | AE x128 Baseline (Trainable) | 0.431 | 0.815 | 0.455 | [notebook](experiments/anomaly_detection/autoencoder/x128/baseline/notebook.ipynb) |
| 6 | AE x224 Main (Trainable) | 0.510 | 0.901 | 0.596 | [notebook](experiments/anomaly_detection/autoencoder/x224/main/notebook.ipynb) |

### VAE

| # | Model | F1 | AUROC | AUPRC | Notebook |
|---|---|---|---|---|---|
| 1 | VAE x64 Baseline (Trainable) | 0.340 | 0.771 | 0.372 | [notebook](experiments/anomaly_detection/vae/x64/baseline/notebook.ipynb) |
| 2 | VAE x64 Beta Sweep (Trainable) | 0.343 | 0.770 | 0.336 | [notebook](experiments/anomaly_detection/vae/x64/beta_sweep/notebook.ipynb) |
| 3 | VAE x64 Latent Dim Sweep (Trainable) | 0.357 | 0.776 | 0.389 | [notebook](experiments/anomaly_detection/vae/x64/latent_dim_sweep/notebook.ipynb) |
| 4 | VAE x224 Main (Trainable) | 0.339 | 0.772 | 0.362 | [notebook](experiments/anomaly_detection/vae/x224/main/notebook.ipynb) |

> Sweep rows show best result. Set `RETRAIN=False` to load saved results for all variants.

### SVDD

| # | Model | F1 | AUROC | AUPRC | Notebook |
|---|---|---|---|---|---|
| 1 | Deep SVDD x64 (Trainable) | 0.360 | 0.788 | 0.213 | [notebook](experiments/anomaly_detection/svdd/x64/baseline/notebook.ipynb) |

### Backbone Embedding

| # | Model | F1 | AUROC | AUPRC | Notebook |
|---|---|---|---|---|---|
| 1 | Backbone Embedding ResNet18 x64 | 0.236 | 0.685 | 0.195 | [notebook](experiments/anomaly_detection/backbone_embedding/resnet18/x64/baseline/notebook.ipynb) |
| 2 | Backbone Embedding WideResNet50-2 x64 | 0.243 | 0.677 | 0.142 | [notebook](experiments/anomaly_detection/backbone_embedding/wide_resnet50_2/x64/baseline/notebook.ipynb) |

### Teacher-Student

| # | Model | F1 | AUROC | AUPRC | Notebook |
|---|---|---|---|---|---|
| 1 | TS ResNet18 x64 (Trainable) | 0.495 | 0.894 | 0.519 | [notebook](experiments/anomaly_detection/teacher_student/resnet18/x64/main/notebook.ipynb) |
| 2 | TS ResNet18 x224 (Trainable) *(missing artifacts — set RETRAIN=True)* | 0.511 | 0.907 | 0.503 | [notebook](experiments/anomaly_detection/teacher_student/resnet18/x224/main/notebook.ipynb) |
| 3 | TS ResNet50 x64 Main (Trainable) | 0.525 | 0.909 | 0.599 | [notebook](experiments/anomaly_detection/teacher_student/resnet50/x64/main/notebook.ipynb) |
| 4 | TS ResNet50 x64 Layer Ablation (Trainable) *(10-epoch run; see note)* | 0.520 | 0.877 | 0.550 | [notebook](experiments/anomaly_detection/teacher_student/resnet50/x64/layer_ablation/notebook.ipynb) |
| 5 | TS ResNet50 x224 Main (Trainable) | 0.399 | 0.828 | 0.361 | [notebook](experiments/anomaly_detection/teacher_student/resnet50/x224/main/notebook.ipynb) |
| 6 | TS ResNet50 x224 Feature AE Dim Sweep (Trainable) *(10-epoch run; see note)* | 0.362 | 0.813 | 0.331 | [notebook](experiments/anomaly_detection/teacher_student/resnet50/x224/feature_autoencoder_dim_sweep/notebook.ipynb) |
| 7 | TS ViT-B/16 x224 (Trainable) *(10-epoch run; see note)* | 0.051 | 0.606 | 0.064 | [notebook](experiments/anomaly_detection/teacher_student/vit_b16/x224/main/notebook.ipynb) |
| 8 | TS WideResNet50-2 x64 Layer2 (Trainable) | 0.508 | 0.920 | 0.540 | [notebook](experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/layer2_self_contained/notebook.ipynb) |
| 9 | TS WideResNet50-2 x64 Multilayer (Trainable) | 0.524 | 0.923 | 0.546 | [notebook](experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/multilayer_self_contained/notebook.ipynb) |
| 10 | TS WideResNet50-2 x224 Multilayer (Trainable) | 0.315 | 0.789 | 0.269 | [notebook](experiments/anomaly_detection/teacher_student/wideresnet50_2/x224/multilayer_self_contained/notebook.ipynb) |

- Row 3 shows best sweep result (topk_mean r=0.20).
- Rows 4, 6, and 7 were retrained for 10 epochs (quick mode); metrics are lower than the original full runs.
- Row 7 (ViT-B/16) is a known failure regardless of training duration due to architectural mismatch between ViT and the CNN student; the original 30-epoch run also only reached F1=0.163.

### PatchCore

| # | Model | F1 | AUROC | AUPRC | Notebook |
|---|---|---|---|---|---|
| 1 | PatchCore AE-BN Backbone x64 (Trainable) | 0.336 | 0.851 | 0.226 | [notebook](experiments/anomaly_detection/patchcore/ae_bn/x64/main/notebook.ipynb) |
| 2 | PatchCore ResNet18 x64 | 0.401 | 0.842 | 0.411 | [notebook](experiments/anomaly_detection/patchcore/resnet18/x64/main/notebook.ipynb) |
| 3 | PatchCore ResNet50 x64 | 0.420 | 0.821 | 0.363 | [notebook](experiments/anomaly_detection/patchcore/resnet50/x64/main/notebook.ipynb) |
| 4 | PatchCore WideResNet50-2 x64 | 0.532 | 0.917 | 0.562 | [notebook](experiments/anomaly_detection/patchcore/wideresnet50/x64/main/notebook.ipynb) |
| 5 | PatchCore WideResNet50-2 x64 Labeled 120k | ~0.480 | — | — | [notebook](experiments/anomaly_detection/patchcore/wideresnet50/x64/labeled_120k/notebook.ipynb) |
| 6 | PatchCore WideResNet50-2 x224 Layer2 | 0.512 | 0.918 | 0.623 | [notebook](experiments/anomaly_detection/patchcore/wideresnet50/x224/layer2/notebook.ipynb) |
| 7 | PatchCore WideResNet50-2 x224 Layer3 | 0.498 | 0.906 | 0.595 | [notebook](experiments/anomaly_detection/patchcore/wideresnet50/x224/layer3/notebook.ipynb) |
| 8 | PatchCore WideResNet50-2 x224 Multilayer | 0.549 | 0.931 | 0.659 | [notebook](experiments/anomaly_detection/patchcore/wideresnet50/x224/multilayer_umap/notebook.ipynb) |
| 9 | PatchCore WideResNet50-2 x224 Weighted | 0.528 | 0.928 | 0.654 | [notebook](experiments/anomaly_detection/patchcore/wideresnet50/x224/weighted/notebook.ipynb) |
| 10 | PatchCore EfficientNet-B0 x224 | 0.544 | 0.925 | 0.483 | [notebook](experiments/anomaly_detection/patchcore/efficientnet_b0/x224/main/notebook.ipynb) |
| 11 | PatchCore EfficientNet-B1 x240 (one layer) | 0.591 | 0.935 | 0.609 | [notebook](experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main_one_layer/notebook.ipynb) |
| 12 | PatchCore EfficientNet-B1 x240 (layer 3+5) | 0.562 | 0.929 | 0.592 | [notebook](experiments/anomaly_detection/patchcore/efficientnet_b1/x240/layer3_5/notebook.ipynb) |
| 13 | PatchCore EfficientNet-B1 x240 (layer 3+5, no defect tuning) | 0.541 | 0.920 | 0.563 | [notebook](experiments/anomaly_detection/patchcore/efficientnet_b1/x240/layer3_5_no_defect_tuning/notebook.ipynb) |
| 14 | PatchCore ViT-B/16 x224 Main | 0.595 | 0.956 | 0.671 | [notebook](experiments/anomaly_detection/patchcore/vit_b16/x224/main/notebook.ipynb) |
| 15 | PatchCore ViT-B/16 x224 Block Depth Sweep (best: block 6) | 0.580 | 0.954 | 0.642 | [notebook](experiments/anomaly_detection/patchcore/vit_b16/x224/block_depth_sweep/notebook.ipynb) |
| 16 | PatchCore ViT-B/16 x224 One Layer Defect Tuning | 0.595 | 0.956 | 0.671 | [notebook](experiments/anomaly_detection/patchcore/vit_b16/x224/one_layer_defect_tuning/notebook.ipynb) |
| 17 | PatchCore ViT-B/16 x224 One Layer No Defect Tuning | 0.573 | 0.953 | 0.643 | [notebook](experiments/anomaly_detection/patchcore/vit_b16/x224/one_layer_no_defect_tuning/notebook.ipynb) |
| 18 | PatchCore ViT-B/16 x224 Two Block | 0.569 | 0.950 | 0.628 | [notebook](experiments/anomaly_detection/patchcore/vit_b16/x224/two_block/notebook.ipynb) |
| 19 | PatchCore ViT-B/16 x224 Two Block No Defect Tuning | 0.551 | 0.943 | 0.602 | [notebook](experiments/anomaly_detection/patchcore/vit_b16/x224/two_block_no_defect_tuning/notebook.ipynb) |
| 20 | PatchCore ViT-B/16 x64 | 0.342 | 0.832 | 0.348 | [notebook](experiments/anomaly_detection/patchcore/vit_b16/x64/main/notebook.ipynb) |
| 21 | PatchCore DINOv2 ViT-B/14 x224 (block sweep, best: block 6) | 0.521 | 0.926 | 0.549 | [notebook](experiments/anomaly_detection/patchcore/dinov2_vit_b14/x224/notebook.ipynb) |

> Row 4 shows best sweep result (r=0.10). Row 5 F1 is approximate (~0.48, AUROC/AUPRC not recorded). PatchCore AE-BN is trainable; all others use frozen pretrained backbones.

### RD4AD

| # | Model | F1 | AUROC | AUPRC | Notebook |
|---|---|---|---|---|---|
| 1 | RD4AD WideResNet50-2 x224 (Trainable) | 0.477 | 0.877 | 0.414 | [notebook](experiments/anomaly_detection/rd4ad/wideresnet50/x224/main/notebook.ipynb) |

### FastFlow

| # | Model | F1 | AUROC | AUPRC | Notebook |
|---|---|---|---|---|---|
| 1 | FastFlow x64 sweep (Trainable) | 0.482 | 0.871 | 0.489 | [notebook](experiments/anomaly_detection/fastflow/x64/main/notebook.ipynb) |

> Shows best sweep result (WRN50-2 layer2+3, 4 flow steps).

### Ensemble

| # | Model | F1 | AUROC | AUPRC | Notebook |
|---|---|---|---|---|---|
| 1 | Ensemble: ViT-B/16 + DINOv2 ViT-B/14 (max-fusion) | 0.623 | 0.967 | 0.716 | [notebook](experiments/anomaly_detection/patchcore/dinov2_vit_b14/x224/ensemble.ipynb) |
| 2 | Ensemble: ViT-B/16 + DINOv2 ViT-B/14 (Mahalanobis) | 0.592 | 0.968 | 0.762 | [notebook](experiments/anomaly_detection/patchcore/dinov2_vit_b14/x224/ensemble.ipynb) |
| 3 | Ensemble: ViT-B/16 + EfficientNet-B1 (80/20) | 0.576 | 0.952 | 0.643 | [notebook](experiments/anomaly_detection/ensemble/x224/vit_effnetb1_ensemble/notebook.ipynb) |
| 4 | Score Ensemble x64 (PatchCore-WRN50 + TS-Res50) | 0.529 | 0.916 | 0.611 | [notebook](experiments/anomaly_detection/ensemble/x64/score_ensemble/notebook.ipynb) |

### Report Figures

[experiments/anomaly_detection/report_figures/](experiments/anomaly_detection/report_figures/)

### Supervised Multiclass Classifier

| # | Model | Notebook |
|---|---|---|
| 1 | Multiclass Classifier x64 Training (Trainable) | [notebook](experiments/classifier/multiclass/x64/training/notebook.ipynb) |
| 2 | Multiclass Classifier x64 Seed 07 (Trainable) | [notebook](experiments/classifier/multiclass/x64/seed07/notebook.ipynb) |
| 3 | Multiclass Classifier x64 Final Labeling | [notebook](experiments/classifier/multiclass/x64/final_labeling/notebook.ipynb) |
| 4 | Multiclass Classifier x64 Showcase (Trainable) | [notebook](experiments/classifier/multiclass/x64/showcase/notebook.ipynb) |

### Supervised Defect Detection

| # | Model | Notebook |
|---|---|---|
| 1 | Supervised Sweep ViT-B/16 x224 (Trainable) | [notebook](experiments/anomaly_detection_defect/supervised_sweep/vit_b16/x224/main/notebook.ipynb) |
| 2 | Supervised CNN Full Defect | [notebook](experiments/anomaly_detection_defect/supervised_cnn/full_defect/notebook.ipynb) |
| 3 | Supervised CNN Half Defect | [notebook](experiments/anomaly_detection_defect/supervised_cnn/half_defect/notebook.ipynb) |

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
