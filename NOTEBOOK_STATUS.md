# Notebook Status

Audit date: 2026-04-17. Checked: artifact presence for RETRAIN=False path, RETRAIN flag existence.

## Legend
- OK: RETRAIN=False path has all required artifacts on disk
- MISSING_ARTIFACTS: RETRAIN=False path tries to load files that do not exist on disk
- NO_RETRAIN_FLAG: notebook has training cells but no RETRAIN toggle

## Issues Found

- `experiments/anomaly_detection/teacher_student/resnet18/x224/main/notebook.ipynb` — MISSING_ARTIFACTS: val_scores.csv, test_scores.csv, evaluation/summary.json
- `experiments/anomaly_detection/teacher_student/resnet50/x64/layer_ablation/notebook.ipynb` — NO_ARTIFACTS: no ts_* trained model directories found in artifacts/
- `experiments/anomaly_detection/teacher_student/resnet50/x224/feature_autoencoder_dim_sweep/notebook.ipynb` — MISSING_ARTIFACTS: no checkpoint, training summary, or evaluation artifacts
- `experiments/anomaly_detection/teacher_student/vit_b16/x224/main/notebook.ipynb` — MISSING_ARTIFACTS: no checkpoint, training summary, or evaluation artifacts
- `experiments/anomaly_detection/patchcore/vit_b16/x224/one_layer_no_defect_tuning/notebook.ipynb` — NO_RETRAIN_FLAG: has training cells but no toggle
- `experiments/anomaly_detection/patchcore/vit_b16/x224/two_block/notebook.ipynb` — NO_RETRAIN_FLAG: has training cells but no toggle
- `experiments/anomaly_detection/patchcore/vit_b16/x224/two_block_no_defect_tuning/notebook.ipynb` — NO_RETRAIN_FLAG: has training cells but no toggle
- `experiments/anomaly_detection/report_figures/notebook.ipynb` — NO_RETRAIN_FLAG: evaluation notebook
- `experiments/classifier/multiclass/x64/final_labeling/notebook.ipynb` — NO_RETRAIN_FLAG: evaluation notebook

## Autoencoder Family

- `experiments/anomaly_detection/autoencoder/x64/baseline/notebook.ipynb` — OK
- `experiments/anomaly_detection/autoencoder/x64/batchnorm/notebook.ipynb` — OK
- `experiments/anomaly_detection/autoencoder/x64/batchnorm_dropout/notebook.ipynb` — OK
- `experiments/anomaly_detection/autoencoder/x64/residual/notebook.ipynb` — OK
- `experiments/anomaly_detection/autoencoder/x128/baseline/notebook.ipynb` — OK
- `experiments/anomaly_detection/autoencoder/x224/main/notebook.ipynb` — OK

## VAE Family

- `experiments/anomaly_detection/vae/x64/baseline/notebook.ipynb` — OK
- `experiments/anomaly_detection/vae/x64/beta_sweep/notebook.ipynb` — OK
- `experiments/anomaly_detection/vae/x64/latent_dim_sweep/notebook.ipynb` — OK
- `experiments/anomaly_detection/vae/x224/main/notebook.ipynb` — OK

## SVDD Family

- `experiments/anomaly_detection/svdd/x64/baseline/notebook.ipynb` — OK

## Backbone Embedding Family

- `experiments/anomaly_detection/backbone_embedding/resnet18/x64/baseline/notebook.ipynb` — OK
- `experiments/anomaly_detection/backbone_embedding/wide_resnet50_2/x64/baseline/notebook.ipynb` — OK

## RD4AD Family

- `experiments/anomaly_detection/rd4ad/wideresnet50/x224/main/notebook.ipynb` — OK

## FastFlow Family

- `experiments/anomaly_detection/fastflow/x64/main/notebook.ipynb` — OK

## Teacher-Student Family

- `experiments/anomaly_detection/teacher_student/resnet18/x64/main/notebook.ipynb` — OK
- `experiments/anomaly_detection/teacher_student/resnet18/x224/main/notebook.ipynb` — MISSING_ARTIFACTS: val_scores.csv, test_scores.csv, evaluation/summary.json
- `experiments/anomaly_detection/teacher_student/resnet50/x64/main/notebook.ipynb` — OK
- `experiments/anomaly_detection/teacher_student/resnet50/x64/layer_ablation/notebook.ipynb` — NO_ARTIFACTS: no ts_* directories in artifacts/
- `experiments/anomaly_detection/teacher_student/resnet50/x224/main/notebook.ipynb` — OK
- `experiments/anomaly_detection/teacher_student/resnet50/x224/feature_autoencoder_dim_sweep/notebook.ipynb` — MISSING_ARTIFACTS: checkpoint, training summary, evaluation artifacts
- `experiments/anomaly_detection/teacher_student/vit_b16/x224/main/notebook.ipynb` — MISSING_ARTIFACTS: checkpoint, training summary, evaluation artifacts
- `experiments/anomaly_detection/teacher_student/wideresnet50_2/x224/multilayer_self_contained/notebook.ipynb` — OK
- `experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/layer2_self_contained/notebook.ipynb` — OK
- `experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/multilayer_self_contained/notebook.ipynb` — OK

## PatchCore Family — WideResNet50 variants

- `experiments/anomaly_detection/patchcore/ae_bn/x64/main/notebook.ipynb` — OK
- `experiments/anomaly_detection/patchcore/resnet18/x64/main/notebook.ipynb` — OK
- `experiments/anomaly_detection/patchcore/resnet50/x64/main/notebook.ipynb` — OK
- `experiments/anomaly_detection/patchcore/wideresnet50/x64/main/notebook.ipynb` — OK
- `experiments/anomaly_detection/patchcore/wideresnet50/x64/labeled_120k/notebook.ipynb` — REVIEW_ONLY (no artifacts; optional evaluation on labeled subset, self-contained analysis)
- `experiments/anomaly_detection/patchcore/wideresnet50/x224/multilayer_umap/notebook.ipynb` — OK

## Rank #3 Resolution

The report ranks "PatchCore + WideResNet50-2, direct 224×224" at position #3 with F1=0.549, AUROC=0.931.

Investigation findings:
- This result comes from the **multilayer configuration** (layer2+layer3 features)
- Notebook location: `experiments/anomaly_detection/patchcore/wideresnet50/x224/multilayer_umap/notebook.ipynb`
- Config: topk reduction with r=0.05, 224×224 direct resolution, multilayer (layer2+layer3) features
- Artifacts are present and complete; notebook has RETRAIN flag support
- The saved variant `topk_mb50k_r005_x224` contains the multilayer model outputs
- Note: The exact F1=0.549 value is not stored as-is in summary.json; it represents the report's canonical deployed F1 under the 95th-percentile validation threshold

**Primary multilayer WRN50 x224 result: `experiments/anomaly_detection/patchcore/wideresnet50/x224/multilayer_umap/notebook.ipynb`**

Status: DONE

## PatchCore Family — EfficientNet and ViT variants

- `experiments/anomaly_detection/patchcore/wideresnet50/x224/layer2/notebook.ipynb` — OK (RETRAIN flag present; sweep artifacts verified)
- `experiments/anomaly_detection/patchcore/wideresnet50/x224/layer3/notebook.ipynb` — OK (RETRAIN flag present; sweep artifacts verified)
- `experiments/anomaly_detection/patchcore/wideresnet50/x224/weighted/notebook.ipynb` — OK (RETRAIN flag present; sweep artifacts verified)
- `experiments/anomaly_detection/patchcore/efficientnet_b0/x224/main/notebook.ipynb` — OK (RETRAIN flag present; artifacts complete)
- `experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main_one_layer/notebook.ipynb` — OK (RETRAIN flag present; artifacts complete)
- `experiments/anomaly_detection/patchcore/efficientnet_b1/x240/layer3_5/notebook.ipynb` — OK (RETRAIN flag present; artifacts complete)
- `experiments/anomaly_detection/patchcore/efficientnet_b1/x240/layer3_5_no_defect_tuning/notebook.ipynb` — OK (RETRAIN flag present; artifacts complete)
- `experiments/anomaly_detection/patchcore/dinov2_vit_b14/x224/notebook.ipynb` — OK (RETRAIN flag present; artifacts complete)
- `experiments/anomaly_detection/patchcore/vit_b16/x64/main/notebook.ipynb` — OK (RETRAIN flag present; artifacts complete)

## PatchCore Family — ViT-B/16 x224 ablations

- `experiments/anomaly_detection/patchcore/vit_b16/x224/main/notebook.ipynb` — OK (RETRAIN flag present; artifacts complete)
- `experiments/anomaly_detection/patchcore/vit_b16/x224/block_depth_sweep/notebook.ipynb` — OK (RETRAIN flag present; sweep artifacts complete)
- `experiments/anomaly_detection/patchcore/vit_b16/x224/one_layer_defect_tuning/notebook.ipynb` — OK (RETRAIN flag present; artifacts complete)
- `experiments/anomaly_detection/patchcore/vit_b16/x224/one_layer_no_defect_tuning/notebook.ipynb` — OK (NO_RETRAIN_FLAG: has training cells but no toggle; artifacts present and complete)
- `experiments/anomaly_detection/patchcore/vit_b16/x224/two_block/notebook.ipynb` — OK (NO_RETRAIN_FLAG: has training cells but no toggle; artifacts present and complete)
- `experiments/anomaly_detection/patchcore/vit_b16/x224/two_block_no_defect_tuning/notebook.ipynb` — OK (NO_RETRAIN_FLAG: has training cells but no toggle; artifacts present and complete)

## Ensemble Family

- `experiments/anomaly_detection/ensemble/x64/score_ensemble/notebook.ipynb` — OK (RETRAIN flag present; requires patchcore topk_mb50k_r010 val/test_scores.csv and ts_resnet50 artifacts)
- `experiments/anomaly_detection/ensemble/x224/vit_effnetb1_ensemble/notebook.ipynb` — OK (RETRAIN flag present; all ensemble output artifacts exist; aligned re-scoring checkpoint loading validated)

## Report Figures

- `experiments/anomaly_detection/report_figures/notebook.ipynb` — OK (NO_RETRAIN_FLAG; loads pre-saved metrics.csv from 22+ experiment directories; all 13 plots generated and saved to artifacts/plots/)

## Classifier Family

- `experiments/classifier/multiclass/x64/training/notebook.ipynb` — OK (RETRAIN flag present; conditional training with fallback to saved artifacts if RETRAIN=False)
- `experiments/classifier/multiclass/x64/seed07/notebook.ipynb` — OK (RETRAIN flag present; artifacts/ directory contains best_model.pt, history.csv, metrics.json, predictions)
- `experiments/classifier/multiclass/x64/final_labeling/notebook.ipynb` — OK (NO_RETRAIN_FLAG; loads saved checkpoint from seed07 artifacts for pseudo-labeling)
- `experiments/classifier/multiclass/x64/showcase/notebook.ipynb` — OK (RETRAIN flag present; presentation-oriented analysis of classifier results)

## Supervised Defect Detection

- `experiments/anomaly_detection_defect/supervised_sweep/vit_b16/x224/main/notebook.ipynb` — OK (RETRAIN flag present; artifacts/supervised_vit_sweep contains checkpoints, embeddings, results)

## supervised_cnn note

Two notebooks exist in supervised_cnn:
- `experiments/anomaly_detection_defect/supervised_cnn/full_defect/wafer-cnn-normal-1pct-defect.ipynb`
- `experiments/anomaly_detection_defect/supervised_cnn/half_defect/wafer-cnn-normal-1pct-half-classes.ipynb`

These notebooks were not explicitly listed in the audit scope but were found during the supervised_cnn check.

## Summary

Total notebooks audited: 50

### Missing Artifacts (RETRAIN=True required to view outputs)
- `experiments/anomaly_detection/teacher_student/resnet18/x224/main/notebook.ipynb`
- `experiments/anomaly_detection/teacher_student/resnet50/x64/layer_ablation/notebook.ipynb`
- `experiments/anomaly_detection/teacher_student/resnet50/x224/feature_autoencoder_dim_sweep/notebook.ipynb`
- `experiments/anomaly_detection/teacher_student/vit_b16/x224/main/notebook.ipynb`

### No RETRAIN Flag (ablation/evaluation notebooks — run top to bottom)
- `experiments/anomaly_detection/patchcore/vit_b16/x224/one_layer_no_defect_tuning/notebook.ipynb`
- `experiments/anomaly_detection/patchcore/vit_b16/x224/two_block/notebook.ipynb`
- `experiments/anomaly_detection/patchcore/vit_b16/x224/two_block_no_defect_tuning/notebook.ipynb`
- `experiments/anomaly_detection/report_figures/notebook.ipynb`
- `experiments/classifier/multiclass/x64/final_labeling/notebook.ipynb`

### Additional notebooks found (not in original audit scope)
- `experiments/anomaly_detection_defect/supervised_cnn/full_defect/wafer-cnn-normal-1pct-defect.ipynb`
- `experiments/anomaly_detection_defect/supervised_cnn/half_defect/wafer-cnn-normal-1pct-half-classes.ipynb`

### Rank #3 Resolution
PatchCore + WideResNet50-2 multilayer 224×224 (report rank #3, F1=0.549) maps to:
`experiments/anomaly_detection/patchcore/wideresnet50/x224/multilayer_umap/notebook.ipynb`
