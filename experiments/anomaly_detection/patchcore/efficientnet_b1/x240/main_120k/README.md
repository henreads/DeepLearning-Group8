# EfficientNet-B1 PatchCore (`x240`, 120k main follow-up)

This branch scales the original one-layer EfficientNet-B1 PatchCore main recipe from the standard `50k / 5%` benchmark to a larger `120k / 5%` normal-only split.

What stays the same:

- pretrained `EfficientNet-B1`
- one-layer feature extraction from block `3`
- random projection to `512` patch dimensions
- PatchCore memory bank capped at `240k` patches
- `nn_k = 3`
- wafer score = `topk_mean` over the top `3%` patch distances
- validation threshold = `95th` percentile of validation-normal scores

What changes:

- the data split now uses `120,000` total normals before the `80 / 10 / 10` split
- test anomalies remain fixed to `5%` of the test-normal count

Key files:

- `train_config.toml`
  Script-driven training config for the 120k follow-up.
- `artifacts/patchcore_efficientnet_b1_one_layer_120k/`
  Output root populated by the runner or the Modal app.
- `scripts/run_patchcore_effb1_x240_main.py`
  Canonical training entrypoint used by the Modal app.

## Modal Run Log

Run recorded on `2026-04-07`:

- Modal app id: `ap-FUG8aknXjs2utPUUrwcm18`
- Status: `complete`
- Selected variant: `topk_mb240k_r003_x240_120k`
- Backbone: `EfficientNet-B1`, feature block `3`
- Memory bank: `240000 x 512`
- Memory subset images: `267`
- Checkpoint size: `0.518 GB`

Main metrics from the downloaded artifact summary:

- validation-threshold: `0.508993`
- precision: `0.449541`
- recall: `0.816667`
- F1: `0.579882`
- AUROC: `0.942541`
- AUPRC: `0.659086`
- best sweep threshold: `0.543127`
- best sweep F1: `0.670112`

Downloaded artifact bundle:

- `artifacts/patchcore_efficientnet_b1_one_layer_120k/run_manifest.json`
- `artifacts/patchcore_efficientnet_b1_one_layer_120k/results/config.json`
- `artifacts/patchcore_efficientnet_b1_one_layer_120k/results/patchcore_sweep_results.csv`
- `artifacts/patchcore_efficientnet_b1_one_layer_120k/results/patchcore_sweep_summary.json`
- `artifacts/patchcore_efficientnet_b1_one_layer_120k/results/selected_checkpoint.json`
- `artifacts/patchcore_efficientnet_b1_one_layer_120k/topk_mb240k_r003_x240_120k/checkpoints/best_model.pt`
- `artifacts/patchcore_efficientnet_b1_one_layer_120k/topk_mb240k_r003_x240_120k/results/summary.json`
- `artifacts/patchcore_efficientnet_b1_one_layer_120k/topk_mb240k_r003_x240_120k/results/evaluation/val_scores.csv`
- `artifacts/patchcore_efficientnet_b1_one_layer_120k/topk_mb240k_r003_x240_120k/results/evaluation/test_scores.csv`
- `artifacts/patchcore_efficientnet_b1_one_layer_120k/topk_mb240k_r003_x240_120k/results/evaluation/threshold_sweep.csv`
- `artifacts/patchcore_efficientnet_b1_one_layer_120k/topk_mb240k_r003_x240_120k/results/evaluation/defect_breakdown.csv`
- `artifacts/patchcore_efficientnet_b1_one_layer_120k/topk_mb240k_r003_x240_120k/plots/score_distribution.png`
- `artifacts/patchcore_efficientnet_b1_one_layer_120k/topk_mb240k_r003_x240_120k/plots/threshold_sweep.png`
- `artifacts/patchcore_efficientnet_b1_one_layer_120k/topk_mb240k_r003_x240_120k/plots/confusion_matrix.png`
- `artifacts/patchcore_efficientnet_b1_one_layer_120k/topk_mb240k_r003_x240_120k/plots/defect_breakdown.png`

Log retrieval note:

- `modal app logs ap-FUG8aknXjs2utPUUrwcm18 --timestamps` returned no retained stdout when queried after the run completed, so this section records the finished run from the synced manifests and summaries instead.
