# VAE Latent Dimension Sweep Modal App

Detached training runner for the VAE latent dimension sweep on the shared wafer defect anomaly benchmark.

## Purpose

Tests whether 128 is the optimal latent dimension or if KL regularization compresses useful defect signals regardless of latent capacity. Sweeps across [32, 64, 128, 256, 512] to find the sweet spot.

## Quick Start

```bash
# Run training detached (no sync-back; download later)
modal run --detach modal_apps/vae_latent_dim_sweep/app.py::main --no-sync-back

# Download finished artifacts to local repo
modal run modal_apps/vae_latent_dim_sweep/app.py::download_artifacts
```

## Configuration

See `experiments/anomaly_detection/vae/x64/latent_dim_sweep/train_config.toml`:
- **Image size**: 64×64 (baseline resolution)
- **Batch size**: 64
- **Latent dimensions tested**: [32, 64, 128, 256, 512]
- **Beta (KL weight)**: 0.005 (same as baseline)
- **Training per model**: 30 epochs, early stopping patience=5

## Dataset

Uses the shared `wm811k` 50k train / 5% test split at 64×64 resolution:
- 40,000 normal wafers (training)
- 5,000 normal wafers (validation)
- 5,000 normal + 250 anomaly wafers (test)

## Outputs

**Local artifact directory**: `experiments/anomaly_detection/vae/x64/latent_dim_sweep/artifacts/vae_latent_dim_sweep/`

Contains per-latent-dim subdirectories `latent_dim_<N>/`:
- `checkpoints/best_model.pt` — Best checkpoint by validation loss
- `results/history.json` — Training history
- `results/evaluation/summary.json` — Evaluation metrics
- `results/evaluation/val_scores.csv` — Validation anomaly scores
- `results/evaluation/test_scores.csv` — Test anomaly scores
- `results/evaluation/threshold_sweep.csv` — Threshold sweep

Plus aggregate outputs:
- `results/latent_dim_sweep_summary.json` — Aggregated metrics across all dims
- `plots/latent_dim_sweep_metrics.png` — F1/AUROC/AUPRC vs latent_dim
- `plots/latent_dim_sweep_training_curves.png` — Validation loss curves overlaid
- `plots/best_latent_dim_distribution_sweep_confusion.png` — Best model's detailed results

## Expected Baseline

VAE x64 latent_dim=128 for comparison:
- **F1** (val threshold): 0.340
- **F1** (best sweep): 0.418
- **AUROC**: 0.772
- **AUPRC**: 0.372

## Execution Time

- Data preparation: ~2 minutes (reuses cached x64 dataset)
- Training (5 models): ~3 hours total (30 min per model on A10G)
- Evaluation + sweep: ~1.5 hours (15 min per model)
- **Total**: ~4-6 hours

## Integration

Once complete, update `family_reports/vae.md` with:
1. New "Experiment 3: VAE x64 Latent Dimension Sweep" section
2. Metrics table: latent_dim vs F1/AUROC/AUPRC
3. Training curves comparison
4. Best latent dimension summary and interpretation
