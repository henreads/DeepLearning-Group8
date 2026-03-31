# VAE x224 Main Modal App

Detached training runner for the VAE x224 resolution study on the shared wafer defect anomaly benchmark.

## Purpose

Investigates whether higher input resolution (224×224 vs 64×64) improves VAE anomaly detection by preserving spatial detail before KL-divergence compression smooths the representation.

## Quick Start

```bash
# Run training detached (no sync-back; download later)
modal run --detach modal_apps/vae_x224_main/app.py::main --no-sync-back

# Download finished artifacts to local repo
modal run modal_apps/vae_x224_main/app.py::download_artifacts
```

## Configuration

See `experiments/anomaly_detection/vae/x224/main/train_config.toml`:
- **Image size**: 224×224 (vs 64 for baseline)
- **Batch size**: 32 (reduced from 64 to fit memory)
- **Latent dim**: 128 (same as baseline)
- **Beta (KL weight)**: 0.005 (same as baseline)
- **Training**: 30 epochs, early stopping patience=5

## Dataset

Uses the shared `wm811k` 50k train / 5% test split at 224×224 resolution:
- 40,000 normal wafers (training)
- 5,000 normal wafers (validation)
- 5,000 normal + 250 anomaly wafers (test)

## Outputs

**Local artifact directory**: `experiments/anomaly_detection/vae/x224/main/artifacts/vae_x224/`

Contains:
- `checkpoints/best_model.pt` — Best checkpoint by validation loss
- `results/history.json` — Training history (epoch-wise losses)
- `results/evaluation/summary.json` — Evaluation metrics
- `results/evaluation/val_scores.csv` — Validation anomaly scores
- `results/evaluation/test_scores.csv` — Test anomaly scores  
- `results/evaluation/threshold_sweep.csv` — Threshold sweep (P/R/F1)
- `plots/training_curves.png` — Training/val/KL loss curves
- `plots/score_distribution_sweep_confusion.png` — Score dist + threshold sweep + confusion matrix
- `plots/reconstruction_examples.png` — Qualitative reconstructions

## Expected Metrics

Baseline VAE x64 for comparison:
- **F1** (val threshold): 0.340
- **F1** (best sweep): 0.418
- **AUROC**: 0.772
- **AUPRC**: 0.372

## Execution Time

- Data preparation: ~5 minutes
- Training: ~1.5 hours (30 epochs on A10G)
- Evaluation + sweep: ~30 minutes
- **Total**: ~2-3 hours

## Integration

Once complete, update `family_reports/vae.md` with:
1. New "Experiment 3: VAE x224 Main" section
2. Metrics table and plots
3. Comparison discussion vs x64 baseline
