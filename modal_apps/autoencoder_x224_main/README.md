# Autoencoder x224 Main Modal App

Detached training runner for the autoencoder x224 resolution study on the shared wafer defect anomaly benchmark.

## Purpose

Investigates whether higher input resolution (224×224 vs 64×64) improves autoencoder anomaly detection by preserving spatial detail before reconstruction-based scoring loses local information.

## Quick Start

```bash
# Run training detached (no sync-back; download later)
modal run --detach modal_apps/autoencoder_x224_main/app.py::main --no-sync-back

# Download finished artifacts to local repo
modal run modal_apps/autoencoder_x224_main/app.py::download_artifacts
```

## Configuration

See `experiments/anomaly_detection/autoencoder/x224/main/train_config.toml`:
- **Image size**: 224×224 (vs 64 for baseline)
- **Batch size**: 128 (supported by A10G GPU)
- **Latent dim**: 128 (same as baseline)
- **Training**: 50 epochs, early stopping patience=5
- **Score method**: topk_abs_mean (1% of highest absolute errors)

## Dataset

Uses the shared `wm811k` 50k train / 5% test split at 224×224 resolution:
- 40,000 normal wafers (training)
- 5,000 normal wafers (validation)
- 5,000 normal + 250 anomaly wafers (test)

## Outputs

**Local artifact directory**: `experiments/anomaly_detection/autoencoder/x224/main/artifacts/autoencoder_x224/`

Contains:
- `checkpoints/best_model.pt` — Best checkpoint by validation loss
- `results/history.json` — Training history (epoch-wise losses)
- `results/summary.json` — Training summary (best epoch, loss)
- `results/test_scores.csv` — Test set anomaly scores
- `results/metrics.csv` — Precision/recall/F1/AUROC/AUPRC
- `results/threshold_sweep.csv` — Threshold sweep (P/R/F1)
- `results/failure_analysis.csv` — Per-sample error analysis
- `plots/training_curves.png` — Training/val loss curves
- `plots/confusion_matrix.png` — Confusion matrix at val threshold
- `plots/threshold_sweep.png` — P/R/F1 vs threshold
- `plots/score_distribution.png` — Score histogram
- `plots/reconstruction_examples.png` — Qualitative reconstructions
- `plots/failure_examples_*.png` — FP/FN/TP/TN examples

## Expected Metrics

Baseline autoencoder x64 for comparison:
- **F1** (val threshold): 0.467
- **AUROC**: 0.839
- **AUPRC**: 0.522

Best x64 variant (AE + BatchNorm):
- **F1** (val threshold): 0.502
- **AUROC**: 0.834
- **AUPRC**: 0.568

## Execution Time

- Data preparation: ~5 minutes
- Training: ~1-1.5 hours (50 epochs on A10G, batch_size=128)
- Evaluation + sweep: ~30 minutes
- **Total**: ~2-2.5 hours

## Integration

Once complete, update `family_reports/autoencoder.md` with:
1. New "Experiment 3: Autoencoder x224 Main" section
2. Metrics table and plots
3. Comparison discussion vs x64 baseline

## Research Question

Does using higher resolution (224×224) help autoencoder preserve spatial detail and improve anomaly detection over downsampled 64×64? Is reconstruction-based scoring bottlenecked by information loss at low resolution?
