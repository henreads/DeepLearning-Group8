# Teacher-Student ResNet50 x224 Feature Autoencoder Dimension Sweep

## Overview

This experiment tests whether **larger feature autoencoder bottlenecks can fix the resolution degradation** observed at x224 resolution for large backbones like ResNet50.

**Research Question:** Can increasing feature autoencoder hidden dimension from 128 → 256/512/768 improve ResNet50 x224 performance?

**Dimensions tested:** [64, 128, 256, 512, 768]

## Motivation

ResNet50 at x224 shows catastrophic degradation (-18.9% F1) compared to x64 baseline. This is hypothesized to be due to:

1. ResNet50 layer2 outputs 512 channels at x224
2. Fixed 128-dim autoencoder bottleneck is too small (1:4 compression ratio)
3. Cannot adequately bridge feature-space gap between teacher and student at high resolution

**Hypothesis:** Larger bottlenecks (256, 512, 768 dims) will improve x224 performance.

## Directory Structure

```
experiments/anomaly_detection/teacher_student/resnet50/x224/feature_autoencoder_dim_sweep/
├── train_config.toml                 # Base config (used for all dimensions)
├── notebook.ipynb                    # Analysis notebook
└── artifacts/
    └── ts_resnet50_x224_ae_dim_sweep/
        ├── ae_dim_64/                # Per-dimension results
        │   ├── checkpoints/
        │   │   └── best_model.pt
        │   ├── results/
        │   │   ├── history.json
        │   │   └── evaluation/
        │   │       ├── summary.json
        │   │       ├── val_scores.csv
        │   │       ├── test_scores.csv
        │   │       └── threshold_sweep.csv
        │   └── plots/
        ├── ae_dim_128/
        ├── ae_dim_256/
        ├── ae_dim_512/
        ├── ae_dim_768/
        └── ae_dimension_sweep_summary.csv  # Aggregated results
```

## Run Commands

### Local Execution (CPU - for testing)

```bash
cd experiments/anomaly_detection/teacher_student/resnet50/x224/feature_autoencoder_dim_sweep
jupyter notebook notebook.ipynb
```

Edit `RETRAIN = False` to load cached checkpoints, or `RETRAIN = True` to retrain.

### Modal Execution (GPU - recommended)

#### Start the sweep (detached, 14-hour timeout)

```bash
modal run modal_apps/teacher_student_resnet50_x224_ae_dim_sweep/app.py::main --no-sync-back --detach
```

This will:
1. Train 5 models (ae_dim = 64, 128, 256, 512, 768) sequentially on A10G GPU
2. Evaluate each model after training
3. Save all results to the volume
4. Total compute time: ~5-6 hours (1 hour per dimension)

#### Download artifacts after completion

```bash
modal run modal_apps/teacher_student_resnet50_x224_ae_dim_sweep/app.py::download_artifacts
```

This will display the artifact location in the shared volume.

#### Force fresh retrain (skip cached checkpoints)

```bash
modal run modal_apps/teacher_student_resnet50_x224_ae_dim_sweep/app.py::main --fresh-train --no-sync-back --detach
```

## Configuration

Base configuration: `experiments/anomaly_detection/teacher_student/resnet50/x224/feature_autoencoder_dim_sweep/train_config.toml`

Key settings:
- **Resolution:** 224×224 (native backbone resolution)
- **Batch size:** 256 (properly scaled from x64's 16)
- **Learning rate:** 0.0003
- **Teacher backbone:** ResNet50 (layer2)
- **Teacher layers:** layer2 only (single-layer, unlike WideResNet50-2)
- **Feature autoencoder hidden dims:** [64, 128, 256, 512, 768] (swept)
- **Score weights:** student=1.0, autoencoder=0.0 (student-only scoring)

## Expected Outputs

### Per-dimension results

Each dimension folder (`ae_dim_64/`, `ae_dim_128/`, etc.) contains:

- **checkpoints/best_model.pt** - Trained model
- **results/history.json** - Training curves
- **results/evaluation/summary.json** - Evaluation metrics
- **results/evaluation/val_scores.csv** - Validation anomaly scores
- **results/evaluation/test_scores.csv** - Test anomaly scores
- **results/evaluation/threshold_sweep.csv** - Threshold sweep metrics

### Aggregated summary

**artifacts/ts_resnet50_x224_ae_dim_sweep/ae_dimension_sweep_summary.csv**

Contains one row per dimension with:
- `ae_hidden_dim`: The tested bottleneck dimension
- `f1`: Validation-threshold F1 score
- `auroc`: Area under ROC curve
- `auprc`: Area under precision-recall curve
- `precision`, `recall`: At validation threshold

## Baseline Comparison

| Configuration | F1 | AUROC | AUPRC | Status |
|---|---|---|---|---|
| ResNet50 x64 (ae_dim=128) | 0.488 | 0.913 | 0.581 | Baseline ✓ |
| ResNet50 x224 (ae_dim=128) | 0.3988 | 0.8277 | 0.3608 | Current (fails) ❌ |
| ResNet50 x224 (ae_dim=256) | TBD | TBD | TBD | Testing ⏳ |
| ResNet50 x224 (ae_dim=512) | TBD | TBD | TBD | Testing ⏳ |
| ResNet50 x224 (ae_dim=768) | TBD | TBD | TBD | Testing ⏳ |

## Analysis Notebook

The included `notebook.ipynb` will:

1. Iterate through all dimensions
2. Load cached evaluations or run evaluations if needed
3. Aggregate results into a comparison table
4. Plot F1, AUROC, AUPRC vs AE hidden dimension
5. Compare best x224 result to x64 baseline

To run locally:
```bash
jupyter notebook experiments/anomaly_detection/teacher_student/resnet50/x224/feature_autoencoder_dim_sweep/notebook.ipynb
```

## Key Metrics to Track

1. **Best F1 score across dimensions** - Which bottleneck size works best?
2. **F1 relative to x64 baseline (0.488)** - Can we match or exceed x64?
3. **AUPRC improvement** - Is ranking quality better at higher dimensions?
4. **Convergence speed** - Do larger models train faster/slower?

## Expected Timeline

- **Setup:** Complete ✓
- **Training:** ~1 hour per dimension (5 dimensions total = 5 hours)
- **Evaluation:** ~5 minutes per dimension
- **Total:** ~5-6 hours on A10G GPU
- **Download + analysis:** ~30 minutes

## Related Experiments

- **WideResNet50-2 x224 Multilayer** - Tests multi-layer feature extraction at x224 (catastrophic -37.5% F1)
- **ResNet50 x224 main** - Single baseline for x224 with ae_dim=128 (-18.9% F1)
- **ResNet18 x224 main** - Shows x224 works for small backbones (+3.2% F1)

## Future Work

1. If sweep shows improvement: retrain WideResNet50-2 x224 with optimal dimension
2. If sweep shows no improvement: investigate alternative architectures (skip connections, feature normalization)
3. Test optimal dimension on other backbones (ResNet18, WideResNet50-2)

## Contact & Notes

This sweep is designed to isolate the **feature bottleneck hypothesis** as the root cause of x224 degradation.

If results suggest bottleneck size isn't the issue, we'll need to investigate:
- Feature alignment/mismatch between teacher and student at x224
- Student network capacity limitations
- Different training dynamics at higher resolution
- Feature normalization or standardization issues
