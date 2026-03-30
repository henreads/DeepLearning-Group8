# EfficientNet-B1 PatchCore (`x240`, one-layer main run)

This branch contains the local EfficientNet-B1 one-layer PatchCore follow-up at the backbone's native `x240` scale.

The main notebook trains the benchmark run, saves the fitted checkpoint, evaluates the standard `50k / 5%` protocol, reevaluates the same checkpoint on the `70k / 3.5k` holdout, and exports UMAP diagnostics from the saved feature manifold.

## Files

- `notebook.ipynb`
  Canonical local training and evaluation workflow for the one-layer EfficientNet-B1 run.
- `artifacts/patchcore_efficientnet_b1_one_layer/checkpoints/`
  Saved model checkpoint for the benchmark run.
- `artifacts/patchcore_efficientnet_b1_one_layer/results/`
  Benchmark summaries, score CSVs, holdout evaluation files, UMAP exports, and config snapshots.
- `artifacts/patchcore_efficientnet_b1_one_layer/plots/`
  Benchmark and UMAP figures regenerated from the saved local artifacts.
