# ViT-B/16 PatchCore

This branch stores the local ViT-B/16 PatchCore `x224` benchmark and holdout runs.

Files:
- `notebook.ipynb`: canonical train-or-reuse notebook for the benchmark run
- `artifacts/patchcore_vit_b16_5pct/main_5pct/checkpoints/`: canonical checkpoint location for the benchmark run
- `artifacts/patchcore_vit_b16_5pct/main_5pct/results/`: benchmark summaries, score files, evaluation CSVs, and UMAP exports
- `artifacts/patchcore_vit_b16_5pct/main_5pct/plots/`: benchmark figures regenerated from the saved local artifacts
- `artifacts/patchcore_vit_b16_5pct/holdout70k_3p5k/`: secondary holdout evaluation bundle
- `scripts/run_patchcore_vit_b16_x224_main_notebook.py`: local runner used by the notebook when `RUN_TRAINING = True`

Notes:
- The canonical benchmark run is the outer `main_5pct/` bundle.
- `legacy_nested_run/` preserves an older nested export that is no longer the primary artifact source.
- The notebook is intended to run top to bottom in both modes: reuse the saved benchmark checkpoint and saved score artifacts by default, or launch the local runner to rebuild the benchmark bundle before the later analysis cells execute.

