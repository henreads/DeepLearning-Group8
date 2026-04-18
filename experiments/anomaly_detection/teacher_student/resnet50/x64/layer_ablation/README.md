# TS-ResNet50 Layer Ablation

This notebook is the follow-up analysis branch for the main TS-ResNet50 local run. It compares different teacher layers and summarizes how the selected layer changes downstream anomaly performance.

## What This Notebook Does

- starts from the local TS-ResNet50 configuration and saved main-run artifacts
- compares alternative teacher-layer variants
- reuses the saved `layer2` main run plus the saved `layer1` reference snapshots by default
- can rerun local variants when explicitly enabled
- writes comparison tables into `artifacts/results/`
- saves comparison figures into `artifacts/plots/`

## Notes

This folder is more analysis-oriented than the main `main/` run folder. For the primary saved checkpoint and evaluation outputs, start with `../main/` first.

- main reference run: `../main/`
- saved comparison tables: `artifacts/results/teacher_layer_comparison.csv` and `artifacts/results/best_by_teacher_layer.csv`
- saved comparison figures: `artifacts/plots/top_variants_f1.png`, `artifacts/plots/precision_recall_scatter.png`, and `artifacts/plots/best_by_teacher_layer.png`

The default notebook path is lightweight and submission-friendly. It does not retrain unless you explicitly enable the local rerun flag.
