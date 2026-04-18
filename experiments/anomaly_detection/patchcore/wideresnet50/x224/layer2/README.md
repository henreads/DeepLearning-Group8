# WRN50-2 layer2 PatchCore (`x224`)

This branch is curated as a reproducibility and results-review notebook built from saved PatchCore sweep artifacts.

Notebook execution modes:
- `RETRAIN = False`: artifact-review mode. Load saved checkpoints, CSVs, and plots directly.
- `RETRAIN = True`: full rerun mode. The notebook launches `scripts/run_patchcore_wrn50_x224_umap.py` with the same runner, train config, data config, output dir, and worker argument pattern used by the Modal app.

In review mode the notebook reloads the saved per-variant score CSVs, recomputes defect analysis using the local `x224` metadata, and repopulates each variant folder with plots.

Key files:
- notebook: `experiments/anomaly_detection/patchcore/wideresnet50/x224/layer2/notebook.ipynb`
- train config: `experiments/anomaly_detection/patchcore/wideresnet50/x224/layer2/train_config.toml`
- artifact root: `experiments/anomaly_detection/patchcore/wideresnet50/x224/layer2/artifacts/patchcore-wideresnet50-layer2`

Artifact layout:
- `results/`: branch-level sweep tables and selected-variant review outputs
- `plots/`: branch-level plots for the selected variant
- `topk_mb50k_r005_l2_x224/`: per-variant results/evaluation/plots generated from cached CSVs
- `checkpoints/`: note explaining that checkpoints were not checked in

