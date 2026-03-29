# ResNet50 PatchCore (`x64`)

This is the curated submission-facing PatchCore sweep notebook for the `64x64` benchmark split.

What this notebook does:
- loads the processed `50k / 5%` benchmark dataset
- reuses the saved PatchCore sweep artifacts by default
- loads the selected local checkpoint from `checkpoints/best_model.pt`
- regenerates the main plots and saves them into `plots/`
- saves failure-analysis CSVs into `results/evaluation/`
- can repopulate each saved variant folder with cached plots and failure-analysis outputs

Key files:
- notebook: `experiments/anomaly_detection/patchcore/resnet50/x64/main/notebook.ipynb`
- train config: `experiments/anomaly_detection/patchcore/resnet50/x64/main/train_config.toml`
- data config: `experiments/anomaly_detection/patchcore/resnet50/x64/main/data_config.toml`
- artifact root: `experiments/anomaly_detection/patchcore/resnet50/x64/main/artifacts/patchcore_resnet50`

Artifact layout:
- `checkpoints/`: canonical selected checkpoint for this run
- `results/`: sweep summary, selected-run summary, and evaluation CSVs
- `plots/`: saved figures recreated by the notebook
- per-variant folders such as `mean_mb50k/`: saved checkpoints and evaluation files for each sweep option

Default behavior:
- open the notebook and run top to bottom
- it will reuse cached artifacts unless `FORCE_RERUN_SWEEP = True`
- it will render per-variant plots from cached CSVs unless you turn `RENDER_ALL_SAVED_VARIANTS` off
