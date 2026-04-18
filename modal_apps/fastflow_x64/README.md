# FastFlow x64 Modal Runner

This app runs the curated FastFlow x64 notebook remotely through the Modal CLI, stores outputs in a Modal volume, and syncs the artifacts back into this repo.

## What It Runs

- Remote entrypoint: `modal_apps/fastflow_x64/app.py`
- Headless notebook runner: `scripts/run_fastflow_x64_notebook.py`
- Source notebook: `experiments/anomaly_detection/fastflow/x64/main/notebook.ipynb`

## Prerequisites

- Activate the repo virtual environment.
- Install and authenticate the Modal CLI in that environment.
- Make sure the processed dataset exists locally under `data/processed/x64/wm811k/`.

Example:

```powershell
.\venv\Scripts\activate
pip install modal
modal setup
```

## First-Time Setup

Upload only the FastFlow-required processed data into the Modal data volume once:

```powershell
modal run modal_apps/fastflow_x64/app.py::upload_data
```

By default this uploads only:

- `data/processed/x64/wm811k/metadata_50k_5pct.csv`
- `data/processed/x64/wm811k/arrays_50k_5pct/`

into the Modal volume mounted remotely at:

- `/root/project/data/processed/x64/wm811k/metadata_50k_5pct.csv`
- `/root/project/data/processed/x64/wm811k/arrays_50k_5pct/`

This is enough for the current FastFlow config because `train_config.toml` points to `metadata_50k_5pct.csv`, and that metadata only references files inside `arrays_50k_5pct/`.

If you ever want to upload the entire processed `wm811k` tree instead, use:

```powershell
modal run modal_apps/fastflow_x64/app.py::upload_data --upload-all=true
```

## Run The Sweep

Run the FastFlow sweep on Modal and sync the resulting artifacts back into the repo:

```powershell
modal run modal_apps/fastflow_x64/app.py::main
```

By default this:

- trains only missing variants
- reuses existing saved CSV artifacts when present
- downloads the artifact volume back into `experiments/anomaly_detection/fastflow/x64/main/artifacts/fastflow_variant_sweep/`

## Useful Variants

Train only missing variants, which is the default:

```powershell
modal run modal_apps/fastflow_x64/app.py::main --run-missing-variants
```

Force a clean retrain of every variant:

```powershell
modal run modal_apps/fastflow_x64/app.py::main --force-retrain-variants
```

Skip the local artifact download step:

```powershell
modal run modal_apps/fastflow_x64/app.py::main --no-sync-back
```

Download the latest artifact volume contents again without rerunning training:

```powershell
modal run modal_apps/fastflow_x64/app.py::download_artifacts
```

## Volumes

- Data volume: `wafer-defect-fastflow-x64-data`
- Artifact volume: `wafer-defect-fastflow-x64-artifacts`

The remote app mounts:

- `wafer-defect-fastflow-x64-data` at `/root/project/data`
- `wafer-defect-fastflow-x64-artifacts` at `/root/project/experiments/anomaly_detection/fastflow/x64/main/artifacts/fastflow_variant_sweep`

## Notes

- The first Modal run may take a while because it has to build the container image.
- Modal currently emits some Windows asyncio deprecation warnings in local CLI output; they are noisy but not the blocker for this workflow.
- The artifact sync step downloads from the Modal artifact volume into the same experiment folder used by the local notebook.
- The app now mounts only the minimal FastFlow dataset subset directly under `/root/project/data`, so runs start immediately without a long per-run copy step.
