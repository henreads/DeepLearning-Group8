# PatchCore EfficientNet-B1 x240 Main 120k Modal Runner

This app runs the script-driven `120k / 5%` EfficientNet-B1 one-layer PatchCore follow-up remotely on Modal.

It keeps the original EfficientNet-B1 main settings:

- `x240` inputs
- EfficientNet-B1 feature block `3`
- projection dim `512`
- memory bank cap `240k`
- `nn_k = 3`
- wafer score = `top 3%` patch-distance mean
- validation threshold = `95th` percentile of validation-normal scores

## Prerequisites

```powershell
.\venv\Scripts\activate
pip install modal
modal setup
```

## First-Time Setup

Upload `data/raw/LSWMD.pkl` once:

```powershell
modal run modal_apps/patchcore_effb1_x240_main_120k/app.py::upload_raw_data
```

## Run

Prepare the shared `x240` `120k / 5%` processed dataset only:

```powershell
modal run modal_apps/patchcore_effb1_x240_main_120k/app.py::prepare_processed_data
```

Launch the training run and sync artifacts back locally:

```powershell
modal run modal_apps/patchcore_effb1_x240_main_120k/app.py::main
```

Use a different worker count if needed:

```powershell
modal run modal_apps/patchcore_effb1_x240_main_120k/app.py::main --num-workers 4
```

Skip the local artifact download step:

```powershell
modal run modal_apps/patchcore_effb1_x240_main_120k/app.py::main --no-sync-back
```

Download the latest artifacts again without rerunning:

```powershell
modal run modal_apps/patchcore_effb1_x240_main_120k/app.py::download_artifacts
```

## Volumes

- Raw-data volume: `wafer-defect-lswmd-raw`
- Processed-data volume: `wafer-defect-wm811k-x240-processed`
- Artifact volume: `wafer-defect-patchcore-effb1-x240-main-120k-artifacts`

The remote app mounts:

- `wafer-defect-lswmd-raw` at `/root/project/data/raw`
- `wafer-defect-wm811k-x240-processed` at `/root/project/data/processed/x240/wm811k`
- `wafer-defect-patchcore-effb1-x240-main-120k-artifacts` at `/root/project/experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main_120k/artifacts`
