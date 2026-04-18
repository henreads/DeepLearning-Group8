# PatchCore WRN50 x64 Modal Runner

This app runs `experiments/anomaly_detection/patchcore/wideresnet50/x64/main/notebook.ipynb` remotely through the Modal CLI.

## Prerequisites

```powershell
.\venv\Scripts\activate
pip install modal
modal setup
```

## First-Time Setup

Upload `data/raw/LSWMD.pkl` once:

```powershell
modal run modal_apps/patchcore_wrn50_x64/app.py::upload_raw_data
```

## Run

Run the notebook remotely on Modal and sync artifacts back into the notebook folder:

```powershell
modal run modal_apps/patchcore_wrn50_x64/app.py::main
```

Use a different loader worker count if needed:

```powershell
modal run modal_apps/patchcore_wrn50_x64/app.py::main --num-workers 4
```

Skip the local artifact download step:

```powershell
modal run modal_apps/patchcore_wrn50_x64/app.py::main --no-sync-back
```

Download the latest artifacts again without rerunning:

```powershell
modal run modal_apps/patchcore_wrn50_x64/app.py::download_artifacts
```

## Volumes

- Raw-data volume: `wafer-defect-lswmd-raw`
- Artifact volume: `wafer-defect-patchcore-wrn50-x64-artifacts`

The remote app mounts:

- `wafer-defect-lswmd-raw` at `/root/project/data/raw`
- `wafer-defect-patchcore-wrn50-x64-artifacts` at `/root/project/experiments/anomaly_detection/patchcore/wideresnet50/x64/main/artifacts`
