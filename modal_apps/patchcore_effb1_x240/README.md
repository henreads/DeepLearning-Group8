# PatchCore EfficientNet-B1 x240 Modal Runner

This app runs `experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main/notebook.ipynb` remotely through the Modal CLI.

By default it runs only the main benchmark phase first. You can also ask it to continue into the later holdout and UMAP-heavy sections after the main benchmark artifacts have already been committed to the Modal artifact volume.

## Prerequisites

```powershell
.\venv\Scripts\activate
pip install modal
modal setup
```

## First-Time Setup

Upload `data/raw/LSWMD.pkl` once:

```powershell
modal run modal_apps/patchcore_effb1_x240/app.py::upload_raw_data
```

## Run

Run the notebook remotely on Modal and sync artifacts back into the notebook folder:

```powershell
modal run modal_apps/patchcore_effb1_x240/app.py::main
```

Run the main benchmark first, commit those artifacts, then continue into holdout + UMAP in the same remote job:

```powershell
modal run modal_apps/patchcore_effb1_x240/app.py::main --run-extras
```

Use a different loader worker count if needed:

```powershell
modal run modal_apps/patchcore_effb1_x240/app.py::main --num-workers 4
```

Skip the local artifact download step:

```powershell
modal run modal_apps/patchcore_effb1_x240/app.py::main --no-sync-back
```

Download the latest artifacts again without rerunning:

```powershell
modal run modal_apps/patchcore_effb1_x240/app.py::download_artifacts
```

Prepare the shared x240 processed dataset cache only:

```powershell
modal run modal_apps/patchcore_effb1_x240/app.py::prepare_processed_data
```

If you launch `--run-extras`, the remote run commits the main benchmark artifacts before starting holdout and UMAP. That means you can open a second terminal and pull the already-finished main artifacts immediately with:

```powershell
modal run modal_apps/patchcore_effb1_x240/app.py::download_artifacts
```

## Volumes

- Raw-data volume: `wafer-defect-lswmd-raw`
- Processed-data volume: `wafer-defect-wm811k-x240-processed`
- Artifact volume: `wafer-defect-patchcore-effb1-x240-artifacts`

The remote app mounts:

- `wafer-defect-lswmd-raw` at `/root/project/data/raw`
- `wafer-defect-wm811k-x240-processed` at `/root/project/data/processed/x240/wm811k`
- `wafer-defect-patchcore-effb1-x240-artifacts` at `/root/project/experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main/artifacts`
