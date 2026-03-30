# PatchCore EfficientNet-B0 x224 Main Modal Runner

This app runs the x224 EfficientNet-B0 PatchCore source notebook remotely and writes normalized artifacts back into `experiments/anomaly_detection/patchcore/efficientnet_b0/x224/main/artifacts/`.

## Prerequisites

```powershell
.\venv\Scripts\activate
pip install modal
modal setup
```

## First-Time Setup

If `LSWMD.pkl` is already in the shared Modal raw-data volume, you can skip this step.

Otherwise upload only the raw pickle once:

```powershell
modal run modal_apps/patchcore_effb0_x224_main/app.py::upload_raw_data
```

The app now prepares `metadata_50k_5pct.csv` and `arrays_50k_5pct/` remotely on first run and caches them in the x224 processed-data volume.

## Run

Run the job and download artifacts back into the experiment folder:

```powershell
modal run modal_apps/patchcore_effb0_x224_main/app.py::main
```

Detached/background run:

```powershell
modal run --detach modal_apps/patchcore_effb0_x224_main/app.py::main --no-sync-back
```

Download artifacts later:

```powershell
modal run modal_apps/patchcore_effb0_x224_main/app.py::download_artifacts
```

Use different loader settings if needed:

```powershell
modal run modal_apps/patchcore_effb0_x224_main/app.py::main --num-workers 4 --batch-size 64
```
