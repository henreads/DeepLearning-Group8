# PatchCore ViT-B/16 x224 Main Modal Runner

This app runs the ViT-B/16 x224 PatchCore source notebook remotely and writes normalized artifacts back into `experiments/anomaly_detection/patchcore/vit_b16/x224/main/artifacts/`.

## Prerequisites

```powershell
.\venv\Scripts\activate
pip install modal
modal setup
```

## First-Time Setup

Upload only `data/raw/LSWMD.pkl` once:

```powershell
modal run modal_apps/patchcore_vit_b16_x224_main/app.py::upload_raw_data
```

## Run

Run the job and download artifacts back into the experiment folder:

```powershell
modal run modal_apps/patchcore_vit_b16_x224_main/app.py::main
```

Detached/background run:

```powershell
modal run --detach modal_apps/patchcore_vit_b16_x224_main/app.py::main --no-sync-back
```

Download artifacts later:

```powershell
modal run modal_apps/patchcore_vit_b16_x224_main/app.py::download_artifacts
```

Use different loader settings if needed:

```powershell
modal run modal_apps/patchcore_vit_b16_x224_main/app.py::main --num-workers 0 --batch-size 128
```
