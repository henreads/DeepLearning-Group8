# PatchCore DINOv2 ViT-B/14 x224

Modal app for:

`experiments/anomaly_detection/patchcore/dinov2_vit_b14/x224/main/notebook.ipynb`

## First-time setup

Upload raw data to the current Modal account:

```powershell
modal run modal_apps/patchcore_dinov2_vit_b14_x224_main/app.py::upload_raw_data
```

Prepare the shared processed `x224` dataset cache explicitly if you want:

```powershell
modal run modal_apps/patchcore_dinov2_vit_b14_x224_main/app.py::prepare_processed_data
```

The app now reuses the shared processed volume:

`wafer-defect-wm811k-x224-processed`

## Run

Detached mode:

```powershell
modal run --detach modal_apps/patchcore_dinov2_vit_b14_x224_main/app.py::main --no-sync-back
```

Foreground mode:

```powershell
modal run modal_apps/patchcore_dinov2_vit_b14_x224_main/app.py::main
```

Runs automatically validate and reuse the processed `x224` cache. If it is missing or incomplete, the app rebuilds it from the uploaded raw `LSWMD.pkl` first.

## Download artifacts

```powershell
modal run modal_apps/patchcore_dinov2_vit_b14_x224_main/app.py::download_artifacts
```

Artifacts sync back into:

`experiments/anomaly_detection/patchcore/dinov2_vit_b14/x224/main/artifacts`

## Optional flags

Force a full rebuild of scores and UMAP:

```powershell
modal run --detach modal_apps/patchcore_dinov2_vit_b14_x224_main/app.py::main --force-rerun --no-sync-back
```
