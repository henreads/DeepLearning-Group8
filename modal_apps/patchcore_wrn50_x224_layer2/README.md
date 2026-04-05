# PatchCore WRN50-2 x224 Layer2-only Modal Runner

Trains the single-layer `layer2` WRN50-2 PatchCore x224 experiment.

- Teacher layers: `layer2` only (feature dim = 512)
- Variants: `topk_mb50k_r005_l2_x224`, `topk_mb50k_r010_l2_x224`
- Artifacts: `experiments/anomaly_detection/patchcore/wideresnet50/x224/layer2/artifacts/`

## Run

Detached run:

```powershell
modal run --detach modal_apps/patchcore_wrn50_x224_layer2/app.py::main --no-sync-back
```

Download artifacts later:

```powershell
modal run modal_apps/patchcore_wrn50_x224_layer2/app.py::download_artifacts
```

## Shared volumes

| Volume | Purpose |
|---|---|
| `wafer-defect-lswmd-raw` | Raw LSWMD.pkl (upload once, shared with all apps) |
| `wafer-defect-wm811k-x224-processed` | Processed x224 arrays (shared with all x224 apps) |
| `wafer-defect-patchcore-wrn50-x224-layer2-artifacts` | This experiment's outputs |

Raw data upload (first time only, shared):
```powershell
modal run modal_apps/patchcore_wrn50_x224_layer2/app.py::upload_raw_data
```
