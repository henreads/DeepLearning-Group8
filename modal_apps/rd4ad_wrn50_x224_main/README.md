# RD4AD WRN50-2 x224 Modal Runner

Trains the RD4AD WideResNet50-2 `x224` experiment at:

- `experiments/anomaly_detection/rd4ad/wideresnet50/x224/main/notebook.ipynb`
- `experiments/anomaly_detection/rd4ad/wideresnet50/x224/main/train_config.toml`

Artifacts sync back to:

- `experiments/anomaly_detection/rd4ad/wideresnet50/x224/main/artifacts/`

## Run

Detached run:

```powershell
modal run --detach modal_apps/rd4ad_wrn50_x224_main/app.py::main --no-sync-back
```

Download artifacts later:

```powershell
modal run modal_apps/rd4ad_wrn50_x224_main/app.py::download_artifacts
```

## Shared volumes

| Volume | Purpose |
|---|---|
| `wafer-defect-lswmd-raw` | Raw `LSWMD.pkl` uploaded once and reused |
| `wafer-defect-wm811k-x224-processed` | Shared processed x224 arrays and metadata |
| `wafer-defect-rd4ad-wrn50-x224-main-artifacts` | This experiment's outputs |

Raw data upload, first time only:

```powershell
modal run modal_apps/rd4ad_wrn50_x224_main/app.py::upload_raw_data
```

Optional dataset cache prep without training:

```powershell
modal run modal_apps/rd4ad_wrn50_x224_main/app.py::prepare_processed_data
```
