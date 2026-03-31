# FastFlow x64 Holdout

Builds or reuses the shared `70k / 3.5k` x64 holdout metadata on Modal, then reevaluates the saved FastFlow variant on that split.

Commands:

```powershell
modal run --detach modal_apps/fastflow_x64_holdout/app.py::main --no-sync-back
modal run modal_apps/fastflow_x64_holdout/app.py::download_artifacts
```

Optional:

```powershell
modal run modal_apps/fastflow_x64_holdout/app.py::main --variant-name wrn50_l23_s6
```
