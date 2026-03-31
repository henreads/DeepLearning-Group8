# PatchCore WRN50 x64 Holdout

Builds or reuses the shared `70k / 3.5k` x64 holdout metadata on Modal, then reevaluates the saved WRN50 PatchCore checkpoint on that split.

Commands:

```powershell
modal run --detach modal_apps/patchcore_wrn50_x64_holdout/app.py::main --no-sync-back
modal run modal_apps/patchcore_wrn50_x64_holdout/app.py::download_artifacts
```

Optional:

```powershell
modal run modal_apps/patchcore_wrn50_x64_holdout/app.py::main --variant-name topk_mb50k_r010
```
