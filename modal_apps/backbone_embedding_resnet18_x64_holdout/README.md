# ResNet18 Backbone x64 Holdout

Builds or reuses the shared `70k / 3.5k` x64 holdout metadata on Modal, then reevaluates the frozen ResNet18 embedding baseline on that split.

Commands:

```powershell
modal run --detach modal_apps/backbone_embedding_resnet18_x64_holdout/app.py::main --no-sync-back
modal run modal_apps/backbone_embedding_resnet18_x64_holdout/app.py::download_artifacts
```
