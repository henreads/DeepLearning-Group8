# PatchCore WRN50-2 x224 Modal Runner

Runs the WideResNet50-2 multilayer PatchCore x224 experiment (UMAP follow-up) on Modal GPU.

Trains both sweep variants, saves checkpoints, embeddings, scores, and UMAP plots,
then downloads everything to the local repo folder.

## Prerequisites

```powershell
.\venv\Scripts\activate
pip install modal
modal setup
```

## First-Time Setup

Upload `data/raw/LSWMD.pkl` once (shared with other apps via the same volume):

```powershell
modal run modal_apps/patchcore_wrn50_x224_umap/app.py::upload_raw_data
```

Optionally pre-upload the locally processed x224 arrays to skip on-Modal data prep
(saves ~10–15 min on the first run, but requires ~10 GB upload):

```powershell
modal run modal_apps/patchcore_wrn50_x224_umap/app.py::upload_processed_data
```

If you skip this, the runner generates the processed arrays from the raw pickle automatically
on the first run and caches them in the `wafer-defect-processed-x224-wm811k` volume.

## Run

Detached run:

```powershell
modal run --detach modal_apps/patchcore_wrn50_x224_umap/app.py::main --no-sync-back
```

Download artifacts later:

```powershell
modal run modal_apps/patchcore_wrn50_x224_umap/app.py::download_artifacts
```

Skip UMAP (faster — useful if you only need updated model weights):

```powershell
modal run modal_apps/patchcore_wrn50_x224_umap/app.py::main --skip-umap
```

Skip the local artifact download:

```powershell
modal run modal_apps/patchcore_wrn50_x224_umap/app.py::main --no-sync-back
```

Download the latest artifacts without rerunning:

```powershell
modal run modal_apps/patchcore_wrn50_x224_umap/app.py::download_artifacts
```

## What Gets Saved

For each sweep variant (`topk_mb50k_r005_x224`, `topk_mb50k_r010_x224`):

```
artifacts/patchcore-wideresnet50-multilayer-umap/
  {variant_name}/
    checkpoints/best_model.pt          ← full model + memory bank
    results/
      summary.json
      evaluation/
        val_scores.csv
        test_scores.csv
        threshold_sweep.csv
        selected_defect_breakdown.csv
      umap/
        train_embeddings.npy           ← mean-pooled patch embeddings (train normals)
        val_embeddings.npy
        val_labels.npy
        test_embeddings.npy
        test_labels.npy
        umap_by_split.png
        umap_by_score.png
        umap_points.csv
        umap_summary.json
    plots/
      score_distribution.png
      threshold_sweep.png
      confusion_matrix.png
      defect_breakdown.png
      umap_by_split.png
  results/
    patchcore_sweep_results.csv
    patchcore_sweep_summary.json
    selected_checkpoint.json
    config.json
  plots/
    sweep_metrics.png
  run_manifest.json
```

## Volumes

| Volume                                             | Purpose                                     |
| -------------------------------------------------- | ------------------------------------------- |
| `wafer-defect-lswmd-raw`                           | Raw LSWMD.pkl (shared)                      |
| `wafer-defect-processed-x224-wm811k`               | Processed x224 arrays (cached between runs) |
| `wafer-defect-patchcore-wrn50-x224-umap-artifacts` | Training outputs                            |

## After Running

1. Open `experiments/anomaly_detection/patchcore/wideresnet50/x224/multilayer_umap/notebook.ipynb`
2. Set `REGENERATE_UMAP = False` (embeddings already saved)
3. Run the notebook to review results and regenerate plots with new colours
