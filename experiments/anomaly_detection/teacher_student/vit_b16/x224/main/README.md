# Teacher-Student ViT-B/16 x224 Main

Teacher-student distillation experiment using a frozen ViT-B/16 teacher at `224x224`.

## Research Goal

Test whether ViT's strength in Phase 3 comes mainly from:

- the ViT backbone itself
- PatchCore's direct patch matching
- or the combination of both

This branch gives a direct ViT teacher-student comparison against PatchCore ViT-B/16.

## Configuration

| parameter | value |
|---|---|
| teacher backbone | `vit_b16` |
| teacher layer | `block6` |
| teacher input size | `224` |
| feature autoencoder hidden dim | `512` |
| batch size | `256` |
| learning rate | `3e-4` |
| epochs | `30` |
| early stopping | `patience=5` |

## Modal Run

From repo root:

```powershell
modal run modal_apps/teacher_student_vit_b16_x224_main/app.py::main
```

Detached run:

```powershell
modal run --detach modal_apps/teacher_student_vit_b16_x224_main/app.py::main --no-sync-back
```

Fresh retrain:

```powershell
modal run --detach modal_apps/teacher_student_vit_b16_x224_main/app.py::main --fresh-train --no-sync-back
```

Download artifacts later:

```powershell
modal run modal_apps/teacher_student_vit_b16_x224_main/app.py::download_artifacts
```

If the raw pickle is not yet in the shared raw-data volume:

```powershell
modal run modal_apps/teacher_student_vit_b16_x224_main/app.py::upload_raw_data
```

## Artifact Layout

Artifacts are written under:

`artifacts/ts_vit_b16_x224/`

Key outputs:

- `checkpoints/best_model.pt`
- `checkpoints/last_model.pt`
- `checkpoints/latest_checkpoint.pt`
- `results/summary.json`
- `results/history.json`
- `results/evaluation/summary.json`
- `results/evaluation/score_sweep_summary.csv`
- `results/evaluation/selected_score_variant.json`
- `results/evaluation/selected_defect_breakdown.csv`
- `results/configs/train_config.toml`
- `results/configs/train_config_effective.toml`
- `plots/training_curves.png`
- `plots/score_variant_comparison.png`
- `plots/defect_breakdown.png`

## Notes

- Processed dataset creation is handled remotely through `prepare_wm811k.py` and cached in a Modal volume.
- The run is resume-first by default. If `latest_checkpoint.pt` exists, training resumes automatically.
- `--fresh-train` is destructive for this branch and should only be used when you want to ignore saved checkpoints.
