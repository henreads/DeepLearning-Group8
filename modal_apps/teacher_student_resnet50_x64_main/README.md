# Teacher-Student ResNet50 x64 Main Modal Runner

This app runs `experiments/anomaly_detection/teacher_student/resnet50/x64/main/notebook.ipynb` remotely through the Modal CLI.

It runs in three committed phases:
- training
- default evaluation
- score sweep + defect breakdown

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
modal run modal_apps/teacher_student_resnet50_x64_main/app.py::upload_raw_data
```

The app now prepares `metadata_50k_5pct.csv` and `arrays_50k_5pct/` remotely on first run and caches them in the x64 processed-data volume.

## Run

Run the full job and download artifacts back into the experiment folder:

```powershell
modal run modal_apps/teacher_student_resnet50_x64_main/app.py::main
```

Detached/background run:

```powershell
modal run --detach modal_apps/teacher_student_resnet50_x64_main/app.py::main --no-sync-back
```

Download artifacts later:

```powershell
modal run modal_apps/teacher_student_resnet50_x64_main/app.py::download_artifacts
```

The downloader skips any top-level `processed/` directory if it ever appears.
