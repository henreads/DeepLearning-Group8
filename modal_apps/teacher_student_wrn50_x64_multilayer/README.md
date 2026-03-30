# Teacher-Student WideResNet50-2 x64 Multilayer Modal Runner

This app runs `experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/multilayer_self_contained/notebook.ipynb` remotely through the Modal CLI.

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

Upload only `data/raw/LSWMD.pkl` once:

```powershell
modal run modal_apps/teacher_student_wrn50_x64_multilayer/app.py::upload_raw_data
```

## Run

Run the full job and download artifacts back into the experiment folder:

```powershell
modal run modal_apps/teacher_student_wrn50_x64_multilayer/app.py::main
```

Detached/background run:

```powershell
modal run --detach modal_apps/teacher_student_wrn50_x64_multilayer/app.py::main --no-sync-back
```

Download artifacts later:

```powershell
modal run modal_apps/teacher_student_wrn50_x64_multilayer/app.py::download_artifacts
```

Use a different worker count if needed:

```powershell
modal run modal_apps/teacher_student_wrn50_x64_multilayer/app.py::main --num-workers 8
```
