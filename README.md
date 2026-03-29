# DeepLearning-Group8

# Automated IC Wafer Defect Classification

**Group Number:** 08

**Group Members**

- Henry Lee Jun, 1004219
- Chia Tang, 1007200
- Genson Low, 1006931

## Project Summary

This project studies wafer-map defect detection and classification on the WM-811K dataset using PyTorch.

The repository is organized around a reproducible notebook workflow:

1. place the raw WM-811K pickle in `data/raw/`
2. build the processed dataset variants from `data/dataset/`
3. run experiment notebooks from `experiments/`
4. reuse saved checkpoints, plots, and result files from each experiment branch when available

## Main Entry Points

- `data/dataset/`
  Dataset construction and validation notebooks, grouped by resolution and split protocol.
- `experiments/anomaly_detection/`
  Anomaly-detection experiment families such as autoencoder, VAE, SVDD, teacher-student, PatchCore, and FastFlow.
- `experiments/classifier/`
  Supervised multiclass classifier experiments.
- `src/wafer_defect/`
  Shared package code used by the notebooks and helper scripts.
- `scripts/`
  Supporting CLI utilities for dataset preparation, training helpers, evaluation, and repo maintenance.

## Quick Start

1. Create a Python 3.11 virtual environment and install the project in editable mode.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
pip install jupyter matplotlib
```

2. Place the raw dataset at:

```text
data/raw/LSWMD.pkl
```

3. Open Jupyter from the repository root and run a dataset notebook, for example:

- `data/dataset/x64/benchmark_50k_5pct/notebook.ipynb`
- `data/dataset/x64/holdout70k_3p5k/notebook.ipynb`

4. Run the desired experiment notebook from `experiments/`.

Recommended starting points:

- `experiments/anomaly_detection/autoencoder/x64/baseline/notebook.ipynb`
- `experiments/anomaly_detection/vae/x64/baseline/notebook.ipynb`
- `experiments/anomaly_detection/svdd/x64/baseline/notebook.ipynb`
- `experiments/anomaly_detection/backbone_embedding/resnet18/x64/baseline/notebook.ipynb`
- `experiments/anomaly_detection/teacher_student/resnet18/x64/main/notebook.ipynb`
- `experiments/anomaly_detection/patchcore/resnet18/x64/main/notebook.ipynb`
- `experiments/anomaly_detection/fastflow/x64/main/notebook.ipynb`
- `experiments/classifier/multiclass/x64/training/notebook.ipynb`

## Experiment Organization

The cleaned submission path lives under `experiments/`.

Each branch is intended to be self-contained and typically includes:

- `README.md`
- `notebook.ipynb`
- `train_config.toml` and/or `data_config.toml`
- `artifacts/`

The artifact folders are organized so results can be inspected or reproduced without searching across unrelated directories:

- `artifacts/checkpoints/`
- `artifacts/plots/`
- `artifacts/results/`

## Repository Layout

```text
data/dataset/          Dataset-building notebooks and configs
data/raw/              Local raw dataset files
data/processed/        Local processed outputs
experiments/           Submission-facing experiment notebooks, configs, and artifacts
configs/               Shared config snapshots and reusable defaults
src/wafer_defect/      Shared package code
scripts/               Supporting CLI utilities
```

## Notes On Execution

- The notebooks are the primary submission surface.
- Jupyter should be launched from the repository root so relative paths resolve correctly.
- Some branches are artifact-first and reuse saved outputs by default.
- Some heavier follow-up branches are designed to retrain or regenerate outputs when checkpoints are absent.
