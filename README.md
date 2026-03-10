# DeepLearning-Group8

# Automated IC Wafer Defect Classification
**Group Number:** 08
**Group Members:**
- Henry Lee Jun, 1004219
- Chia Tang, 1007200
- Genson Low, 1005931

## Project Goal

This project studies anomaly detection on semiconductor wafer maps using the WM-811K dataset.
The initial codebase is organized around a simple baseline workflow:

1. Place the raw WM-811K pickle under `data/raw/`
2. Build processed arrays plus metadata CSV files
3. Train a PyTorch autoencoder baseline on normal wafers
4. Save weights and training history for reproducibility

## Repository Layout

```text
configs/               Experiment and data settings
scripts/               Entry points for dataset prep and training
src/wafer_defect/      Package code
data/raw/              Local raw dataset files (ignored by git)
data/processed/        Local processed outputs (ignored by git)
artifacts/             Saved model outputs (ignored by git if desired later)
```

## Setup

Install the package in editable mode inside your virtual environment:

```powershell
pip install -e .
```

## Expected Dataset Input

The current preparation script expects the WM-811K pickle at:

```text
data/raw/LSWMD.pkl
```

The script assumes the pickle can be loaded by `pandas.read_pickle` and contains columns compatible
with common WM-811K distributions, including:

- `waferMap`
- `failureType`
- `trianTestLabel`

If your downloaded dataset uses different column names or a different file format, adjust
[scripts/prepare_wm811k.py](c:\Users\User\Desktop\Term%208\Deep%20Learning\Project\DeepLearning-Group8\scripts\prepare_wm811k.py).

## First Commands

Prepare a small development subset:

```powershell
python scripts/prepare_wm811k.py --dev
```

Train the baseline autoencoder:

```powershell
python scripts/train_autoencoder.py
```

## Immediate Next Steps

- Verify the exact WM-811K file format after download
- Tighten label parsing in the preparation script against the real dataset
- Add anomaly-score evaluation on the test split
- Add a saved-model reload script for report reproducibility
