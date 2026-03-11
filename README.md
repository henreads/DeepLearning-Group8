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
4. Train a convolutional VAE on the same split for comparison
5. Save weights, evaluation outputs, and training history for reproducibility

## Repository Layout

```text
configs/data/          Dataset preparation settings
configs/training/      Model training settings
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

For notebooks used in this repo, also install:

```powershell
pip install jupyter matplotlib
```

## Fresh Clone Setup

Use these steps on a new machine or fresh clone.

1. Clone the repo and enter it:

```powershell
git clone <repo-url>
cd DeepLearning-Group8
```

2. Create and activate a Python 3.11 virtual environment:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

3. Install the project in editable mode:

```powershell
pip install -e .
pip install jupyter matplotlib
```

4. Place the raw WM-811K pickle here:

```text
data/raw/LSWMD.pkl
```

5. Build the processed dataset used by the main experiments:

```powershell
python scripts/prepare_wm811k.py
```

6. Verify the expected processed metadata exists:

```powershell
dir data\processed\x64\wm811k
```

The main `64x64` training configs expect:

```text
data/processed/x64/wm811k/metadata_50k_5pct.csv
```

7. Train or open notebooks only after the processed metadata exists:

```powershell
python scripts/train_autoencoder.py
jupyter notebook
```

Open notebooks from the repo root so relative paths resolve correctly.

## Common Path Issues

If a fresh setup fails, check these first:

- you are running commands from the repository root
- you ran `pip install -e .`
- the raw dataset file is exactly `data/raw/LSWMD.pkl`
- you already ran `python scripts/prepare_wm811k.py`
- `data/processed/x64/wm811k/metadata_50k_5pct.csv` exists before training
- Jupyter was launched from the repo root, not from another folder

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

The default config locations are now:

- data prep: `configs/data/data.toml`
- training: `configs/training/*.toml`

Train the baseline autoencoder:

```powershell
python scripts/train_autoencoder.py
```

Train the VAE follow-up model:

```powershell
python scripts/train_vae.py
```

Evaluate any reconstruction-based checkpoint on the shared validation/test protocol:

```powershell
python scripts/evaluate_reconstruction_model.py --checkpoint artifacts/x64/autoencoder_baseline/best_model.pt
python scripts/evaluate_reconstruction_model.py --checkpoint artifacts/x64/vae_baseline/best_model.pt
```
