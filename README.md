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

7. Open Jupyter only after the processed metadata exists:

```powershell
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

Then create the main training metadata used by the default `64x64` configs:

```powershell
python scripts/prepare_wm811k.py
```

Notes:

- `--dev` switches to the small development subset and writes a variant-specific file such as `data/processed/x64/wm811k/metadata_dev_2kn_400d.csv`
- with the current default config, `python scripts/prepare_wm811k.py` writes `data/processed/x64/wm811k/metadata_50k_5pct.csv`
- the script also writes arrays to a matching variant folder such as `data/processed/x64/wm811k/arrays_50k_5pct`
- `--metadata-path` still works as an override; when you use it, the arrays folder name is derived from that metadata filename
- with the current `configs/data/data.toml`, the main non-dev build uses `50000` normal wafers and samples test defects at `5%` of the test-normal count

The default config locations are now:

- data prep: `configs/data/data.toml`
- training: `configs/training/*.toml`

After preparing the dataset, use the notebooks for the main experiments:

- `notebooks/01_data_exploration.ipynb`
  Explore the processed metadata, class balance, and sample wafer maps.
- `notebooks/02_autoencoder_training.ipynb`
  Train and evaluate the baseline autoencoder.
- `notebooks/03_vae_training.ipynb`
  Train and evaluate the convolutional VAE.
- `notebooks/04_svdd_training.ipynb`
  Train and evaluate the Deep SVDD experiment.
- `notebooks/05_autoencoder_batchnorm_training.ipynb`
  Train and evaluate the BatchNorm autoencoder variant on the same `64x64` 5% dataset and evaluation protocol.
- `notebooks/06_autoencoder_batchnorm_dropout_training.ipynb`
  Train and evaluate the BatchNorm + Dropout autoencoder variant on the same `64x64` 5% dataset and evaluation protocol.
- `notebooks/07_patchcore_training.ipynb`
  Fit and evaluate a PatchCore-style local nearest-neighbor detector on the same `64x64` 5% dataset using the BatchNorm autoencoder checkpoint as the frozen feature backbone.
- `notebooks/08_autoencoder_residual_training.ipynb`
  Train and evaluate a stronger residual autoencoder backbone on the same `64x64` 5% dataset and evaluation protocol.
- `notebooks/09_resnet18_backbone_baseline.ipynb`
  Evaluate a frozen pretrained ResNet18 backbone with simple center-distance scoring on the same `64x64` 5% dataset.
- `notebooks/10_patchcore_resnet18_training.ipynb`
  Fit and evaluate PatchCore on a frozen pretrained ResNet18 backbone using the same `64x64` 5% dataset and validation-threshold protocol.
- `notebooks/11_patchcore_resnet50_training.ipynb`
  Fit and evaluate PatchCore on a frozen pretrained ResNet50 backbone using the same `64x64` 5% dataset and validation-threshold protocol.

Recommended run order for a fresh setup:

1. Run `notebooks/01_data_exploration.ipynb` to confirm the processed dataset looks correct.
2. Run `notebooks/02_autoencoder_training.ipynb` for the baseline autoencoder.
3. Run `notebooks/03_vae_training.ipynb` if you want the VAE comparison.
4. Run `notebooks/04_svdd_training.ipynb` if you want the Deep SVDD comparison.
5. Run `notebooks/05_autoencoder_batchnorm_training.ipynb` if you want the BatchNorm autoencoder comparison.
6. Run `notebooks/06_autoencoder_batchnorm_dropout_training.ipynb` if you want the BatchNorm + Dropout autoencoder comparison.
7. Run `notebooks/07_patchcore_training.ipynb` if you want the PatchCore-style comparison.
8. Run `notebooks/08_autoencoder_residual_training.ipynb` if you want the stronger residual autoencoder backbone comparison.
9. Run `notebooks/09_resnet18_backbone_baseline.ipynb` if you want the plain pretrained ResNet18 backbone baseline.
10. Run `notebooks/10_patchcore_resnet18_training.ipynb` if you want PatchCore with a pretrained ResNet18 backbone.
11. Run `notebooks/11_patchcore_resnet50_training.ipynb` if you want PatchCore with a pretrained ResNet50 backbone.

How to run them:

- open the notebook you want
- select the virtual-environment kernel where `pip install -e .` was run
- run cells top to bottom
- keep the config paths unchanged unless you intentionally want a different dataset variant

## Changing the Test Defect Ratio

The test defect ratio is controlled in [configs/data/data.toml](DeepLearning-Group8\configs\data\data.toml):

```toml
[train_subset]
normal_count = 50000
use_all_defects_for_test = false
test_defect_fraction_of_test_normals = 0.05
```

What this means:

- the train and validation splits contain only normal wafers
- the test split contains `5000` normal wafers
- `test_defect_fraction_of_test_normals = 0.05` means `5% of 5000 = 250` defect wafers in test

Examples:

- `0.01` means about `50` test defects
- `0.05` means `250` test defects
- `1.0` means `5000` test defects
- `use_all_defects_for_test = true` means use all available defect wafers in test

## When Retraining Is Needed

If you only change the test defect ratio and keep the same train/validation split:

- you usually do not need to retrain
- you can keep the trained checkpoint
- you only need to reevaluate the frozen model on the new test metadata

This is the recommended workflow when you want:

- `5%` as a development benchmark
- `1%` or closer-to-real-world prevalence as a final evaluation

## Changing the Ratio by Re-running Prep

You can change the ratio directly in `configs/data/data.toml` and rerun:

```powershell
python scripts/prepare_wm811k.py
```

Output names follow the active config:

- `0.05` produces `metadata_50k_5pct.csv` and `arrays_50k_5pct`
- `0.01` produces `metadata_50k_1pct.csv` and `arrays_50k_1pct`
- `use_all_defects_for_test = true` produces a variant such as `metadata_50k_all.csv`

This keeps different ratio variants in separate metadata files and arrays folders.

## Safe Way to Evaluate a New Ratio

If you only changed the test defect ratio:

1. Update `test_defect_fraction_of_test_normals` in `configs/data/data.toml`.
2. Run `python scripts/prepare_wm811k.py`.
3. Point the notebook or config `metadata_csv` field to the newly generated metadata file.
4. Reuse the same trained checkpoint if the train/validation split did not change.
5. Re-run evaluation on the new test metadata.

## Notes on Re-running Dataset Prep

Re-running `scripts/prepare_wm811k.py` after a config change is safe as long as each dataset variant keeps its own metadata file and arrays folder.

Reason:

- the script now derives output names from the active config
- different ratio settings produce different metadata filenames
- different ratio settings also produce different arrays folders

You can still overwrite an older variant if:

- you manually pass the same `--metadata-path` as an existing variant
- you change the config and then point training back to an older metadata file by mistake

So after changing the ratio, make sure your training or evaluation config points to the new metadata file.

## Recommended Use

- For model development: keep the current `5%` split if you are already comparing experiments on it.
- For final realism: make a separate `1%` metadata file and evaluate the final frozen model on that.
- For fair reporting: do not keep changing the test ratio during tuning. Pick the final evaluation ratio first and report it clearly.
