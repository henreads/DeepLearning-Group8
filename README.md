# DeepLearning-Group8

# Automated IC Wafer Defect Classification

**Group Number:** 08
**Group Members:**

- Henry Lee Jun, 1004219
- Chia Tang, 1007200
- Genson Low, 1006931

## Project Goal

This project studies anomaly detection on semiconductor wafer maps using the WM-811K dataset.
The initial codebase is organized around a simple baseline workflow:

1. Place the raw WM-811K pickle under `data/raw/`
2. Build processed arrays plus metadata CSV files
3. Open the notebooks as the main workflow for training and analysis
4. Use helper scripts only for data preparation or optional automation
5. Save weights, evaluation outputs, and training history for reproducibility

## Repository Layout

```text
configs/data/          Dataset preparation settings
configs/training/      Model training settings
scripts/               Notebook support scripts plus optional helpers
src/wafer_defect/      Package code
data/raw/              Local raw dataset files (ignored by git)
data/processed/        Local processed outputs (ignored by git)
artifacts/             Saved model outputs (ignored by git if desired later)
```

The processed data now has two separate anomaly-detection families:

- `data/processed/x64/wm811k/` for the original `50k`-normal benchmark workflow used by most notebooks
- `data/processed/x64/wm811k_patchcore_custom/` for the larger labeled WRN50 PatchCore split such as `120k / 10k / 20k`

The notebooks are the primary way to run experiments in this repo.
Top-level scripts are kept small on purpose:

- `scripts/prepare_wm811k.py` prepares the dataset used by the notebooks
- `scripts/evaluate_reconstruction_model.py` runs shared evaluation from saved checkpoints
- `scripts/evaluate_autoencoder_scores.py` runs the autoencoder score-ablation evaluation used by several notebooks
- `scripts/train_vae.py` stays at the top level because it is called directly from the VAE notebook
- `scripts/train_ts_distillation.py` stays at the top level because notebook `12` calls it directly

Optional ad hoc inspection helpers live under:

- `scripts/dev/`
- `scripts/anomaly_120k_labeled/` for repo-friendly CLI wrappers around the larger labeled WRN50 PatchCore notebooks

Older standalone experiment-runner scripts were removed during cleanup.
The experiment logic still lives primarily inside the notebooks plus reusable code under `src/wafer_defect/`, with thin `scripts/anomaly_120k_labeled/` wrappers added for the `120k` PatchCore runs.

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

If you also want the larger labeled WRN50 PatchCore split, prepare it separately:

```powershell
python scripts/prepare_wm811k.py --config configs/data/data_patchcore_wrn50_120k.toml
```

That writes to:

```text
data/processed/x64/wm811k_patchcore_custom/metadata_train120000_a6000_val10000_a500_test20000_a1000.csv
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
- `python scripts/prepare_wm811k.py --config configs/data/data_patchcore_wrn50_120k.toml` writes the larger labeled WRN50 PatchCore split under `data/processed/x64/wm811k_patchcore_custom/`
- `--metadata-path` still works as an override; when you use it, the arrays folder name is derived from that metadata filename
- with the current `configs/data/data.toml`, the main non-dev build uses `50000` normal wafers and samples test defects at `5%` of the test-normal count

The default config locations are now:

- data prep: `configs/data/data.toml`
- large WRN50 labeled split: `configs/data/data_patchcore_wrn50_120k.toml`
- training: `configs/training/*.toml`

After preparing the dataset, use the notebooks for the main experiments. The notebook folders are now split by workflow:

- `notebooks/anomaly_50k/` for the original anomaly-detection sequence on the `50k` benchmark split
- `notebooks/anomaly_120k_labeled/` for larger labeled anomaly-detection workflows, including the root-level WRN50 PatchCore sequence and `ts_resnet50/`
- `notebooks/classifier/` for the multiclass classification workflow

Main anomaly-detection notebooks on the original `50k` benchmark split:

- `notebooks/anomaly_50k/01_data_exploration.ipynb`
  Explore the processed metadata, class balance, and sample wafer maps. If you switch dataset variants, update the metadata path in the first code cell.
- `notebooks/anomaly_50k/02_autoencoder_training.ipynb`
  Train and evaluate the baseline autoencoder.
- `notebooks/anomaly_50k/03_vae_training.ipynb`
  Train and evaluate the convolutional VAE.
- `notebooks/anomaly_50k/04_svdd_training.ipynb`
  Train and evaluate the Deep SVDD experiment.
- `notebooks/anomaly_50k/05_autoencoder_batchnorm_training.ipynb`
  Train and evaluate the BatchNorm autoencoder variant on the same `64x64` 5% dataset and evaluation protocol.
- `notebooks/anomaly_50k/06_autoencoder_batchnorm_dropout_training.ipynb`
  Train and evaluate the BatchNorm + Dropout autoencoder variant on the same `64x64` 5% dataset and evaluation protocol.
- `notebooks/anomaly_50k/07_patchcore_training.ipynb`
  Fit and evaluate a PatchCore-style local nearest-neighbor detector on the same `64x64` 5% dataset using the BatchNorm autoencoder checkpoint as the frozen feature backbone.
- `notebooks/anomaly_50k/08_autoencoder_residual_training.ipynb`
  Train and evaluate a stronger residual autoencoder backbone on the same `64x64` 5% dataset and evaluation protocol.
- `notebooks/anomaly_50k/09_resnet18_backbone_baseline.ipynb`
  Evaluate a frozen pretrained ResNet18 backbone with simple center-distance scoring on the same `64x64` 5% dataset.
- `notebooks/anomaly_50k/10_patchcore_resnet18_training.ipynb`
  Fit and evaluate PatchCore on a frozen pretrained ResNet18 backbone using the same `64x64` 5% dataset and validation-threshold protocol.
- `notebooks/anomaly_50k/11_patchcore_resnet50_training.ipynb`
  Fit and evaluate PatchCore on a frozen pretrained ResNet50 backbone using the same `64x64` 5% dataset and validation-threshold protocol.
- `notebooks/anomaly_50k/12_ts_distillation_training.ipynb`
  Train and evaluate the teacher-student distillation detector, including optional shared evaluation and ablation cells.
- `notebooks/anomaly_50k/13_ts_resnet50_kaggle_import_analysis.ipynb`
  Inspect the Kaggle-imported teacher-student ResNet50 results and compare imported artifacts before final reporting.
- `notebooks/anomaly_120k_labeled/`
  Dedicated labeled-split anomaly folder containing the WRN50 PatchCore workflow at the folder root plus the `ts_resnet50/` pilot notebook.
- `notebooks/classifier/1_multiclass_classifier_training.ipynb`
  Prepare the `50k` labeled multiclass subset and train/evaluate the classifier without generating unlabeled predictions automatically.
  The current training config runs for up to `80` epochs with learning-rate decay and early stopping, saves the best checkpoint by validation balanced accuracy, and marks only high-confidence unlabeled predictions as safe pseudo-label candidates.
- `notebooks/classifier/1_multiclass_classifier_methodology.md`
  Summarize the research-aligned rationale behind the current multiclass classifier design and pseudo-labeling workflow.
- `notebooks/classifier/3_multiclass_classifier_final_labeling.ipynb`
  Generate unlabeled pseudo-labels only after you have selected the final classifier checkpoint you want to trust.
- `notebooks/classifier/4_multiclass_classifier_ensemble_workflow.ipynb`
  Evaluate and use an ensemble of multiple multiclass classifier checkpoints trained with different random seeds.
- `notebooks/classifier/5_multiclass_classifier_all_labeled_kaggle.ipynb`
  Rebuild the multiclass classifier on all labeled WM-811K rows using the stratified `80 / 10 / 10` split prepared for Kaggle training.
- `notebooks/classifier/6_seed07_unlabeled_pseudolabeling.ipynb`
  Use the saved `seed07` all-labeled classifier checkpoint to pseudo-label the unlabeled WM-811K rows and export confidence-scored CSV files for later validation work.
- `scripts/classifier/ensemble_multiclass_classifier.py`
  Evaluate either a simple averaged ensemble or a validation-fitted stacking ensemble from multiple classifier checkpoints.
  The stacking mode saves a reusable `stacking_combiner.json` file for later inference.
- `scripts/classifier/predict_unlabeled_multiclass_ensemble.py`
  Run unlabeled inference with the same checkpoint set, optionally using `--combiner-json` to apply a saved stacking combiner.
- `notebooks/classifier/2_multiclass_classifier_showcase.ipynb`
  Present the multiclass classifier results, plots, and example predictions after notebook `1` has produced the artifacts.

Current all-labeled classifier snapshot from the saved `seed07` Kaggle run:

- split: all labeled rows with stratified `80 / 10 / 10`
- best epoch: `29`
- validation balanced accuracy: `0.9463`
- test accuracy: `0.9587`
- test balanced accuracy: `0.9180`
- test macro F1: `0.8456`
- test weighted F1: `0.9619`

Per-class test metrics for that `seed07` run:

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| `none` | 0.9979 | 0.9619 | 0.9796 | 14,743 |
| `Center` | 0.8004 | 0.9698 | 0.8770 | 430 |
| `Donut` | 0.8246 | 0.8545 | 0.8393 | 55 |
| `Edge-Loc` | 0.6771 | 0.9171 | 0.7791 | 519 |
| `Edge-Ring` | 0.9704 | 0.9835 | 0.9769 | 968 |
| `Loc` | 0.6008 | 0.8468 | 0.7029 | 359 |
| `Near-full` | 1.0000 | 0.8667 | 0.9286 | 15 |
| `Random` | 0.7281 | 0.9540 | 0.8259 | 87 |
| `Scratch` | 0.5714 | 0.9076 | 0.7013 | 119 |

This checkpoint is currently the strongest fully exported artifact from the interrupted all-labeled Kaggle run. The main remaining weakness is precision on the local defect families such as `Loc`, `Scratch`, and `Edge-Loc`, with most large-support mistakes still coming from normal wafers being predicted as local defects.

<!-- BEGIN: NOTEBOOK6_KAGGLE_SYNC -->
Notebook `6` Kaggle pseudo-label snapshot from `jikutopepega/notebook6010fb082e`:

- synced output folder: `outputs\seed07_pseudolabel_bundle_kaggle_outputs`
- summary file: `outputs\seed07_pseudolabel_bundle_kaggle_outputs\unlabeled_predictions.seed07.symmary.json`
- rows scored: `638,507`
- confidence threshold: `0.90`
- accepted pseudo-labels: `417,831` (65.44%)
- predicted defect fraction: `36.12%`
- accepted defect fraction: `30.61%`
- mean confidence: `0.8637`
- mean accepted confidence: `0.9475`

Confidence bucket review:

| Threshold | Accepted Rows | Accepted Fraction | Defect Rows | `none` Rows |
| --- | --- | --- | --- | --- |
| 50% | 609,690 | 95.49% | 211,426 | 398,264 |
| 75% | 525,337 | 82.28% | 168,629 | 356,708 |
| 90% | 417,831 | 65.44% | 127,918 | 289,913 |

Standard classifier UMAP snapshot:

- labeled reference points: `3,349`
- pseudo-labeled points plotted: `6,865`
- mean plotted pseudo confidence: `0.9139`
- UMAP settings: `n_neighbors = 30`, `min_dist = 0.1`, `metric = cosine`

10A-style classifier UMAP snapshot:

- labeled normal points: `4,000`
- labeled defect points: `4,000`
- pseudo-labeled points plotted: `8,000`
- PCA dimension before UMAP: `50`
- UMAP settings: `n_neighbors = 15`, `min_dist = 0.1`, `metric = euclidean`

Label distribution across the full pseudo-label export, accepted subset, and both UMAP views:

| Label | All Scored | Accepted | Std UMAP | 10A UMAP |
| --- | --- | --- | --- | --- |
| `none` | 407,882 | 289,913 | 800 | 5,405 |
| `Center` | 21,294 | 13,825 | 800 | 285 |
| `Donut` | 613 | 364 | 465 | 12 |
| `Edge-Loc` | 33,053 | 10,958 | 800 | 304 |
| `Edge-Ring` | 15,074 | 6,254 | 800 | 147 |
| `Loc` | 25,583 | 7,248 | 800 | 233 |
| `Near-full` | 90,485 | 72,101 | 800 | 1,164 |
| `Random` | 13,223 | 9,486 | 800 | 182 |
| `Scratch` | 31,300 | 7,682 | 800 | 268 |
<!-- END: NOTEBOOK6_KAGGLE_SYNC -->

Recommended run order for a fresh setup:

1. Run `notebooks/anomaly_50k/01_data_exploration.ipynb` to confirm the processed dataset looks correct.
2. Run `notebooks/anomaly_50k/02_autoencoder_training.ipynb` for the baseline autoencoder.
3. Run `notebooks/anomaly_50k/03_vae_training.ipynb` if you want the VAE comparison.
4. Run `notebooks/anomaly_50k/04_svdd_training.ipynb` if you want the Deep SVDD comparison.
5. Run `notebooks/anomaly_50k/05_autoencoder_batchnorm_training.ipynb` if you want the BatchNorm autoencoder comparison.
6. Run `notebooks/anomaly_50k/06_autoencoder_batchnorm_dropout_training.ipynb` if you want the BatchNorm + Dropout autoencoder comparison.
7. Run `notebooks/anomaly_50k/07_patchcore_training.ipynb` if you want the PatchCore-style comparison.
8. Run `notebooks/anomaly_50k/08_autoencoder_residual_training.ipynb` if you want the stronger residual autoencoder backbone comparison.
9. Run `notebooks/anomaly_50k/09_resnet18_backbone_baseline.ipynb` if you want the plain pretrained ResNet18 backbone baseline.
10. Run `notebooks/anomaly_50k/10_patchcore_resnet18_training.ipynb` if you want PatchCore with a pretrained ResNet18 backbone.
11. Run `notebooks/anomaly_50k/11_patchcore_resnet50_training.ipynb` if you want PatchCore with a pretrained ResNet50 backbone.

How to run them:

- open the notebook you want
- select the virtual-environment kernel where `pip install -e .` was run
- run cells top to bottom
- keep the config paths unchanged unless you intentionally want a different dataset variant

## Changing the Test Defect Ratio

This section applies to the original `50k` benchmark family under `configs/data/data.toml`.

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

