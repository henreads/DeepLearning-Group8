# Wafer Defect Autoencoder Baseline Report

## Scope

This report summarizes the current anomaly-detection baseline built for the WM-811K / LSWMD wafer map project.

The goal of the baseline is:

- train only on normal wafers (`failureType == none`)
- treat defect wafers as anomalies at test time
- use reconstruction error from a convolutional autoencoder as the anomaly score

## Current Project Setup

Relevant files:

- [configs/data/data.toml](configs/data/data.toml)
- [configs/training/train_autoencoder.toml](configs/training/train_autoencoder.toml)
- [scripts/prepare_wm811k.py](scripts/prepare_wm811k.py)
- [scripts/train_autoencoder.py](scripts/train_autoencoder.py)
- [src/wafer_defect/models/autoencoder.py](src/wafer_defect/models/autoencoder.py)
- [src/wafer_defect/training/autoencoder.py](src/wafer_defect/training/autoencoder.py)
- [src/wafer_defect/training/vae.py](src/wafer_defect/training/vae.py)
- [notebooks/02_autoencoder_training.ipynb](notebooks/02_autoencoder_training.ipynb)

### Data preparation

The dataset is prepared by [prepare_wm811k.py](scripts/prepare_wm811k.py).

Key preprocessing behavior:

- reads the legacy `LSWMD.pkl` format
- keeps only explicitly labeled rows
- labels `failureType == none` as normal
- labels all other explicit failure types as defect / anomaly
- resizes wafer maps to `64 x 64`
- saves processed arrays as `.npy`
- writes repo-relative paths into metadata CSV files

### Split logic

Normal wafers are split by `split_normals(...)` into:

- `80%` train
- `10%` val
- `10%` test

Defects are added only to the `test` split by `sample_test_defects(...)`.

## Active Metadata Configuration

The active training config uses:

- [metadata_50k_5pct.csv](data/processed/x64/wm811k/metadata_50k_5pct.csv)

This metadata was created from:

- `50,000` sampled normal wafers
- test anomalies capped at `5%` of the number of test-normal wafers

Current effective split:

- train: `40,000` normal
- val: `5,000` normal
- test: `5,000` normal
- test: `250` anomaly

This was chosen to avoid evaluating on all anomalies at once and to produce a more controlled test distribution.

## Model and Training Configuration

Current training settings from [train_autoencoder.toml](configs/training/train_autoencoder.toml):

- model: convolutional autoencoder
- latent dimension: `128`
- optimizer: Adam
- learning rate: `0.001`
- weight decay: `0.0001`
- device: `auto` (`cuda` if available)
- max epochs: `25`
- early stopping patience: `5`
- early stopping min delta: `0.00005`
- checkpoint interval: every `5` epochs

Training now supports:

- `best_model.pt`
- `last_model.pt`
- `latest_checkpoint.pt`
- `checkpoint_epoch_5.pt`, `checkpoint_epoch_10.pt`, ...
- resume training through `resume_from` in the config

## Training Results

Saved history:

- [history.json](artifacts/x64/autoencoder_baseline/history.json)

Observed loss trend from the saved history:

- epoch 1: train `0.026390`, val `0.024768`
- epoch 10: train `0.024169`, val `0.024185`
- epoch 20: train `0.020241`, val `0.020260`
- epoch 25: train `0.019691`, val `0.019755`

Interpretation:

- training remained stable through 25 epochs
- validation loss kept improving
- there was no obvious overfitting in this run

Note:

- [summary.json](artifacts/x64/autoencoder_baseline/summary.json) currently records `best_epoch = 24` and `best_val_loss = 0.019792`
- however, [history.json](artifacts/x64/autoencoder_baseline/history.json) shows epoch 25 reached a slightly lower validation loss of `0.019755`
- this indicates the saved summary is slightly stale relative to the saved history

## Test Evaluation

The notebook evaluates the best checkpoint on the test split using reconstruction MSE as the anomaly score.

Validation-derived threshold:

- threshold from validation normals (95th percentile): `0.031658`

Metrics at the validation threshold:

- precision: `0.346154`
- recall: `0.504000`
- F1: `0.410423`
- AUROC: `0.809694`
- AUPRC: `0.447970`

Confusion matrix at the validation threshold:

|              | pred_normal | pred_anomaly |
| ------------ | ----------- | ------------ |
| true_normal  | 4762        | 238          |
| true_anomaly | 124         | 126          |

Interpretation:

- the autoencoder learned a real anomaly signal
- AUROC around `0.81` is a reasonable baseline
- precision and recall are both moderate
- the model catches about half of anomalies and misses the other half

## 128x128 Follow-Up Experiment

A second experiment was run with the same overall split logic but higher image resolution.

Configuration changes:

- metadata: `data/processed/x128/wm811k/metadata_50k_5pct.csv`
- image size: `128 x 128`
- batch size: `32`
- max epochs: `50`
- output dir: `artifacts/x128/autoencoder_baseline`

Observed training outcome:

- early stopped at epoch `22`
- saved best epoch: `17`
- best saved validation loss: `0.020438`
- validation threshold from normal scores: `0.032356`

Metrics at the validation threshold:

- precision: `0.309973`
- recall: `0.460000`
- F1: `0.370370`
- AUROC: `0.795673`
- AUPRC: `0.393266`

Confusion matrix at the validation threshold:

|              | pred_normal | pred_anomaly |
| ------------ | ----------- | ------------ |
| true_normal  | 4744        | 256          |
| true_anomaly | 135         | 115          |

Best observed test-set F1 from threshold sweep:

- threshold: `0.034747`
- precision: `0.462617`
- recall: `0.396000`
- F1: `0.426724`
- predicted anomalies: `213`

Important note on the saved best epoch:

- later epochs reached slightly lower validation losses than epoch `17`
- however, those improvements were smaller than `early_stopping_min_delta = 0.00005`
- so they were not counted as checkpoint-improving epochs

## 64x64 vs 128x128 Comparison

Comparison using the current runs:

| setup | val-threshold precision | val-threshold recall | val-threshold F1 | AUROC | AUPRC | best sweep F1 |
| ----- | ----------------------- | -------------------- | ---------------- | ----- | ----- | ------------- |
| 64x64 | `0.346154` | `0.504000` | `0.410423` | `0.809694` | `0.447970` | `0.473318` |
| 128x128 | `0.309973` | `0.460000` | `0.370370` | `0.795673` | `0.393266` | `0.426724` |

Interpretation:

- the `128x128` run was slower and more expensive
- it did not improve anomaly detection in this architecture/configuration
- the current `64x64` baseline remains the stronger result

## Threshold Sweep

The notebook also sweeps thresholds on the test set to analyze operating points.

Best observed test-set F1:

- threshold: `0.035031`
- precision: `0.563536`
- recall: `0.408000`
- F1: `0.473318`
- predicted anomalies: `180`

Interpretation:

- a slightly higher threshold than the validation cutoff improved F1
- this threshold reduces false positives somewhat
- recall falls to about `41%`, so many anomalies are still missed
- the baseline remains only moderately effective

## Score Distribution Interpretation

The histogram in the notebook shows:

- normal wafers concentrated mostly around lower reconstruction error
- anomaly wafers extending further to the right
- substantial overlap between the two distributions

This means:

- the model does separate the classes somewhat
- but the overlap is still large
- reconstruction error alone is not yet giving strong anomaly discrimination

## What Was Implemented

Completed work:

- WM-811K legacy pickle loading
- explicit normal-only training setup
- processed metadata generation with repo-relative paths
- resolution-specific processed folders for `x64` and `x128`
- 50k-normal subset generation
- alternate metadata with anomaly-capped test split
- convolutional autoencoder baseline
- notebook-based end-to-end training
- best checkpoint saving
- resumable periodic checkpoints
- validation-threshold metrics
- threshold sweep analysis
- higher-resolution `128x128` comparison experiment
- convolutional VAE baseline
- scriptable reconstruction-model evaluation
- VAE beta-sweep automation

## VAE Follow-Up Experiment

A `64x64` VAE was trained on the same split as the strongest autoencoder baseline, then tuned with a small beta sweep.

Initial VAE run with `beta = 0.01`:

- validation threshold: `0.035629`
- precision: `0.280323`
- recall: `0.416000`
- F1: `0.334944`
- AUROC: `0.766392`
- AUPRC: `0.369030`
- best test-sweep F1: `0.416667`

Best VAE setting from the beta sweep:

- chosen beta: `0.005`
- validation threshold: `0.034248`
- precision: `0.286104`
- recall: `0.420000`
- F1: `0.340357`
- AUROC: `0.771391`
- AUPRC: `0.372184`
- confusion matrix: `[[4738, 262], [145, 105]]`
- best test-sweep threshold: `0.038787`
- best test-sweep F1: `0.420253`

Comparison against the strongest autoencoder baseline:

| model | val-threshold precision | val-threshold recall | val-threshold F1 | AUROC | AUPRC | best sweep F1 |
| ----- | ----------------------- | -------------------- | ---------------- | ----- | ----- | ------------- |
| Autoencoder `64x64` | `0.346154` | `0.504000` | `0.410423` | `0.809694` | `0.447970` | `0.473318` |
| VAE `beta = 0.01` | `0.280323` | `0.416000` | `0.334944` | `0.766392` | `0.369030` | `0.416667` |
| VAE `beta = 0.005` | `0.286104` | `0.420000` | `0.340357` | `0.771391` | `0.372184` | `0.420253` |

Interpretation:

- the VAE learned a real anomaly signal
- lowering `beta` from `0.01` to `0.005` improved the VAE slightly
- even the best VAE setting remained clearly below the `64x64` autoencoder baseline
- reconstruction-based anomaly detection is useful, but still only moderately separable on this task

## Threshold Selection Rule

For the main reported result:

- use the threshold derived from validation-normal scores

For analysis only:

- report the best test-set threshold sweep as an operating-point study

Reason:

- the validation threshold is the fair evaluation threshold
- the best sweep threshold uses test labels and should not be treated as the deployment threshold

## Current Baseline Conclusion

The autoencoder is a valid first baseline, but it is not strong enough to be the final project result on its own.

Current conclusion:

- training is stable
- anomaly scores carry useful information
- AUROC is acceptable for a baseline
- thresholded detection quality is still limited
- the model misses a substantial fraction of defect wafers
- increasing resolution to `128x128` did not improve results in the current model
- VAE beta tuning improved results slightly, but still did not beat the `64x64` autoencoder baseline

## Recommended Next Steps

Recommended follow-up experiments:

- inspect false negatives visually
- compare with Deep SVDD
- try a stronger anomaly method such as PatchCore if time allows
- report both validation-threshold metrics and threshold-sweep analysis, but keep the validation-derived threshold as the main result

## VAE Beta Sweep

The repo now includes a simple sweep script:

- [run_vae_beta_sweep.py](scripts/run_vae_beta_sweep.py)

Default beta values:

- `0.001`
- `0.005`
- `0.01`
- `0.05`

Example command:

```powershell
python scripts/run_vae_beta_sweep.py
```

Outputs:

- per-beta training artifacts under `artifacts/x64/vae_beta_sweep/`
- per-beta evaluation summaries under each run's `evaluation/` folder
- aggregated summary at `artifacts/x64/vae_beta_sweep/beta_sweep_summary.json`

Observed sweep ranking:

1. `beta = 0.005`
2. `beta = 0.001`
3. `beta = 0.01`
4. `beta = 0.05`

This indicates that a small amount of KL regularization helps, but heavier regularization degrades anomaly-detection performance in the current VAE setup.
