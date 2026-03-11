# Wafer Defect Anomaly Detection Report

## Scope

This report summarizes the anomaly-detection experiments run so far for the WM-811K / LSWMD wafer map project.

Shared goal across experiments:

- train only on normal wafers (`failureType == none`)
- treat labeled defect wafers as anomalies at test time
- compare anomaly-scoring approaches under one consistent split and evaluation protocol

## Shared Setup

Relevant files:

- [configs/data/data.toml](configs/data/data.toml)
- [configs/training/train_autoencoder.toml](configs/training/train_autoencoder.toml)
- [configs/training/train_vae.toml](configs/training/train_vae.toml)
- [configs/training/train_svdd.toml](configs/training/train_svdd.toml)
- [scripts/prepare_wm811k.py](scripts/prepare_wm811k.py)
- [scripts/train_autoencoder.py](scripts/train_autoencoder.py)
- [scripts/train_vae.py](scripts/train_vae.py)
- [scripts/train_svdd.py](scripts/train_svdd.py)
- [scripts/evaluate_reconstruction_model.py](scripts/evaluate_reconstruction_model.py)
- [scripts/run_vae_beta_sweep.py](scripts/run_vae_beta_sweep.py)
- [src/wafer_defect/models/autoencoder.py](src/wafer_defect/models/autoencoder.py)
- [src/wafer_defect/models/vae.py](src/wafer_defect/models/vae.py)
- [src/wafer_defect/models/svdd.py](src/wafer_defect/models/svdd.py)
- [src/wafer_defect/training/autoencoder.py](src/wafer_defect/training/autoencoder.py)
- [src/wafer_defect/training/vae.py](src/wafer_defect/training/vae.py)
- [src/wafer_defect/training/svdd.py](src/wafer_defect/training/svdd.py)
- [notebooks/02_autoencoder_training.ipynb](notebooks/02_autoencoder_training.ipynb)
- [notebooks/03_vae_training.ipynb](notebooks/03_vae_training.ipynb)
- [notebooks/04_svdd_training.ipynb](notebooks/04_svdd_training.ipynb)

Data preparation:

- [prepare_wm811k.py](scripts/prepare_wm811k.py) reads the legacy `LSWMD.pkl` file
- only explicitly labeled rows are kept
- `failureType == none` is treated as normal
- all other explicit failure types are treated as anomaly
- wafer maps are resized and saved as `.npy`
- metadata CSV files store repo-relative array paths

Primary metadata used by the main experiments:

- [metadata_50k_5pct.csv](data/processed/x64/wm811k/metadata_50k_5pct.csv)

Effective split for the main `64x64` experiments:

- train: `40,000` normal
- val: `5,000` normal
- test: `5,000` normal
- test: `250` anomaly

Split rule:

- normals are split `80 / 10 / 10`
- defects are added only to the test split
- test anomalies are capped at `5%` of the number of test-normal wafers

## Overall Comparison

Main comparison across completed experiments:

| experiment    | model       | image size | val-threshold precision | val-threshold recall | val-threshold F1 | AUROC      | AUPRC      | best sweep F1 |
| ------------- | ----------- | ---------- | ----------------------- | -------------------- | ---------------- | ---------- | ---------- | ------------- |
| AE-64         | Autoencoder | `64x64`    | `0.346154`              | `0.504000`           | `0.410423`       | `0.809694` | `0.447970` | `0.473318`    |
| AE-128        | Autoencoder | `128x128`  | `0.309973`              | `0.460000`           | `0.370370`       | `0.795673` | `0.393266` | `0.426724`    |
| VAE-64-b0.01  | VAE         | `64x64`    | `0.280323`              | `0.416000`           | `0.334944`       | `0.766392` | `0.369030` | `0.416667`    |
| VAE-64-b0.005 | VAE         | `64x64`    | `0.286104`              | `0.420000`           | `0.340357`       | `0.771391` | `0.372184` | `0.420253`    |
| SVDD-64       | Deep SVDD   | `64x64`    | `0.304709`              | `0.440000`           | `0.360065`       | `0.787506` | `0.213108` | `0.366288`    |

![Overall experiment comparison](artifacts/report_plots/overall_experiment_comparison.png)

Current ranking:

1. Autoencoder `64x64`
2. Autoencoder `128x128`
3. Deep SVDD `64x64`
4. VAE `64x64`, `beta = 0.005`
5. VAE `64x64`, `beta = 0.01`

High-level interpretation:

- the `64x64` autoencoder remains the strongest experiment overall
- increasing the autoencoder resolution to `128x128` did not improve results
- VAE beta tuning helped slightly, but the VAE remained below both autoencoder runs
- Deep SVDD beat the tuned VAE on validation-threshold F1 and AUROC, but still did not beat the best autoencoder
- Deep SVDD had especially weak AUPRC, which suggests poorer ranking quality under class imbalance
- all tested approaches learn a real anomaly signal, but class separation is still only moderate

## Evaluation Rule

Main reported threshold:

- use the threshold derived from validation-normal scores at the `95th` percentile

Analysis-only threshold:

- also report the best test-set threshold sweep as an operating-point study

Reason:

- the validation threshold is the fair deployment-style threshold
- the best threshold sweep uses test labels and should not be treated as the main result

## Experiment 1: Autoencoder `64x64`

Purpose:

- establish the first convolutional anomaly-detection baseline on the shared split

Configuration:

- config: [train_autoencoder.toml](configs/training/train_autoencoder.toml)
- artifact dir: [artifacts/x64/autoencoder_baseline](artifacts/x64/autoencoder_baseline)
- latent dimension: `128`
- optimizer: Adam
- learning rate: `0.001`
- weight decay: `0.0001`
- max epochs: `25`
- early stopping patience: `5`
- early stopping min delta: `0.00005`

Training observations:

- saved history: [history.json](artifacts/x64/autoencoder_baseline/history.json)
- epoch 1: train `0.026390`, val `0.024768`
- epoch 10: train `0.024169`, val `0.024185`
- epoch 20: train `0.020241`, val `0.020260`
- epoch 25: train `0.019691`, val `0.019755`

Evaluation:

- validation threshold: `0.031658`
- precision: `0.346154`
- recall: `0.504000`
- F1: `0.410423`
- AUROC: `0.809694`
- AUPRC: `0.447970`
- confusion matrix: `[[4762, 238], [124, 126]]`
- best test-sweep threshold: `0.035031`
- best test-sweep F1: `0.473318`

![AE-64 training and evaluation plots](artifacts/report_plots/ae64_training_and_evaluation.png)

Interpretation:

- training was stable and validation loss kept improving
- the model learned a useful anomaly signal
- this remains the strongest experiment so far
- false positives and false negatives are still substantial, so the baseline is not yet strong enough to be the final project result by itself

Note:

- [summary.json](artifacts/x64/autoencoder_baseline/summary.json) records `best_epoch = 24` and `best_val_loss = 0.019792`
- [history.json](artifacts/x64/autoencoder_baseline/history.json) shows epoch 25 reached `0.019755`
- the saved summary appears slightly stale relative to the final history

## Experiment 2: Autoencoder `128x128`

Purpose:

- test whether higher image resolution improves the same autoencoder baseline

Configuration changes from Experiment 1:

- metadata: `data/processed/x128/wm811k/metadata_50k_5pct.csv`
- image size: `128 x 128`
- batch size: `32`
- max epochs: `50`
- output dir: `artifacts/x128/autoencoder_baseline`

Training observations:

- early stopped at epoch `22`
- saved best epoch: `17`
- best saved validation loss: `0.020438`

Evaluation:

- validation threshold: `0.032356`
- precision: `0.309973`
- recall: `0.460000`
- F1: `0.370370`
- AUROC: `0.795673`
- AUPRC: `0.393266`
- confusion matrix: `[[4744, 256], [135, 115]]`
- best test-sweep threshold: `0.034747`
- best test-sweep F1: `0.426724`

Interpretation:

- the `128x128` run was slower and more expensive
- it did not improve anomaly detection relative to the `64x64` autoencoder
- the current autoencoder evidence favors `64x64`, not `128x128`

![Autoencoder resolution comparison](artifacts/report_plots/autoencoder_resolution_comparison.png)

## Experiment 3: VAE `64x64`, `beta = 0.01`

Purpose:

- test whether a variational latent space improves over the reconstruction-only autoencoder baseline

Configuration:

- config: [train_vae.toml](configs/training/train_vae.toml)
- image size: `64x64`
- latent dimension: `128`
- beta: `0.01`

Evaluation:

- validation threshold: `0.035629`
- precision: `0.280323`
- recall: `0.416000`
- F1: `0.334944`
- AUROC: `0.766392`
- AUPRC: `0.369030`
- best test-sweep F1: `0.416667`

Interpretation:

- the VAE learned a real anomaly signal
- this first VAE run was clearly below the `64x64` autoencoder baseline
- this motivated a small beta sweep rather than dropping the VAE immediately

## Experiment 4: VAE `64x64` Beta Sweep

Purpose:

- tune KL regularization strength to see whether the VAE can close the gap to the autoencoder

Sweep script:

- [run_vae_beta_sweep.py](scripts/run_vae_beta_sweep.py)

Default beta values:

- `0.001`
- `0.005`
- `0.01`
- `0.05`

Outputs:

- per-beta artifacts under `artifacts/x64/vae_beta_sweep/`
- per-beta evaluation summaries under each run's `evaluation/` directory
- aggregated summary at [beta_sweep_summary.json](artifacts/x64/vae_beta_sweep/beta_sweep_summary.json)

Observed sweep ranking:

1. `beta = 0.001` by validation-threshold F1
2. `beta = 0.005` by AUROC, AUPRC, and best-sweep F1
3. `beta = 0.01`
4. `beta = 0.05`

![VAE beta sweep metrics](artifacts/report_plots/vae_beta_sweep.png)

Best VAE result from the sweep:

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

Interpretation:

- reducing beta from `0.01` to `0.005` improved the VAE slightly
- `beta = 0.001` gave the strongest validation-threshold F1 inside the saved sweep runs
- `beta = 0.005` gave the stronger overall ranking metrics and threshold-sweep behavior
- some KL regularization helps, but heavier regularization hurts in this setup
- even the best VAE remained clearly below the `64x64` autoencoder

## Experiment 5: Deep SVDD `64x64`

Purpose:

- compare a one-class distance-based model against the reconstruction-based baselines

Implementation:

- config: [train_svdd.toml](configs/training/train_svdd.toml)
- notebook: [04_svdd_training.ipynb](notebooks/04_svdd_training.ipynb)
- model: fixed-center Deep SVDD
- encoder: three strided convolution blocks
- latent dimension: `128`
- anomaly score: squared distance to the learned SVDD center
- center initialization: mean embedding over training-normal wafers with `center_eps` clipping

Evaluation:

- validation threshold: `0.000304`
- precision: `0.304709`
- recall: `0.440000`
- F1: `0.360065`
- AUROC: `0.787506`
- AUPRC: `0.213108`
- predicted anomalies: `361`
- confusion matrix: `[[4749, 251], [140, 110]]`
- best test-sweep threshold: `0.000302`
- best test-sweep precision: `0.307902`
- best test-sweep recall: `0.452000`
- best test-sweep F1: `0.366288`

![SVDD training and evaluation plots](artifacts/report_plots/svdd_training_and_evaluation.png)

Interpretation:

- Deep SVDD learned a usable anomaly signal on the shared split
- it improved over the tuned VAE on validation-threshold precision, recall, F1, and AUROC
- it still remained below the `64x64` autoencoder on validation-threshold F1, AUROC, AUPRC, and best sweep F1
- the especially low AUPRC suggests weaker score ranking under class imbalance
- this makes Deep SVDD a useful comparison result, but not the current best model

## Overall Interpretation

Across all completed experiments:

- the `64x64` autoencoder is still the best-performing model
- simply increasing autoencoder resolution did not help
- the VAE underperformed the autoencoder even after beta tuning
- Deep SVDD was a stronger alternative than the tuned VAE in some thresholded metrics, but not enough to replace the autoencoder baseline
- all tested models show overlap between normal and anomaly score distributions, which explains the moderate F1 values and missed anomalies
- the bottleneck looks more like limited class separation than threshold selection alone

## What Was Implemented

Completed work:

- WM-811K legacy pickle loading
- explicit normal-only training setup
- processed metadata generation with repo-relative paths
- resolution-specific processed folders for `x64` and `x128`
- `50k`-normal subset generation
- anomaly-capped test split generation
- convolutional autoencoder baseline
- convolutional VAE baseline
- Deep SVDD baseline
- notebook-based end-to-end training for AE, VAE, and SVDD
- scriptable reconstruction-model evaluation
- VAE beta-sweep automation
- best-checkpoint saving
- resumable periodic checkpoints
- validation-threshold metrics
- threshold sweep analysis

## Recommended Next Steps

Recommended follow-up work:

- inspect false negatives and false positives visually for the `64x64` autoencoder and Deep SVDD
- compare where the autoencoder and Deep SVDD disagree on the same test wafers
- try a stronger anomaly method such as PatchCore if time allows
- keep the validation-derived threshold as the main reported result, and treat test-set threshold sweeps as analysis only
