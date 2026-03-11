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
- [scripts/evaluate_autoencoder_scores.py](scripts/evaluate_autoencoder_scores.py)
- [scripts/train_vae.py](scripts/train_vae.py)
- [scripts/train_svdd.py](scripts/train_svdd.py)
- [scripts/evaluate_reconstruction_model.py](scripts/evaluate_reconstruction_model.py)
- [scripts/run_vae_beta_sweep.py](scripts/run_vae_beta_sweep.py)
- [src/wafer_defect/models/autoencoder.py](src/wafer_defect/models/autoencoder.py)
- [src/wafer_defect/models/vae.py](src/wafer_defect/models/vae.py)
- [src/wafer_defect/models/svdd.py](src/wafer_defect/models/svdd.py)
- [src/wafer_defect/scoring.py](src/wafer_defect/scoring.py)
- [src/wafer_defect/evaluation.py](src/wafer_defect/evaluation.py)
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

| experiment    | model       | score | image size | val-threshold precision | val-threshold recall | val-threshold F1 | AUROC      | AUPRC      | best sweep F1 |
| ------------- | ----------- | ----- | ---------- | ----------------------- | -------------------- | ---------------- | ---------- | ---------- | ------------- |
| AE-64-topk    | Autoencoder | `topk_abs_mean` | `64x64`    | `0.390374`              | `0.584000`           | `0.467949`       | `0.839282` | `0.522171` | `0.509091`    |
| AE-64-topk-43ep | Autoencoder | `topk_abs_mean` | `64x64`    | `0.381579`              | `0.580000`           | `0.460317`       | `0.834819` | `0.525162` | `0.520661`    |
| AE-64-mse     | Autoencoder | `mse_mean` | `64x64`    | `0.346154`              | `0.504000`           | `0.410423`       | `0.809694` | `0.447970` | `0.473318`    |
| AE-128-mse    | Autoencoder | `mse_mean` | `128x128`  | `0.309973`              | `0.460000`           | `0.370370`       | `0.795673` | `0.393266` | `0.426724`    |
| VAE-64-b0.01  | VAE         | `vae_score` | `64x64`    | `0.280323`              | `0.416000`           | `0.334944`       | `0.766392` | `0.369030` | `0.416667`    |
| VAE-64-b0.005 | VAE         | `vae_score` | `64x64`    | `0.286104`              | `0.420000`           | `0.340357`       | `0.771391` | `0.372184` | `0.420253`    |
| SVDD-64       | Deep SVDD   | `latent_distance` | `64x64`    | `0.304709`              | `0.440000`           | `0.360065`       | `0.787506` | `0.213108` | `0.366288`    |

![Overall experiment comparison](artifacts/report_plots/overall_experiment_comparison.png)

Current ranking:

1. Autoencoder `64x64` with `topk_abs_mean`
2. Autoencoder `64x64` with `topk_abs_mean`, longer-epoch rerun
3. Autoencoder `64x64` with `mse_mean`
4. Autoencoder `128x128` with `mse_mean`
5. Deep SVDD `64x64`
6. VAE `64x64`, `beta = 0.005`
7. VAE `64x64`, `beta = 0.01`

High-level interpretation:

- the strongest result is now the `64x64` autoencoder with `topk_abs_mean` scoring
- the same `64x64` autoencoder improved materially when the scoring rule changed, even without retraining
- retraining that same autoencoder longer produced only marginal changes, which suggests epoch count alone is not the main bottleneck
- increasing the autoencoder resolution to `128x128` did not improve results
- VAE beta tuning helped slightly, but the VAE remained below both autoencoder runs
- Deep SVDD beat the tuned VAE on validation-threshold F1 and AUROC, but still did not beat the best autoencoder
- Deep SVDD had especially weak AUPRC, which suggests poorer ranking quality under class imbalance
- local-error-focused scoring appears more effective than full-image averaging on wafer maps
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

- original run history was later overwritten by a longer rerun in the same artifact directory
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
- this was the strongest model architecture at the time of the first run
- later score ablation showed that the same checkpoint can perform substantially better with a different anomaly score
- false positives and false negatives are still substantial, so the baseline is not yet strong enough to be the final project result by itself

Note:

- the current [summary.json](artifacts/x64/autoencoder_baseline/summary.json) and [history.json](artifacts/x64/autoencoder_baseline/history.json) now correspond to the later longer-epoch rerun, not this original `25`-epoch baseline
- the original `25`-epoch baseline metrics above are kept for comparison because they were the first completed AE result on the shared split

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

## Experiment 3: Autoencoder `64x64` Score Ablation

Purpose:

- test whether the current best `64x64` autoencoder checkpoint can be improved by changing only the anomaly score
- determine whether score design is part of the bottleneck before retraining new models

Implementation:

- script: [evaluate_autoencoder_scores.py](scripts/evaluate_autoencoder_scores.py)
- scoring helpers: [scoring.py](src/wafer_defect/scoring.py)
- artifacts: [artifacts/x64/autoencoder_baseline/score_ablation](artifacts/x64/autoencoder_baseline/score_ablation)
- checkpoint used: `artifacts/x64/autoencoder_baseline/best_model.pt`

Scoring rules evaluated:

- `mse_mean`: mean squared reconstruction error over all pixels
- `mae_mean`: mean absolute reconstruction error over all pixels
- `max_abs`: maximum absolute reconstruction error over any single pixel
- `topk_abs_mean`: mean absolute reconstruction error over the top `1%` highest-error pixels
- `foreground_mse`: mean squared reconstruction error only on non-background pixels
- `foreground_mae`: mean absolute reconstruction error only on non-background pixels
- `pooled_mae_mean`: mean absolute reconstruction error after local average pooling of the error map

What these mean in practice:

- full-image means ask whether average reconstruction quality is enough to separate classes
- max-error asks whether the single worst pixel is informative
- top-k error asks whether small local high-error regions carry more signal than the global average
- foreground-only scores ask whether background pixels are diluting anomaly evidence
- pooled error asks whether smoothed local regions rank better than raw noisy pixel spikes

Results:

| score name | val-threshold F1 | AUROC | AUPRC | best sweep F1 |
| ---------- | ---------------- | ----- | ----- | ------------- |
| `topk_abs_mean` | `0.467949` | `0.839282` | `0.522171` | `0.509091` |
| `mse_mean` | `0.410423` | `0.809694` | `0.447970` | `0.473318` |
| `max_abs` | `0.323944` | `0.778442` | `0.239525` | `0.330341` |
| `foreground_mse` | `0.317263` | `0.760905` | `0.354126` | `0.377358` |
| `mae_mean` | `0.299835` | `0.762006` | `0.326166` | `0.352645` |
| `pooled_mae_mean` | `0.292763` | `0.754371` | `0.315182` | `0.342342` |
| `foreground_mae` | `0.239203` | `0.727066` | `0.278870` | `0.301370` |

![AE-64 score ablation](artifacts/report_plots/ae64_score_ablation.png)

Best score from the ablation:

- score: `topk_abs_mean`
- validation-threshold precision: `0.390374`
- validation-threshold recall: `0.584000`
- validation-threshold F1: `0.467949`
- AUROC: `0.839282`
- AUPRC: `0.522171`
- best threshold-sweep F1: `0.509091`

Interpretation:

- `topk_abs_mean` clearly outperformed the original `mse_mean` score on every main metric
- this shows the current `64x64` autoencoder checkpoint already contained more anomaly information than the original score extracted from it
- local defect regions matter more than full-image averaging on this dataset
- background dilution is real, but foreground-only averaging alone did not beat the top-k score
- `max_abs` was too unstable, which suggests that a single extreme pixel is noisier than a small cluster of high-error pixels
- this is the strongest result in the report so far, and it was achieved without retraining the model

Longer-epoch rerun using the selected score:

- notebook: [02_autoencoder_training.ipynb](notebooks/02_autoencoder_training.ipynb)
- artifact dir: [artifacts/x64/autoencoder_baseline](artifacts/x64/autoencoder_baseline)
- training override: `50` max epochs
- actual epochs run: `43`
- best epoch from [summary.json](artifacts/x64/autoencoder_baseline/summary.json): `38`
- best validation loss from [summary.json](artifacts/x64/autoencoder_baseline/summary.json): `0.019262`
- evaluation score: `topk_abs_mean`

Longer-rerun evaluation:

- validation threshold: `0.630657`
- precision: `0.381579`
- recall: `0.580000`
- F1: `0.460317`
- AUROC: `0.834819`
- AUPRC: `0.525162`
- confusion matrix: `[[4765, 235], [105, 145]]`
- best test-sweep threshold: `0.666356`
- best test-sweep precision: `0.538462`
- best test-sweep recall: `0.504000`
- best test-sweep F1: `0.520661`

Interpretation of the rerun:

- longer training kept the model in essentially the same performance band as the earlier `topk_abs_mean` result
- validation-threshold F1 dropped slightly from `0.467949` to `0.460317`
- AUPRC improved slightly from `0.522171` to `0.525162`
- best threshold-sweep F1 improved slightly from `0.509091` to `0.520661`
- the rerun shifted the thresholded behavior toward slightly better ranking and sweep performance, but not a clear overall breakthrough
- this suggests that simply extending training is a lower-leverage change than score design or more targeted model changes

Failure-mode analysis from the selected AE run:

- notebook section: [02_autoencoder_training.ipynb](notebooks/02_autoencoder_training.ipynb)
- evaluated on the validation-threshold predictions from the longer rerun
- error-type counts and mean scores:
  - true positive: `145`, mean score `0.758864`
  - false negative: `105`, mean score `0.540085`
  - false positive: `235`, mean score `0.671556`
  - true negative: `4765`, mean score `0.512200`

Defect-type recall on the anomaly test set:

| defect type | count | detected | recall | mean score |
| ----------- | ----- | -------- | ------ | ---------- |
| `Edge-Ring` | `84` | `68` | `0.809524` | `0.734189` |
| `Center` | `50` | `36` | `0.720000` | `0.726195` |
| `Edge-Loc` | `53` | `23` | `0.433962` | `0.605984` |
| `Loc` | `34` | `7` | `0.205882` | `0.552777` |
| `Donut` | `7` | `4` | `0.571429` | `0.646222` |
| `Scratch` | `15` | `3` | `0.200000` | `0.581900` |
| `Random` | `5` | `3` | `0.600000` | `0.656655` |
| `Near-full` | `2` | `1` | `0.500000` | `0.657818` |

Failure-analysis interpretation:

- the selected AE score separates the classes meaningfully, but the normal and anomaly score ranges still overlap
- the model is much stronger on broad or globally visible defects such as `Edge-Ring` and `Center`
- the weakest categories are more local or subtle patterns such as `Loc`, `Edge-Loc`, and `Scratch`
- false positives are all normal wafers by label, but their mean score `0.671556` is still relatively close to the operating region used for anomaly decisions
- this suggests that the current AE pipeline captures coarse structural deviations well, but still struggles on smaller localized defects
- that failure pattern makes one more focused AE tuning pass reasonable, but it also supports moving to a stronger local-anomaly method if the next AE change does not improve `Loc` / `Scratch` recall

## Experiment 4: VAE `64x64`, `beta = 0.01`

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

## Experiment 5: VAE `64x64` Beta Sweep

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

## Experiment 6: Deep SVDD `64x64`

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

- the best current result is the `64x64` autoencoder scored with `topk_abs_mean`
- the original `64x64` autoencoder checkpoint improved substantially just by changing the scoring rule
- retraining that autoencoder longer did not materially change the outcome, so epoch count alone is unlikely to be the key lever
- simply increasing autoencoder resolution did not help
- the VAE underperformed the autoencoder even after beta tuning
- Deep SVDD was a stronger alternative than the tuned VAE in some thresholded metrics, but not enough to replace the autoencoder baseline
- all tested models show overlap between normal and anomaly score distributions, which explains the moderate F1 values and missed anomalies
- the score-ablation result shows that part of the bottleneck was the scoring rule, not only the model architecture
- after fixing the score, the remaining bottleneck still looks more like limited class separation than threshold selection alone
- the AE failure analysis shows that the remaining weakness is concentrated in smaller local defects rather than large global defect patterns
- this makes the next decision clearer: either tune the AE specifically for local defects, or move to a method that is naturally stronger on local anomaly structure

## What Was Implemented

Completed work:

- WM-811K legacy pickle loading
- explicit normal-only training setup
- processed metadata generation with repo-relative paths
- resolution-specific processed folders for `x64` and `x128`
- `50k`-normal subset generation
- anomaly-capped test split generation
- convolutional autoencoder baseline
- autoencoder score-ablation evaluation
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

- compare the `topk_abs_mean` and `mse_mean` rankings on the same wafers
- avoid spending more time on longer-epoch reruns alone unless another change is paired with them
- run one focused AE follow-up aimed at local defects, such as smaller latent size, denoising training, or `L1` reconstruction loss
- compare where the autoencoder and Deep SVDD disagree on the same test wafers
- move to a stronger local-anomaly method such as PatchCore if the next focused AE run does not improve `Loc`, `Edge-Loc`, and `Scratch` recall
- keep the validation-derived threshold as the main reported result, and treat test-set threshold sweeps as analysis only
