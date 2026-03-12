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
- [configs/training/train_autoencoder_batchnorm.toml](configs/training/train_autoencoder_batchnorm.toml)
- [configs/training/train_autoencoder_batchnorm_dropout.toml](configs/training/train_autoencoder_batchnorm_dropout.toml)
- [configs/training/train_autoencoder_residual.toml](configs/training/train_autoencoder_residual.toml)
- [configs/training/train_patchcore.toml](configs/training/train_patchcore.toml)
- [configs/training/train_vae.toml](configs/training/train_vae.toml)
- [configs/training/train_svdd.toml](configs/training/train_svdd.toml)
- [scripts/prepare_wm811k.py](scripts/prepare_wm811k.py)
- [scripts/train_autoencoder.py](scripts/train_autoencoder.py)
- [scripts/evaluate_autoencoder_scores.py](scripts/evaluate_autoencoder_scores.py)
- [scripts/train_patchcore.py](scripts/train_patchcore.py)
- [scripts/train_vae.py](scripts/train_vae.py)
- [scripts/train_svdd.py](scripts/train_svdd.py)
- [scripts/evaluate_reconstruction_model.py](scripts/evaluate_reconstruction_model.py)
- [scripts/run_vae_beta_sweep.py](scripts/run_vae_beta_sweep.py)
- [src/wafer_defect/models/autoencoder.py](src/wafer_defect/models/autoencoder.py)
- [src/wafer_defect/models/patchcore.py](src/wafer_defect/models/patchcore.py)
- [src/wafer_defect/models/vae.py](src/wafer_defect/models/vae.py)
- [src/wafer_defect/models/svdd.py](src/wafer_defect/models/svdd.py)
- [src/wafer_defect/scoring.py](src/wafer_defect/scoring.py)
- [src/wafer_defect/evaluation.py](src/wafer_defect/evaluation.py)
- [src/wafer_defect/training/autoencoder.py](src/wafer_defect/training/autoencoder.py)
- [src/wafer_defect/training/patchcore.py](src/wafer_defect/training/patchcore.py)
- [src/wafer_defect/training/vae.py](src/wafer_defect/training/vae.py)
- [src/wafer_defect/training/svdd.py](src/wafer_defect/training/svdd.py)
- [notebooks/02_autoencoder_training.ipynb](notebooks/02_autoencoder_training.ipynb)
- [notebooks/05_autoencoder_batchnorm_training.ipynb](notebooks/05_autoencoder_batchnorm_training.ipynb)
- [notebooks/06_autoencoder_batchnorm_dropout_training.ipynb](notebooks/06_autoencoder_batchnorm_dropout_training.ipynb)
- [notebooks/07_patchcore_training.ipynb](notebooks/07_patchcore_training.ipynb)
- [notebooks/08_autoencoder_residual_training.ipynb](notebooks/08_autoencoder_residual_training.ipynb)
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
| AE-64-BN-max  | Autoencoder + BatchNorm | `max_abs` | `64x64`    | `0.401442`              | `0.668000`           | `0.501502`       | `0.834023` | `0.568039` | `0.629808`    |
| AE-64-BN-topk | Autoencoder + BatchNorm | `topk_abs_mean` | `64x64`    | `0.346247`              | `0.572000`           | `0.431373`       | `0.790020` | `0.603447` | `0.655172`    |
| AE-64-BN-DO0.00 | Autoencoder + BatchNorm + Dropout `0.00` | `max_abs` | `64x64`    | `0.393120`              | `0.640000`           | `0.487062`       | `0.850790` | `0.616946` | `0.656642`    |
| AE-64-BN-DO0.05 | Autoencoder + BatchNorm + Dropout `0.05` | `max_abs` | `64x64`    | `0.377828`              | `0.668000`           | `0.482659`       | `0.835035` | `0.551700` | `0.609959`    |
| AE-64-BN-DO0.10 | Autoencoder + BatchNorm + Dropout `0.10` | `max_abs` | `64x64`    | `0.385343`              | `0.652000`           | `0.484398`       | `0.844670` | `0.570245` | `0.634615`    |
| AE-64-BN-DO0.20 | Autoencoder + BatchNorm + Dropout `0.20` | `max_abs` | `64x64`    | `0.370115`              | `0.644000`           | `0.470073`       | `0.841431` | `0.574973` | `0.633929`    |
| AE-64-Res-max  | Residual Autoencoder | `max_abs` | `64x64`    | `0.374419`              | `0.644000`           | `0.473529`       | `0.843360` | `0.588907` | `0.625592`    |
| AE-64-topk    | Autoencoder | `topk_abs_mean` | `64x64`    | `0.390374`              | `0.584000`           | `0.467949`       | `0.839282` | `0.522171` | `0.509091`    |
| AE-64-topk-43ep | Autoencoder | `topk_abs_mean` | `64x64`    | `0.381579`              | `0.580000`           | `0.460317`       | `0.834819` | `0.525162` | `0.520661`    |
| AE-64-Res-topk | Residual Autoencoder | `topk_abs_mean` | `64x64`    | `0.356974`              | `0.604000`           | `0.448737`       | `0.804607` | `0.626014` | `0.678133`    |
| AE-64-mse     | Autoencoder | `mse_mean` | `64x64`    | `0.346154`              | `0.504000`           | `0.410423`       | `0.809694` | `0.447970` | `0.473318`    |
| AE-128-mse    | Autoencoder | `mse_mean` | `128x128`  | `0.309973`              | `0.460000`           | `0.370370`       | `0.795673` | `0.393266` | `0.426724`    |
| SVDD-64       | Deep SVDD   | `latent_distance` | `64x64`    | `0.304709`              | `0.440000`           | `0.360065`       | `0.787506` | `0.213108` | `0.366288`    |
| PatchCore-mean-mb50k | PatchCore | `mean` | `64x64`    | `0.283747`              | `0.412000`           | `0.336052`       | `0.850786` | `0.226325` | `0.389447`    |
| VAE-64-b0.005 | VAE         | `vae_score` | `64x64`    | `0.286104`              | `0.420000`           | `0.340357`       | `0.771391` | `0.372184` | `0.420253`    |
| VAE-64-b0.01  | VAE         | `vae_score` | `64x64`    | `0.280323`              | `0.416000`           | `0.334944`       | `0.766392` | `0.369030` | `0.416667`    |
| PatchCore-topk-mb50k-r010 | PatchCore | `topk_mean` | `64x64`    | `0.166134`              | `0.208000`           | `0.184725`       | `0.808633` | `0.148827` | `0.304950`    |
| PatchCore-topk-mb50k-r005 | PatchCore | `topk_mean` | `64x64`    | `0.112583`              | `0.136000`           | `0.123188`       | `0.777215` | `0.120862` | `0.241529`    |
| PatchCore-topk-mb10k-r005 | PatchCore | `topk_mean` | `64x64`    | `0.053004`              | `0.060000`           | `0.056285`       | `0.659112` | `0.072701` | `0.157971`    |
| PatchCore-max-mb50k | PatchCore | `max` | `64x64`    | `0.052632`              | `0.060000`           | `0.056075`       | `0.678692` | `0.080039` | `0.152152`    |
| PatchCore-max-mb10k | PatchCore | `max` | `64x64`    | `0.029412`              | `0.036000`           | `0.032374`       | `0.587003` | `0.061002` | `0.120301`    |

How to read these metrics:

- `val-threshold precision`: of the wafers predicted as anomalies, how many were actually anomalous
- `val-threshold recall`: of the true anomalous wafers, how many the model successfully detected
- `val-threshold F1`: the main thresholded summary metric used in this report; it balances precision and recall at the deployed validation-derived threshold
- `AUROC`: ranking quality across all possible thresholds; useful to see whether anomalous wafers generally receive higher scores than normal wafers
- `AUPRC`: ranking quality under class imbalance; often more informative than AUROC when anomalies are rare
- `best sweep F1`: best possible F1 if the threshold were chosen using test labels; useful for analysis, but not the main reported result

Metric priority for this project:

1. `val-threshold F1`
2. `val-threshold precision` and `val-threshold recall`
3. `AUPRC`
4. `AUROC`
5. `best sweep F1`

Why this order:

- the project needs a real anomaly decision rule, so thresholded metrics matter most
- the threshold is chosen from validation normals, which makes the thresholded result the fairest deployment-style comparison
- `AUPRC` and `AUROC` are still useful, but they summarize score ranking rather than one actual operating point
- `best sweep F1` uses test labels, so it is an optimistic diagnostic metric and should not drive the main conclusion

![Overall experiment comparison](artifacts/report_plots/overall_experiment_comparison.png)

Current ranking:

This ranking is based mainly on `val-threshold F1`, with the other metrics used as supporting evidence.

1. Autoencoder + BatchNorm `64x64` with `max_abs`
2. Autoencoder + BatchNorm + Dropout `0.00` `64x64` with `max_abs`
3. Autoencoder + BatchNorm + Dropout `0.10` `64x64` with `max_abs`
4. Autoencoder + BatchNorm + Dropout `0.05` `64x64` with `max_abs`
5. Autoencoder + BatchNorm + Dropout `0.20` `64x64` with `max_abs`
6. Residual autoencoder `64x64` with `max_abs`
7. Autoencoder `64x64` with `topk_abs_mean`
8. Autoencoder `64x64` with `topk_abs_mean`, longer-epoch rerun
9. Residual autoencoder `64x64` with `topk_abs_mean`
10. Autoencoder + BatchNorm `64x64` with `topk_abs_mean`
11. Autoencoder `64x64` with `mse_mean`
12. Autoencoder `128x128` with `mse_mean`
13. Deep SVDD `64x64`
14. VAE `64x64`, `beta = 0.005`
15. PatchCore `64x64`, `mean`, memory bank `50k`
16. VAE `64x64`, `beta = 0.01`
17. PatchCore `64x64`, `topk_mean`, memory bank `50k`, top-k ratio `0.10`
18. PatchCore `64x64`, `topk_mean`, memory bank `50k`, top-k ratio `0.05`
19. PatchCore `64x64`, `topk_mean`, memory bank `10k`, top-k ratio `0.05`
20. PatchCore `64x64`, `max`, memory bank `50k`
21. PatchCore `64x64`, `max`, memory bank `10k`

High-level interpretation:

- adding BatchNorm changed the scoring behavior of the autoencoder substantially
- BatchNorm with the old `topk_abs_mean` score was weaker than the baseline autoencoder on F1 and AUROC, even though it improved AUPRC
- once the BatchNorm checkpoint was rescored, `max_abs` became the strongest validation-threshold result in the report so far
- the same `64x64` autoencoder improved materially when the scoring rule changed, even without retraining
- retraining that same autoencoder longer produced only marginal changes, which suggests epoch count alone is not the main bottleneck
- the new evidence suggests architecture and scoring interact strongly; the best score for one checkpoint is not necessarily the best score for another
- the dropout sweep produced several meaningful AE variants, but none beat the no-dropout BatchNorm model
- the residual autoencoder was a meaningful architecture upgrade over the plain `topk_abs_mean` AE path, but it still did not beat the best BatchNorm AE when both were scored with their best thresholded rule
- increasing the autoencoder resolution to `128x128` did not improve results
- VAE beta tuning helped slightly, but the VAE remained below both autoencoder runs
- Deep SVDD beat the tuned VAE on validation-threshold F1 and AUROC, but still did not beat the best autoencoder
- PatchCore worked only when the wafer-level reduction became less brittle; `mean` reduction with a `50k` memory bank clearly beat the `max` variants
- the best PatchCore variant reached competitive AUROC (`0.850786`), but its validation-threshold F1 (`0.336052`) still stayed below the best AE and below Deep SVDD
- this suggests the current PatchCore setup has usable ranking quality but a weaker deployed operating point under the shared threshold rule
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

## Autoencoder Experiment Family

### Baseline: Autoencoder `64x64`

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

### Variant: Autoencoder `64x64` with BatchNorm

Purpose:

- test whether inserting BatchNorm into the same `64x64` autoencoder improves anomaly detection on the shared `5%` test-defect split
- keep the same dataset, optimizer family, threshold rule, and evaluation notebook flow as the baseline

Configuration:

- config: [train_autoencoder_batchnorm.toml](configs/training/train_autoencoder_batchnorm.toml)
- notebook: [05_autoencoder_batchnorm_training.ipynb](notebooks/05_autoencoder_batchnorm_training.ipynb)
- artifact dir: [artifacts/x64/autoencoder_batchnorm](artifacts/x64/autoencoder_batchnorm)
- metadata: `data/processed/x64/wm811k/metadata_50k_5pct.csv`
- latent dimension: `128`
- BatchNorm: enabled in encoder and decoder
- optimizer: Adam
- learning rate: `0.001`
- weight decay: `0.0001`
- max epochs: `50`
- early stopping patience: `5`
- early stopping min delta: `0.00005`

Training observations:

- early stopped at epoch `13`
- best epoch: `8`
- best validation loss: `0.014935`
- epoch 1: train `0.020315`, val `0.016544`
- epoch 8: train `0.014960`, val `0.014935`
- epoch 13: train `0.014813`, val `0.014998`

Evaluation with the same main score as the baseline notebook (`topk_abs_mean`):

- validation threshold: `0.532667`
- precision: `0.346247`
- recall: `0.572000`
- F1: `0.431373`
- AUROC: `0.790020`
- AUPRC: `0.603447`
- confusion matrix: `[[4730, 270], [107, 143]]`
- best test-sweep threshold: `0.600826`
- best test-sweep precision: `0.852564`
- best test-sweep recall: `0.532000`
- best test-sweep F1: `0.655172`

Score ablation on the BatchNorm checkpoint:

| score name | val-threshold F1 | AUROC | AUPRC | best sweep F1 |
| ---------- | ---------------- | ----- | ----- | ------------- |
| `max_abs` | `0.501502` | `0.834023` | `0.568039` | `0.629808` |
| `topk_abs_mean` | `0.431373` | `0.790020` | `0.603447` | `0.655172` |
| `mse_mean` | `0.326733` | `0.779451` | `0.345216` | `0.385000` |
| `foreground_mse` | `0.278317` | `0.738702` | `0.278022` | `0.329114` |
| `mae_mean` | `0.259567` | `0.728685` | `0.284041` | `0.330969` |
| `pooled_mae_mean` | `0.257095` | `0.722959` | `0.278594` | `0.323760` |
| `foreground_mae` | `0.242017` | `0.700971` | `0.244630` | `0.296512` |

Best BatchNorm score under the main validation-threshold rule:

- score: `max_abs`
- validation-threshold precision: `0.401442`
- validation-threshold recall: `0.668000`
- validation-threshold F1: `0.501502`
- AUROC: `0.834023`
- AUPRC: `0.568039`
- best threshold-sweep F1: `0.629808`

Failure-mode analysis from the BatchNorm notebook under `topk_abs_mean`:

- true positive: `143`, mean score `0.740248`
- false negative: `107`, mean score `0.478619`
- false positive: `270`, mean score `0.562314`
- true negative: `4730`, mean score `0.475181`

Defect-type recall under `topk_abs_mean`:

- `Edge-Ring`: `0.833333`
- `Center`: `0.700000`
- `Edge-Loc`: `0.396226`
- `Loc`: `0.176471`
- `Scratch`: `0.200000`
- `Donut`: `0.428571`
- `Random`: `0.600000`
- `Near-full`: `1.000000`

Interpretation:

- BatchNorm did not help when paired with the old baseline score `topk_abs_mean`; validation-threshold F1 fell from `0.460317` in the longer baseline rerun to `0.431373`
- the BatchNorm checkpoint still learned a useful anomaly signal, shown by its high AUPRC (`0.603447`) and very strong best-sweep behavior
- the score ablation is the key result: BatchNorm changed the error distribution enough that `max_abs`, which was weak on the baseline model, became the best fair-threshold score for this checkpoint
- under the shared validation-threshold rule, `max_abs` on the BatchNorm checkpoint is the strongest completed result in the report so far by F1, recall, and AUPRC
- AUROC for BatchNorm + `max_abs` is essentially tied with the stronger baseline autoencoder runs, so the gain is mainly better thresholded operating behavior rather than dramatically better ranking
- the remaining weak classes are still `Loc`, `Scratch`, and parts of `Edge-Loc`, so BatchNorm alone does not solve the hardest defect patterns

Note:

- the current [summary.json](artifacts/x64/autoencoder_baseline/summary.json) and [history.json](artifacts/x64/autoencoder_baseline/history.json) now correspond to the later longer-epoch rerun, not this original `25`-epoch baseline
- the original `25`-epoch baseline metrics above are kept for comparison because they were the first completed AE result on the shared split

### Variant: Autoencoder `64x64` with BatchNorm + Dropout Sweep

Purpose:

- test whether light dropout improves the BatchNorm autoencoder on the same shared `64x64` 5% test-defect split
- keep the same data, optimizer family, threshold rule, and evaluation flow while sweeping only the dropout rate

Configuration:

- config: [train_autoencoder_batchnorm_dropout.toml](configs/training/train_autoencoder_batchnorm_dropout.toml)
- notebook: [06_autoencoder_batchnorm_dropout_training.ipynb](notebooks/06_autoencoder_batchnorm_dropout_training.ipynb)
- artifact root: [artifacts/x64/autoencoder_batchnorm_dropout](artifacts/x64/autoencoder_batchnorm_dropout)
- metadata: `data/processed/x64/wm811k/metadata_50k_5pct.csv`
- latent dimension: `128`
- BatchNorm: enabled
- dropout sweep: `0.00`, `0.05`, `0.10`, `0.20`
- selection rule: lowest validation loss

Sweep summary:

| dropout | best epoch | best val loss | epochs ran |
| ------- | ---------- | ------------- | ---------- |
| `0.00` | `11` | `0.014824` | `16` |
| `0.10` | `25` | `0.014978` | `30` |
| `0.05` | `16` | `0.015063` | `21` |
| `0.20` | `20` | `0.015660` | `25` |

Best score-ablation result for each dropout setting:

| dropout | best score | precision | recall | F1 | AUROC | AUPRC | best sweep F1 |
| ------- | ---------- | --------- | ------ | -- | ----- | ----- | ------------- |
| `0.00` | `max_abs` | `0.393120` | `0.640000` | `0.487062` | `0.850790` | `0.616946` | `0.656642` |
| `0.05` | `max_abs` | `0.377828` | `0.668000` | `0.482659` | `0.835035` | `0.551700` | `0.609959` |
| `0.10` | `max_abs` | `0.385343` | `0.652000` | `0.484398` | `0.844670` | `0.570245` | `0.634615` |
| `0.20` | `max_abs` | `0.370115` | `0.644000` | `0.470073` | `0.841431` | `0.574973` | `0.633929` |

Selected run:

- selected dropout: `0.00`
- selected output dir: `artifacts/x64/autoencoder_batchnorm_dropout/dropout_0p00`

Score ablation on the selected `0.00` run:

| score name | val-threshold F1 | AUROC | AUPRC | best sweep F1 |
| ---------- | ---------------- | ----- | ----- | ------------- |
| `max_abs` | `0.487062` | `0.850790` | `0.616946` | `0.656642` |
| `topk_abs_mean` | `0.435703` | `0.799805` | `0.602296` | `0.668380` |
| `mse_mean` | `0.336601` | `0.784481` | `0.343180` | `0.394432` |
| `foreground_mse` | `0.272131` | `0.734939` | `0.263042` | `0.302083` |
| `mae_mean` | `0.256494` | `0.727365` | `0.262883` | `0.318408` |
| `pooled_mae_mean` | `0.251634` | `0.721275` | `0.256669` | `0.310502` |
| `foreground_mae` | `0.236887` | `0.694310` | `0.228523` | `0.276029` |

Interpretation:

- dropout did not help this autoencoder family; the best sweep result was `0.00`, not a positive dropout value
- `0.05` and `0.10` stayed close but still underperformed the no-dropout run, while `0.20` was clearly too strong
- the selected no-dropout run behaved similarly to the BatchNorm notebook, which suggests the dropout sweep mostly confirmed that latent dropout is not a useful lever here
- even after score ablation, the best dropout-sweep result `max_abs` with F1 `0.487062` remained below the earlier BatchNorm-only best result `0.501502`
- this is still a useful negative result because it narrows the AE search space: BatchNorm is promising, but dropout is not

### Variant: Residual Autoencoder `64x64`

Purpose:

- test whether a stronger residual encoder-decoder architecture improves the `64x64` autoencoder family on the shared `5%` test-defect split
- keep the same training and evaluation protocol so the architecture change is isolated from the rest of the pipeline

Configuration:

- config: [train_autoencoder_residual.toml](configs/training/train_autoencoder_residual.toml)
- notebook: [08_autoencoder_residual_training.ipynb](notebooks/08_autoencoder_residual_training.ipynb)
- artifact dir: [artifacts/x64/autoencoder_residual](artifacts/x64/autoencoder_residual)
- architecture: residual autoencoder with residual down/up blocks
- latent dimension: `128`
- BatchNorm: enabled
- optimizer: Adam
- learning rate: `0.001`
- weight decay: `0.0001`
- max epochs: `50`
- early stopping patience: `5`
- early stopping min delta: `0.00005`

Training observations:

- early stopped at epoch `20`
- best epoch: `15`
- best validation loss: `0.014504`
- epoch 1: train `0.018846`, val `0.016567`
- epoch 10: train `0.014621`, val `0.014580`
- epoch 15: train `0.014508`, val `0.014504`
- epoch 20: train `0.014442`, val `0.014488`

Evaluation with the notebook default score (`topk_abs_mean`):

- validation threshold: `0.537005`
- precision: `0.356974`
- recall: `0.604000`
- F1: `0.448737`
- AUROC: `0.804607`
- AUPRC: `0.626014`
- confusion matrix: `[[4728, 272], [99, 151]]`
- best test-sweep threshold: `0.637794`
- best test-sweep precision: `0.878981`
- best test-sweep recall: `0.552000`
- best test-sweep F1: `0.678133`

Score ablation on the residual checkpoint:

| score name | val-threshold F1 | AUROC | AUPRC | best sweep F1 |
| ---------- | ---------------- | ----- | ----- | ------------- |
| `max_abs` | `0.473529` | `0.843360` | `0.588907` | `0.625592` |
| `topk_abs_mean` | `0.448737` | `0.804607` | `0.626014` | `0.678133` |
| `mse_mean` | `0.392220` | `0.806132` | `0.426133` | `0.463415` |
| `foreground_mse` | `0.322581` | `0.778402` | `0.353748` | `0.410758` |
| `mae_mean` | `0.269360` | `0.734534` | `0.263283` | `0.315789` |
| `pooled_mae_mean` | `0.260000` | `0.728333` | `0.255770` | `0.308824` |
| `foreground_mae` | `0.245791` | `0.710838` | `0.230483` | `0.270784` |

Best residual score under the main validation-threshold rule:

- score: `max_abs`
- validation-threshold precision: `0.374419`
- validation-threshold recall: `0.644000`
- validation-threshold F1: `0.473529`
- AUROC: `0.843360`
- AUPRC: `0.588907`
- best threshold-sweep F1: `0.625592`

Failure-mode analysis under `topk_abs_mean`:

- true positive: `151`, mean score `0.847068`
- false negative: `99`, mean score `0.480705`
- false positive: `272`, mean score `0.576203`
- true negative: `4728`, mean score `0.477503`

Defect-type recall under `topk_abs_mean`:

- `Edge-Ring`: `0.857143`
- `Center`: `0.720000`
- `Edge-Loc`: `0.433962`
- `Loc`: `0.235294`
- `Scratch`: `0.133333`
- `Donut`: `0.571429`
- `Random`: `0.800000`
- `Near-full`: `1.000000`

Interpretation:

- the residual architecture is a real improvement over the weaker plain-AE scoring path, especially for `topk_abs_mean`
- under score ablation, `max_abs` again became the strongest thresholded score for the checkpoint
- the best residual result (`F1 = 0.473529`) is competitive with several AE-family variants, but it still does not beat the BatchNorm AE + `max_abs` winner (`F1 = 0.501502`)
- the residual model still struggles with `Loc` and `Scratch`, so it does not remove the main local-defect weakness
- this makes it a useful stronger backbone candidate, but not the new best end-to-end detector

### Variant: Autoencoder `128x128`

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

### Score Ablation: Autoencoder `64x64`

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

## Experiment 7: PatchCore Sweep `64x64`

Purpose:

- test a local nearest-neighbor anomaly method on the same shared `64x64` 5% test-defect split
- check whether a patch-based method can recover the smaller local defects that remain hard for the autoencoder family

Implementation:

- config: [train_patchcore.toml](configs/training/train_patchcore.toml)
- notebook: [07_patchcore_training.ipynb](notebooks/07_patchcore_training.ipynb)
- artifact dir: [artifacts/x64/patchcore_ae_bn](artifacts/x64/patchcore_ae_bn)
- backbone checkpoint: [best_model.pt](artifacts/x64/autoencoder_batchnorm/best_model.pt)
- backbone source: frozen BatchNorm autoencoder encoder
- compared variants:
  - `max`, memory bank `10k`
  - `max`, memory bank `50k`
  - `topk_mean`, memory bank `10k`, top-k ratio `0.05`
  - `topk_mean`, memory bank `50k`, top-k ratio `0.05`
  - `topk_mean`, memory bank `50k`, top-k ratio `0.10`
  - `mean`, memory bank `50k`

Sweep summary:

| variant | reduction | memory bank | top-k ratio | val-threshold F1 | AUROC | AUPRC | best sweep F1 |
| ------- | --------- | ----------- | ----------- | ---------------- | ----- | ----- | ------------- |
| `mean_mb50k` | `mean` | `50000` | `0.10` | `0.336052` | `0.850786` | `0.226325` | `0.389447` |
| `topk_mb50k_r010` | `topk_mean` | `50000` | `0.10` | `0.184725` | `0.808633` | `0.148827` | `0.304950` |
| `topk_mb50k_r005` | `topk_mean` | `50000` | `0.05` | `0.123188` | `0.777215` | `0.120862` | `0.241529` |
| `topk_mb10k_r005` | `topk_mean` | `10000` | `0.05` | `0.056285` | `0.659112` | `0.072701` | `0.157971` |
| `max_mb50k` | `max` | `50000` | `0.10` | `0.056075` | `0.678692` | `0.080039` | `0.152152` |
| `max_mb10k` | `max` | `10000` | `0.10` | `0.032374` | `0.587003` | `0.061002` | `0.120301` |

Best PatchCore variant under the main validation-threshold rule:

- variant: `mean_mb50k`
- precision: `0.283747`
- recall: `0.412000`
- F1: `0.336052`
- AUROC: `0.850786`
- AUPRC: `0.226325`
- best test-sweep threshold: `0.146545`
- best test-sweep F1: `0.389447`

Failure analysis for `mean_mb50k`:

- true positive: `103`, mean score `0.185981`
- false negative: `147`, mean score `0.132794`
- false positive: `260`, mean score `0.191426`
- true negative: `4740`, mean score `0.105194`

Defect-type recall for `mean_mb50k`:

- `Center`: `0.680000`
- `Edge-Ring`: `0.369048`
- `Edge-Loc`: `0.358491`
- `Loc`: `0.235294`
- `Scratch`: `0.133333`
- `Donut`: `0.428571`
- `Random`: `0.800000`
- `Near-full`: `1.000000`

Interpretation:

- PatchCore only became competitive when the wafer-level score moved away from the brittle `max` reduction
- the larger `50k` memory bank helped substantially; both `10k` variants were clearly weaker
- `mean_mb50k` produced the best PatchCore result by every main metric in the sweep
- the best PatchCore AUROC (`0.850786`) is strong and shows that the score ranking is useful overall
- the validation-threshold F1 stayed moderate, which means the operating point under the shared threshold rule is still weaker than the best AE family run
- PatchCore did not solve the hardest local defect types yet; `Scratch`, `Loc`, and parts of `Edge-Loc` remain weak
- this makes the next improvement path clear: keep the PatchCore protocol, but replace the current frozen AE encoder with a stronger backbone

## Overall Interpretation

Across all completed experiments:

- the best current result is the `64x64` BatchNorm autoencoder scored with `max_abs`
- the original `64x64` autoencoder checkpoint improved substantially just by changing the scoring rule
- retraining that autoencoder longer did not materially change the outcome, so epoch count alone is unlikely to be the key lever
- adding BatchNorm changed the best score choice for the autoencoder from `topk_abs_mean` to `max_abs`
- the dropout sweep did not help; the best run selected `dropout = 0.00`, so latent dropout is not a promising next AE lever in this setup
- simply increasing autoencoder resolution did not help
- the VAE underperformed the autoencoder even after beta tuning
- Deep SVDD was a stronger alternative than the tuned VAE in some thresholded metrics, but not enough to replace the autoencoder baseline
- the residual autoencoder was a stronger architecture than the plain baseline in several metrics, but it still did not overtake the BatchNorm AE + `max_abs` result
- PatchCore with the frozen BatchNorm AE encoder did produce a usable anomaly signal, but it still fell short of the best AE operating point
- the best PatchCore result came from `mean` reduction with a `50k` memory bank, which suggests that more stable patch aggregation matters more than emphasizing a single worst patch
- all tested models show overlap between normal and anomaly score distributions, which explains the moderate F1 values and missed anomalies
- the score-ablation result shows that part of the bottleneck was the scoring rule, not only the model architecture
- after fixing the score, the remaining bottleneck still looks more like limited class separation than threshold selection alone
- the AE failure analysis shows that the remaining weakness is concentrated in smaller local defects rather than large global defect patterns
- this makes the next decision clearer: keep the current AE winner as the benchmark, and move the next effort into a stronger encoder backbone for local-anomaly methods

## What Was Implemented

Completed work:

- WM-811K legacy pickle loading
- explicit normal-only training setup
- processed metadata generation with repo-relative paths
- resolution-specific processed folders for `x64` and `x128`
- `50k`-normal subset generation
- anomaly-capped test split generation
- convolutional autoencoder baseline
- BatchNorm autoencoder variant
- BatchNorm + dropout sweep
- residual autoencoder variant
- autoencoder score-ablation evaluation
- convolutional VAE baseline
- Deep SVDD baseline
- PatchCore baseline sweep with a frozen BatchNorm AE encoder
- notebook-based end-to-end training for AE, VAE, and SVDD
- notebook-based PatchCore sweep
- scriptable reconstruction-model evaluation
- VAE beta-sweep automation
- best-checkpoint saving
- resumable periodic checkpoints
- validation-threshold metrics
- threshold sweep analysis

## Recommended Next Steps

Recommended follow-up work:

- avoid spending more time on longer-epoch reruns alone unless another change is paired with them
- do not spend more time on dropout tuning for the current AE family unless another structural change is introduced
- keep the current AE + BatchNorm + `max_abs` result as the benchmark to beat
- move the next model-development effort into a stronger non-AE encoder backbone for PatchCore, such as a pretrained ResNet family model
- keep the residual autoencoder as a logged comparison result, but stop using AE encoders as the main PatchCore improvement path
- keep the validation-derived threshold as the main reported result, and treat test-set threshold sweeps as analysis only
