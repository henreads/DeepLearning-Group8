# Wafer Defect Anomaly Detection Report

## Scope

This report summarizes the anomaly-detection experiments for the WM-811K wafer map project.

The cleaned repository is notebook-first and uses PyTorch throughout. The canonical execution path is:

1. build or validate a dataset branch from `data/dataset/`
2. run the corresponding experiment notebook from `experiments/anomaly_detection/`
3. inspect saved checkpoints, plots, and result files under the local `artifacts/` folder of that branch

Historical experiment numbers are still used in the narrative below because they reflect the actual order in which the work progressed, but the runnable notebooks now live under `experiments/` rather than the older `notebooks/anomaly_50k/` tree.

## Team

- Henry Lee Jun, 1004219
- Chia Tang, 1007200
- Genson Low, 1006931

## Shared Goal

All anomaly branches in this report aim to:

- train only on normal wafers when the method is unsupervised or one-class
- treat labeled defect wafers as anomalies at evaluation time
- compare anomaly-scoring approaches under a consistent train / validation / test protocol

## Reproducibility And Dependencies

Main reproducibility surface:

- [README.md](README.md)
- [data/dataset/](data/dataset/)
- [experiments/anomaly_detection/](experiments/anomaly_detection/)
- [src/wafer_defect/](src/wafer_defect/)
- [scripts/](scripts/)

Key shared implementation files:

- [scripts/prepare_wm811k.py](scripts/prepare_wm811k.py)
- [scripts/evaluate_autoencoder_scores.py](scripts/evaluate_autoencoder_scores.py)
- [scripts/evaluate_reconstruction_model.py](scripts/evaluate_reconstruction_model.py)
- [scripts/train_vae.py](scripts/train_vae.py)
- [scripts/train_svdd.py](scripts/train_svdd.py)
- [scripts/train_ts_distillation.py](scripts/train_ts_distillation.py)
- [src/wafer_defect/models/autoencoder.py](src/wafer_defect/models/autoencoder.py)
- [src/wafer_defect/models/patchcore.py](src/wafer_defect/models/patchcore.py)
- [src/wafer_defect/models/ts_distillation.py](src/wafer_defect/models/ts_distillation.py)
- [src/wafer_defect/models/vae.py](src/wafer_defect/models/vae.py)
- [src/wafer_defect/models/svdd.py](src/wafer_defect/models/svdd.py)
- [src/wafer_defect/training/autoencoder.py](src/wafer_defect/training/autoencoder.py)
- [src/wafer_defect/training/patchcore.py](src/wafer_defect/training/patchcore.py)
- [src/wafer_defect/training/ts_distillation.py](src/wafer_defect/training/ts_distillation.py)
- [src/wafer_defect/training/vae.py](src/wafer_defect/training/vae.py)
- [src/wafer_defect/training/svdd.py](src/wafer_defect/training/svdd.py)
- [src/wafer_defect/scoring.py](src/wafer_defect/scoring.py)
- [src/wafer_defect/evaluation/reconstruction_metrics.py](src/wafer_defect/evaluation/reconstruction_metrics.py)

The project uses PyTorch for all model development. Environment and package installation are documented in [README.md](README.md), with the package installed in editable mode through `pip install -e .`.

## Dataset Protocol

Dataset creation and validation now live under `data/dataset/`.

Canonical dataset notebooks:

- [data/dataset/x64/benchmark_50k_5pct/notebook.ipynb](data/dataset/x64/benchmark_50k_5pct/notebook.ipynb)
- [data/dataset/x64/holdout70k_3p5k/notebook.ipynb](data/dataset/x64/holdout70k_3p5k/notebook.ipynb)
- [data/dataset/x128/benchmark_50k_5pct/notebook.ipynb](data/dataset/x128/benchmark_50k_5pct/notebook.ipynb)
- [data/dataset/x224/benchmark_50k_5pct/notebook.ipynb](data/dataset/x224/benchmark_50k_5pct/notebook.ipynb)
- [data/dataset/x240/benchmark_50k_5pct/notebook.ipynb](data/dataset/x240/benchmark_50k_5pct/notebook.ipynb)

Data preparation follows the same core rule across the report:

- [prepare_wm811k.py](scripts/prepare_wm811k.py) reads the legacy `LSWMD.pkl` file
- only explicitly labeled rows are kept
- `failureType == none` is treated as normal
- all other explicit failure types are treated as anomaly
- wafer maps are resized and saved as `.npy`
- metadata CSV files store repo-relative array paths

Primary metadata used by the main benchmark experiments:

- [metadata_50k_5pct.csv](data/processed/x64/wm811k/metadata_50k_5pct.csv)

Effective split for the main `64x64` benchmark:

- train: `40,000` normal
- validation: `5,000` normal
- test: `5,000` normal
- test: `250` anomaly

Split rule:

- normals are split `80 / 10 / 10`
- defects are excluded from training and validation for the main anomaly benchmark
- test anomalies are capped at `5%` of the number of test-normal wafers

## Canonical Experiment Branches

The narrative below still uses historical experiment numbering, but the current runnable notebooks are organized by family:

- [experiments/anomaly_detection/autoencoder/README.md](experiments/anomaly_detection/autoencoder/README.md)
- [experiments/anomaly_detection/vae/README.md](experiments/anomaly_detection/vae/README.md)
- [experiments/anomaly_detection/svdd/README.md](experiments/anomaly_detection/svdd/README.md)
- [experiments/anomaly_detection/backbone_embedding/README.md](experiments/anomaly_detection/backbone_embedding/README.md)
- [experiments/anomaly_detection/teacher_student/README.md](experiments/anomaly_detection/teacher_student/README.md)
- [experiments/anomaly_detection/patchcore/README.md](experiments/anomaly_detection/patchcore/README.md)
- [experiments/anomaly_detection/fastflow/README.md](experiments/anomaly_detection/fastflow/README.md)
- [experiments/anomaly_detection/ensemble/README.md](experiments/anomaly_detection/ensemble/README.md)

Representative notebooks used by the cleaned repo structure:

- [experiments/anomaly_detection/autoencoder/x64/baseline/notebook.ipynb](experiments/anomaly_detection/autoencoder/x64/baseline/notebook.ipynb)
- [experiments/anomaly_detection/vae/x64/baseline/notebook.ipynb](experiments/anomaly_detection/vae/x64/baseline/notebook.ipynb)
- [experiments/anomaly_detection/svdd/x64/baseline/notebook.ipynb](experiments/anomaly_detection/svdd/x64/baseline/notebook.ipynb)
- [experiments/anomaly_detection/backbone_embedding/resnet18/x64/baseline/notebook.ipynb](experiments/anomaly_detection/backbone_embedding/resnet18/x64/baseline/notebook.ipynb)
- [experiments/anomaly_detection/teacher_student/resnet18/x64/main/notebook.ipynb](experiments/anomaly_detection/teacher_student/resnet18/x64/main/notebook.ipynb)
- [experiments/anomaly_detection/teacher_student/resnet50/x64/main/notebook.ipynb](experiments/anomaly_detection/teacher_student/resnet50/x64/main/notebook.ipynb)
- [experiments/anomaly_detection/patchcore/resnet18/x64/main/notebook.ipynb](experiments/anomaly_detection/patchcore/resnet18/x64/main/notebook.ipynb)
- [experiments/anomaly_detection/patchcore/resnet50/x64/main/notebook.ipynb](experiments/anomaly_detection/patchcore/resnet50/x64/main/notebook.ipynb)
- [experiments/anomaly_detection/patchcore/wideresnet50/x224/multilayer/notebook.ipynb](experiments/anomaly_detection/patchcore/wideresnet50/x224/multilayer/notebook.ipynb)
- [experiments/anomaly_detection/patchcore/efficientnet_b0/x224/main/notebook.ipynb](experiments/anomaly_detection/patchcore/efficientnet_b0/x224/main/notebook.ipynb)
- [experiments/anomaly_detection/patchcore/vit_b16/x224/main/notebook.ipynb](experiments/anomaly_detection/patchcore/vit_b16/x224/main/notebook.ipynb)
- [experiments/anomaly_detection/fastflow/x64/main/notebook.ipynb](experiments/anomaly_detection/fastflow/x64/main/notebook.ipynb)

Each branch now keeps its own documentation, notebook, config snapshots, and local `artifacts/` folder for reproducibility.

## Experiment Progression

The report narrative below keeps the original experiment numbering because that numbering reflects the real order in which the work developed. Where a section still names an older notebook path, the current canonical runnable version should be understood as the matching branch under `experiments/anomaly_detection/`.

The experiment sequence was intentionally staged rather than random. We started with the simplest reconstruction baseline in `Experiment 1` to establish a shared `64x64` protocol, a fair validation-threshold rule, and a first realistic failure pattern. Once that baseline showed that the model could detect broad defects but still struggled on smaller local patterns, the next steps stayed close to the same family first: `Experiment 2` tested whether BatchNorm changed feature stability, `Experiment 3` checked whether dropout improved generalization, and the residual / resolution variants tested whether the bottleneck was architecture depth or image scale. In parallel, `Experiments 4` and `5` asked whether a probabilistic reconstruction model could beat the plain autoencoder, while `Experiment 6` tested a one-class distance model to see whether reconstruction itself was the main advantage.

After those early branches, the results pointed to a clearer hypothesis: the main weakness was not learning anomalies at all, but missing smaller local defects and having weak local scoring. That is why `Experiment 7` moved to PatchCore on top of the best AE-family encoder, and why `Experiments 8`, `9`, and `10` shifted from reconstruction backbones to pretrained ResNet backbones with stronger local anomaly scoring. The weak plain ResNet18 center-distance baseline showed that backbone quality alone was not enough, while the PatchCore follow-ups showed that local patch aggregation mattered much more than global embedding distance. That progression led directly to `Experiments 11` and `12`, the teacher-distillation-resnet family: once stronger frozen backbones and local discrepancy maps looked promising, the next logical step was a teacher-student model that could preserve spatial anomaly information better than the older global or purely reconstruction-based scoring rules. In short, each step was chosen to isolate one question at a time: first whether reconstruction worked, then whether that family could be strengthened, then whether local scoring mattered more than reconstruction, and finally whether a stronger teacher-student local detector could combine the best parts of both directions.

## Overall Comparison

Main comparison across completed experiments:

| experiment    | model       | score | image size | val-threshold precision | val-threshold recall | val-threshold F1 | AUROC      | AUPRC      | best sweep F1 |
| ------------- | ----------- | ----- | ---------- | ----------------------- | -------------------- | ---------------- | ---------- | ---------- | ------------- |
| PatchCore-ViT-B16-x224-topk10-mb400k | PatchCore + ViT-B/16 (block `6`) | `top10% mean` | `224x224`   | `0.463252`              | `0.832000`           | `0.595136`       | `0.956301` | `0.670907` | `0.650206`    |
| PatchCore-EffNetB1-x240-topk-mb240k-r003 | PatchCore + EfficientNet-B1 (feature block `3`) | `top3% mean` | `240x240`   | `0.475610`              | `0.780000`           | `0.590909`       | `0.935374` | `0.608633` | `0.650699`    |
| PatchCore-WideRes50-x224-topk-mb600k-r005 | PatchCore + WideResNet50-2 (`layer2` + `layer3`) | `topk_mean` | `224x224`   | `0.432184`              | `0.752000`           | `0.548905`       | `0.930680` | `0.659063` | `0.634146`    |
| PatchCore-EffNetB0-x224-topk-mb240k-r002 | PatchCore + EfficientNet-B0 (`mid=3`, `deep=6`) | `topk_mean` | `224x224`   | `0.438725`              | `0.716000`           | `0.544073`       | `0.924586` | `0.483186` | `0.566667`    |
| PatchCore-WideRes50-topk-mb50k-r010 | PatchCore + WideResNet50-2 (`layer2` + `layer3`) | `topk_mean` | `64x64`    | `0.421546`              | `0.720000`           | `0.531758`       | `0.916943` | `0.561855` | `0.585774`    |
| Ensemble-PatchCore-TSRes50-maxpair | Score ensemble of PatchCore-WideRes50-topk-mb50k-r010 + TS-Res50-mixed-topk20 | `max` on normalized scores | `64x64`    | `0.422434`              | `0.708000`           | `0.529148`       | `0.916330` | `0.611277` | `0.619718`    |
| PatchCore-WideRes50-topk-mb50k-r015 | PatchCore + WideResNet50-2 (`layer2` + `layer3`) | `topk_mean` | `64x64`    | `0.420673`              | `0.700000`           | `0.525526`       | `0.911559` | `0.548919` | `0.592902`    |
| PatchCore-WideRes50-topk-mb50k-r005 | PatchCore + WideResNet50-2 (`layer2` + `layer3`) | `topk_mean` | `64x64`    | `0.415888`              | `0.712000`           | `0.525074`       | `0.920518` | `0.563612` | `0.579281`    |
| TS-Res50-mixed-topk20 | Teacher-Student Distillation + ResNet50 Teacher Backbone | `topk_mean` | `64x64`    | `0.418052`              | `0.704000`           | `0.524590`       | `0.909189` | `0.599169` | `0.606299`    |
| TS-WideRes50-multilayer-mixed-topk15 | Teacher-Student Distillation + WideResNet50-2 Teacher Backbone (`layer2` + `layer3`) | `topk_mean` | `64x64`    | `0.421951`              | `0.692000`           | `0.524242`       | `0.923114` | `0.546305` | `0.560928`    |
| PatchCore-WideRes50-topk-mb50k-r020 | PatchCore + WideResNet50-2 (`layer2` + `layer3`) | `topk_mean` | `64x64`    | `0.408451`              | `0.696000`           | `0.514793`       | `0.906730` | `0.533386` | `0.585657`    |
| TS-WideRes50-layer2-mixed-topk25 | Teacher-Student Distillation + WideResNet50-2 Teacher Backbone | `topk_mean` | `64x64`    | `0.404651`              | `0.696000`           | `0.511765`       | `0.903371` | `0.512148` | `0.526316`    |
| PatchCore-WideRes50-topk-mb50k-r025 | PatchCore + WideResNet50-2 (`layer2` + `layer3`) | `topk_mean` | `64x64`    | `0.398169`              | `0.696000`           | `0.506550`       | `0.902348` | `0.518213` | `0.575488`    |
| TS-Res50-layer1-mixed-topk10 | Teacher-Student Distillation + ResNet50 Teacher Backbone | `topk_mean` | `64x64`    | `0.407862`              | `0.664000`           | `0.505327`       | `0.872754` | `0.527526` | `0.547284`    |
| AE-64-BN-max  | Autoencoder + BatchNorm | `max_abs` | `64x64`    | `0.401442`              | `0.668000`           | `0.501502`       | `0.834023` | `0.568039` | `0.629808`    |
| TS-Res18-student-topk20 | Teacher-Student Distillation + ResNet18 Teacher Backbone | `topk_mean` | `64x64`    | `0.402500`              | `0.644000`           | `0.495385`       | `0.894076` | `0.519445` | `0.520548`    |
| AE-64-BN-DO0.00 | Autoencoder + BatchNorm + Dropout `0.00` | `max_abs` | `64x64`    | `0.393120`              | `0.640000`           | `0.487062`       | `0.850790` | `0.616946` | `0.656642`    |
| AE-64-BN-DO0.10 | Autoencoder + BatchNorm + Dropout `0.10` | `max_abs` | `64x64`    | `0.385343`              | `0.652000`           | `0.484398`       | `0.844670` | `0.570245` | `0.634615`    |
| PatchCore-WideRes50-mean-mb50k | PatchCore + WideResNet50-2 (`layer2` + `layer3`) | `mean` | `64x64`    | `0.386635`              | `0.648000`           | `0.484305`       | `0.873711` | `0.413971` | `0.510714`    |
| AE-64-BN-DO0.05 | Autoencoder + BatchNorm + Dropout `0.05` | `max_abs` | `64x64`    | `0.377828`              | `0.668000`           | `0.482659`       | `0.835035` | `0.551700` | `0.609959`    |
| FastFlow-WideRes50-l23-s4-mean | FastFlow + WideResNet50-2 (`layer2` + `layer3`), `4` flow steps | `mean` | `64x64`    | `0.385167`              | `0.644000`           | `0.482036`       | `0.870692` | `0.488619` | `0.482036`    |
| AE-64-Res-max  | Residual Autoencoder | `max_abs` | `64x64`    | `0.374419`              | `0.644000`           | `0.473529`       | `0.843360` | `0.588907` | `0.625592`    |
| AE-64-BN-DO0.20 | Autoencoder + BatchNorm + Dropout `0.20` | `max_abs` | `64x64`    | `0.370115`              | `0.644000`           | `0.470073`       | `0.841431` | `0.574973` | `0.633929`    |
| AE-64-topk    | Autoencoder | `topk_abs_mean` | `64x64`    | `0.390374`              | `0.584000`           | `0.467949`       | `0.839282` | `0.522171` | `0.509091`    |
| PatchCore-EffNetB0-x64-topk-mb240k-r002 | PatchCore + EfficientNet-B0 (`mid=3`, `deep=6`) | `topk_mean` | `64x64`    | `0.381313`              | `0.604000`           | `0.467492`       | `0.905171` | `0.489141` | `0.504132`    |
| PatchCore-WideRes50-mean-mb20k | PatchCore + WideResNet50-2 (`layer2` + `layer3`) | `mean` | `64x64`    | `0.386842`              | `0.588000`           | `0.466667`       | `0.875149` | `0.389518` | `0.471002`    |
| AE-64-topk-43ep | Autoencoder | `topk_abs_mean` | `64x64`    | `0.381579`              | `0.580000`           | `0.460317`       | `0.834819` | `0.525162` | `0.520661`    |
| AE-64-Res-topk | Residual Autoencoder | `topk_abs_mean` | `64x64`    | `0.356974`              | `0.604000`           | `0.448737`       | `0.804607` | `0.626014` | `0.678133`    |
| AE-64-BN-topk | Autoencoder + BatchNorm | `topk_abs_mean` | `64x64`    | `0.346247`              | `0.572000`           | `0.431373`       | `0.790020` | `0.603447` | `0.655172`    |
| PatchCore-Res50-mean-mb50k | PatchCore + ResNet50 | `mean` | `64x64`    | `0.339950`              | `0.548000`           | `0.419602`       | `0.821402` | `0.362657` | `0.439604`    |
| AE-64-mse     | Autoencoder | `mse_mean` | `64x64`    | `0.346154`              | `0.504000`           | `0.410423`       | `0.809694` | `0.447970` | `0.473318`    |
| PatchCore-Res18-mean-mb50k | PatchCore + ResNet18 | `mean` | `64x64`    | `0.345930`              | `0.476000`           | `0.400673`       | `0.842266` | `0.410729` | `0.445344`    |
| PatchCore-WideRes50-max-mb50k | PatchCore + WideResNet50-2 (`layer2` + `layer3`) | `max` | `64x64`    | `0.321782`              | `0.520000`           | `0.397554`       | `0.876209` | `0.397211` | `0.451327`    |
| PatchCore-Res18-mean-mb10k | PatchCore + ResNet18 | `mean` | `64x64`    | `0.345133`              | `0.468000`           | `0.397284`       | `0.831191` | `0.409682` | `0.425439`    |
| PatchCore-Res50-mean-mb10k | PatchCore + ResNet50 | `mean` | `64x64`    | `0.323232`              | `0.512000`           | `0.396285`       | `0.804225` | `0.310237` | `0.405738`    |
| AE-128-mse    | Autoencoder | `mse_mean` | `128x128`  | `0.309973`              | `0.460000`           | `0.370370`       | `0.795673` | `0.393266` | `0.426724`    |
| SVDD-64       | Deep SVDD   | `latent_distance` | `64x64`    | `0.304709`              | `0.440000`           | `0.360065`       | `0.787506` | `0.213108` | `0.366288`    |
| VAE-64-b0.005 | VAE         | `vae_score` | `64x64`    | `0.286104`              | `0.420000`           | `0.340357`       | `0.771391` | `0.372184` | `0.420253`    |
| PatchCore-AEBN-mean-mb50k | PatchCore + AE-BN Backbone | `mean` | `64x64`    | `0.283747`              | `0.412000`           | `0.336052`       | `0.850786` | `0.226325` | `0.389447`    |
| VAE-64-b0.01  | VAE         | `vae_score` | `64x64`    | `0.280323`              | `0.416000`           | `0.334944`       | `0.766392` | `0.369030` | `0.416667`    |
| PatchCore-Res18-topk-mb50k-r010 | PatchCore + ResNet18 | `topk_mean` | `64x64`    | `0.296875`              | `0.380000`           | `0.333333`       | `0.803171` | `0.329613` | `0.361991`    |
| PatchCore-Res18-topk-mb10k-r005 | PatchCore + ResNet18 | `topk_mean` | `64x64`    | `0.290520`              | `0.380000`           | `0.329289`       | `0.795090` | `0.323395` | `0.365000`    |
| PatchCore-Res18-topk-mb50k-r005 | PatchCore + ResNet18 | `topk_mean` | `64x64`    | `0.291667`              | `0.364000`           | `0.323843`       | `0.795596` | `0.318155` | `0.345263`    |
| PatchCore-Res18-max-mb50k | PatchCore + ResNet18 | `max` | `64x64`    | `0.281553`              | `0.348000`           | `0.311270`       | `0.786144` | `0.303307` | `0.331183`    |
| PatchCore-Res50-topk-mb50k-r005 | PatchCore + ResNet50 | `topk_mean` | `64x64`    | `0.259053`              | `0.372000`           | `0.305419`       | `0.797955` | `0.279352` | `0.312618`    |
| PatchCore-Res50-topk-mb50k-r010 | PatchCore + ResNet50 | `topk_mean` | `64x64`    | `0.256198`              | `0.372000`           | `0.303426`       | `0.800964` | `0.291924` | `0.317757`    |
| PatchCore-Res50-max-mb50k | PatchCore + ResNet50 | `max` | `64x64`    | `0.233618`              | `0.328000`           | `0.272879`       | `0.780863` | `0.208679` | `0.282528`    |
| PatchCore-Res50-topk-mb10k-r005 | PatchCore + ResNet50 | `topk_mean` | `64x64`    | `0.221932`              | `0.340000`           | `0.268562`       | `0.785025` | `0.217361` | `0.285319`    |
| WideRes50-center | Pretrained WideResNet50-2 Backbone | `center_l2` | `64x64`    | `0.221854`              | `0.268000`           | `0.242754`       | `0.677274` | `0.142323` | `0.269504`    |
| ResNet18-center | Pretrained ResNet18 Backbone | `center_l2` | `64x64`    | `0.201705`              | `0.284000`           | `0.235880`       | `0.684746` | `0.194977` | `0.259740`    |
| PatchCore-AEBN-topk-mb50k-r010 | PatchCore + AE-BN Backbone | `topk_mean` | `64x64`    | `0.166134`              | `0.208000`           | `0.184725`       | `0.808633` | `0.148827` | `0.304950`    |
| PatchCore-AEBN-topk-mb50k-r005 | PatchCore + AE-BN Backbone | `topk_mean` | `64x64`    | `0.112583`              | `0.136000`           | `0.123188`       | `0.777215` | `0.120862` | `0.241529`    |
| PatchCore-AEBN-topk-mb10k-r005 | PatchCore + AE-BN Backbone | `topk_mean` | `64x64`    | `0.053004`              | `0.060000`           | `0.056285`       | `0.659112` | `0.072701` | `0.157971`    |
| PatchCore-AEBN-max-mb50k | PatchCore + AE-BN Backbone | `max` | `64x64`    | `0.052632`              | `0.060000`           | `0.056075`       | `0.678692` | `0.080039` | `0.152152`    |
| PatchCore-AEBN-max-mb10k | PatchCore + AE-BN Backbone | `max` | `64x64`    | `0.029412`              | `0.036000`           | `0.032374`       | `0.587003` | `0.061002` | `0.120301`    |

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

The overview plot below is split into two views to stay readable as the experiment list grows: the left panel shows the top runs by deployment-style F1, while the right panel shows all completed runs in AUPRC-vs-F1 space with point size scaled by AUROC.

![Overall experiment comparison](artifacts/report_plots/overall_experiment_comparison.png)

Current top ranking:

This ranking is based mainly on `val-threshold F1`, with the other metrics used as supporting evidence.

1. PatchCore + ViT-B/16 direct `224x224`, block `6`, memory bank `400k`, wafer score = mean of top `10%` patch scores
2. PatchCore + EfficientNet-B1 direct `240x240`, feature block `3`, memory bank `240k`, wafer score = mean of top `3%` patch scores
3. PatchCore + WideResNet50-2 multilayer direct `224x224`, `topk_mean`, memory bank `600k`, top-k ratio `0.05`
4. PatchCore + EfficientNet-B0 direct `224x224`, `topk_mean`, memory bank `240k`, top-k ratio `0.02`
5. PatchCore + WideResNet50-2 multilayer `layer2 + layer3` `64x64`, `topk_mean`, memory bank `50k`, top-k ratio `0.10`
6. Score ensemble of WRN PatchCore + `TS-Res50` with normalized-score `max` fusion
7. PatchCore + WideResNet50-2 multilayer `layer2 + layer3` `64x64`, `topk_mean`, memory bank `50k`, top-k ratio `0.15`
8. PatchCore + WideResNet50-2 multilayer `layer2 + layer3` `64x64`, `topk_mean`, memory bank `50k`, top-k ratio `0.05`
9. Teacher-student distillation + ResNet50 teacher `64x64` with mixed student+autoencoder `topk_mean`, top-k ratio `0.20`
10. Teacher-student distillation + WideResNet50-2 teacher `64x64` with multilayer `layer2 + layer3`, mixed student+autoencoder `topk_mean`, top-k ratio `0.15`

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
- PatchCore improved in stages: the AE-backed version was weak, the ResNet18 and ResNet50 versions validated the pretrained-backbone direction, the later WideResNet50-2 multilayer follow-up became the strongest `64x64` PatchCore result, the direct-`224x224` EfficientNet-B0 follow-up then pushed the branch higher, the local EfficientNet-B1 `x240` one-layer run climbed further still, the newer direct-`224x224` WideResNet50-2 all-in-one run remained a strong CNN reference, and the ViT-B/16 `x224` follow-up is now the strongest main-benchmark PatchCore result overall
- the best completed PatchCore variant on the main benchmark is now the direct-`224x224` ViT-B/16 one-layer run with block `6`, memory bank `400k`, and top-`10%` wafer-level reduction, reaching `F1 = 0.595136`, `AUROC = 0.956301`, and `AUPRC = 0.670907`
- the local EfficientNet-B1 `x240` one-layer follow-up is now the second-strongest completed main-benchmark run overall, reaching `F1 = 0.590909`, `AUROC = 0.935374`, and `AUPRC = 0.608633`; it is the strongest completed CNN PatchCore run on deployed F1 in the current repo snapshot
- within the WideResNet50-2 PatchCore sweep, `topk_mean` clearly beat both `mean` and `max`; the strongest operating region was the narrow top-k range around `0.05` to `0.10`
- the frozen pretrained ResNet18 backbone with simple center-distance scoring was weak, which suggests the backbone alone is not enough without a stronger local-anomaly scoring rule
- ResNet18 + PatchCore fixed that issue materially; the best ResNet18 PatchCore variant reached `F1 = 0.400673` and clearly outperformed the plain ResNet18 center-distance baseline
- scaling that same PatchCore direction to a pretrained ResNet50 backbone helped further; the best ResNet50 PatchCore variant reached `F1 = 0.419602`
- the first all-in-one EfficientNet-B0 PatchCore run at `64x64` preprocessing was only a mid-tier report result, reaching `F1 = 0.467492` and `AUROC = 0.905171`
- moving that same EfficientNet-B0 branch to direct `224x224` preprocessing changed the picture materially: the report-compatible `x224` follow-up reached `F1 = 0.544073` and `AUROC = 0.924586`, overtaking the earlier `64x64` WRN PatchCore leader on deployed F1 before the later WRN `x224` follow-up moved the bar again
- the later EfficientNet-B1 direct-`240x240` one-layer PatchCore run pushed the main benchmark materially higher than the earlier CNN PatchCore baselines, and the later ViT-B/16 direct-`224x224` PatchCore run still remains the strongest overall on deployed F1, AUROC, and AUPRC among the fair `5%` main-split runs currently in the repo
- this EfficientNet-B0 branch is still useful for two reasons: it shows that direct `224x224` preprocessing matters for pretrained-backbone PatchCore, and it provides a strong report-compatible bridge between the weaker `64x64` run and the newer WRN `x224` leader
- the ResNet50 PatchCore sweep was a useful intermediate step, but the larger WideResNet50-2 PatchCore follow-up is the result that finally pushed PatchCore above both the AE family and the teacher-student baselines on deployed F1
- the original combined teacher-student score looked weak, but a post-training score sweep showed that the main bottleneck was scoring rather than the checkpoint itself
- once the teacher-student checkpoint was rescored with a student-only `topk_mean` rule and a wider top-k ratio, it became a genuinely competitive validation-threshold result rather than a failed branch
- this is the strongest evidence so far that architecture and anomaly scoring interact very strongly in this project; a weak default score can hide a strong checkpoint
- teacher-student distillation still remains extremely strong, but it no longer holds the report's best ranking metrics; the direct-`224x224` ViT-B/16 PatchCore follow-up now leads on deployed F1, AUROC, AUPRC, and best-sweep F1 among the main-benchmark runs
- the refreshed FastFlow `19A` artifacts still select the multilayer `WideResNet50-2` model with `4` flow steps and plain `mean` reduction, but the cleaned rerun is weaker than the older snapshot, reaching `F1 = 0.482036`
- this keeps FastFlow competitive with the stronger autoencoder-family runs, but it now sits below the leading PatchCore and teacher-student branches by a clearer margin
- the first score-level ensemble study was competitive but not decisive: the best true fusion, normalized-score `max` over WRN PatchCore and `TS-Res50`, reached `F1 = 0.529148` and a much stronger `AUPRC = 0.611277`, but it still does not beat the new direct-`224x224` WRN PatchCore leader on the main deployed F1 metric
- the FastFlow family still follows the same defect pattern seen elsewhere in the project: broad defects such as `Edge-Ring` are much easier than smaller local defects such as `Scratch`, `Loc`, and `Edge-Loc`
- Deep SVDD had especially weak AUPRC, which suggests poorer ranking quality under class imbalance
- local-error-focused scoring appears more effective than full-image averaging on wafer maps
- all tested approaches learn a real anomaly signal, but class separation is still only moderate

## Cross-Model Defect Breakdown

To compare what each strong family is actually good at, I compiled the selected per-defect breakdowns for the strongest representative runs that have trustworthy saved artifact outputs:

- `AE-64-BN-max`
- `TS-Res18-student-topk20`
- `TS-Res50-s2_a1_topk_mean_r0.20`
- `WideRes50-center`
- `PatchCore-WideRes50-topk-mb50k-r010`
- `FastFlow-WideRes50-l23-s4-mean`

Per-defect recall comparison:

| defect type | AE-64-BN-max | TS-Res18-student-topk20 | TS-Res50-s2_a1_topk_mean_r0.20 | WideRes50-center | PatchCore-WideRes50-topk-mb50k-r010 | FastFlow-WideRes50-l23-s4-mean |
| ----------- | ------------ | ----------------------- | ------------------------------ | ---------------- | ----------------------------------- | ------------------------------ |
| `Scratch` | `0.466667` | `0.333333` | `0.333333` | `0.200000` | `0.533333` | `0.133333` |
| `Loc` | `0.294118` | `0.441176` | `0.558824` | `0.176471` | `0.558824` | `0.588235` |
| `Edge-Loc` | `0.471698` | `0.490566` | `0.490566` | `0.132075` | `0.584906` | `0.528302` |
| `Center` | `0.800000` | `0.620000` | `0.720000` | `0.140000` | `0.620000` | `0.720000` |
| `Donut` | `0.571429` | `0.714286` | `0.714286` | `0.285714` | `1.000000` | `0.857143` |
| `Edge-Ring` | `0.892857` | `0.857143` | `0.928571` | `0.464286` | `0.916667` | `0.738095` |
| `Random` | `0.800000` | `1.000000` | `1.000000` | `0.600000` | `1.000000` | `1.000000` |
| `Near-full` | `1.000000` | `1.000000` | `1.000000` | `0.000000` | `1.000000` | `1.000000` |

Best model by defect type:

| defect type | best model | best recall |
| ----------- | ---------- | ----------- |
| `Scratch` | `PatchCore-WideRes50-topk-mb50k-r010` | `0.533333` |
| `Loc` | `FastFlow-WideRes50-l23-s4-mean` | `0.588235` |
| `Edge-Loc` | `PatchCore-WideRes50-topk-mb50k-r010` | `0.584906` |
| `Center` | `AE-64-BN-max` | `0.800000` |
| `Donut` | `PatchCore-WideRes50-topk-mb50k-r010` | `1.000000` |
| `Edge-Ring` | `TS-Res50-s2_a1_topk_mean_r0.20` | `0.928571` |
| `Random` | multiple models tie | `1.000000` |
| `Near-full` | multiple models tie | `1.000000` |

Interpretation:

- the strongest `Scratch` and `Edge-Loc` detector among the compared selected runs is the best multilayer `WideResNet50-2` PatchCore variant, which supports the idea that local patch aggregation is still the most reliable approach for small localized defects
- `Loc` remains one of the few categories where the refreshed FastFlow branch is strongest, reaching `0.588235`; that suggests flow-based density modeling can still be useful on some medium-scale local defects even when its overall F1 trails the best PatchCore and teacher-student runs
- `Center` is the one clear category where the simpler `AE-64-BN-max` run is still strongest, reaching `0.800000`; this matches the broader observation that broader and more centrally distributed anomaly structure is easier for reconstruction-style models than tiny local defects
- `Edge-Ring` is led by `TS-Res50` at `0.928571`, with the best `PatchCore` very close behind at `0.916667`; both are clearly strong on larger structured rings
- the plain `WideRes50-center` baseline remains weak on almost every defect type, which reinforces the earlier conclusion that a strong pretrained backbone alone is not enough without a better local anomaly scoring rule
- `FastFlow` is still useful on `Loc`, `Center`, and the easier broad-defect categories, but it remains clearly weak on `Scratch` and no longer matches the stronger PatchCore run on `Edge-Loc`
- the saved multilayer `TS-WideRes50` defect-breakdown recomputation was not included in this cross-model table because the reconstructed per-defect output was internally inconsistent with its saved summary metrics; it should be recomputed cleanly before being used in the final apples-to-apples defect comparison

## Evaluation Rule

Main reported threshold:

- use the threshold derived from validation-normal scores at the `95th` percentile

Analysis-only threshold:

- also report the best test-set threshold sweep as an operating-point study

Reason:

- the validation threshold is the fair deployment-style threshold
- the best threshold sweep uses test labels and should not be treated as the main result

## Autoencoder Experiment Family

This family covers the core autoencoder line of experiments and follow-up variants. The numbered experiments in this family are:

- `Experiment 1`: baseline autoencoder `64x64`
- `Experiment 2`: autoencoder `64x64` with BatchNorm
- `Experiment 3`: autoencoder `64x64` with BatchNorm + dropout sweep

Additional family analysis and architectural variants are kept in the same section because they build directly on the same reconstruction baseline.

The autoencoder family figure below is split into two views so the growing number of AE variants stays readable: the left panel ranks all AE-family runs by deployment-style F1, while the right panel shows the precision-recall tradeoff within the family, with point size scaled by AUPRC.

![Autoencoder family comparison](artifacts/report_plots/autoencoder_family_comparison.png)

### Experiment 1: Baseline Autoencoder `64x64`

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

### Experiment 2: Autoencoder `64x64` with BatchNorm

Purpose:

- test whether inserting BatchNorm into the same `64x64` autoencoder improves anomaly detection on the shared `5%` test-defect split
- keep the same dataset, optimizer family, threshold rule, and evaluation notebook flow as the baseline

Configuration:

- config: [train_autoencoder_batchnorm.toml](configs/training/train_autoencoder_batchnorm.toml)
- notebook: [05_autoencoder_batchnorm_training.ipynb](notebooks/anomaly_50k/05_autoencoder_batchnorm_training.ipynb)
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

### Experiment 3: Autoencoder `64x64` with BatchNorm + Dropout Sweep

Purpose:

- test whether light dropout improves the BatchNorm autoencoder on the same shared `64x64` 5% test-defect split
- keep the same data, optimizer family, threshold rule, and evaluation flow while sweeping only the dropout rate

Configuration:

- config: [train_autoencoder_batchnorm_dropout.toml](configs/training/train_autoencoder_batchnorm_dropout.toml)
- notebook: [06_autoencoder_batchnorm_dropout_training.ipynb](notebooks/anomaly_50k/06_autoencoder_batchnorm_dropout_training.ipynb)
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
- notebook: [08_autoencoder_residual_training.ipynb](notebooks/anomaly_50k/08_autoencoder_residual_training.ipynb)
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

- notebook: [02_autoencoder_training.ipynb](notebooks/anomaly_50k/02_autoencoder_training.ipynb)
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

- notebook section: [02_autoencoder_training.ipynb](notebooks/anomaly_50k/02_autoencoder_training.ipynb)
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

## VAE Experiment Family

This family covers the first VAE run and the follow-up beta sweep while keeping the original experiment numbering.

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
- notebook: [04_svdd_training.ipynb](notebooks/anomaly_50k/04_svdd_training.ipynb)
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

## Experiment 7: PatchCore Sweep with AE-BN Backbone `64x64`

Purpose:

- test a local nearest-neighbor anomaly method on the same shared `64x64` 5% test-defect split
- check whether a patch-based method can recover the smaller local defects that remain hard for the autoencoder family

Implementation:

- config: [train_patchcore.toml](configs/training/train_patchcore.toml)
- notebook: [07_patchcore_training.ipynb](notebooks/anomaly_50k/07_patchcore_training.ipynb)
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

Best AE-BN PatchCore variant under the main validation-threshold rule:

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

## Pretrained-ResNet Experiment Family

This family groups the pretrained ResNet backbone baseline and the follow-up PatchCore variants while keeping the original experiment numbering and notebook references.

## Experiment 8: Pretrained ResNet18 Backbone Baseline `64x64`

Purpose:

- test a non-autoencoder backbone baseline before combining the stronger backbone with PatchCore
- check whether a frozen ImageNet-pretrained `ResNet18` embedding space is already useful with a very simple one-class scoring rule

Implementation:

- config: [train_resnet18_backbone.toml](configs/training/train_resnet18_backbone.toml)
- notebook: [09_resnet18_backbone_baseline.ipynb](notebooks/anomaly_50k/09_resnet18_backbone_baseline.ipynb)
- artifact dir: [artifacts/x64/resnet18_embedding_baseline](artifacts/x64/resnet18_embedding_baseline)
- backbone: ImageNet-pretrained `ResNet18`
- input adaptation: RGB stem averaged into a single-channel wafer input stem
- backbone mode: frozen
- input size inside the backbone: `224x224`
- anomaly score: L2 distance from each wafer embedding to the train-normal feature center

Evaluation:

- validation threshold: `12.720178`
- precision: `0.201705`
- recall: `0.284000`
- F1: `0.235880`
- AUROC: `0.684746`
- AUPRC: `0.194977`
- predicted anomalies: `352`
- confusion matrix: `[[4719, 281], [179, 71]]`
- best test-sweep threshold: `14.406645`
- best test-sweep precision: `0.370370`
- best test-sweep recall: `0.200000`
- best test-sweep F1: `0.259740`

Failure analysis:

- true positive: `71`, mean score `15.933155`
- false negative: `179`, mean score `9.165589`
- false positive: `281`, mean score `14.271239`
- true negative: `4719`, mean score `8.471154`

Defect-type recall:

- `Edge-Ring`: `0.559524`
- `Edge-Loc`: `0.150943`
- `Center`: `0.080000`
- `Loc`: `0.117647`
- `Random`: `0.800000`
- `Scratch`: `0.133333`
- `Donut`: `0.142857`
- `Near-full`: `0.500000`

Interpretation:

- the pretrained ResNet18 backbone worked technically, but the simple center-distance baseline was weak
- this result is well below the best AE family run and also below the better PatchCore variants
- the weakness appears to be the scoring rule, not only the backbone itself; global embedding distance is too crude for the local defect patterns in WM-811K
- this result still supports the broader backbone direction, because it shows that a stronger backbone alone is not enough without a local anomaly method on top

The compact baseline figure below keeps the earlier non-leading baselines readable in one place: the left panel ranks the VAE, SVDD, and simple backbone runs by deployment-style F1, while the right panel shows their precision-recall tradeoff with point size scaled by AUROC.

![Compact baseline comparison](artifacts/report_plots/compact_baseline_comparison.png)

## Experiment 9: PatchCore Sweep with Pretrained ResNet18 `64x64`

Purpose:

- test whether PatchCore can extract a stronger local-anomaly signal from the pretrained `ResNet18` backbone than the plain center-distance baseline
- compare several patch aggregation and memory-bank settings on the same shared `64x64` 5% test-defect split

Implementation:

- config: [train_patchcore_resnet18.toml](configs/training/train_patchcore_resnet18.toml)
- notebook: [10_patchcore_resnet18_training.ipynb](notebooks/anomaly_50k/10_patchcore_resnet18_training.ipynb)
- artifact dir: [artifacts/x64/patchcore_resnet18](artifacts/x64/patchcore_resnet18)
- backbone: frozen ImageNet-pretrained `ResNet18`
- input adaptation: single-channel wafer maps resized internally to `224x224`
- compared variants:
  - `mean`, memory bank `10k`
  - `mean`, memory bank `50k`
  - `topk_mean`, memory bank `10k`, top-k ratio `0.05`
  - `topk_mean`, memory bank `50k`, top-k ratio `0.05`
  - `topk_mean`, memory bank `50k`, top-k ratio `0.10`
  - `max`, memory bank `50k`

Sweep summary:

| variant | reduction | memory bank | top-k ratio | val-threshold F1 | AUROC | AUPRC | best sweep F1 |
| ------- | --------- | ----------- | ----------- | ---------------- | ----- | ----- | ------------- |
| `mean_mb50k` | `mean` | `50000` | `0.10` | `0.400673` | `0.842266` | `0.410729` | `0.445344` |
| `mean_mb10k` | `mean` | `10000` | `0.10` | `0.397284` | `0.831191` | `0.409682` | `0.425439` |
| `topk_mb50k_r010` | `topk_mean` | `50000` | `0.10` | `0.333333` | `0.803171` | `0.329613` | `0.361991` |
| `topk_mb10k_r005` | `topk_mean` | `10000` | `0.05` | `0.329289` | `0.795090` | `0.323395` | `0.365000` |
| `topk_mb50k_r005` | `topk_mean` | `50000` | `0.05` | `0.323843` | `0.795596` | `0.318155` | `0.345263` |
| `max_mb50k` | `max` | `50000` | `0.10` | `0.311270` | `0.786144` | `0.303307` | `0.331183` |

Best ResNet18 PatchCore variant under the main validation-threshold rule:

- variant: `mean_mb50k`
- precision: `0.345930`
- recall: `0.476000`
- F1: `0.400673`
- AUROC: `0.842266`
- AUPRC: `0.410729`
- best test-sweep threshold: `0.369899`
- best test-sweep F1: `0.445344`

Failure analysis for `mean_mb50k`:

- true positive: `119`, mean score `0.412849`
- false negative: `131`, mean score `0.337854`
- false positive: `225`, mean score `0.378711`
- true negative: `4775`, mean score `0.322284`

Defect-type recall for `mean_mb50k`:

- `Edge-Ring`: `0.523810`
- `Center`: `0.480000`
- `Edge-Loc`: `0.264151`
- `Loc`: `0.411765`
- `Scratch`: `0.600000`
- `Donut`: `1.000000`
- `Random`: `1.000000`
- `Near-full`: `1.000000`

Interpretation:

- ResNet18 + PatchCore is a major improvement over the plain ResNet18 center-distance baseline
- `mean` reduction was best again, and the larger `50k` memory bank gave a small but consistent gain over `10k`
- the ResNet18 PatchCore path is much stronger than the older AE-backed PatchCore path on validation-threshold F1
- especially important: the selected ResNet18 PatchCore variant improved on several local defect types, including `Loc` and `Scratch`
- even with that gain, the best ResNet18 PatchCore run still did not beat the AE + BatchNorm + `max_abs` winner, so it is a strong challenger but not the new leader

UMAP check:

- canonical artifact path: [artifacts/x64/patchcore_resnet18_10A/max_mb50k/evaluation/plots](artifacts/x64/patchcore_resnet18_10A/max_mb50k/evaluation/plots)
- this UMAP comes from the saved `max_mb50k` ResNet18 PatchCore artifact, not the selected best-row `mean_mb50k`; it should therefore be read as a diagnostic geometry check rather than as the main selected result above
- generation method:
  - reference-fit UMAP learned from sampled train-normal embeddings only
  - `5000` train-reference points used to fit PCA + UMAP
  - UMAP-space KNN threshold calibrated on all `5000` validation normals at the `95th` percentile
  - exported plotting subset capped to `5000` train reference, `4000` validation normals, `8000` test normals, and `3500` test anomalies
- how to read it:
  - split-colored plot: asks whether normals and anomalies occupy the same broad regions or peel into different branches/islands
  - score-colored plot: asks where PatchCore assigns higher anomaly evidence inside that same geometry
  - the absolute UMAP axes do not have physical meaning; the useful signal is overlap, branching, tail structure, and localized score hot spots
- geometry result:
  - validation normals, test normals, and anomalies still share broad overlapping islands
  - anomalies are not cleanly isolated into one separate cluster, so geometry alone does not give an easy threshold
  - the score-colored view still shows localized hotter bands and pockets inside the manifold, which is consistent with PatchCore learning useful local patch structure even when the global embedding geometry remains mixed
- UMAP-KNN thresholding result for this saved `max_mb50k` artifact:
  - threshold policy: validation-normal `95th` percentile in UMAP-KNN space
  - deployed threshold: `0.233744`
  - precision: `0.117193`
  - recall: `0.151714`
  - F1: `0.132238`
  - AUROC: `0.566825`
  - AUPRC: `0.082963`
  - predicted anomalies: `4531`
- interpretation:
  - this confirms that the UMAP is useful here mainly as an explanatory diagnostic, not as a stronger replacement threshold for the original PatchCore wafer score
  - for this ResNet18 branch, the original PatchCore score remains the meaningful deployment signal; the UMAP-space KNN score is substantially weaker

![ResNet18 PatchCore UMAP by split](experiments/anomaly_detection/patchcore/resnet18/x64/holdout70k_3p5k_umap_followup/artifacts/patchcore_resnet18_holdout70k_3p5k/max_mb50k/evaluation/plots/umap_by_split.png)

![ResNet18 PatchCore UMAP by anomaly score](experiments/anomaly_detection/patchcore/resnet18/x64/holdout70k_3p5k_umap_followup/artifacts/patchcore_resnet18_holdout70k_3p5k/max_mb50k/evaluation/plots/umap_by_score.png)

## Experiment 10: PatchCore Sweep with Pretrained ResNet50 `64x64`

Purpose:

- test whether a larger pretrained `ResNet50` backbone can improve the non-AE PatchCore branch beyond the `ResNet18` result
- keep the same shared `64x64` 5% split and the same validation-threshold evaluation rule

Implementation:

- config: [train_patchcore_resnet50.toml](configs/training/train_patchcore_resnet50.toml)
- notebook: [11_patchcore_resnet50_training.ipynb](notebooks/anomaly_50k/11_patchcore_resnet50_training.ipynb)
- artifact dir: [artifacts/x64/patchcore_resnet50](artifacts/x64/patchcore_resnet50)
- backbone: frozen ImageNet-pretrained `ResNet50`
- input adaptation: single-channel wafer maps resized internally to `224x224`
- feature dimension: `2048`
- compared variants:
  - `mean`, memory bank `10k`
  - `mean`, memory bank `50k`
  - `topk_mean`, memory bank `10k`, top-k ratio `0.05`
  - `topk_mean`, memory bank `50k`, top-k ratio `0.05`
  - `topk_mean`, memory bank `50k`, top-k ratio `0.10`
  - `max`, memory bank `50k`

Sweep summary:

| variant | reduction | memory bank | top-k ratio | val-threshold F1 | AUROC | AUPRC | best sweep F1 |
| ------- | --------- | ----------- | ----------- | ---------------- | ----- | ----- | ------------- |
| `mean_mb50k` | `mean` | `50000` | `0.10` | `0.419602` | `0.821402` | `0.362657` | `0.439604` |
| `mean_mb10k` | `mean` | `10000` | `0.10` | `0.396285` | `0.804225` | `0.310237` | `0.405738` |
| `topk_mb50k_r005` | `topk_mean` | `50000` | `0.05` | `0.305419` | `0.797955` | `0.279352` | `0.312618` |
| `topk_mb50k_r010` | `topk_mean` | `50000` | `0.10` | `0.303426` | `0.800964` | `0.291924` | `0.317757` |
| `max_mb50k` | `max` | `50000` | `0.10` | `0.272879` | `0.780863` | `0.208679` | `0.282528` |
| `topk_mb10k_r005` | `topk_mean` | `10000` | `0.05` | `0.268562` | `0.785025` | `0.217361` | `0.285319` |

Best ResNet50 PatchCore variant under the main validation-threshold rule:

- variant: `mean_mb50k`
- precision: `0.339950`
- recall: `0.548000`
- F1: `0.419602`
- AUROC: `0.821402`
- AUPRC: `0.362657`
- best test-sweep threshold: `0.565182`
- best test-sweep F1: `0.439604`

Failure analysis for `mean_mb50k`:

- true positive: `137`, mean score `0.598213`
- false negative: `113`, mean score `0.504366`
- false positive: `266`, mean score `0.574844`
- true negative: `4734`, mean score `0.488310`

Defect-type recall for `mean_mb50k`:

- `Edge-Ring`: `0.833333`
- `Center`: `0.360000`
- `Edge-Loc`: `0.320755`
- `Loc`: `0.382353`
- `Scratch`: `0.466667`
- `Donut`: `0.857143`
- `Random`: `0.800000`
- `Near-full`: `1.000000`

Interpretation:

- ResNet50 + PatchCore improved the best non-AE validation-threshold F1 over the ResNet18 PatchCore branch
- the improvement came mainly from higher recall at the selected validation threshold
- `mean` reduction remained the clear winner, and `50k` memory bank again beat `10k`
- the larger backbone did not improve every metric at once; `ResNet50` improved F1 over `ResNet18`, but `ResNet18` remained slightly stronger on AUROC and AUPRC
- even with that improvement, the best ResNet50 PatchCore result still stayed below the AE + BatchNorm + `max_abs` winner and was later overtaken clearly by the WideResNet50-2 PatchCore follow-up

### Experiment 16: PatchCore Sweep with WideResNet50-2 Multilayer `64x64`

Purpose:

- test whether the stronger `WideResNet50-2` backbone can improve PatchCore once the model uses two spatial feature scales together
- keep the same shared `64x64` 5% split and the same validation-threshold evaluation rule as the earlier PatchCore notebooks
- compare a small critical sweep first, then a targeted follow-up only around the strongest `topk_mean` region

Implementation:

- config: [train_patchcore_wideresnet50_multilayer.toml](configs/training/train_patchcore_wideresnet50_multilayer.toml)
- notebook: [notebook.ipynb](experiments/anomaly_detection/patchcore/wideresnet50/x64/main/notebook.ipynb)
- artifact dir: [patchcore_wideresnet50_multilayer](experiments/anomaly_detection/patchcore/wideresnet50/x64/main/artifacts/patchcore_wideresnet50_multilayer)
- execution note: the canonical repo notebook is now fully local, and the cleaned artifact bundle includes the combined sweep table plus per-variant evaluation plots
- backbone: frozen ImageNet-pretrained `WideResNet50-2`
- feature layers: `layer2` + `layer3`
- input adaptation: single-channel wafer maps resized internally to `224x224`
- feature dimension: `1536`
- query chunk size: `1024`
- memory chunk size: `4096`
- compared variants:
  - `mean`, memory bank `20k`
  - `mean`, memory bank `50k`
  - `topk_mean`, memory bank `50k`, top-k ratio `0.05`
  - `topk_mean`, memory bank `50k`, top-k ratio `0.10`
  - `topk_mean`, memory bank `50k`, top-k ratio `0.15`
  - `topk_mean`, memory bank `50k`, top-k ratio `0.20`
  - `topk_mean`, memory bank `50k`, top-k ratio `0.25`
  - `max`, memory bank `50k`

Sweep summary:

| variant | reduction | memory bank | top-k ratio | val-threshold F1 | AUROC | AUPRC | best sweep F1 |
| ------- | --------- | ----------- | ----------- | ---------------- | ----- | ----- | ------------- |
| `topk_mb50k_r010` | `topk_mean` | `50000` | `0.10` | `0.531758` | `0.916943` | `0.561855` | `0.585774` |
| `topk_mb50k_r015` | `topk_mean` | `50000` | `0.15` | `0.525526` | `0.911559` | `0.548919` | `0.592902` |
| `topk_mb50k_r005` | `topk_mean` | `50000` | `0.05` | `0.525074` | `0.920518` | `0.563612` | `0.579281` |
| `topk_mb50k_r020` | `topk_mean` | `50000` | `0.20` | `0.514793` | `0.906730` | `0.533386` | `0.585657` |
| `topk_mb50k_r025` | `topk_mean` | `50000` | `0.25` | `0.506550` | `0.902348` | `0.518213` | `0.575488` |
| `mean_mb50k` | `mean` | `50000` | `0.10` | `0.484305` | `0.873711` | `0.413971` | `0.510714` |
| `mean_mb20k` | `mean` | `20000` | `0.10` | `0.466667` | `0.875149` | `0.389518` | `0.471002` |
| `max_mb50k` | `max` | `50000` | `0.10` | `0.397554` | `0.876209` | `0.397211` | `0.451327` |

Best WideResNet50-2 PatchCore variant under the main validation-threshold rule:

- variant: `topk_mb50k_r010`
- precision: `0.421546`
- recall: `0.720000`
- F1: `0.531758`
- AUROC: `0.916943`
- AUPRC: `0.561855`
- validation-derived threshold: `0.529660`
- best test-sweep threshold: `0.548772`
- best test-sweep F1: `0.585774`

Failure analysis for `topk_mb50k_r010`:

- true positive: `180`
- false negative: `70`
- false positive: `247`
- true negative: `4753`
- predicted anomalies at deployed threshold: `427`

Interpretation:

- this run is the first PatchCore result in the project that clearly beats the strongest AE family run and also edges past the best teacher-student result on deployed F1
- the gain did not come from backbone scale alone, because the plain `WideResNet50-2` center-distance baseline stayed weak; it came from combining the larger backbone with multilayer local PatchCore scoring
- `topk_mean` became the decisive reduction family for this backbone; all `topk_mean` variants beat both `mean` variants, and `max` remained clearly weaker
- the best operating region was narrow rather than broad: `r = 0.10` gave the best deployed F1, while `r = 0.05` slightly improved AUROC and AUPRC but not the deployed thresholded result
- memory bank size still mattered: `mean_mb50k` was clearly better than `mean_mb20k`, which is consistent with the earlier ResNet PatchCore runs

### Experiment 18A: PatchCore with WideResNet50-2 Multilayer All-in-One `x224`

Purpose:

- test whether the same multilayer `WideResNet50-2` PatchCore branch improves materially when wafer preprocessing is cached directly at `224x224`
- keep the report-compatible split and the same validation-normal threshold rule, while reusing the all-in-one Modal-friendly workflow
- determine whether the strongest `64x64` WRN PatchCore operating region remains strong after moving the full preprocessing path to `224x224`

Implementation:

- notebook: [notebook.ipynb](experiments/anomaly_detection/patchcore/wideresnet50/x224/multilayer/notebook.ipynb)
- artifact dir: [patchcore-wideresnet50-multilayer](experiments/anomaly_detection/patchcore/wideresnet50/x224/multilayer/artifacts/patchcore-wideresnet50-multilayer)
- execution note: the canonical repo notebook now runs locally and reuses the cleaned artifact bundle under the experiment folder
- split and evaluation:
  - same shared `40,000 / 5,000 / 5,000 + 250` report split from raw `LSWMD.pkl`
  - threshold selected as the `95th` percentile of validation-normal raw scores
- key model settings:
  - wafer preprocessing cached directly at `224x224`
  - frozen ImageNet-pretrained `WideResNet50-2`
  - multilayer patch embedding from `layer2` and `layer3`
  - feature dimension `1536`
  - memory bank cap `600,000`
  - wafer score = mean of top `5%` or top `10%` patch anomaly scores

Sweep summary:

| variant | reduction | memory bank | top-k ratio | val-threshold precision | val-threshold recall | val-threshold F1 | AUROC | AUPRC | best sweep F1 |
| ------- | --------- | ----------- | ----------- | ----------------------- | -------------------- | ---------------- | ----- | ----- | ------------- |
| `topk_mb50k_r005_x224` | `topk_mean` | `600000` | `0.05` | `0.432184` | `0.752000` | `0.548905` | `0.930680` | `0.659063` | `0.634146` |
| `topk_mb50k_r010_x224` | `topk_mean` | `600000` | `0.10` | `0.418345` | `0.748000` | `0.536585` | `0.920673` | `0.619938` | `0.598753` |

Selected result:

| variant | val-threshold precision | val-threshold recall | val-threshold F1 | AUROC | AUPRC | best sweep F1 |
| ------- | ----------------------- | -------------------- | ---------------- | ----- | ----- | ------------- |
| `PatchCore-WideRes50-x224-topk-mb600k-r005` | `0.432184` | `0.752000` | `0.548905` | `0.930680` | `0.659063` | `0.634146` |

Per-defect recall:

| defect type | recall |
| ----------- | ------ |
| `Edge-Loc` | `0.622642` |
| `Scratch` | `0.666667` |
| `Loc` | `0.735294` |
| `Center` | `0.780000` |
| `Edge-Ring` | `0.797619` |
| `Donut` | `1.000000` |
| `Random` | `1.000000` |
| `Near-full` | `1.000000` |

Interpretation:

- at that stage, this follow-up was the strongest completed deployment-style result in the report, improving over both the earlier `64x64` WRN PatchCore leader and the direct-`224x224` EfficientNet-B0 PatchCore run on deployed F1
- the gain is not just thresholded: this run also sets the current best `AUROC`, `AUPRC`, and best-sweep F1 in the report, which means the improvement is visible in both ranking quality and the deployed operating point
- direct `224x224` preprocessing appears to matter for the WRN PatchCore branch just as it did for EfficientNet-B0, but the larger WRN backbone now converts that extra spatial detail into a stronger final operating point
- the best ratio moved slightly toward a narrower top-k region than the earlier `64x64` winner: here `r = 0.05` beats `r = 0.10` on both thresholded F1 and ranking metrics
- defect behavior is especially encouraging in the small-local regime: `Scratch`, `Loc`, and `Edge-Loc` all improve over the older `64x64` WRN PatchCore pattern, while broad defects remain consistently easy

![WRN PatchCore x224 selected variant score distribution](experiments/anomaly_detection/patchcore/wideresnet50/x224/multilayer/artifacts/patchcore-wideresnet50-multilayer/topk_mb50k_r005_x224/plots/score_distribution.png)

![WRN PatchCore x224 selected variant threshold sweep](experiments/anomaly_detection/patchcore/wideresnet50/x224/multilayer/artifacts/patchcore-wideresnet50-multilayer/topk_mb50k_r005_x224/plots/threshold_sweep.png)

![WRN PatchCore x224 variant comparison](experiments/anomaly_detection/patchcore/wideresnet50/x224/multilayer/artifacts/patchcore-wideresnet50-multilayer/plots/variant_comparison_metrics.png)

### Experiment 18A2: UMAP Export Follow-Up for the Selected WRN PatchCore `x224` Checkpoint

Purpose:

- export the selected `18A` checkpoint's saved embeddings into a stable UMAP visualization without changing the trained model or the reported evaluation protocol
- inspect whether the stronger `x224` PatchCore result is coming from a healthier embedding geometry than the weak plain WRN center-distance baseline

Implementation:

- notebook: [notebook.ipynb](experiments/anomaly_detection/patchcore/wideresnet50/x224/multilayer_umap_followup/notebook.ipynb)
- artifact dir: [patchcore-wideresnet50-multilayer-umap](experiments/anomaly_detection/patchcore/wideresnet50/x224/multilayer_umap_followup/artifacts/patchcore-wideresnet50-multilayer-umap)
- checkpoint reused: selected `18A` variant `topk_mb50k_r005_x224`
- visualization note: this follow-up is interpretive only; it does not introduce a new training run or change the reported metrics above

UMAP interpretation:

- validation-normal and test-normal points still overlap heavily, which is useful because it suggests the stronger result is not coming from a train/test distribution shift artifact
- unlike the plain WRN embedding baseline, anomaly wafers no longer blanket the full dominant manifold; they appear more often on branch edges, thinner tails, and smaller side islands
- that pattern is still not a clean class split, so the task remains hard, but it is much more compatible with a local PatchCore score than with one global center-distance threshold
- in other words, the UMAP supports the same conclusion as the metrics: the win came from stronger local geometry plus local scoring, not from backbone scale alone

![WRN PatchCore x224 selected-checkpoint UMAP](experiments/anomaly_detection/patchcore/wideresnet50/x224/multilayer_umap_followup/artifacts/patchcore-wideresnet50-multilayer-umap/plots/selected_variant_umap_by_split.png)

The PatchCore family figure below now summarizes the full branch, including the newer WideResNet50-2 multilayer follow-up: the left panel compares the best PatchCore result from each backbone or source, while the right panel shows all PatchCore variants colored by backbone/source and marked by wafer-level reduction.

![PatchCore family comparison](artifacts/report_plots/patchcore_family_comparison.png)

## Experiment Family: Teacher-Distillation-ResNet `64x64`

This is one family of experiments built around the same teacher-student distillation design:

- frozen pretrained ResNet teacher backbone
- lightweight student CNN trained on normal wafers
- auxiliary feature autoencoder branch
- shared `64x64` wafer split and the same validation-threshold evaluation rule

Within this family, two backbone variants were tested:

- `Teacher-Distillation-ResNet18`
- `Teacher-Distillation-ResNet50`

The teacher-distillation family figure below compares the two backbone variants directly: the left panel shows their deployed F1, AUPRC, and AUROC side by side, while the right panel shows the precision-recall operating point reached by each selected score rule.

![Teacher-distillation family comparison](artifacts/report_plots/ts_family_comparison.png)

### Experiment 11: Teacher-Distillation-ResNet18

Purpose:

- test whether a teacher-student distillation model can improve sensitivity to smaller local defects on the shared split
- keep the same shared `64x64` 5% test-defect setup and the same validation-threshold evaluation rule

Implementation:

- config: [train_ts_resnet18.toml](configs/training/train_ts_resnet18.toml)
- notebook: [12_ts_distillation_training.ipynb](notebooks/anomaly_50k/12_ts_distillation_training.ipynb)
- training script: [train_ts_distillation.py](scripts/train_ts_distillation.py)
- artifact dir: [artifacts/x64/ts_resnet18](artifacts/x64/ts_resnet18)
- model: teacher-student distillation detector with a frozen teacher backbone, a lightweight student CNN, and an auxiliary feature autoencoder
- teacher backbone: frozen ImageNet-pretrained `ResNet18`
- teacher feature layer: `layer2`
- input adaptation: single-channel wafer maps resized internally to `224x224` through the shared ResNet preprocessing path
- student branch: lightweight convolutional student network trained to match teacher feature maps on normal wafers
- auxiliary branch: feature autoencoder trained to reconstruct teacher feature maps
- reported anomaly map: normalized student-teacher discrepancy only
- auxiliary branch status in the selected score: kept for training, but not used in the primary reported wafer-level score
- wafer-level score: `topk_mean`
- selected top-k ratio: `0.20`
- selected score came from a post-training score sweep over branch weights and wafer-level reductions in notebook `12`

Training observations:

- training was stable across the full `30` epochs
- epoch 1: train `0.050254`, val `0.033348`
- epoch 10: train `0.024442`, val `0.024482`
- epoch 20: train `0.023601`, val `0.023673`
- best epoch 27: train `0.023342`, val `0.023353`
- epoch 30: train `0.023275`, val `0.023421`
- early stopping did not trigger before the configured max epoch count
- post-training error-map scales: student `0.011238`, feature autoencoder `0.012071`

Evaluation:

- validation threshold: `2.342451`
- precision: `0.402500`
- recall: `0.644000`
- F1: `0.495385`
- AUROC: `0.894076`
- AUPRC: `0.519445`
- predicted anomalies: `400`
- confusion matrix: `[[4761, 239], [89, 161]]`
- best test-sweep threshold: `2.459717`
- best test-sweep precision: `0.509579`
- best test-sweep recall: `0.532000`
- best test-sweep F1: `0.520548`

Score-sweep summary:

- full sweep artifact: [score_sweep_summary.csv](artifacts/x64/ts_resnet18/evaluation/score_sweep_summary.csv)

| variant | student weight | autoencoder weight | reduction | top-k ratio | val-threshold F1 | AUROC | AUPRC | best sweep F1 |
| ------- | -------------- | ------------------ | --------- | ----------- | ---------------- | ----- | ----- | ------------- |
| `s1_a0_topk_mean_r0.20` | `1.0` | `0.0` | `topk_mean` | `0.20` | `0.495385` | `0.894076` | `0.519445` | `0.520548` |
| `s1_a0_topk_mean_r0.10` | `1.0` | `0.0` | `topk_mean` | `0.10` | `0.488189` | `0.890692` | `0.496916` | `0.493023` |
| `s2_a1_topk_mean_r0.20` | `2.0` | `1.0` | `topk_mean` | `0.20` | `0.476043` | `0.893469` | `0.453071` | `0.487141` |
| `s1_a0.5_topk_mean_r0.20` | `1.0` | `0.5` | `topk_mean` | `0.20` | `0.476043` | `0.893469` | `0.453071` | `0.487141` |
| `s1_a1_topk_mean_r0.20` | `1.0` | `1.0` | `topk_mean` | `0.20` | `0.465331` | `0.891553` | `0.406017` | `0.476879` |
| `s1_a2_topk_mean_r0.20` | `1.0` | `2.0` | `topk_mean` | `0.20` | `0.463453` | `0.888585` | `0.359451` | `0.467719` |

Focused teacher-layer ablation summary:

- full ablation artifact: [ablation_sweep_summary.csv](artifacts/x64/ts_resnet18/evaluation/ablation_sweep_summary.csv)

| variant | teacher layer | top-k ratio | epochs | val-threshold F1 | AUROC | AUPRC | best sweep F1 |
| ------- | ------------- | ----------- | ------ | ---------------- | ----- | ----- | ------------- |
| `ts_resnet18_layer2_topk25` | `layer2` | `0.25` | `15` | `0.493865` | `0.889740` | `0.514988` | `0.521452` |
| `ts_resnet18_layer2_topk20` | `layer2` | `0.20` | `15` | `0.493827` | `0.888565` | `0.509206` | `0.511194` |
| `ts_resnet18_layer2_topk15` | `layer2` | `0.15` | `15` | `0.483077` | `0.889012` | `0.506745` | `0.501754` |
| `ts_resnet18_layer1_topk15` | `layer1` | `0.15` | `15` | `0.411483` | `0.823581` | `0.339756` | `0.436782` |
| `ts_resnet18_layer1_topk20` | `layer1` | `0.20` | `15` | `0.386581` | `0.817318` | `0.319118` | `0.419708` |
| `ts_resnet18_layer1_topk25` | `layer1` | `0.25` | `15` | `0.378467` | `0.807248` | `0.299344` | `0.405680` |

Failure analysis:

- true positive: `161`, mean score not recomputed in this rerun summary
- false negative: `89`, mean score not recomputed in this rerun summary
- false positive: `239`, mean score not recomputed in this rerun summary
- true negative: `4761`, mean score not recomputed in this rerun summary

Defect-type recall:

- defect-type recall was not recomputed in the current notebook output and should be regenerated before making per-class claims for this rerun

Interpretation:

- the current rerun confirms that the teacher-student branch is stable and genuinely competitive, but it no longer leads the full report on validation-threshold F1
- the selected student-only `topk_mean` score remains the right deployed score; the score sweep in the current notebook matched the default result rather than improving it further
- even in this `TS-Res18` rerun, teacher-student distillation still showed a stronger ranking signal than the AE baseline; that direction was later strengthened further by the `TS-Res50` variation
- compared with the BatchNorm autoencoder, the teacher-student model now looks like a higher-recall, lower-precision alternative with a stronger ranking signal but less clean threshold separation
- the optional teacher-layer ablation sweep in notebook `12` now supports non-`layer2` variants after the feature-map alignment fix
- the completed focused ablation results did not beat the main `30`-epoch base run: all `layer1` variants were clearly weaker, while `layer2` with top-k ratios `0.15`, `0.20`, and `0.25` stayed close to the default result without improving on it

### Experiment 12: Teacher-Distillation-ResNet50

Purpose:

- test whether scaling the teacher backbone from `ResNet18` to `ResNet50` improves local anomaly ranking and thresholded detection
- keep the main `TS-ResNet50` experiment reproducible from this repo, even though the original heavy training run was executed on Kaggle
- evaluate the imported Kaggle-trained checkpoint locally under the same shared protocol and the same post-training score sweep logic used for the `ResNet18` teacher run

Implementation:

- main repo notebook: [13_ts_resnet50_kaggle_import_analysis.ipynb](notebooks/anomaly_50k/13_ts_resnet50_kaggle_import_analysis.ipynb)
- repo layer-comparison notebook: [14_ts_resnet50_teacher_layer_ablation_analysis.ipynb](notebooks/anomaly_50k/14_ts_resnet50_teacher_layer_ablation_analysis.ipynb)
- local import config: [train_ts_resnet50_kaggle.toml](configs/training/train_ts_resnet50_kaggle.toml)
- training entry point: [train_ts_distillation.py](scripts/train_ts_distillation.py)
- evaluation entry point: [evaluate_reconstruction_model.py](scripts/evaluate_reconstruction_model.py)
- artifact dir: [artifacts/x64/ts_resnet50](artifacts/x64/ts_resnet50)
- raw focused Kaggle ablation notebooks: [resNet50_abal_1.ipynb](artifacts/x64/kaggle_rn50_ablation/resNet50_abal_1.ipynb), [resnet50-abal-2.ipynb](artifacts/x64/kaggle_rn50_ablation/resnet50-abal-2.ipynb)
- focused Kaggle ablation leaderboards: [leaderboard_live.csv](artifacts/x64/kaggle_rn50_ablation/leaderboard_live.csv), [leaderboard_live2.csv](artifacts/x64/kaggle_rn50_ablation/leaderboard_live2.csv)
- model: teacher-student distillation detector with a frozen `ResNet50` teacher, a lightweight student CNN, and an auxiliary feature autoencoder
- teacher backbone: frozen ImageNet-pretrained `ResNet50`
- teacher feature layer: `layer2`
- input adaptation: single-channel wafer maps resized internally to `224x224`
- student branch width: `512`
- feature-autoencoder hidden width: `128`
- imported Kaggle default score: student-only wafer-level `topk_mean`, top-k ratio `0.20`
- selected local score after import analysis: mixed student+autoencoder score with weights `2.0 : 1.0`, wafer-level `topk_mean`, top-k ratio `0.20`

Notebook roles:

- notebook `13` is now the main runnable `TS-ResNet50` notebook in the repo; it can launch local training, reuse the imported Kaggle artifact, remap the checkpoint into the repo model format, rerun local evaluation, and perform the score sweep
- notebook `14` is the focused follow-up notebook for the teacher-layer comparison; it keeps the `layer1` versus `layer2` analysis separate from the main base-run narrative
- this keeps the experiment reproducible from the repo while still acknowledging that the original best `ResNet50` teacher run was trained on external hardware

Training observations:

- Kaggle training was stable through all `30` epochs
- best epoch: `29`
- best validation loss: `0.370080`
- epoch 26: train `0.370199`, val `0.370970`
- epoch 27: train `0.369908`, val `0.371948`
- epoch 28: train `0.369559`, val `0.371207`
- epoch 29: train `0.369285`, val `0.370080`
- epoch 30: train `0.368939`, val `0.370501`
- post-training error-map scales: student `0.177813`, feature autoencoder `0.191531`

Imported Kaggle evaluation:

- validation threshold: `2.255374`
- precision: `0.382353`
- recall: `0.676000`
- F1: `0.488439`
- AUROC: `0.912691`
- AUPRC: `0.581770`
- predicted anomalies: `442`
- confusion matrix: `[[4727, 273], [81, 169]]`
- best test-sweep threshold: `2.403453`
- best test-sweep precision: `0.538462`
- best test-sweep recall: `0.588000`
- best test-sweep F1: `0.562141`

Local import-analysis findings:

- the imported checkpoint remapped cleanly into the repo model format and reproduced the Kaggle result almost exactly
- local re-evaluation kept the same thresholded operating point: precision `0.382353`, recall `0.676000`, F1 `0.488439`
- local AUROC and AUPRC matched closely: AUROC `0.912730`, AUPRC `0.581275`

Local score-sweep result:

- full local sweep artifact: [score_sweep_summary.csv](artifacts/x64/ts_resnet50/evaluation_local/score_sweep_summary.csv)

| variant | student weight | autoencoder weight | reduction | top-k ratio | val-threshold F1 | AUROC | AUPRC | best sweep F1 |
| ------- | -------------- | ------------------ | --------- | ----------- | ---------------- | ----- | ----- | ------------- |
| `s2_a1_topk_mean_r0.20` | `2.0` | `1.0` | `topk_mean` | `0.20` | `0.524590` | `0.909189` | `0.599169` | `0.606299` |
| `s1_a0.5_topk_mean_r0.20` | `1.0` | `0.5` | `topk_mean` | `0.20` | `0.524590` | `0.909189` | `0.599169` | `0.606299` |
| `s1_a2_topk_mean_r0.20` | `1.0` | `2.0` | `topk_mean` | `0.20` | `0.522523` | `0.904408` | `0.569921` | `0.564756` |
| `s0.5_a1_topk_mean_r0.20` | `0.5` | `1.0` | `topk_mean` | `0.20` | `0.522523` | `0.904408` | `0.569921` | `0.564756` |
| `s1_a1_topk_mean_r0.20` | `1.0` | `1.0` | `topk_mean` | `0.20` | `0.521610` | `0.905117` | `0.588096` | `0.584222` |
| `s2_a1_mean` | `2.0` | `1.0` | `mean` | `-` | `0.512121` | `0.889755` | `0.501980` | `0.552962` |
| `s2_a1_topk_mean_r0.10` | `2.0` | `1.0` | `topk_mean` | `0.10` | `0.505848` | `0.912989` | `0.584444` | `0.562937` |
| `s1_a1_topk_mean_r0.10` | `1.0` | `1.0` | `topk_mean` | `0.10` | `0.503639` | `0.912523` | `0.558833` | `0.543253` |

- best selected variant: `s2_a1_topk_mean_r0.20`
- student weight: `2.0`
- autoencoder weight: `1.0`
- reduction: `topk_mean`
- top-k ratio: `0.20`
- validation threshold: `7.020788`
- precision: `0.418052`
- recall: `0.704000`
- F1: `0.524590`
- AUROC: `0.909189`
- AUPRC: `0.599169`
- predicted anomalies: `421`
- best test-sweep F1: `0.606299`

Focused teacher-layer comparison:

The completed `TS-ResNet50` record now includes both a main imported `layer2` run and a separate focused Kaggle ablation sweep covering additional `layer1` settings. The verified saved leaderboard files currently available in the repo are the `layer1` ablation snapshots below; together with the imported `layer2` result, they show that both teacher-feature layers were actually run and compared.

| variant | teacher layer | score setup | top-k ratio | precision | recall | F1 | AUROC | AUPRC | best sweep F1 |
| ------- | ------------- | ----------- | ----------- | --------- | ------ | -- | ----- | ----- | ------------- |
| TS-Res50 imported default | `layer2` | student-only `topk_mean` | `0.20` | `0.382353` | `0.676000` | `0.488439` | `0.912691` | `0.581770` | `0.562141` |
| TS-Res50 selected local score | `layer2` | mixed `2.0 : 1.0` `topk_mean` | `0.20` | `0.418052` | `0.704000` | `0.524590` | `0.909189` | `0.599169` | `0.606299` |
| tsres50_layer1_topk0p1_sw1p0_aw0p5 | `layer1` | mixed `1.0 : 0.5` `topk_mean` | `0.10` | `0.407862` | `0.664000` | `0.505327` | `0.872754` | `0.527526` | `0.547284` |
| tsres50_layer1_topk0p1_sw2p0_aw1p0 | `layer1` | mixed `2.0 : 1.0` `topk_mean` | `0.10` | `0.401460` | `0.660000` | `0.499244` | `0.872678` | `0.521696` | `0.545906` |
| tsres50_layer1_topk0p1_sw2p0_aw0p5 | `layer1` | mixed `2.0 : 0.5` `topk_mean` | `0.10` | `0.396226` | `0.672000` | `0.498516` | `0.867110` | `0.518197` | `0.564516` |
| tsres50_layer1_topk0p2_sw2p0_aw1p0 | `layer1` | mixed `2.0 : 1.0` `topk_mean` | `0.20` | `0.399491` | `0.628000` | `0.488336` | `0.863550` | `0.453591` | `0.535865` |

Cross-layer takeaway:

- `layer2` remained the strongest completed `TS-Res50` direction overall
- the best verified `layer1` run reached `F1 = 0.505327`, which is competitive but still below the selected `layer2` score at `F1 = 0.524590`
- `layer1` could still deliver useful recall, but it consistently gave weaker ranking quality than the main `layer2` run
- this makes the teacher feature layer a real experimental axis for `ResNet50`, not just an implementation detail

Interpretation:

- `TS-ResNet50` is the second tested variation inside the same teacher-distillation-resnet family, not a separate method family
- the imported Kaggle default score already set the strongest AUROC in the project and a stronger AUPRC than the `ResNet18` teacher run
- unlike `TS-Res18`, the `ResNet50` teacher checkpoint benefited from bringing the feature-autoencoder branch back into the deployed score
- at that stage, after local score selection, `TS-ResNet50` became the best completed result in the report on validation-threshold F1
- it also remains one of the strongest ranking models in the report, with AUROC still above `0.90`
- the main gain over `TS-Res18` came from better recall and better class-imbalance ranking quality, not from a cleaner threshold alone
- this result suggests that teacher scale and score composition interact strongly: the larger teacher backbone made the auxiliary branch useful again rather than harmful

### Secondary Holdout Benchmark: Expanded `70k` Normal / `3.5k` Defect Test Set

The main report benchmark remains the shared `50k_5pct` split, but we also built a larger disjoint secondary holdout to check whether saved checkpoints stay stable when the test set is much larger.

Evaluation bundle:

- leaderboard: [artifacts/x64/holdout70k_3p5k_evaluations/leaderboard.csv](artifacts/x64/holdout70k_3p5k_evaluations/leaderboard.csv)
- full compiled results: [artifacts/x64/holdout70k_3p5k_evaluations/compiled_full.csv](artifacts/x64/holdout70k_3p5k_evaluations/compiled_full.csv)
- histogram outputs: [artifacts/x64/holdout70k_3p5k_evaluations/_histograms_all](artifacts/x64/holdout70k_3p5k_evaluations/_histograms_all)
- metadata: [data/processed/x64/wm811k/metadata_50k_5pct_holdout70k_3p5k.csv](data/processed/x64/wm811k/metadata_50k_5pct_holdout70k_3p5k.csv)

Dataset breakdown:

| split | original `50k_5pct` | secondary holdout |
| ----- | ------------------- | ----------------- |
| train normals | `40,000` | `40,000` |
| val normals | `5,000` | `5,000` |
| test normals | `5,000` | `70,000` |
| test defects | `250` | `3,500` |

Test defect-type breakdown:

| defect type | original test | secondary holdout test |
| ----------- | ------------- | ---------------------- |
| Edge-Ring | `84` | `1,302` |
| Edge-Loc | `53` | `739` |
| Center | `50` | `603` |
| Loc | `34` | `492` |
| Scratch | `15` | `169` |
| Random | `5` | `108` |
| Donut | `7` | `71` |
| Near-full | `2` | `16` |

Rerun scope:

- this benchmark reuses saved checkpoints from the autoencoder, PatchCore, teacher-student, VAE, and SVDD branches
- each run keeps the original `40k / 5k` train/validation normals fixed and applies the same validation-derived thresholding rule as in the main report
- the leaderboard below reflects what was actually rerun on this expanded holdout; some newer report winners, especially the later WRN `x224` PatchCore follow-ups, are not part of this bundle yet

Leaderboard on the expanded holdout:

| rank | model | family | selected score | val-threshold F1 | AUROC | AUPRC | best sweep F1 |
| ---- | ----- | ------ | -------------- | ---------------- | ----- | ----- | ------------- |
| `1` | `ts_resnet50` | `ts_distillation` | checkpoint default | `0.515853` | `0.925607` | `0.621967` | `0.584928` |
| `2` | `autoencoder_batchnorm` | `autoencoder` | `max_abs` | `0.507116` | `0.858353` | `0.637568` | `0.662647` |
| `3` | `autoencoder_residual` | `autoencoder` | `max_abs` | `0.503347` | `0.871261` | `0.636548` | `0.643216` |
| `4` | `autoencoder_batchnorm_dropout_d0p00` | `autoencoder` | `max_abs` | `0.500634` | `0.857073` | `0.654263` | `0.672813` |
| `5` | `ts_resnet18` | `ts_distillation` | checkpoint default | `0.494564` | `0.902961` | `0.544659` | `0.514396` |
| `6` | `ts_resnet18_layer2_topk25` | `ts_distillation` | checkpoint default | `0.494083` | `0.898711` | `0.540878` | `0.515752` |
| `7` | `ts_resnet18_layer2_topk20` | `ts_distillation` | checkpoint default | `0.486365` | `0.898085` | `0.533561` | `0.502040` |
| `8` | `ts_resnet18_layer2_topk15` | `ts_distillation` | checkpoint default | `0.483770` | `0.899496` | `0.536538` | `0.501002` |
| `9` | `autoencoder_baseline` | `autoencoder` | `topk_abs_mean` | `0.455000` | `0.839591` | `0.534693` | `0.525665` |
| `10` | `patchcore_resnet50__mean_mb50k` | `patchcore` | checkpoint default | `0.426488` | `0.823888` | `0.372599` | `0.449047` |

Selected histogram views from the expanded-holdout bundle:

- `TS-ResNet50` remains the strongest deployed run in this bundle; the histogram shows meaningful right-shift for anomalies, but still with substantial overlap near the operating threshold
- `AE-BN` with `max_abs` remains unusually competitive on the larger holdout and actually posts stronger `AUPRC` and best-sweep `F1` than the top teacher-student run, even though its deployed-threshold `F1` stays slightly lower
- the best evaluated PatchCore rerun in this bundle is `PatchCore-ResNet50-mean-mb50k`, and its histogram shows much heavier overlap around the threshold than the top teacher-student and autoencoder representatives

![Expanded-holdout histogram: TS-ResNet50](artifacts/x64/holdout70k_3p5k_evaluations/_histograms_all/ts_resnet50.png)

![Expanded-holdout histogram: AE-BN `max_abs`](artifacts/x64/holdout70k_3p5k_evaluations/_histograms_all/autoencoder_batchnorm__max_abs.png)

![Expanded-holdout histogram: PatchCore-ResNet50 `mean_mb50k`](artifacts/x64/holdout70k_3p5k_evaluations/_histograms_all/patchcore_resnet50__mean_mb50k.png)

Interpretation:

- this larger holdout is intentionally a secondary validation only; it does not replace the original `50k_5pct` benchmark table
- the original `40k / 5k` train/val normals stayed fixed across these reruns, so the checkpoint choice and threshold policy did not change
- `TS-ResNet50` still leads this expanded-holdout bundle on deployed `F1` and `AUROC`, which makes it the most stable top performer among the models actually reevaluated here
- the best autoencoder variants stayed much closer than the main report ranking might suggest; on this larger holdout they trail `TS-ResNet50` slightly on deployed `F1`, but they beat it on `AUPRC` and on optimistic best-sweep `F1`
- the strongest evaluated PatchCore rerun is much weaker here than the top teacher-student and autoencoder entries, but that should be interpreted narrowly: this bundle does not yet include the later WRN `x224` PatchCore winner from the main report
- the false-positive rates stay close to the expected `~5%` implied by the `95th`-percentile validation thresholds, which is reassuring given the jump from `5,000` to `70,000` test-normal wafers
- the much larger defect pool also makes rare-class recall estimates far more meaningful than on the original `250`-defect test set
- this bundle is therefore most useful as a robustness check on the saved checkpoints we did reevaluate, not yet as a full replacement leaderboard for every strong model in the project

## Experiment Family: WideResNet50-2 `64x64`

This family extends the pretrained-backbone anomaly experiments beyond `ResNet18` and `ResNet50` by testing a larger `WideResNet50-2` backbone in plain embedding, teacher-student, and PatchCore forms.

The sequence of runs was:

- first, a plain frozen `WideResNet50-2` embedding baseline to check whether backbone scale alone improves the weak center-distance direction
- second, a single-layer teacher-student `WideResNet50-2` run with score ablations over branch weights and wafer-level reductions
- third, a multilayer teacher-student run using both `layer2` and `layer3` to test whether combining two teacher feature scales improves detection further
- fourth, a multilayer PatchCore follow-up using the same `layer2 + layer3` pair to test whether the wider backbone works even better with a memory-bank local-anomaly method
- that PatchCore follow-up was also sweep-based: notebook `18` first ran a small critical sweep, then a short follow-up around the strongest `topk_mean` region

The WideResNet50-2 family figure below summarizes that branch directly using the representative selected row from each completed WRN experiment: the plain backbone baseline, the single-layer teacher-student run, the multilayer teacher-student run, and the selected multilayer PatchCore run. The left panel ranks those experiment-level results by deployed F1, while the right panel shows their precision-recall operating points with point size scaled by best-sweep F1.

![WideResNet50-2 family comparison](artifacts/report_plots/wrn_family_comparison.png)

### Experiment 13: Pretrained WideResNet50-2 Backbone Baseline `64x64`

Purpose:

- test whether a larger frozen pretrained backbone improves the plain embedding-distance baseline beyond the earlier `ResNet18` version
- keep the same shared `64x64` split and validation-threshold rule

Implementation:

- config: [train_wideresnet50_backbone.toml](configs/training/train_wideresnet50_backbone.toml)
- notebook: [15_wideresnet50_2_backbone_baseline.ipynb](notebooks/anomaly_50k/15_wideresnet50_2_backbone_baseline.ipynb)
- artifact dir: [artifacts/x64/wideresnet50_embedding_baseline](artifacts/x64/wideresnet50_embedding_baseline)
- model: frozen ImageNet-pretrained `WideResNet50-2` adapted to single-channel wafer maps
- score: embedding center-distance from the train-normal feature center
- teacher input size: `224x224`
- embedding dimension: `2048`

Evaluation:

- validation threshold: `8.660283`
- precision: `0.221854`
- recall: `0.268000`
- F1: `0.242754`
- AUROC: `0.677274`
- AUPRC: `0.142323`
- predicted anomalies: `302`
- confusion matrix: `[[4765, 235], [183, 67]]`
- best test-sweep threshold: `9.530535`
- best test-sweep precision: `0.329480`
- best test-sweep recall: `0.228000`
- best test-sweep F1: `0.269504`

Failure analysis:

- strongest meaningful defect class in the notebook output was `Edge-Ring`, with recall `0.464286`
- recall stayed weak on `Edge-Loc`, `Center`, and `Loc`
- `Near-full` was missed entirely in this run

Interpretation:

- the larger pretrained backbone by itself did not solve the center-distance weakness
- `WideResNet50-2` remained only marginally better than the earlier plain ResNet embedding idea in thresholded performance
- this confirmed again that stronger local anomaly scoring matters more than backbone scale alone

UMAP check:

- the saved embedding UMAP for this baseline shows why the metric ceiling stayed low: anomaly wafers still spread through the same large islands occupied by both validation and test normals rather than peeling off into clearly abnormal regions
- there are a few anomaly-heavy side clumps, but the dominant picture is overlap inside the main manifold, which is exactly what a single global center-distance score struggles with

![WideResNet50-2 embedding baseline UMAP](experiments/anomaly_detection/backbone_embedding/wide_resnet50_2/x64/baseline/artifacts/umaps/wideresnet50A_embedding_baseline/evaluation/plots/embedding_umap.png)

### Experiment 14: Teacher-Student Distillation with WideResNet50-2 `layer2`

Purpose:

- test whether the stronger `WideResNet50-2` teacher becomes useful once paired with the local anomaly-map structure of teacher-student distillation
- sweep score composition and wafer-level reduction without changing the trained checkpoint

Implementation:

- config reference: [train_ts_wideresnet50.toml](configs/training/train_ts_wideresnet50.toml)
- notebook: [16-ts-wideresnet50-training-all-in-one.ipynb](notebooks/anomaly_50k/16-ts-wideresnet50-training-all-in-one.ipynb)
- teacher backbone: frozen ImageNet-pretrained `WideResNet50-2`
- teacher feature layer: `layer2`
- input adaptation: single-channel wafer maps resized internally to `224x224`
- auxiliary branch: feature autoencoder
- default configured score: student-only `topk_mean`, top-k ratio `0.20`

Training observations:

- training remained stable through the completed `16` epochs shown in the notebook output
- the best validation loss visible in the run summary was `0.279331` at epoch `11`
- student and feature-autoencoder losses both decreased substantially from the early epochs

Default evaluation:

- validation threshold: `2.076181`
- precision: `0.396313`
- recall: `0.688000`
- F1: `0.502924`
- AUROC: `0.904550`
- AUPRC: `0.516065`
- predicted anomalies: `434`
- confusion matrix: `[[4738, 262], [78, 172]]`
- best test-sweep F1: `0.534846`

Score-sweep summary:

- notebook `16` did run a full score sweep over branch weights and wafer-level reductions
- the selected best row is recorded below
- unlike notebooks `17` and `18`, this single-layer WRN run does not currently have a separate exported sweep-summary CSV copied into the repo artifacts, so the report keeps the selected row and the notebook itself as the source of the wider sweep leaderboard

Score-sweep result:

- best selected variant: `s2_a1_topk_mean_r0.25`
- student weight: `2.0`
- autoencoder weight: `1.0`
- reduction: `topk_mean`
- top-k ratio: `0.25`
- validation threshold: `6.277527`
- precision: `0.404651`
- recall: `0.696000`
- F1: `0.511765`
- AUROC: `0.903371`
- AUPRC: `0.512148`
- best test-sweep F1: `0.526316`

Interpretation:

- moving from the plain `WideResNet50-2` embedding baseline to teacher-student distillation produced a very large jump in every practical metric
- the best selected score beat the default configured score only modestly on F1, which suggests the single-layer run was already reasonably calibrated
- compared with the plain backbone baseline, the real gain came from local anomaly-map learning, not just from the stronger teacher backbone

### Experiment 15: Teacher-Student Distillation with WideResNet50-2 `layer2 + layer3`

Purpose:

- test whether combining two teacher feature scales improves detection beyond the single-layer `WideResNet50-2` teacher-student run
- preserve the same shared split, threshold rule, and score-sweep evaluation logic

Implementation:

- notebook: [17-ts-wideresnet50-training-multilayer.ipynb](notebooks/anomaly_50k/17-ts-wideresnet50-training-multilayer.ipynb)
- artifact dir: [artifacts/x64/wideresnet50_2_modal](artifacts/x64/wideresnet50_2_modal)
- teacher backbone: frozen ImageNet-pretrained `WideResNet50-2`
- teacher layers: `layer2` and `layer3`
- feature dimensions in the notebook output: `layer2 = 512`, `layer3 = 1024`
- configured score: mixed student+autoencoder `topk_mean`, top-k ratio `0.25`

Training observations:

- the best saved training summary reported best epoch `5`
- best validation loss: `0.768422`
- the multilayer run was materially heavier than the single-layer run, but it still produced a stable saved checkpoint and evaluation artifact

Default evaluation:

- validation threshold: `4.006928`
- precision: `0.405728`
- recall: `0.680000`
- F1: `0.508221`
- AUROC: `0.919741`
- AUPRC: `0.539738`
- predicted anomalies: `419`
- confusion matrix: `[[4751, 249], [80, 170]]`
- best test-sweep F1: `0.559387`

Score-sweep summary:

| variant | student weight | autoencoder weight | reduction | top-k ratio | val-threshold F1 | AUROC | AUPRC | best sweep F1 |
| ------- | -------------- | ------------------ | --------- | ----------- | ---------------- | ----- | ----- | ------------- |
| `s2_a1_topk_mean_r0.15` | `2.0` | `1.0` | `topk_mean` | `0.15` | `0.524242` | `0.923114` | `0.546305` | `0.560928` |
| `s1_a1_topk_mean_r0.25` | `1.0` | `1.0` | `topk_mean` | `0.25` | `0.517611` | `0.926123` | `0.537986` | `0.558052` |
| `s2_a1_topk_mean_r0.20` | `2.0` | `1.0` | `topk_mean` | `0.20` | `0.512048` | `0.921612` | `0.542974` | `0.560748` |
| `s2_a1_topk_mean_r0.10` | `2.0` | `1.0` | `topk_mean` | `0.10` | `0.510638` | `0.923445` | `0.543885` | `0.556150` |
| `s1_a1_topk_mean_r0.20` | `1.0` | `1.0` | `topk_mean` | `0.20` | `0.509036` | `0.927437` | `0.541561` | `0.551595` |
| `s1_a2_topk_mean_r0.25` | `1.0` | `2.0` | `topk_mean` | `0.25` | `0.508271` | `0.928086` | `0.533389` | `0.543890` |
| `default_config_score` | `2.0` | `1.0` | `topk_mean` | `0.25` | `0.508221` | `0.919741` | `0.539738` | `0.559387` |
| `s2_a1_topk_mean_r0.25` | `2.0` | `1.0` | `topk_mean` | `0.25` | `0.508221` | `0.919741` | `0.539738` | `0.559387` |
| `s1_a1_topk_mean_r0.10` | `1.0` | `1.0` | `topk_mean` | `0.10` | `0.505988` | `0.927355` | `0.532729` | `0.539450` |
| `s1_a2_topk_mean_r0.20` | `1.0` | `2.0` | `topk_mean` | `0.20` | `0.504505` | `0.929183` | `0.531342` | `0.542714` |
| `s1_a1_topk_mean_r0.15` | `1.0` | `1.0` | `topk_mean` | `0.15` | `0.503726` | `0.928318` | `0.540707` | `0.547368` |
| `s1_a2_topk_mean_r0.10` | `1.0` | `2.0` | `topk_mean` | `0.10` | `0.503012` | `0.927842` | `0.512081` | `0.522088` |

Score-sweep result:

- best selected variant: `s2_a1_topk_mean_r0.15`
- student weight: `2.0`
- autoencoder weight: `1.0`
- reduction: `topk_mean`
- top-k ratio: `0.15`
- validation threshold: `4.263504`
- precision: `0.421951`
- recall: `0.692000`
- F1: `0.524242`
- AUROC: `0.923114`
- AUPRC: `0.546305`
- best test-sweep F1: `0.560928`

Interpretation:

- the multilayer teacher produced the best `WideResNet50-2` result in the report so far
- compared with the single-layer `WideResNet50-2` run, the multilayer version improved F1, AUROC, and AUPRC at the selected operating point
- the gain was not huge, but it was consistent enough to support the idea that combining two teacher feature scales helps on this dataset
- this made the `WideResNet50-2` branch a serious challenger to the best `ResNet50` and AE-family runs rather than just a backbone curiosity

### Experiment 16: PatchCore with WideResNet50-2 `layer2 + layer3`

Purpose:

- test whether the same multilayer `WideResNet50-2` feature pair becomes even stronger when used in a memory-bank PatchCore setup
- keep the same shared `64x64` split and validation-threshold rule so the comparison against notebooks `16` and `17` stays fair
- run the experiment as a controlled sweep rather than a single configuration, so reduction choice, memory-bank size, and the critical top-k ratio band are all evaluated explicitly

Implementation:

- config: [train_patchcore_wideresnet50_multilayer.toml](configs/training/train_patchcore_wideresnet50_multilayer.toml)
- notebook: [notebook.ipynb](experiments/anomaly_detection/patchcore/wideresnet50/x64/main/notebook.ipynb)
- artifact dir: [patchcore_wideresnet50_multilayer](experiments/anomaly_detection/patchcore/wideresnet50/x64/main/artifacts/patchcore_wideresnet50_multilayer)
- execution note: the canonical local notebook now writes into a cleaned artifact layout with root sweep plots and per-variant `checkpoints`, `results`, and `plots`
- teacher backbone: frozen ImageNet-pretrained `WideResNet50-2`
- teacher layers: `layer2` and `layer3`
- reduction sweep: `mean`, `topk_mean`, and `max`
- memory bank sweep: `20k` and `50k`, with follow-up top-k ratio ablations around the best `50k` setting

Sweep summary:

- full sweep artifacts: [patchcore_combined_sweep_results.csv](experiments/anomaly_detection/patchcore/wideresnet50/x64/main/artifacts/patchcore_wideresnet50_multilayer/patchcore_combined_sweep_results.csv)
- cleaned review plots: [sweep_metrics.png](experiments/anomaly_detection/patchcore/wideresnet50/x64/main/artifacts/patchcore_wideresnet50_multilayer/plots/sweep_metrics.png), [selected_variant_score_distribution_sweep.png](experiments/anomaly_detection/patchcore/wideresnet50/x64/main/artifacts/patchcore_wideresnet50_multilayer/plots/selected_variant_score_distribution_sweep.png), [selected_variant_confusion_matrix.png](experiments/anomaly_detection/patchcore/wideresnet50/x64/main/artifacts/patchcore_wideresnet50_multilayer/plots/selected_variant_confusion_matrix.png), [selected_variant_defect_breakdown.png](experiments/anomaly_detection/patchcore/wideresnet50/x64/main/artifacts/patchcore_wideresnet50_multilayer/plots/selected_variant_defect_breakdown.png)

| variant | reduction | memory bank | top-k ratio | val-threshold F1 | AUROC | AUPRC | best sweep F1 |
| ------- | --------- | ----------- | ----------- | ---------------- | ----- | ----- | ------------- |
| `topk_mb50k_r010` | `topk_mean` | `50000` | `0.10` | `0.531758` | `0.916943` | `0.561855` | `0.585774` |
| `topk_mb50k_r015` | `topk_mean` | `50000` | `0.15` | `0.525526` | `0.911559` | `0.548919` | `0.592902` |
| `topk_mb50k_r005` | `topk_mean` | `50000` | `0.05` | `0.525074` | `0.920518` | `0.563612` | `0.579281` |
| `topk_mb50k_r020` | `topk_mean` | `50000` | `0.20` | `0.514793` | `0.906730` | `0.533386` | `0.585657` |
| `topk_mb50k_r025` | `topk_mean` | `50000` | `0.25` | `0.506550` | `0.902348` | `0.518213` | `0.575488` |
| `mean_mb50k` | `mean` | `50000` | `0.10` | `0.484305` | `0.873711` | `0.413971` | `0.510714` |
| `mean_mb20k` | `mean` | `20000` | `0.10` | `0.466667` | `0.875149` | `0.389518` | `0.471002` |
| `max_mb50k` | `max` | `50000` | `0.10` | `0.397554` | `0.876209` | `0.397211` | `0.451327` |

Selected result:

- best selected variant: `topk_mb50k_r010`
- reduction: `topk_mean`
- memory bank: `50000`
- top-k ratio: `0.10`
- validation threshold: `0.529660`
- precision: `0.421546`
- recall: `0.720000`
- F1: `0.531758`
- AUROC: `0.916943`
- AUPRC: `0.561855`
- best test-sweep F1: `0.585774`

Interpretation:

- this completed the `WideResNet50-2` story cleanly: backbone scale alone was weak, teacher-student made the wider teacher competitive, and multilayer PatchCore finally turned the wider backbone into the best deployed result in the report
- compared with the best multilayer teacher-student `WideResNet50-2` run, multilayer PatchCore improved deployed recall and F1 while staying competitive on AUROC and AUPRC
- the winning WRN PatchCore setting also stayed very consistent with the notebook sweep pattern: `topk_mean` won clearly, `50k` memory bank stayed worthwhile, and the best ratio region remained narrow around `0.10`

## Experiment Family: FastFlow `64x64`

This family starts a new flow-based anomaly branch built on frozen pretrained spatial features rather than reconstruction or memory-bank nearest-neighbor scoring.

The initial goal was:

- test whether a FastFlow-style density model can compete with the stronger non-AE anomaly branches without relying on a PatchCore memory bank
- keep the same shared `64x64` split and validation-threshold rule so comparison against the current leaderboard stays fair
- run the first study as a local all-in-one notebook with a small set of layer-choice and flow-depth ablations in one sitting

### Experiment 19: FastFlow with WideResNet50-2 flow ablations

Purpose:

- test whether multilayer `WideResNet50-2` features remain strong when paired with flow-based density estimation instead of PatchCore local nearest-neighbor scoring
- preserve the same shared `64x64` 5% test-defect setup and the same validation-normal threshold rule
- compare a small, cost-aware set of FastFlow ablations in one local sweep:
  - multilayer `layer2 + layer3`, `6` flow steps
  - `layer2` only, `6` flow steps
  - multilayer `layer2 + layer3`, `4` flow steps

Implementation:

- notebook: [notebook.ipynb](experiments/anomaly_detection/fastflow/x64/main/notebook.ipynb)
- artifact dir: [fastflow_variant_sweep](experiments/anomaly_detection/fastflow/x64/main/artifacts/fastflow_variant_sweep)
- execution note: the canonical repo notebook now runs locally and writes a cleaned artifact layout with branch-level `results` and `plots` plus per-variant `checkpoints` and `results`
- backbone: frozen ImageNet-pretrained `WideResNet50-2`
- input adaptation: single-channel wafer maps resized internally to `224x224`
- model head: FastFlow-style affine coupling flow on each selected feature map
- training stabilizers used in the canonical notebook:
  - reduced learning rate
  - gradient clipping
  - disabled AMP
  - more conservative flow initialization and scale clamp
- compared wafer-level reductions:
  - `mean`
  - `topk_mean`, top-k ratio `0.05`
  - `topk_mean`, top-k ratio `0.10`
  - `topk_mean`, top-k ratio `0.15`
  - `max`

Ablation summary:

| training variant | feature layers | flow steps | best score variant | val-threshold precision | val-threshold recall | val-threshold F1 | AUROC | AUPRC |
| ---------------- | -------------- | ---------- | ------------------ | ----------------------- | -------------------- | ---------------- | ----- | ----- |
| `wrn50_l23_s4` | `layer2` + `layer3` | `4` | `mean` | `0.385167` | `0.644000` | `0.482036` | `0.870692` | `0.488619` |
| `wrn50_l23_s6` | `layer2` + `layer3` | `6` | `mean` | `0.374408` | `0.632000` | `0.470238` | `0.869890` | `0.479070` |
| `wrn50_l2_s6` | `layer2` | `6` | `topk_mean`, ratio `0.15` | `0.364583` | `0.560000` | `0.441640` | `0.884224` | `0.459659` |

Evaluation for the best training variant `wrn50_l23_s4`:

| score variant | reduction | top-k ratio | val-threshold F1 | AUROC | AUPRC | balanced accuracy |
| ------------- | --------- | ----------- | ---------------- | ----- | ----- | ----------------- |
| `mean` | `mean` | `0.10` | `0.482036` | `0.870692` | `0.488619` | `0.796300` |

Selected result:

- selected training variant: `wrn50_l23_s4`
- selected wafer-level reduction: `mean`
- validation threshold: `0.412847`
- precision: `0.385167`
- recall: `0.664000`
- F1: `0.482036`
- AUROC: `0.870692`
- AUPRC: `0.488619`
- balanced accuracy: `0.796300`
- true positive: `161`
- false negative: `89`
- false positive: `257`
- true negative: `4743`

Defect-type recall for the selected `mean` score:

| defect type | count | detected | recall |
| ----------- | ----- | -------- | ------ |
| `Scratch` | `15` | `2` | `0.133333` |
| `Edge-Loc` | `53` | `28` | `0.528302` |
| `Loc` | `34` | `20` | `0.588235` |
| `Donut` | `7` | `5` | `0.714286` |
| `Center` | `50` | `36` | `0.720000` |
| `Edge-Ring` | `84` | `62` | `0.738095` |
| `Random` | `5` | `5` | `1.000000` |
| `Near-full` | `2` | `2` | `1.000000` |

Interpretation:

- the refreshed `19A` FastFlow artifacts still favor the multilayer `WideResNet50-2` model with `4` flow steps and plain `mean` reduction, so the broad ablation conclusion is unchanged
- however, the cleaned rerun is weaker than the earlier saved snapshot: deployed F1 falls to `0.482036`, so this branch now sits closer to the stronger autoencoder-family results than to the leading PatchCore and teacher-student branches
- within this family, multilayer features clearly mattered more than switching to `layer2` only; the `wrn50_l2_s6` ablation produced the weakest deployed F1 of the three training variants
- unlike the best `WideResNet50-2` PatchCore result, the best FastFlow operating point still prefers plain `mean` reduction rather than `topk_mean`
- the failure pattern matches the rest of the report closely: `Scratch` remains the weakest defect class by far, while `Loc` and `Edge-Loc` are still only moderate
- broader or structurally clearer defects such as `Edge-Ring`, `Center`, `Random`, and `Near-full` remain much easier than the smaller local defect families
- this suggests that simply replacing PatchCore with flow-based density estimation on the same backbone is promising but still not enough by itself; the key remaining challenge is still the small local-defect regime

![FastFlow variant comparison](experiments/anomaly_detection/fastflow/x64/main/artifacts/fastflow_variant_sweep/plots/variant_comparison_metrics.png)

![FastFlow best-variant defect breakdown](experiments/anomaly_detection/fastflow/x64/main/artifacts/fastflow_variant_sweep/plots/best_variant_defect_breakdown.png)

## Experiment 20: Post-Hoc Score Ensemble of WRN PatchCore + TS-Res50

Purpose:

- test whether the strongest PatchCore and teacher-student branches make complementary enough errors that a cheap score-level ensemble can beat the best standalone WRN PatchCore run
- keep the experiment post-hoc and low-cost by reusing saved artifacts rather than introducing a new training pipeline
- preserve the same validation-normal threshold rule so the comparison stays fair

Implementation:

- notebook: [20_score_ensemble_analysis.ipynb](notebooks/anomaly_50k/20_score_ensemble_analysis.ipynb)
- artifact dir: [artifacts/x64/ensemble_patchcore_ts_res50](artifacts/x64/ensemble_patchcore_ts_res50)
- base models:
  - `PatchCore-WideRes50-topk-mb50k-r010`
  - `TS-Res50-s2_a1_topk_mean_r0.20`
- procedure:
  - load saved PatchCore `val` and `test` wafer scores
  - recompute the selected `TS-Res50` score from the saved checkpoint
  - normalize both score streams using validation-normal statistics
  - compare weighted means and `max` fusion

Ensemble sweep summary:

| fusion variant | val-threshold precision | val-threshold recall | val-threshold F1 | AUROC | AUPRC | best sweep F1 |
| -------------- | ----------------------- | -------------------- | ---------------- | ----- | ----- | ------------- |
| `patchcore_only` | `0.421546` | `0.720000` | `0.531758` | `0.916943` | `0.561855` | `0.585774` |
| `max_pair` | `0.422434` | `0.708000` | `0.529148` | `0.916330` | `0.611277` | `0.619718` |
| `mean_70_30_patchcore` | `0.418224` | `0.716000` | `0.528024` | `0.917946` | `0.595837` | `0.616052` |
| `mean_50_50` | `0.416279` | `0.716000` | `0.526471` | `0.916686` | `0.606252` | `0.615385` |
| `ts_only` | `0.418052` | `0.704000` | `0.524590` | `0.909189` | `0.599169` | `0.606299` |
| `mean_30_70_patchcore` | `0.408676` | `0.716000` | `0.520349` | `0.914454` | `0.607724` | `0.621677` |

Interpretation:

- among the true fusion strategies, normalized-score `max_pair` was the strongest ensemble result
- `max_pair` slightly improved precision and improved ranking quality substantially, pushing `AUPRC` to `0.611277`, which is higher than the standalone WRN PatchCore result
- however, its deployed `F1 = 0.529148` was still below the standalone WRN PatchCore `F1 = 0.531758`
- the best overall row in the notebook was still `patchcore_only`, which means the ensemble study did not change the final deployment-style winner
- this suggests that `TS-Res50` does add some complementary ranking information, but not enough thresholded decision improvement to justify replacing the standalone WRN PatchCore benchmark

## Experiment 21A: PatchCore with EfficientNet-B0 All-in-One `x64`

Purpose:

- test whether an all-in-one PatchCore notebook built around pretrained `EfficientNet-B0` can stay competitive while still following the report's shared split and validation-threshold policy
- keep the experiment report-compatible rather than CT-faithful, so the result can be compared directly with the other completed `64x64` runs
- establish the initial all-in-one EfficientNet-B0 baseline before testing whether direct `224x224` preprocessing materially changes the outcome

Implementation:

- notebook: [21A_patchcore_efficientnet_b0_all-in-one.ipynb](notebooks/anomaly_50k/21A_patchcore_efficientnet_b0_all-in-one.ipynb)
- artifact dir: [artifacts/x64/patchcore_efficientnet_b0](artifacts/x64/patchcore_efficientnet_b0)
- execution note: the current canonical notebook is local and writes into the cleaned experiment artifact folder
- split and evaluation:
  - same shared `40,000 / 5,000 / 5,000 + 250` report split from raw `LSWMD.pkl`
  - threshold selected as the `95th` percentile of validation-normal raw scores
- key model settings:
  - wafer preprocessing cached at `64x64`, then resized inside the model path to `224x224`
  - `EfficientNet-B0`, `model_input_size = 224`
  - multilayer patch embedding from feature blocks `3` and `6`
  - memory bank cap `240,000`
  - wafer score = mean of top `2%` patch anomaly scores

Selected result:

| variant | val-threshold precision | val-threshold recall | val-threshold F1 | AUROC | AUPRC | best sweep F1 |
| ------- | ----------------------- | -------------------- | ---------------- | ----- | ----- | ------------- |
| `PatchCore-EffNetB0-topk-mb240k-r002` | `0.381313` | `0.604000` | `0.467492` | `0.905171` | `0.489141` | `0.504132` |

Per-defect recall:

| defect type | recall |
| ----------- | ------ |
| `Scratch` | `0.533333` |
| `Loc` | `0.558824` |
| `Edge-Loc` | `0.490566` |
| `Center` | `0.400000` |
| `Edge-Ring` | `0.773810` |
| `Donut` | `0.857143` |
| `Random` | `1.000000` |
| `Near-full` | `1.000000` |

Interpretation:

- this run is a respectable mid-tier PatchCore result under the report's stricter deployment-style protocol, with strong ranking quality for a non-WRN backbone and `AUROC > 0.90`
- it did not reproduce the much stronger CT-branch numbers because this notebook uses the report split and validation-normal threshold rule rather than CT's labeled anomaly tuning split
- defect behavior is mixed but informative: `Scratch` and `Loc` are reasonably strong, `Edge-Ring` and broad defects remain easy, but `Center` recall is clearly weaker than in the strongest WRN and teacher-student runs
- the cost is high for the gain: the saved run used a `240k` memory bank and `28 x 28 = 784` patches per image, so it is much heavier than the older `64x64` ResNet-family PatchCore sweeps without becoming the new leader at this stage
- in hindsight, the main limitation of this first run was the `64x64 -> 224x224` preprocessing path rather than the backbone choice itself

## Experiment 21B: PatchCore with EfficientNet-B0 All-in-One `x224`

Purpose:

- test whether keeping the same report-compatible split and threshold policy, but moving wafer preprocessing directly to `224x224`, improves the EfficientNet-B0 PatchCore branch materially
- isolate the impact of direct high-resolution preprocessing on a pretrained local-anomaly backbone
- determine whether the earlier EfficientNet-B0 result was limited more by the backbone or by the `64x64` cache path

Implementation:

- notebook: [notebook.ipynb](experiments/anomaly_detection/patchcore/efficientnet_b0/x224/main/notebook.ipynb)
- artifact dir: [artifacts/](experiments/anomaly_detection/patchcore/efficientnet_b0/x224/main/artifacts)
- execution note: the canonical notebook now runs locally, skips retraining by default, and regenerates plots from the saved checkpoint-backed artifacts
- split and evaluation:
  - same shared `40,000 / 5,000 / 5,000 + 250` report split from raw `LSWMD.pkl`
  - threshold selected as the `95th` percentile of validation-normal raw scores
- key model settings:
  - wafer preprocessing cached directly at `224x224`
  - `EfficientNet-B0`, `model_input_size = 224`
  - multilayer patch embedding from feature blocks `3` and `6`
  - memory bank cap `240,000`
  - wafer score = mean of top `2%` patch anomaly scores

Selected result:

| variant | val-threshold precision | val-threshold recall | val-threshold F1 | AUROC | AUPRC | best sweep F1 |
| ------- | ----------------------- | -------------------- | ---------------- | ----- | ----- | ------------- |
| `PatchCore-EffNetB0-x224-topk-mb240k-r002` | `0.438725` | `0.716000` | `0.544073` | `0.924586` | `0.483186` | `0.566667` |

Per-defect recall:

| defect type | recall |
| ----------- | ------ |
| `Center` | `0.540000` |
| `Loc` | `0.558824` |
| `Scratch` | `0.600000` |
| `Edge-Loc` | `0.641509` |
| `Edge-Ring` | `0.904762` |
| `Donut` | `1.000000` |
| `Random` | `1.000000` |
| `Near-full` | `1.000000` |

Evaluation artifacts:

- run summary: [summary.json](experiments/anomaly_detection/patchcore/efficientnet_b0/x224/main/artifacts/results/summary.json)
- evaluation summary: [summary.json](experiments/anomaly_detection/patchcore/efficientnet_b0/x224/main/artifacts/results/evaluation/summary.json)
- evaluation scores: [val_scores.csv](experiments/anomaly_detection/patchcore/efficientnet_b0/x224/main/artifacts/results/evaluation/val_scores.csv), [test_scores.csv](experiments/anomaly_detection/patchcore/efficientnet_b0/x224/main/artifacts/results/evaluation/test_scores.csv)
- threshold sweep: [threshold_sweep.csv](experiments/anomaly_detection/patchcore/efficientnet_b0/x224/main/artifacts/results/evaluation/threshold_sweep.csv)
- defect breakdown: [defect_breakdown.csv](experiments/anomaly_detection/patchcore/efficientnet_b0/x224/main/artifacts/results/evaluation/defect_breakdown.csv)
- plots: [score_distribution.png](experiments/anomaly_detection/patchcore/efficientnet_b0/x224/main/artifacts/plots/score_distribution.png), [threshold_sweep.png](experiments/anomaly_detection/patchcore/efficientnet_b0/x224/main/artifacts/plots/threshold_sweep.png), [defect_breakdown.png](experiments/anomaly_detection/patchcore/efficientnet_b0/x224/main/artifacts/plots/defect_breakdown.png)

Interpretation:

- this follow-up changed the EfficientNet-B0 story materially: direct `224x224` preprocessing lifted deployed F1 from `0.467492` to `0.544073` and AUROC from `0.905171` to `0.924586`
- the gain came without changing the report split or threshold policy, which strongly suggests the earlier bottleneck was the `64x64 -> 224x224` preprocessing path rather than threshold selection
- this direct-`224x224` run became a strong upper-tier PatchCore result and temporarily set the best deployed operating point in the report, before the later WRN `x224` follow-up pushed the branch higher still
- defect behavior improved in the local regime as well: `Edge-Loc` and `Scratch` both moved up, and `Edge-Ring` became very strong, while `Center` remains more moderate
- this makes direct high-resolution preprocessing the default recommendation for future pretrained-backbone PatchCore follow-ups

![EfficientNet-B0 x224 score distribution](experiments/anomaly_detection/patchcore/efficientnet_b0/x224/main/artifacts/plots/score_distribution.png)

![EfficientNet-B0 x224 threshold sweep](experiments/anomaly_detection/patchcore/efficientnet_b0/x224/main/artifacts/plots/threshold_sweep.png)

![EfficientNet-B0 x224 defect breakdown](experiments/anomaly_detection/patchcore/efficientnet_b0/x224/main/artifacts/plots/defect_breakdown.png)

## Experiment 22A: PatchCore with EfficientNet-B1 One-Layer `x240` on Main Benchmark

Purpose:

- test whether moving to a slightly larger EfficientNet backbone and matching its native `x240` scale can outperform the earlier CNN PatchCore baselines while staying fair to the same main benchmark split
- keep the same deployment-style threshold rule used throughout the report
- check whether the direct high-resolution preprocessing trend seen in `21B` continues when the backbone capacity is increased

Implementation:

- notebook: [notebook.ipynb](experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main/notebook.ipynb)
- artifact dir: [patchcore_efficientnet_b1_one_layer](experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main/artifacts/patchcore_efficientnet_b1_one_layer)
- split and evaluation:
  - train normals: `40,000`
  - validation normals: `5,000`
  - test normals: `5,000`
  - test defects: `250`
  - threshold selected as the `95th` percentile of validation-normal raw scores
- key model settings:
  - direct wafer preprocessing at `240x240`
  - `EfficientNet-B1`, `model_input_size = 240`
  - one-layer feature extraction from feature block `3`
  - patch embedding dimension `512`
  - memory bank cap `240,000`
  - wafer score = mean of top `3%` patch anomaly scores

Selected result:

| variant | val-threshold precision | val-threshold recall | val-threshold F1 | AUROC | AUPRC | best sweep F1 |
| ------- | ----------------------- | -------------------- | ---------------- | ----- | ----- | ------------- |
| `PatchCore-EffNetB1-x240-topk-mb240k-r003` | `0.475610` | `0.780000` | `0.590909` | `0.935374` | `0.608633` | `0.650699` |

Threshold details:

- threshold policy: `val_normal_quantile_raw`
- threshold quantile: `0.95`
- deployed raw-score threshold: `0.508699`
- best test-sweep threshold: `0.535425`

Per-defect recall:

| defect type | recall |
| ----------- | ------ |
| `Scratch` | `0.400000` |
| `Loc` | `0.588235` |
| `Center` | `0.700000` |
| `Edge-Loc` | `0.792453` |
| `Edge-Ring` | `0.928571` |
| `Donut` | `1.000000` |
| `Random` | `1.000000` |
| `Near-full` | `1.000000` |

Evaluation artifacts:

- run summary: [summary.json](experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main/artifacts/patchcore_efficientnet_b1_one_layer/results/summary.json)
- benchmark scores: [val_scores.csv](experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main/artifacts/patchcore_efficientnet_b1_one_layer/results/evaluation/val_scores.csv), [test_scores.csv](experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main/artifacts/patchcore_efficientnet_b1_one_layer/results/evaluation/test_scores.csv)
- benchmark threshold sweep: [threshold_sweep.csv](experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main/artifacts/patchcore_efficientnet_b1_one_layer/results/evaluation/threshold_sweep.csv)
- benchmark defect breakdown: [defect_breakdown.csv](experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main/artifacts/patchcore_efficientnet_b1_one_layer/results/evaluation/defect_breakdown.csv)
- benchmark plots: [benchmark_distribution_sweep.png](experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main/artifacts/patchcore_efficientnet_b1_one_layer/plots/benchmark_distribution_sweep.png), [benchmark_distribution_sweep_confusion.png](experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main/artifacts/patchcore_efficientnet_b1_one_layer/plots/benchmark_distribution_sweep_confusion.png), [benchmark_defect_breakdown.png](experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main/artifacts/patchcore_efficientnet_b1_one_layer/plots/benchmark_defect_breakdown.png)

Interpretation:

- this run materially outperformed the earlier EfficientNet-B0 baselines and also overtook the direct-`224x224` WRN PatchCore benchmark on deployed F1
- among the completed CNN PatchCore runs currently in the repo, this is the strongest main-benchmark operating point by deployed F1
- the improvement came without defect-aware threshold tuning, so the gain appears to come from the backbone-resolution combination rather than from a looser evaluation rule
- defect behavior still follows the broader project pattern: `Edge-Ring` and broad anomalies remain easy, while `Scratch` and `Loc` continue to be harder than the larger defect families

![EfficientNet-B1 x240 benchmark distribution, sweep, and confusion](experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main/artifacts/patchcore_efficientnet_b1_one_layer/plots/benchmark_distribution_sweep_confusion.png)

![EfficientNet-B1 x240 benchmark defect breakdown](experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main/artifacts/patchcore_efficientnet_b1_one_layer/plots/benchmark_defect_breakdown.png)

![EfficientNet-B1 x240 joint UMAP by split](experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main/artifacts/patchcore_efficientnet_b1_one_layer/plots/umap_joint_by_split.png)

## Experiment 22B: PatchCore with EfficientNet-B1 One-Layer `x240` on Expanded Holdout

Purpose:

- test whether the same EfficientNet-B1 `x240` checkpoint remains strong on the larger `70k / 3.5k` evaluation pool
- keep the same validation-derived threshold rule so the holdout result remains directly comparable to the main benchmark result
- use the saved local artifacts rather than a separate branch-specific export

Implementation:

- notebook: [notebook.ipynb](experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main/notebook.ipynb)
- artifact dir: [holdout70k_3p5k](experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main/artifacts/patchcore_efficientnet_b1_one_layer/results/holdout70k_3p5k)
- split and evaluation:
  - train normals: `40,000`
  - validation normals: `5,000`
  - test normals: `70,000`
  - test defects: `3,500`
  - threshold kept at the same validation-normal `95th` percentile raw-score rule from the benchmark run

Selected result:

| variant | evaluation mode | val-threshold precision | val-threshold recall | val-threshold F1 | AUROC | AUPRC | best sweep F1 |
| ------- | --------------- | ----------------------- | -------------------- | ---------------- | ----- | ----- | ------------- |
| `effnet_b1_one_layer_patchcore_x240` | `holdout70k_3p5k` | `0.465778` | `0.828286` | `0.596257` | `0.953229` | `0.655528` | `0.670836` |

Threshold details:

- threshold policy: `val_normal_quantile_raw`
- threshold quantile: `0.95`
- deployed raw-score threshold: `0.508699`
- best test-sweep threshold: `0.532987`

Per-defect recall:

| defect type | recall |
| ----------- | ------ |
| `Scratch` | `0.518987` |
| `Loc` | `0.612705` |
| `Center` | `0.738574` |
| `Edge-Loc` | `0.804979` |
| `Edge-Ring` | `0.961877` |
| `Random` | `1.000000` |
| `Donut` | `1.000000` |
| `Near-full` | `1.000000` |

Evaluation artifacts:

- holdout summary: [summary.json](experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main/artifacts/patchcore_efficientnet_b1_one_layer/results/holdout70k_3p5k/summary.json)
- holdout scores: [val_scores.csv](experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main/artifacts/patchcore_efficientnet_b1_one_layer/results/holdout70k_3p5k/evaluation/val_scores.csv), [test_scores.csv](experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main/artifacts/patchcore_efficientnet_b1_one_layer/results/holdout70k_3p5k/evaluation/test_scores.csv)
- holdout threshold sweep: [threshold_sweep.csv](experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main/artifacts/patchcore_efficientnet_b1_one_layer/results/holdout70k_3p5k/evaluation/threshold_sweep.csv)
- holdout defect breakdown: [defect_breakdown.csv](experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main/artifacts/patchcore_efficientnet_b1_one_layer/results/holdout70k_3p5k/evaluation/defect_breakdown.csv)
- holdout plots: [holdout_distribution_sweep.png](experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main/artifacts/patchcore_efficientnet_b1_one_layer/results/holdout70k_3p5k/plots/holdout_distribution_sweep.png), [holdout_distribution_sweep_confusion.png](experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main/artifacts/patchcore_efficientnet_b1_one_layer/results/holdout70k_3p5k/plots/holdout_distribution_sweep_confusion.png), [holdout_defect_breakdown.png](experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main/artifacts/patchcore_efficientnet_b1_one_layer/results/holdout70k_3p5k/plots/holdout_defect_breakdown.png)

Interpretation:

- the EfficientNet-B1 `x240` checkpoint remains strong on the larger holdout, and its deployed F1 on the holdout (`0.596257`) is slightly higher than its main-benchmark deployed F1
- ranking quality improves further on the larger evaluation pool, with `AUROC = 0.953229` and `AUPRC = 0.655528`
- the same defect pattern remains visible: `Edge-Ring` and broad defects are very strong, while the smaller local anomalies still account for most of the misses
- this makes the branch a strong CNN comparison point against the ViT holdout result, even though the ViT run still remains the main headline benchmark on the fair `5%` main split

![EfficientNet-B1 x240 holdout distribution, sweep, and confusion](experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main/artifacts/patchcore_efficientnet_b1_one_layer/results/holdout70k_3p5k/plots/holdout_distribution_sweep_confusion.png)

![EfficientNet-B1 x240 holdout defect breakdown](experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main/artifacts/patchcore_efficientnet_b1_one_layer/results/holdout70k_3p5k/plots/holdout_defect_breakdown.png)

## Experiment 23A: PatchCore with ViT-B/16 One-Layer `x224` on Main Benchmark

Purpose:

- test whether a ViT-based PatchCore branch can beat the strongest CNN PatchCore runs while staying fully fair to the report's main `40,000 / 5,000 / 5,000 + 250` benchmark split
- preserve the same deployment-style threshold rule used throughout the report
- check whether the ViT branch remains strong even though its auxiliary UMAP diagnostic still uses a ViT-specific `cosine` geometry

Implementation:

- notebook: [notebook.ipynb](experiments/anomaly_detection/patchcore/vit_b16/x224/main/notebook.ipynb)
- artifact dir: [main_5pct](experiments/anomaly_detection/patchcore/vit_b16/x224/main/artifacts/patchcore_vit_b16_5pct/main_5pct)
- split and evaluation:
  - train normals: `40,000`
  - validation normals: `5,000`
  - test normals: `5,000`
  - test defects: `250`
  - threshold selected as the `95th` percentile of validation-normal z-scored wafer scores
- key model settings:
  - backbone: `vit_base_patch16_224.augreg_in21k_ft_in1k`
  - feature source: transformer block `6`
  - projected patch embedding dimension: `128`
  - memory bank cap: `400,000` patches
  - PatchCore nearest-neighbor scoring: `k = 3`
  - wafer score = mean of top `10%` patch scores

Selected result:

| variant | evaluation mode | val-threshold precision | val-threshold recall | val-threshold F1 | AUROC | AUPRC | best sweep F1 |
| ------- | --------------- | ----------------------- | -------------------- | ---------------- | ----- | ----- | ------------- |
| `vit_b16_one_layer_patchcore_x224` | `main_5pct` | `0.463252` | `0.832000` | `0.595136` | `0.956301` | `0.670907` | `0.650206` |

Threshold details:

- threshold policy: `tune_normal_quantile_zscore`
- threshold quantile: `0.95`
- deployed threshold in z-space: `1.693279`
- deployed raw-score threshold: `0.518193`
- best test-sweep F1 was reached at percentile `98.7`, with threshold z `2.528852`

Per-defect recall:

| defect type | count | detected | recall | mean z-score |
| ----------- | ----- | -------- | ------ | ------------ |
| `Center` | `34` | `21` | `0.618` | `2.294` |
| `Edge-Loc` | `44` | `31` | `0.705` | `2.723` |
| `Scratch` | `11` | `8` | `0.727` | `3.016` |
| `Loc` | `41` | `34` | `0.829` | `3.000` |
| `Edge-Ring` | `102` | `96` | `0.941` | `3.438` |
| `Donut` | `2` | `2` | `1.000` | `5.755` |
| `Near-full` | `3` | `3` | `1.000` | `5.556` |
| `Random` | `13` | `13` | `1.000` | `4.927` |

UMAP diagnostic on the main benchmark:

- split-plot artifact: [umap_test_embeddings.png](experiments/anomaly_detection/patchcore/vit_b16/x224/main/artifacts/patchcore_vit_b16_5pct/main_5pct/plots/umap_test_embeddings.png)
- score-plot artifact: [umap_by_score.png](experiments/anomaly_detection/patchcore/vit_b16/x224/main/artifacts/patchcore_vit_b16_5pct/main_5pct/plots/umap_by_score.png)
- UMAP summary: [umap_summary.json](experiments/anomaly_detection/patchcore/vit_b16/x224/main/artifacts/patchcore_vit_b16_5pct/main_5pct/results/umap/umap_summary.json)
- the main wafer score stays strong, but UMAP-space KNN remains weak:
  - threshold `0.256939`
  - precision `0.075410`
  - recall `0.092000`
  - F1 `0.082883`
  - AUROC `0.524505`
  - AUPRC `0.074117`
- interpretation: the ViT branch is strong as a PatchCore score, not as a UMAP-KNN thresholding method

Interpretation:

- this is now the strongest fair main-benchmark result in the repo by deployed F1, AUROC, AUPRC, and best-sweep F1
- the gain is not just threshold luck: per-defect recall is broad, with especially strong `Loc`, `Scratch`, and `Edge-Ring` behavior
- the result strengthens the overall report conclusion that pretrained local-anomaly methods benefit materially from direct `224x224` preprocessing
- the UMAP diagnostic still behaves the same way as in the holdout run: useful for geometric visualization, but not a better deployed threshold than the original ViT PatchCore wafer score

![ViT-B16 x224 main test evaluation](experiments/anomaly_detection/patchcore/vit_b16/x224/main/artifacts/patchcore_vit_b16_5pct/main_5pct/plots/test_evaluation.png)

![ViT-B16 x224 main threshold sweep](experiments/anomaly_detection/patchcore/vit_b16/x224/main/artifacts/patchcore_vit_b16_5pct/main_5pct/plots/threshold_sweep.png)

## Experiment 23B: PatchCore with ViT-B/16 One-Layer `x224` on Expanded Holdout

Purpose:

- test whether a ViT-based PatchCore variant can stay report-compatible while using the same direct-`224x224` preprocessing regime as the stronger `18A2`, `21B`, and `22B` follow-ups
- preserve the same deployment-style threshold policy and the same expanded-holdout evaluation pool used in the later UMAP diagnostics, so the comparison against `10A` and other holdout-based analyses remains meaningful
- check whether a geometry-aware UMAP diagnostic or UMAP-space KNN threshold adds anything beyond the main ViT wafer score

Implementation:

- notebook: [notebook.ipynb](experiments/anomaly_detection/patchcore/vit_b16/x224/main/notebook.ipynb)
- artifact dir: [holdout70k_3p5k](experiments/anomaly_detection/patchcore/vit_b16/x224/main/artifacts/patchcore_vit_b16_5pct/holdout70k_3p5k)
- execution mode: expanded holdout evaluation
- split and evaluation:
  - train normals: `40,000`
  - validation normals: `5,000`
  - test normals: `70,000`
  - test defects: `3,500`
  - threshold selected as the `95th` percentile of validation-normal z-scored wafer scores
- key model settings:
  - backbone: `vit_base_patch16_224.augreg_in21k_ft_in1k`
  - feature source: transformer block `6`
  - projected patch embedding dimension: `128`
  - memory bank cap: `400,000` patches
  - PatchCore nearest-neighbor scoring: `k = 3`
  - wafer score: mean of top `10%` patch scores

Selected result:

| variant | evaluation mode | val-threshold precision | val-threshold recall | val-threshold F1 | AUROC | AUPRC | best sweep F1 |
| ------- | --------------- | ----------------------- | -------------------- | ---------------- | ----- | ----- | ------------- |
| `vit_b16_one_layer_patchcore_x224_holdout70k` | `holdout70k_3p5k` | `0.427521` | `0.764286` | `0.548324` | `0.941495` | `0.614664` | `0.606493` |

Threshold details:

- threshold policy: `tune_normal_quantile_zscore`
- threshold quantile: `0.95`
- deployed threshold in z-space: `1.699944`
- deployed raw-score threshold: `0.593475`
- best test-sweep F1 was reached at percentile `98.3`, with threshold z `2.381549`
- confusion matrix at the deployed threshold: `[[66418, 3582], [825, 2675]]`

Per-defect recall:

| defect type | count | detected | recall | mean z-score |
| ----------- | ----- | -------- | ------ | ------------ |
| `Edge-Loc` | `683` | `385` | `0.564` | `2.287` |
| `Loc` | `508` | `337` | `0.663` | `2.602` |
| `Scratch` | `165` | `114` | `0.691` | `3.584` |
| `Center` | `593` | `415` | `0.700` | `2.618` |
| `Edge-Ring` | `1336` | `1209` | `0.905` | `2.764` |
| `Donut` | `75` | `75` | `1.000` | `4.936` |
| `Near-full` | `16` | `16` | `1.000` | `5.015` |
| `Random` | `124` | `124` | `1.000` | `4.658` |

Evaluation artifacts:

- test-evaluation figure: [test_evaluation.png](experiments/anomaly_detection/patchcore/vit_b16/x224/main/artifacts/patchcore_vit_b16_5pct/holdout70k_3p5k/plots/test_evaluation.png)
- threshold-sweep plot: [threshold_sweep.png](experiments/anomaly_detection/patchcore/vit_b16/x224/main/artifacts/patchcore_vit_b16_5pct/holdout70k_3p5k/plots/threshold_sweep.png)
- saved metrics: [evaluation_metrics.json](experiments/anomaly_detection/patchcore/vit_b16/x224/main/artifacts/patchcore_vit_b16_5pct/holdout70k_3p5k/results/evaluation/evaluation_metrics.json)
- full run summary: [summary.json](experiments/anomaly_detection/patchcore/vit_b16/x224/main/artifacts/patchcore_vit_b16_5pct/holdout70k_3p5k/results/summary.json)

UMAP diagnostic:

- split-plot artifact: [umap_test_embeddings.png](experiments/anomaly_detection/patchcore/vit_b16/x224/main/artifacts/patchcore_vit_b16_5pct/holdout70k_3p5k/plots/umap_test_embeddings.png)
- score-plot artifact: [umap_by_score.png](experiments/anomaly_detection/patchcore/vit_b16/x224/main/artifacts/patchcore_vit_b16_5pct/holdout70k_3p5k/plots/umap_by_score.png)
- UMAP summary: [umap_summary.json](experiments/anomaly_detection/patchcore/vit_b16/x224/main/artifacts/patchcore_vit_b16_5pct/holdout70k_3p5k/results/umap/umap_summary.json)
- UMAP threshold sweep: [umap_knn_threshold_sweep.csv](experiments/anomaly_detection/patchcore/vit_b16/x224/main/artifacts/patchcore_vit_b16_5pct/holdout70k_3p5k/results/umap/umap_knn_threshold_sweep.csv)
- UMAP generation protocol:
  - reference-fit on sampled train normals only
  - validation-normal points used for UMAP-KNN threshold calibration only
  - full scoring counts: `5,000` train reference, `5,000` val normal, `70,000` test normal, `3,500` test anomaly
  - exported plot counts: `5,000` train reference, `5,000` val normal, `8,000` test normal, `3,500` test anomaly
  - ViT-specific choice: UMAP metric = `cosine`
  - note: the CNN PatchCore report runs use `euclidean`, so this is the one intentional geometry difference kept for the ViT branch
- UMAP-KNN threshold result:
  - threshold quantile: `0.95`
  - threshold: `0.262382`
  - precision: `0.064224`
  - recall: `0.081143`
  - F1: `0.071699`
  - AUROC: `0.490688`
  - AUPRC: `0.054165`
  - predicted anomalies: `4,422`
  - confusion matrix: `[[65862, 4138], [3216, 284]]`

Interpretation:

- the main ViT PatchCore wafer score is strong on the expanded holdout: deployed `F1 = 0.548324` and `AUROC = 0.941495` place it immediately alongside the strongest report-era PatchCore results
- the defect profile is also healthy for a local method: `Edge-Ring` is very strong, `Center`, `Loc`, and `Scratch` are clearly above chance, and the rare broad classes are all perfectly detected in this holdout bundle
- the threshold-sweep headroom remains real but moderate, with best-sweep `F1 = 0.606493`, so threshold selection is still leaving performance on the table even for a strong backbone
- the UMAP diagnostic is useful as a geometric explanation of the embedding manifold, but not as a better thresholding rule here: UMAP-space KNN collapses to near-random ranking quality and should be treated as a diagnostic only, not as the deployed decision score
- this makes the ViT branch a meaningful comparison point against the direct-`224x224` CNN PatchCore experiments, but the fair headline is that the ViT wafer score is strong while the auxiliary UMAP-KNN score is not

![ViT-B16 x224 holdout test evaluation](experiments/anomaly_detection/patchcore/vit_b16/x224/main/artifacts/patchcore_vit_b16_5pct/holdout70k_3p5k/plots/test_evaluation.png)

![ViT-B16 x224 holdout threshold sweep](experiments/anomaly_detection/patchcore/vit_b16/x224/main/artifacts/patchcore_vit_b16_5pct/holdout70k_3p5k/plots/threshold_sweep.png)

## Overall Interpretation

Across all completed experiments:

- the best current result on the main deployment-style F1 metric is still the direct-`224x224` ViT-B/16 PatchCore follow-up, which reached `F1 = 0.595136`, but the local EfficientNet-B1 `x240` one-layer run is now very close behind at `F1 = 0.590909`
- the original `64x64` autoencoder checkpoint improved substantially just by changing the scoring rule
- retraining that autoencoder longer did not materially change the outcome, so epoch count alone is unlikely to be the key lever
- adding BatchNorm changed the best score choice for the autoencoder from `topk_abs_mean` to `max_abs`
- the dropout sweep did not help; the best run selected `dropout = 0.00`, so latent dropout is not a promising next AE lever in this setup
- simply increasing autoencoder resolution did not help
- the VAE underperformed the autoencoder even after beta tuning
- Deep SVDD was a stronger alternative than the tuned VAE in some thresholded metrics, but not enough to replace the autoencoder baseline
- the residual autoencoder was a stronger architecture than the plain baseline in several metrics, but it still did not overtake the BatchNorm AE + `max_abs` result
- PatchCore with the frozen BatchNorm AE encoder did produce a usable anomaly signal, but it still fell short of the best AE operating point
- the early PatchCore branch worked best with `mean` reduction and a `50k` memory bank, but the later multilayer `WideResNet50-2` PatchCore follow-up changed that picture materially by making `topk_mean` the strongest reduction family
- the frozen pretrained ResNet18 backbone baseline with center-distance scoring was weak, so simply switching backbones without a stronger local scoring rule is not enough
- ResNet18 + PatchCore validated the non-AE backbone direction; it beat the earlier AE-backed PatchCore sweep and substantially improved over plain ResNet18 center-distance
- ResNet50 + PatchCore pushed that non-AE branch a bit further, but the bigger breakthrough came later from multilayer `WideResNet50-2` PatchCore
- the EfficientNet branch ended up splitting into three clear stories: the first EfficientNet-B0 `x64` cache-path run was only mid-pack, the direct-`224x224` EfficientNet-B0 follow-up validated the high-resolution preprocessing direction, and the later local EfficientNet-B1 `x240` one-layer run pushed the CNN PatchCore line even higher
- the ResNet50 gain was still useful because it showed the backbone direction was working, even before the stronger `WideResNet50-2` follow-up widened the margin
- the teacher-student distillation branch improved materially once its scoring rule was fixed, and the new `TS-Res50` variation became the strongest teacher-student result on deployed F1
- inside the `TS-Res50` family, both `layer1` and `layer2` teacher-feature variants were run; `layer1` stayed competitive, but the best verified `layer2` score remained stronger overall
- the plain `WideResNet50-2` embedding baseline repeated the same lesson as the earlier plain ResNet baseline: backbone scale alone is not enough when the score is still global center-distance
- once `WideResNet50-2` was used as a teacher inside the teacher-student framework, performance jumped sharply and became competitive with the strongest experiments in the project
- the best `WideResNet50-2` multilayer teacher-student run slightly trailed the best `TS-Res50` result on deployed F1, but it improved the wider-teacher branch to `AUROC > 0.92` and became a strong stepping stone to the later WRN PatchCore win
- within the `WideResNet50-2` branch, the multilayer `layer2 + layer3` teacher gave a small but consistent gain over the single-layer `layer2` version, which suggests that combining two spatial scales is useful here
- the direct-`224x224` WRN PatchCore follow-up remains a very strong CNN reference, but the local EfficientNet-B1 `x240` one-layer run is now the strongest completed CNN PatchCore operating point on deployed F1, while the newer direct-`224x224` ViT-B/16 PatchCore run still gives the strongest overall main-benchmark operating point and also the best AUROC, AUPRC, and best-sweep F1
- the first post-hoc score ensemble of WRN PatchCore + `TS-Res50` was close but not enough; the best true fusion improved AUPRC but still fell slightly short of the standalone WRN PatchCore result on deployed F1
- the best BatchNorm autoencoder remains a strong comparison point because it is still simpler, still very competitive on thresholded F1, and now trails the new WRN `x224` leader by only a modest margin
- all tested models show overlap between normal and anomaly score distributions, which explains the moderate F1 values and missed anomalies
- the score-ablation result shows that part of the bottleneck was the scoring rule, not only the model architecture
- after fixing the score, the remaining bottleneck still looks more like limited class separation than threshold selection alone
- the AE failure analysis shows that the remaining weakness is concentrated in smaller local defects rather than large global defect patterns
- this makes the next decision clearer: keep the direct-`224x224` ViT-B/16 PatchCore run as the deployment-style benchmark, keep the local EfficientNet-B1 `x240` one-layer run as the strongest CNN PatchCore reference, keep `TS-Res50` as the strongest teacher-student reference, and keep the BatchNorm autoencoder as the strongest reconstruction baseline

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
- pretrained ResNet18 embedding-distance baseline
- PatchCore sweep with a pretrained ResNet18 backbone
- PatchCore sweep with a pretrained ResNet50 backbone
- PatchCore sweep with a multilayer pretrained WideResNet50-2 backbone
- teacher-student distillation baseline with a pretrained ResNet18 teacher
- teacher-student distillation variation with an imported pretrained ResNet50 teacher run and local score-sweep analysis
- teacher-student score sweep, selected-score confirmation, and layer-sweep support
- notebook-based end-to-end training for AE, VAE, and SVDD
- notebook-based PatchCore sweep
- local all-in-one notebook for multilayer WideResNet50-2 PatchCore
- local all-in-one notebooks for EfficientNet-B0 PatchCore at `x64` and direct `224x224`
- local all-in-one notebook for EfficientNet-B1 PatchCore at direct `240x240`, with benchmark, holdout, and UMAP exports
- notebook-based ViT-B/16 PatchCore at direct `224x224`, with both main-benchmark and expanded-holdout evaluation modes
- notebook-based teacher-student distillation training
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
- keep the direct-`224x224` ViT-B/16 PatchCore run as the new deployment-style benchmark to beat
- keep `TS-Res50` as the main teacher-student benchmark and score-quality reference
- keep the current AE + BatchNorm + `max_abs` result as the strongest reconstruction baseline and a useful AUPRC sanity check
- if more non-AE work is justified, keep it on pretrained ResNet backbones with PatchCore-style local scoring rather than returning to plain global embedding-distance baselines
- do not spend more time on larger backbone changes alone unless they are paired with PatchCore or another local-anomaly scoring method
- keep the residual autoencoder as a logged comparison result, but stop using AE encoders as the main PatchCore improvement path
- keep the completed multilayer WideResNet50-2 PatchCore branch as the strongest `64x64` local-anomaly challenger and the best comparison point for the older report regime
- keep the direct-`224x224` ViT-B/16 PatchCore follow-up as the new report-compatible leader, and use the local EfficientNet-B1 `x240` one-layer run as the strongest CNN reference
- if more teacher-student distillation work is justified, tune it from the new `TS-Res50` selected-score baseline: teacher layer choice, student capacity, branch weighting, and wafer-level reduction are higher-priority than longer training alone
- if more PatchCore work is justified, use the ViT-B/16 `x224` result as the main target to beat and tune the multilayer WideResNet50-2 branch first around the `topk_mean` ratio band near `0.05` to `0.10` as the strongest CNN follow-up
- keep the validation-derived threshold as the main reported result, and treat test-set threshold sweeps as analysis only

