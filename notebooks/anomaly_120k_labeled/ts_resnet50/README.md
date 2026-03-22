# TS-ResNet50 120k Labeled Notebooks

This folder contains the teacher-student ResNet50 workflow for the larger labeled anomaly-detection split.

Current contents:

- `1_ts_resnet50_training_120k.ipynb`
  Training-first notebook seeded from the shared teacher-student workflow and pointed at the `120k / 10k / 20k` metadata family by default.
- `1A_ts_resnet50_training_120k_modal.ipynb`
  Modal-friendly companion notebook that looks for an uploaded raw wafer pickle, prepares the `120k` split automatically if needed, and writes outputs to `/output/ts_resnet50_120k_modal`.

Default config:

- `configs/training/train_ts_resnet50_120k.toml`
- `configs/training/train_ts_resnet50_120k_modal.toml`

Default dataset:

- `data/processed/x64/wm811k_patchcore_custom/metadata_train120000_a6000_val10000_a500_test20000_a1000.csv`

Default artifact directory:

- `artifacts/x64/ts_resnet50_120k`
- `/output/ts_resnet50_120k_modal`

Modal usage:

- upload `LSWMD.pkl`
- open `1A_ts_resnet50_training_120k_modal.ipynb`
- run the notebook top to bottom
- the notebook will copy the pickle into the repo data path if needed, prepare the dataset, train, evaluate, and save outputs under `/output/ts_resnet50_120k_modal`
