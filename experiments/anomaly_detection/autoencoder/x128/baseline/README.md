# X128 Autoencoder Baseline

This folder contains the 128x128 autoencoder baseline experiment.

Use [notebook.ipynb](/c:/Users/User/Desktop/Term%208/Deep%20Learning/Project/DeepLearning-Group8/experiments/anomaly_detection/autoencoder/x128/baseline/notebook.ipynb) as the main entry point. By default it loads the saved checkpoint from `artifacts/autoencoder_baseline/` and only retrains if `FORCE_RETRAIN = True`.

Local files:
- `train_config.toml`: experiment-local training config
- `artifacts/autoencoder_baseline/`: saved checkpoints, metrics, plots, and analysis outputs

Dataset input:
- metadata CSV: `data/processed/x128/wm811k/metadata_50k_5pct.csv`

Notes:
- This branch mirrors the cleaned `x64` baseline notebook structure, but uses `image_size = 128`.
- The first full notebook run may generate score-ablation outputs if they are not already present in the artifact folder.
