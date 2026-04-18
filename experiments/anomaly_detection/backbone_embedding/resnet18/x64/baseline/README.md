# ResNet18 Backbone Baseline

This folder contains the cleaned frozen `ResNet18` embedding baseline on the shared `x64` benchmark split.

Use [notebook.ipynb](/c:/Users/User/Desktop/Term%208/Deep%20Learning/Project/DeepLearning-Group8/experiments/anomaly_detection/backbone_embedding/resnet18/x64/baseline/notebook.ipynb) as the main entry point. By default it reuses saved evaluation artifacts and only recomputes embeddings when explicitly requested.

Local files:
- `train_config.toml`: experiment-local config
- `data_config.toml`: copied dataset reference config
- `artifacts/resnet18_embedding_baseline/`: cached scores, summaries, plots, and optional embedding exports
