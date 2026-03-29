# Wide ResNet50-2 Backbone Baseline

This folder contains the cleaned frozen `Wide ResNet50-2` embedding baseline on the shared `x64` benchmark split.

Use [notebook.ipynb](/c:/Users/User/Desktop/Term%208/Deep%20Learning/Project/DeepLearning-Group8/experiments/anomaly_detection/backbone_embedding/wide_resnet50_2/x64/baseline/notebook.ipynb) as the main entry point. By default it reuses saved benchmark evaluation artifacts and only recomputes embeddings when explicitly requested.

Local files:
- `train_config.toml`: experiment-local config
- `data_config.toml`: copied dataset reference config
- `artifacts/wide_resnet50_2_embedding_baseline/`: canonical benchmark artifacts

Legacy note:
- older holdout-oriented Wide ResNet50-2 outputs are still preserved under sibling legacy artifact folders for provenance, but they are not the default submission path
