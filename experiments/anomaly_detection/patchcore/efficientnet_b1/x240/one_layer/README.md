# PatchCore EfficientNet-B1 One-Layer (`x240`)

This notebook is the canonical training and evaluation workflow for the one-layer EfficientNet-B1 PatchCore experiment.

Files:
- `notebook.ipynb`: the canonical training and evaluation notebook for this experiment
- `artifacts/checkpoints/`: saved model checkpoints written by the notebook
- `artifacts/plots/`: figures saved by the notebook
- `artifacts/results/`: metrics, score files, exported CSVs, and supporting outputs

Notes:
- The notebook uses repo-local dataset paths and writes outputs back into the local artifact folders.
- Extracted outputs from the earlier saved run are preserved under `[experiments/anomaly_detection/patchcore/efficientnet_b1/x240/one_layer/artifacts/results/extracted_notebook_outputs](experiments/anomaly_detection/patchcore/efficientnet_b1/x240/one_layer/artifacts/results/extracted_notebook_outputs)`.
