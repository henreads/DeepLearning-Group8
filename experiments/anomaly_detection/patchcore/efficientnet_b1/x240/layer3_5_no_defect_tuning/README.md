# PatchCore EfficientNet-B1 Layer 3/5 No-Defect-Tuning (`x240`)

This notebook is the canonical training and evaluation workflow for the EfficientNet-B1 PatchCore experiment without defect-aware threshold tuning.

Files:
- `notebook.ipynb`: the canonical training and evaluation notebook for this experiment
- `artifacts/checkpoints/`: saved model checkpoints written by the notebook
- `artifacts/plots/`: figures saved by the notebook
- `artifacts/results/`: metrics, score files, exported CSVs, and supporting outputs

Notes:
- The notebook uses repo-local dataset paths and writes outputs back into the local artifact folders.
- Extracted outputs from the earlier saved run are preserved under `[experiments/anomaly_detection/patchcore/efficientnet_b1/x240/layer3_5_no_defect_tuning/artifacts/results/extracted_notebook_outputs](experiments/anomaly_detection/patchcore/efficientnet_b1/x240/layer3_5_no_defect_tuning/artifacts/results/extracted_notebook_outputs)`.
