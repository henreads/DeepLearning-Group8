# PatchCore ViT-B/16 One-Layer Defect-Tuning (`x224`)

This notebook is the canonical training and evaluation workflow for the one-layer ViT-B/16 PatchCore experiment with defect-aware threshold tuning.

Files:
- `notebook.ipynb`: the canonical training and evaluation notebook for this experiment
- `artifacts/checkpoints/`: saved model checkpoints written by the notebook
- `artifacts/plots/`: figures saved by the notebook
- `artifacts/results/`: metrics, score files, exported CSVs, and supporting outputs

Notes:
- The notebook uses repo-local dataset paths and writes outputs back into the local artifact folders.
- The notebook supports two modes: reuse the saved local checkpoint and result files by default, or recompute the score artifacts and UMAP outputs by setting the force flags in the configuration cell.
