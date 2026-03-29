# x128 Benchmark 50k 5pct

This branch rebuilds and validates the curated `128 x 128` WM-811K benchmark split used by the `x128` autoencoder baseline.

Files:
- `notebook.ipynb`: main dataset build and validation notebook
- `data_config.toml`: local config for generating the processed `x128` arrays and metadata

Outputs written under:
- `data/processed/x128/wm811k/metadata_50k_5pct.csv`
- `data/processed/x128/wm811k/arrays/`

Use this notebook when you want to confirm that the `x128` processed dataset can be recreated from the raw pickle before running the `x128` model notebooks.
