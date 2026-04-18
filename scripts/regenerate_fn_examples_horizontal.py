#!/usr/bin/env python
"""
Regenerate failure_examples_fn.png with horizontal layout instead of vertical.
Horizontal layout = 6 columns (3 per sample: input, recon, error_map), 2 rows (top=normal, bottom=anomaly)
"""

import sys
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Setup paths
cwd = Path.cwd().resolve()
candidate_roots = [cwd, *cwd.parents]
REPO_ROOT = None
for candidate in candidate_roots:
    if (candidate / "src" / "wafer_defect").exists() and (candidate / "configs").exists():
        REPO_ROOT = candidate
        break

if REPO_ROOT is None:
    raise RuntimeError("Could not locate repo root containing src/wafer_defect and configs/")

SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wafer_defect.config import load_toml
from wafer_defect.data.wm811k import WaferMapDataset
from wafer_defect.models.autoencoder import ConvAutoencoder
from wafer_defect.scoring import absolute_error_map

# Configuration
CONFIG_PATH = REPO_ROOT / "experiments/anomaly_detection/autoencoder/x64/baseline/train_config.toml"
ARTIFACT_DIR = REPO_ROOT / "experiments/anomaly_detection/autoencoder/x64/baseline/artifacts/autoencoder_baseline"
BEST_MODEL_PATH = ARTIFACT_DIR / "checkpoints" / "best_model.pt"
OUTPUT_DIR = ARTIFACT_DIR / "plots"

config = load_toml(CONFIG_PATH)
image_size = int(config["data"].get("image_size", 64))
metadata_path = REPO_ROOT / config["data"]["metadata_csv"]

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load datasets
test_dataset = WaferMapDataset(metadata_path, split="test", image_size=image_size)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

# Load model
model = ConvAutoencoder(
    latent_dim=int(config["model"]["latent_dim"]),
    image_size=image_size,
    use_batchnorm=bool(config["model"].get("use_batchnorm", False)),
    dropout_prob=float(config["model"].get("dropout_prob", 0.0)),
).to(device)

checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Load test scores
failure_analysis_path = ARTIFACT_DIR / "results" / "failure_analysis.csv"
analysis_df = pd.read_csv(failure_analysis_path)

# Get false negative examples
fn_examples = analysis_df[analysis_df["error_type"] == "fn"].sort_values("score", ascending=True).head(3).index.tolist()
print(f"False negative examples (lowest scores): {fn_examples}")

# Create horizontal layout: 3 columns per sample, 2 rows (for 3 samples = 6 cols total, 1 row)
# Actually, let's do it more clearly: 3 rows (3 samples), 3 columns (input, recon, error_map)
# But horizontal means we want to show samples side-by-side

# Better approach: 1 row per sample type, 3 columns per sample
# FN examples: 3 samples arranged horizontally
# Each sample takes 3 subplots: input, reconstruction, error map
n_samples = 3
fig, axes = plt.subplots(n_samples, 3, figsize=(3.5 * 3, 2.5 * n_samples))

with torch.no_grad():
    for row_idx, sample_idx in enumerate(fn_examples):
        row = analysis_df.iloc[sample_idx]
        input_tensor, label = test_dataset[sample_idx]
        output_tensor = model(input_tensor.unsqueeze(0).to(device)).squeeze(0).cpu()
        error_map = absolute_error_map(input_tensor.unsqueeze(0), output_tensor.unsqueeze(0)).squeeze(0).squeeze(0).cpu()

        # Column 0: Input
        axes[row_idx, 0].imshow(input_tensor.squeeze(0), cmap="viridis")
        axes[row_idx, 0].set_title(f"Input\ndefect={row.get('defect_type', '?')} | score={row['score']:.3f}", fontsize=10)
        axes[row_idx, 0].axis("off")

        # Column 1: Reconstruction
        axes[row_idx, 1].imshow(output_tensor.squeeze(0), cmap="viridis")
        axes[row_idx, 1].set_title(f"Reconstruction\npred={row['predicted_anomaly']} (FN)", fontsize=10)
        axes[row_idx, 1].axis("off")

        # Column 2: Error map
        axes[row_idx, 2].imshow(error_map, cmap="magma")
        axes[row_idx, 2].set_title(f"Absolute Error\nsample #{sample_idx}", fontsize=10)
        axes[row_idx, 2].axis("off")

fig.suptitle("False Negatives: Small Defects Missed by Global Scoring\n(High error signal in error map, but undetected by threshold)", fontsize=12, fontweight='bold', y=0.98)
plt.tight_layout()

# Save figure
output_path = OUTPUT_DIR / "failure_examples_fn_horizontal.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, bbox_inches="tight", dpi=150)
print(f"Saved horizontal failure examples to {output_path}")
plt.close()

print("Done!")
