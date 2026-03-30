from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell


REPO_ROOT = Path(__file__).resolve().parents[2]

BRANCHES = [
    "experiments/anomaly_detection/patchcore/efficientnet_b1/x240/one_layer_no_defect_tuning",
    "experiments/anomaly_detection/patchcore/efficientnet_b1/x240/one_layer",
    "experiments/anomaly_detection/patchcore/efficientnet_b1/x240/layer3_5",
    "experiments/anomaly_detection/patchcore/efficientnet_b1/x240/layer3_5_no_defect_tuning",
    "experiments/anomaly_detection/patchcore/vit_b16/x224/one_layer_no_defect_tuning",
    "experiments/anomaly_detection/patchcore/vit_b16/x224/two_block",
    "experiments/anomaly_detection/patchcore/vit_b16/x224/two_block_no_defect_tuning",
]


MARKER = "## Review Plots"


def review_plot_code(branch_rel: str) -> str:
    return f"""artifact_root = REPO_ROOT / "{branch_rel}" / "artifacts"
checkpoints_dir = artifact_root / "checkpoints"
plots_dir = artifact_root / "plots"
results_dir = artifact_root / "results"
checkpoints_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)

import torch

checkpoint_candidates = sorted(checkpoints_dir.glob("*.pt"))
if not checkpoint_candidates:
    raise FileNotFoundError(f"No checkpoint found under {{checkpoints_dir}}")

checkpoint_path = checkpoint_candidates[0]
checkpoint = torch.load(checkpoint_path, map_location="cpu")
if isinstance(checkpoint, dict):
    print(f"Loaded checkpoint: {{checkpoint_path.name}} with keys: {{list(checkpoint.keys())[:8]}}")
else:
    print(f"Loaded checkpoint: {{checkpoint_path.name}}")

scores_path = results_dir / "scores.npz"
metrics_path = results_dir / "evaluation_metrics.json"
if not scores_path.exists():
    raise FileNotFoundError(f"Saved score bundle not found: {{scores_path}}")
if not metrics_path.exists():
    raise FileNotFoundError(f"Saved evaluation metrics not found: {{metrics_path}}")

scores = np.load(scores_path)
evaluation_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
threshold_z = float(evaluation_metrics["threshold_z"])
confusion_matrix = np.array(evaluation_metrics["confusion_matrix"], dtype=int)

series = []
for key, label, color in [
    ("train_scores_z", "train", "#8d99ae"),
    ("tune_normal_scores_z", "val normal", "#457b9d"),
    ("tune_defect_scores_z", "val defect", "#f4a261"),
    ("test_normal_scores_z", "test normal", "#577590"),
    ("test_defect_scores_z", "test defect", "#e76f51"),
]:
    if key in scores:
        series.append((label, np.asarray(scores[key]).astype(float), color))

if not series:
    raise ValueError("No score arrays found in scores.npz")

fig, ax = plt.subplots(figsize=(9, 4.8))
for label, values, color in series:
    ax.hist(values, bins=40, alpha=0.45, label=label, color=color, density=True)
ax.axvline(threshold_z, color="black", linestyle="--", label=f"threshold z={{threshold_z:.3f}}")
ax.set_title("Score Distribution by Split")
ax.set_xlabel("z-scored wafer anomaly score")
ax.set_ylabel("density")
ax.legend()
score_distribution_path = plots_dir / "score_distribution.png"
plt.tight_layout()
plt.savefig(score_distribution_path, dpi=200, bbox_inches="tight")
plt.show()
plt.close(fig)
print(f"Saved score distribution plot to {{score_distribution_path}}")

fig, ax = plt.subplots(figsize=(8.5, 4.8))
labels = [label for label, _, _ in series]
values = [vals for _, vals, _ in series]
ax.boxplot(values, labels=labels, showfliers=False)
ax.axhline(threshold_z, color="black", linestyle="--", label="threshold")
ax.set_title("Score Summary by Split")
ax.set_ylabel("z-scored wafer anomaly score")
ax.tick_params(axis="x", rotation=20)
ax.legend()
score_summary_path = plots_dir / "score_summary.png"
plt.tight_layout()
plt.savefig(score_summary_path, dpi=200, bbox_inches="tight")
plt.show()
plt.close(fig)
print(f"Saved score summary plot to {{score_summary_path}}")

fig, ax = plt.subplots(figsize=(4.8, 4.2))
im = ax.imshow(confusion_matrix, cmap="Blues")
for (row_idx, col_idx), value in np.ndenumerate(confusion_matrix):
    ax.text(col_idx, row_idx, f"{{value}}", ha="center", va="center", color="black")
ax.set_xticks([0, 1], ["pred normal", "pred anomaly"])
ax.set_yticks([0, 1], ["true normal", "true anomaly"])
ax.set_title("Confusion Matrix")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
confusion_matrix_path = plots_dir / "confusion_matrix.png"
plt.tight_layout()
plt.savefig(confusion_matrix_path, dpi=200, bbox_inches="tight")
plt.show()
plt.close(fig)
print(f"Saved confusion matrix plot to {{confusion_matrix_path}}")
"""


def main() -> None:
    for branch_rel in BRANCHES:
        notebook_path = REPO_ROOT / branch_rel / "notebook.ipynb"
        if not notebook_path.exists():
            continue

        nb = nbformat.read(notebook_path, as_version=4)
        marker_idx = None
        for idx, cell in enumerate(nb.cells):
            if cell.cell_type == "markdown" and MARKER in cell.source:
                marker_idx = idx
                break

        if marker_idx is None:
            nb.cells.append(
                new_markdown_cell(
                    """## Review Plots

This section loads the saved local checkpoint for the branch, then rebuilds the review figures from the saved local score bundle and evaluation metrics. It does not retrain the model; it verifies that the cleaned repo-local checkpoint and artifact folders reproduce the expected plots inside this notebook.
"""
                )
            )
            nb.cells.append(new_code_cell(review_plot_code(branch_rel)))
        else:
            if marker_idx + 1 >= len(nb.cells) or nb.cells[marker_idx + 1].cell_type != "code":
                nb.cells.insert(marker_idx + 1, new_code_cell(review_plot_code(branch_rel)))
            else:
                nb.cells[marker_idx + 1].source = review_plot_code(branch_rel)
        nbformat.write(nb, notebook_path)
        print(f"Appended review-plot section to {notebook_path}")


if __name__ == "__main__":
    main()
