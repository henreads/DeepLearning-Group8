from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def move_if_present(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    ensure_dir(dst.parent)
    if dst.exists():
        if src.is_file():
            src.unlink()
        return
    shutil.move(str(src), str(dst))


def branch_artifact_root(branch_dir: Path) -> Path:
    art = branch_dir / "artifacts"
    return art if art.exists() else branch_dir


def normalize_branch(branch_dir: Path) -> Path:
    branch_dir = branch_dir.resolve()

    if not branch_dir.exists():
        raise FileNotFoundError(branch_dir)

    if (branch_dir / "artifacts").exists():
        artifact_root = branch_dir / "artifacts"
    else:
        artifact_root = branch_dir / "artifacts"
        ensure_dir(artifact_root)

        for notebook_name in sorted(branch_dir.glob("*.ipynb")):
            notebook_target = branch_dir / "notebook.ipynb"
            if notebook_name.name != "notebook.ipynb" and not notebook_target.exists():
                shutil.move(str(notebook_name), str(notebook_target))

        for file_name in [
            "evaluation_metrics.json",
            "scores.npz",
            "summary.json",
            "umap_test_embeddings.csv",
            "umap_test_embeddings.png",
            "patchcore_vit_b16_model.pt",
            "patchcore_efficientnet_b1_model.pt",
            "best_model.pt",
        ]:
            src = branch_dir / file_name
            if src.exists():
                dst = artifact_root / file_name
                shutil.move(str(src), str(dst))

    checkpoints_dir = artifact_root / "checkpoints"
    plots_dir = artifact_root / "plots"
    results_dir = artifact_root / "results"
    ensure_dir(checkpoints_dir)
    ensure_dir(plots_dir)
    ensure_dir(results_dir)

    move_if_present(artifact_root / "patchcore_vit_b16_model.pt", checkpoints_dir / "patchcore_vit_b16_model.pt")
    move_if_present(artifact_root / "patchcore_efficientnet_b1_model.pt", checkpoints_dir / "patchcore_efficientnet_b1_model.pt")
    move_if_present(artifact_root / "best_model.pt", checkpoints_dir / "best_model.pt")
    move_if_present(artifact_root / "evaluation_metrics.json", results_dir / "evaluation_metrics.json")
    move_if_present(artifact_root / "scores.npz", results_dir / "scores.npz")
    move_if_present(artifact_root / "summary.json", results_dir / "summary.json")
    move_if_present(artifact_root / "umap_test_embeddings.csv", results_dir / "umap_test_embeddings.csv")
    move_if_present(artifact_root / "umap_test_embeddings.png", plots_dir / "umap_test_embeddings.png")

    marker = checkpoints_dir / "MISSING_CHECKPOINT.txt"
    if marker.exists() and any(checkpoints_dir.glob("*.pt")):
        marker.unlink()

    return artifact_root


def generate_review_plots(branch_dir: Path) -> None:
    artifact_root = normalize_branch(branch_dir)
    results_dir = artifact_root / "results"
    plots_dir = artifact_root / "plots"
    scores_path = results_dir / "scores.npz"
    metrics_path = results_dir / "evaluation_metrics.json"

    if not scores_path.exists() or not metrics_path.exists():
        return

    scores = np.load(scores_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    threshold_z = float(metrics["threshold_z"])
    confusion_matrix = np.array(metrics["confusion_matrix"], dtype=int)

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

    if series:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        for label, values, color in series:
            ax.hist(values, bins=40, alpha=0.45, label=label, color=color, density=True)
        ax.axvline(threshold_z, color="black", linestyle="--", label=f"threshold z={threshold_z:.3f}")
        ax.set_title("Score Distribution by Split")
        ax.set_xlabel("z-scored wafer anomaly score")
        ax.set_ylabel("density")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "score_distribution.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        labels = [label for label, _, _ in series]
        values = [vals for _, vals, _ in series]
        ax.boxplot(values, labels=labels, showfliers=False)
        ax.axhline(threshold_z, color="black", linestyle="--", label="threshold")
        ax.set_title("Score Summary by Split")
        ax.set_ylabel("z-scored wafer anomaly score")
        ax.tick_params(axis="x", rotation=20)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "score_summary.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(confusion_matrix, cmap="Blues")
    for (row_idx, col_idx), value in np.ndenumerate(confusion_matrix):
        ax.text(col_idx, row_idx, f"{value}", ha="center", va="center", color="black")
    ax.set_xticks([0, 1], ["pred normal", "pred anomaly"])
    ax.set_yticks([0, 1], ["true normal", "true anomaly"])
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(plots_dir / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        raise SystemExit("Usage: normalize_imported_patchcore_branch_artifacts.py <branch_dir> [<branch_dir> ...]")

    for raw_path in argv[1:]:
        branch_dir = (REPO_ROOT / raw_path).resolve()
        generate_review_plots(branch_dir)
        print(f"Normalized and plotted: {branch_dir}")


if __name__ == "__main__":
    main(sys.argv)
