from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
BRANCH_ROOT = REPO_ROOT / "experiments/anomaly_detection/patchcore/wideresnet50/x64/main"
ARTIFACT_ROOT = BRANCH_ROOT / "artifacts/patchcore_wideresnet50_multilayer"
README_PATH = BRANCH_ROOT / "README.md"
NOTEBOOK_PATH = BRANCH_ROOT / "notebook.ipynb"

VARIANT_NAMES = [
    "mean_mb20k",
    "mean_mb50k",
    "topk_mb50k_r005",
    "topk_mb50k_r010",
    "topk_mb50k_r015",
    "topk_mb50k_r020",
    "topk_mb50k_r025",
    "max_mb50k",
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def metadata_path() -> Path:
    processed = ARTIFACT_ROOT / "processed/metadata_50k_5pct.csv"
    if processed.exists():
        return processed
    return REPO_ROOT / "data/processed/x64/wm811k/metadata_50k_5pct.csv"


def load_test_metadata() -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path())
    return metadata[metadata["split"] == "test"].reset_index(drop=True)


def clean_duplicate_variant_dirs() -> None:
    for name in VARIANT_NAMES:
        direct = ARTIFACT_ROOT / name
        nested = direct / name
        if not nested.exists():
            continue
        same = True
        for filename in ["summary.json", "val_scores.csv", "test_scores.csv", "threshold_sweep.csv"]:
            a = direct / filename
            b = nested / filename
            same = same and a.exists() and b.exists() and a.read_bytes() == b.read_bytes()
        if same:
            shutil.rmtree(nested)


def summarize_confusion(scores_df: pd.DataFrame, threshold: float) -> list[list[int]]:
    predicted = (scores_df["score"].astype(float) >= float(threshold)).astype(int)
    truth = scores_df["is_anomaly"].astype(int)
    tn = int(((truth == 0) & (predicted == 0)).sum())
    fp = int(((truth == 0) & (predicted == 1)).sum())
    fn = int(((truth == 1) & (predicted == 0)).sum())
    tp = int(((truth == 1) & (predicted == 1)).sum())
    return [[tn, fp], [fn, tp]]


def write_confusion_csv(path: Path, cm: list[list[int]]) -> None:
    pd.DataFrame(cm, index=["true_normal", "true_anomaly"], columns=["pred_normal", "pred_anomaly"]).to_csv(path)


def build_defect_breakdown(test_metadata: pd.DataFrame, test_scores_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    analysis_df = test_metadata.copy()
    analysis_df["score"] = test_scores_df["score"].reset_index(drop=True)
    analysis_df["predicted_anomaly"] = (analysis_df["score"] >= float(threshold)).astype(int)
    defect_breakdown_df = (
        analysis_df.loc[analysis_df["is_anomaly"] == 1]
        .groupby("defect_type")
        .agg(
            count=("defect_type", "size"),
            detected=("predicted_anomaly", "sum"),
            mean_score=("score", "mean"),
            median_score=("score", "median"),
        )
        .reset_index()
    )
    defect_breakdown_df["detected"] = defect_breakdown_df["detected"].astype(int)
    defect_breakdown_df["missed"] = defect_breakdown_df["count"] - defect_breakdown_df["detected"]
    defect_breakdown_df["recall"] = defect_breakdown_df["detected"] / defect_breakdown_df["count"]
    return defect_breakdown_df.sort_values(["recall", "count", "defect_type"], ascending=[True, False, True]).reset_index(drop=True)


def plot_variant_bundle(
    *,
    variant_name: str,
    summary: dict,
    val_scores_df: pd.DataFrame,
    test_scores_df: pd.DataFrame,
    threshold_sweep_df: pd.DataFrame,
    defect_breakdown_df: pd.DataFrame,
    variant_dir: Path,
) -> None:
    plots_dir = ensure_dir(variant_dir / "plots")
    threshold = float(summary["threshold"])
    cm = summarize_confusion(test_scores_df, threshold)
    write_confusion_csv(variant_dir / "confusion_matrix.csv", cm)
    defect_breakdown_df.to_csv(variant_dir / "selected_defect_breakdown.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].hist(val_scores_df["score"], bins=40, alpha=0.85, color="#4d908e")
    axes[0].axvline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"threshold={threshold:.4f}")
    axes[0].set_title(f"{variant_name}: Validation Scores")
    axes[0].set_xlabel("Score")
    axes[0].legend()

    axes[1].hist(test_scores_df[test_scores_df["is_anomaly"] == 0]["score"], bins=40, alpha=0.72, label="normal", color="#90be6d")
    axes[1].hist(test_scores_df[test_scores_df["is_anomaly"] == 1]["score"], bins=40, alpha=0.72, label="anomaly", color="#f8961e")
    axes[1].axvline(threshold, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_title(f"{variant_name}: Test Score Distribution")
    axes[1].set_xlabel("Score")
    axes[1].legend()

    axes[2].plot(threshold_sweep_df["threshold"], threshold_sweep_df["precision"], label="precision", linewidth=2)
    axes[2].plot(threshold_sweep_df["threshold"], threshold_sweep_df["recall"], label="recall", linewidth=2)
    axes[2].plot(threshold_sweep_df["threshold"], threshold_sweep_df["f1"], label="f1", linewidth=2)
    axes[2].axvline(threshold, color="red", linestyle="--", linewidth=1.5)
    axes[2].set_title(f"{variant_name}: Threshold Sweep")
    axes[2].set_xlabel("Threshold")
    axes[2].grid(alpha=0.25, linestyle="--")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "score_distribution_sweep.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{variant_name}: Confusion Matrix")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred Normal", "Pred Anomaly"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True Normal", "True Anomaly"])
    for row_index in range(2):
        for col_index in range(2):
            ax.text(col_index, row_index, f"{cm[row_index][col_index]:,}", ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(plots_dir / "confusion_matrix.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    if not defect_breakdown_df.empty:
        plot_df = defect_breakdown_df.sort_values("recall", ascending=True)
        fig, ax = plt.subplots(figsize=(8.5, max(4.0, 0.42 * len(plot_df))))
        ax.barh(plot_df["defect_type"], plot_df["recall"], color="#577590")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Recall")
        ax.set_title(f"{variant_name}: Defect Recall by Type")
        ax.grid(axis="x", alpha=0.25, linestyle="--")
        fig.tight_layout()
        fig.savefig(plots_dir / "defect_breakdown.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def plot_root_bundle(combined_results_df: pd.DataFrame, best_variant_name: str, best_variant_dir: Path) -> None:
    plots_dir = ensure_dir(ARTIFACT_ROOT / "plots")
    plot_df = combined_results_df.copy()
    colors = ["#0a9396" if name == best_variant_name else "#94d2bd" for name in plot_df["name"]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].bar(plot_df["name"], plot_df["f1"], color=colors)
    axes[0].set_title("WRN50 x64 PatchCore Sweep: Validation-Threshold F1")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", alpha=0.25, linestyle="--")

    axes[1].bar(plot_df["name"], plot_df["auroc"], color=colors)
    axes[1].set_title("WRN50 x64 PatchCore Sweep: AUROC")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(plots_dir / "sweep_metrics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    for source_name, dest_name in [
        ("score_distribution_sweep.png", "selected_variant_score_distribution_sweep.png"),
        ("confusion_matrix.png", "selected_variant_confusion_matrix.png"),
        ("defect_breakdown.png", "selected_variant_defect_breakdown.png"),
    ]:
        source = best_variant_dir / "plots" / source_name
        if source.exists():
            shutil.copy2(source, plots_dir / dest_name)


def clean_and_plot() -> None:
    clean_duplicate_variant_dirs()
    test_metadata = load_test_metadata()

    sweep_df = pd.read_csv(ARTIFACT_ROOT / "patchcore_sweep_results.csv")
    follow_up_df = pd.read_csv(ARTIFACT_ROOT / "patchcore_follow_up_sweep_results.csv")
    combined_results_df = (
        pd.concat([sweep_df, follow_up_df], ignore_index=True)
        .sort_values(["f1", "auroc"], ascending=False)
        .reset_index(drop=True)
    )
    combined_results_df.to_csv(ARTIFACT_ROOT / "patchcore_combined_sweep_results.csv", index=False)

    for name in VARIANT_NAMES:
        variant_dir = ARTIFACT_ROOT / name
        if not variant_dir.exists():
            continue
        summary = json.loads((variant_dir / "summary.json").read_text(encoding="utf-8"))
        val_scores_df = pd.read_csv(variant_dir / "val_scores.csv")
        test_scores_df = pd.read_csv(variant_dir / "test_scores.csv")
        threshold_sweep_df = pd.read_csv(variant_dir / "threshold_sweep.csv")
        defect_breakdown_df = build_defect_breakdown(test_metadata, test_scores_df, float(summary["threshold"]))
        plot_variant_bundle(
            variant_name=name,
            summary=summary,
            val_scores_df=val_scores_df,
            test_scores_df=test_scores_df,
            threshold_sweep_df=threshold_sweep_df,
            defect_breakdown_df=defect_breakdown_df,
            variant_dir=variant_dir,
        )

    best_variant_name = str(json.loads((ARTIFACT_ROOT / "patchcore_sweep_summary.json").read_text(encoding="utf-8"))["best_variant"]["name"])
    plot_root_bundle(combined_results_df, best_variant_name, ARTIFACT_ROOT / best_variant_name)


def patch_readme() -> None:
    README_PATH.write_text(
        """# WRN50-2 PatchCore (`x64`, local all-in-one run)

This branch keeps the original `64x64` WideResNet50-2 PatchCore benchmark as a self-contained local notebook.

The notebook rebuilds the benchmark split from the raw `LSWMD.pkl`, trains the multilayer PatchCore model with `layer2 + layer3`, runs the baseline and follow-up scoring sweeps, and writes the saved local artifacts into the branch `artifacts/` folder.

## Files

- `notebook.ipynb`
  Canonical local training and evaluation workflow for the `x64` WRN50 PatchCore run.
- `train_config.toml`
  Snapshot of the branch configuration.
- `data_config.toml`
  Snapshot of the dataset settings used by the run.
- `artifacts/patchcore_wideresnet50_multilayer/`
  Local output root for the WRN50 x64 sweep.
  It contains per-variant score CSVs and summaries, combined sweep tables, the processed benchmark metadata snapshot, and generated plots.

## Saved Outputs

- `patchcore_sweep_results.csv`
  Baseline sweep results for the main variants.
- `patchcore_follow_up_sweep_results.csv`
  Follow-up sweep results around the best configuration.
- `patchcore_combined_sweep_results.csv`
  Combined benchmark table across the baseline and follow-up variants.
- `plots/`
  Sweep-level comparison figures plus selected-variant review plots.
- `<variant>/plots/`
  Per-variant score-distribution, confusion-matrix, and defect-breakdown plots generated from the saved local score files.

## Relationship To The `x224` Branches

The higher-resolution `x224` branches remain the stronger WRN PatchCore follow-ups. This `x64` branch is the direct local benchmark counterpart that matches the main `50k / 5%` experiment setting.
""",
        encoding="utf-8",
    )


def main() -> None:
    clean_and_plot()
    patch_readme()


if __name__ == "__main__":
    main()
