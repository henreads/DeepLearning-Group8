from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
BRANCH_ROOT = REPO_ROOT / "experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main"
BUNDLE_ROOT = BRANCH_ROOT / "artifacts/patchcore_efficientnet_b1_one_layer"
NOTEBOOK_PATH = BRANCH_ROOT / "notebook.ipynb"
README_PATH = BRANCH_ROOT / "README.md"
X240_README_PATH = REPO_ROOT / "experiments/anomaly_detection/patchcore/efficientnet_b1/x240/README.md"
UMAP_FOLLOWUP_README_PATH = REPO_ROOT / "experiments/anomaly_detection/patchcore/efficientnet_b1/x240/umap_followup/README.md"
UMAP_FOLLOWUP_NOTEBOOK_PATH = REPO_ROOT / "experiments/anomaly_detection/patchcore/efficientnet_b1/x240/umap_followup/notebook.ipynb"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def move_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    ensure_dir(dst.parent)
    if dst.exists():
        return
    shutil.move(str(src), str(dst))


def layout() -> dict[str, Path]:
    results_dir = ensure_dir(BUNDLE_ROOT / "results")
    evaluation_dir = ensure_dir(results_dir / "evaluation")
    holdout_dir = ensure_dir(results_dir / "holdout70k_3p5k")
    holdout_eval_dir = ensure_dir(holdout_dir / "evaluation")
    holdout_plots_dir = ensure_dir(holdout_dir / "plots")
    umap_dir = ensure_dir(results_dir / "umap")
    umap_reference_dir = ensure_dir(umap_dir / "reference_fit")
    umap_reference_plots_dir = ensure_dir(umap_reference_dir / "plots")
    umap_joint_dir = ensure_dir(umap_dir / "joint_fit")
    umap_joint_plots_dir = ensure_dir(umap_joint_dir / "plots")
    checkpoints_dir = ensure_dir(BUNDLE_ROOT / "checkpoints")
    plots_dir = ensure_dir(BUNDLE_ROOT / "plots")
    cache_dir = ensure_dir(results_dir / "cache")
    return {
        "results_dir": results_dir,
        "evaluation_dir": evaluation_dir,
        "holdout_dir": holdout_dir,
        "holdout_eval_dir": holdout_eval_dir,
        "holdout_plots_dir": holdout_plots_dir,
        "umap_reference_dir": umap_reference_dir,
        "umap_reference_plots_dir": umap_reference_plots_dir,
        "umap_joint_dir": umap_joint_dir,
        "umap_joint_plots_dir": umap_joint_plots_dir,
        "checkpoints_dir": checkpoints_dir,
        "plots_dir": plots_dir,
        "cache_dir": cache_dir,
    }


def reorganize_bundle() -> dict[str, Path]:
    paths = layout()
    variant_name = "effnet_b1_one_layer_patchcore_x240"

    move_if_exists(BUNDLE_ROOT / f"{variant_name}_best_model.pt", paths["checkpoints_dir"] / "best_model.pt")
    move_if_exists(BUNDLE_ROOT / "dataset_cache.npz", paths["cache_dir"] / "dataset_cache.npz")
    move_if_exists(BUNDLE_ROOT / "config_snapshot.json", paths["results_dir"] / "config_snapshot.json")
    move_if_exists(BUNDLE_ROOT / "main_phase_manifest.json", paths["results_dir"] / "main_phase_manifest.json")
    move_if_exists(BUNDLE_ROOT / "variant_summary.csv", paths["results_dir"] / "variant_summary.csv")
    move_if_exists(BUNDLE_ROOT / f"{variant_name}_summary.json", paths["results_dir"] / "summary.json")
    move_if_exists(BUNDLE_ROOT / f"{variant_name}_best_row.csv", paths["results_dir"] / "best_row.csv")
    move_if_exists(BUNDLE_ROOT / f"{variant_name}_val_scores.csv", paths["evaluation_dir"] / "val_scores.csv")
    move_if_exists(BUNDLE_ROOT / f"{variant_name}_test_scores.csv", paths["evaluation_dir"] / "test_scores.csv")
    move_if_exists(BUNDLE_ROOT / f"{variant_name}_threshold_sweep.csv", paths["evaluation_dir"] / "threshold_sweep.csv")
    move_if_exists(BUNDLE_ROOT / f"{variant_name}_defect_breakdown.csv", paths["evaluation_dir"] / "defect_breakdown.csv")

    holdout_src = BUNDLE_ROOT / "holdout70k_3p5k"
    move_if_exists(holdout_src / "summary.json", paths["holdout_dir"] / "summary.json")
    move_if_exists(holdout_src / "summary.csv", paths["holdout_dir"] / "summary.csv")
    move_if_exists(holdout_src / "val_scores.csv", paths["holdout_eval_dir"] / "val_scores.csv")
    move_if_exists(holdout_src / "test_scores.csv", paths["holdout_eval_dir"] / "test_scores.csv")
    move_if_exists(holdout_src / "threshold_sweep.csv", paths["holdout_eval_dir"] / "threshold_sweep.csv")
    move_if_exists(holdout_src / "defect_breakdown.csv", paths["holdout_eval_dir"] / "defect_breakdown.csv")

    evaluation_src = BUNDLE_ROOT / "evaluation"
    move_if_exists(evaluation_src / "umap_points.csv", paths["umap_reference_dir"] / "umap_points.csv")
    move_if_exists(evaluation_src / "umap_summary.json", paths["umap_reference_dir"] / "umap_summary.json")
    move_if_exists(evaluation_src / "umap_knn_threshold_sweep.csv", paths["umap_reference_dir"] / "umap_knn_threshold_sweep.csv")
    for filename in [
        "train_embeddings.npy",
        "train_labels.npy",
        "val_embeddings.npy",
        "val_labels.npy",
        "val_scores.npy",
        "test_embeddings.npy",
        "test_labels.npy",
        "test_scores.npy",
    ]:
        move_if_exists(evaluation_src / filename, paths["umap_reference_dir"] / filename)
    move_if_exists(evaluation_src / "plots/umap_by_split.png", paths["umap_reference_plots_dir"] / "umap_by_split.png")
    move_if_exists(evaluation_src / "plots/umap_by_score.png", paths["umap_reference_plots_dir"] / "umap_by_score.png")
    move_if_exists(evaluation_src / "joint_fit/umap_points.csv", paths["umap_joint_dir"] / "umap_points.csv")
    move_if_exists(evaluation_src / "joint_fit/umap_summary.json", paths["umap_joint_dir"] / "umap_summary.json")
    move_if_exists(evaluation_src / "joint_fit/plots/umap_by_split.png", paths["umap_joint_plots_dir"] / "umap_by_split.png")
    move_if_exists(evaluation_src / "joint_fit/plots/umap_by_score.png", paths["umap_joint_plots_dir"] / "umap_by_score.png")

    return paths


def summarize_confusion(scores_df: pd.DataFrame, threshold: float) -> list[list[int]]:
    predicted = (scores_df["score"].astype(float) >= float(threshold)).astype(int)
    truth = scores_df["is_anomaly"].astype(int)
    tn = int(((truth == 0) & (predicted == 0)).sum())
    fp = int(((truth == 0) & (predicted == 1)).sum())
    fn = int(((truth == 1) & (predicted == 0)).sum())
    tp = int(((truth == 1) & (predicted == 1)).sum())
    return [[tn, fp], [fn, tp]]


def write_confusion_csv(path: Path, cm: list[list[int]]) -> None:
    df = pd.DataFrame(cm, index=["true_normal", "true_anomaly"], columns=["pred_normal", "pred_anomaly"])
    df.to_csv(path)


def plot_distribution_sweep_confusion(
    *,
    val_scores_df: pd.DataFrame,
    test_scores_df: pd.DataFrame,
    threshold_sweep_df: pd.DataFrame,
    threshold: float,
    title_prefix: str,
    output_path: Path,
) -> list[list[int]]:
    cm = summarize_confusion(test_scores_df, threshold)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    axes[0].hist(val_scores_df["score"], bins=40, alpha=0.85, color="#4d908e")
    axes[0].axvline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"threshold={threshold:.4f}")
    axes[0].set_title(f"{title_prefix}: Validation Scores")
    axes[0].set_xlabel("Score")
    axes[0].legend()

    axes[1].hist(test_scores_df[test_scores_df["is_anomaly"] == 0]["score"], bins=40, alpha=0.7, label="normal", color="#90be6d")
    axes[1].hist(test_scores_df[test_scores_df["is_anomaly"] == 1]["score"], bins=40, alpha=0.7, label="anomaly", color="#f8961e")
    axes[1].axvline(threshold, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_title(f"{title_prefix}: Test Score Distribution")
    axes[1].set_xlabel("Score")
    axes[1].legend()

    axes[2].plot(threshold_sweep_df["threshold"], threshold_sweep_df["precision"], label="precision", linewidth=2)
    axes[2].plot(threshold_sweep_df["threshold"], threshold_sweep_df["recall"], label="recall", linewidth=2)
    axes[2].plot(threshold_sweep_df["threshold"], threshold_sweep_df["f1"], label="f1", linewidth=2)
    axes[2].axvline(threshold, color="red", linestyle="--", linewidth=1.5)
    axes[2].set_title(f"{title_prefix}: Threshold Sweep")
    axes[2].set_xlabel("Threshold")
    axes[2].grid(alpha=0.25, linestyle="--")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{title_prefix}: Confusion Matrix")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred Normal", "Pred Anomaly"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True Normal", "True Anomaly"])
    for row_index in range(2):
        for col_index in range(2):
            ax.text(col_index, row_index, f"{cm[row_index][col_index]:,}", ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path.with_name(output_path.stem + "_confusion.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    return cm


def plot_defect_breakdown(df: pd.DataFrame, title: str, output_path: Path) -> None:
    if df.empty:
        return
    plot_df = df.sort_values("recall", ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, max(4.0, 0.42 * len(plot_df))))
    ax.barh(plot_df["defect_type"], plot_df["recall"], color="#577590")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_plots(paths: dict[str, Path]) -> None:
    summary = json.loads((paths["results_dir"] / "summary.json").read_text(encoding="utf-8"))
    val_scores_df = pd.read_csv(paths["evaluation_dir"] / "val_scores.csv")
    test_scores_df = pd.read_csv(paths["evaluation_dir"] / "test_scores.csv")
    threshold_sweep_df = pd.read_csv(paths["evaluation_dir"] / "threshold_sweep.csv")
    defect_breakdown_df = pd.read_csv(paths["evaluation_dir"] / "defect_breakdown.csv")

    benchmark_cm = plot_distribution_sweep_confusion(
        val_scores_df=val_scores_df,
        test_scores_df=test_scores_df,
        threshold_sweep_df=threshold_sweep_df,
        threshold=float(summary["threshold"]),
        title_prefix="EfficientNet-B1 x240 Benchmark",
        output_path=paths["plots_dir"] / "benchmark_distribution_sweep.png",
    )
    write_confusion_csv(paths["evaluation_dir"] / "confusion_matrix.csv", benchmark_cm)
    plot_defect_breakdown(
        defect_breakdown_df,
        "EfficientNet-B1 x240 Benchmark: Defect Recall by Type",
        paths["plots_dir"] / "benchmark_defect_breakdown.png",
    )

    holdout_summary = json.loads((paths["holdout_dir"] / "summary.json").read_text(encoding="utf-8"))
    holdout_val_scores_df = pd.read_csv(paths["holdout_eval_dir"] / "val_scores.csv")
    holdout_test_scores_df = pd.read_csv(paths["holdout_eval_dir"] / "test_scores.csv")
    holdout_threshold_sweep_df = pd.read_csv(paths["holdout_eval_dir"] / "threshold_sweep.csv")
    holdout_defect_breakdown_df = pd.read_csv(paths["holdout_eval_dir"] / "defect_breakdown.csv")

    holdout_cm = plot_distribution_sweep_confusion(
        val_scores_df=holdout_val_scores_df,
        test_scores_df=holdout_test_scores_df,
        threshold_sweep_df=holdout_threshold_sweep_df,
        threshold=float(holdout_summary["threshold"]),
        title_prefix="EfficientNet-B1 x240 Holdout70k/3.5k",
        output_path=paths["holdout_plots_dir"] / "holdout_distribution_sweep.png",
    )
    write_confusion_csv(paths["holdout_eval_dir"] / "confusion_matrix.csv", holdout_cm)
    plot_defect_breakdown(
        holdout_defect_breakdown_df,
        "EfficientNet-B1 x240 Holdout: Defect Recall by Type",
        paths["holdout_plots_dir"] / "holdout_defect_breakdown.png",
    )

    reference_split_plot = paths["umap_reference_plots_dir"] / "umap_by_split.png"
    reference_score_plot = paths["umap_reference_plots_dir"] / "umap_by_score.png"
    joint_split_plot = paths["umap_joint_plots_dir"] / "umap_by_split.png"
    joint_score_plot = paths["umap_joint_plots_dir"] / "umap_by_score.png"

    if reference_split_plot.exists():
        shutil.copy2(reference_split_plot, paths["plots_dir"] / "umap_reference_by_split.png")
    if reference_score_plot.exists():
        shutil.copy2(reference_score_plot, paths["plots_dir"] / "umap_reference_by_score.png")
    if joint_split_plot.exists():
        shutil.copy2(joint_split_plot, paths["plots_dir"] / "umap_joint_by_split.png")
    if joint_score_plot.exists():
        shutil.copy2(joint_score_plot, paths["plots_dir"] / "umap_joint_by_score.png")


def patch_notebook() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    def replace_in_notebook(old: str, new: str) -> None:
        for cell in notebook["cells"]:
            source = "".join(cell.get("source", []))
            if old in source:
                cell["source"] = source.replace(old, new).splitlines(keepends=True)

    replace_in_notebook(
        "This notebook adapts the external EfficientNet-B1 PatchCore prototype to the repo's report-compatible anomaly protocol.",
        "This notebook runs the local EfficientNet-B1 PatchCore x240 follow-up using the repo's report-compatible anomaly protocol.",
    )
    replace_in_notebook(
        "- saves CSV, JSON, plot, and checkpoint artifacts under `artifacts/x240/patchcore_efficientnet_b1_one_layer`",
        "- saves CSV, JSON, plot, and checkpoint artifacts under `experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main/artifacts/patchcore_efficientnet_b1_one_layer`",
    )
    replace_in_notebook(
        '"output_dir": "artifacts/x240/patchcore_efficientnet_b1_one_layer",',
        '"output_dir": "experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main/artifacts/patchcore_efficientnet_b1_one_layer",',
    )
    replace_in_notebook(
        """def prepare_output_dir(config: dict[str, Any]) -> Path:
    output_dir = resolve_project_path(config["run"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def dataset_cache_path(config: dict[str, Any]) -> Path:
    return prepare_output_dir(config) / "dataset_cache.npz"
""",
        """def prepare_output_dir(config: dict[str, Any]) -> Path:
    output_dir = resolve_project_path(config["run"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def artifact_layout(output_dir: Path) -> dict[str, Path]:
    results_dir = output_dir / "results"
    evaluation_dir = results_dir / "evaluation"
    holdout_dir = results_dir / "holdout70k_3p5k"
    holdout_eval_dir = holdout_dir / "evaluation"
    umap_dir = results_dir / "umap"
    umap_reference_dir = umap_dir / "reference_fit"
    umap_joint_dir = umap_dir / "joint_fit"
    checkpoints_dir = output_dir / "checkpoints"
    plots_dir = output_dir / "plots"
    cache_dir = results_dir / "cache"
    for directory in [
        results_dir,
        evaluation_dir,
        holdout_dir,
        holdout_eval_dir,
        umap_reference_dir,
        umap_reference_dir / "plots",
        umap_joint_dir,
        umap_joint_dir / "plots",
        checkpoints_dir,
        plots_dir,
        cache_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    return {
        "results_dir": results_dir,
        "evaluation_dir": evaluation_dir,
        "holdout_dir": holdout_dir,
        "holdout_eval_dir": holdout_eval_dir,
        "umap_reference_dir": umap_reference_dir,
        "umap_joint_dir": umap_joint_dir,
        "checkpoints_dir": checkpoints_dir,
        "plots_dir": plots_dir,
        "cache_dir": cache_dir,
    }


def dataset_cache_path(config: dict[str, Any]) -> Path:
    return artifact_layout(prepare_output_dir(config))["cache_dir"] / "dataset_cache.npz"
""",
    )
    replace_in_notebook(
        """def save_variant_result(output_dir: Path, variant_name: str, result: dict[str, Any]) -> None:
    result["score_df"].to_csv(output_dir / f"{variant_name}_best_row.csv", index=False)
    result["val_scores_df"].to_csv(output_dir / f"{variant_name}_val_scores.csv", index=False)
    result["test_scores_df"].to_csv(output_dir / f"{variant_name}_test_scores.csv", index=False)
    result["threshold_sweep_df"].to_csv(output_dir / f"{variant_name}_threshold_sweep.csv", index=False)
    result["defect_breakdown_df"].to_csv(output_dir / f"{variant_name}_defect_breakdown.csv", index=False)
    with (output_dir / f"{variant_name}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result["summary"], handle, indent=2)
    torch.save(result["checkpoint"], output_dir / f"{variant_name}_best_model.pt")
    pd.DataFrame([result["summary"]]).to_csv(output_dir / "variant_summary.csv", index=False)
    with (output_dir / "config_snapshot.json").open("w", encoding="utf-8") as handle:
        json.dump(result["summary"]["config"], handle, indent=2)
    print(f"Saved outputs for {variant_name} to {output_dir}")


def run_and_save_variant(dataset: dict[str, np.ndarray], config: dict[str, Any]) -> dict[str, Any]:
    output_dir = prepare_output_dir(config)
    variant_name = str(config["run"]["variant_name"])
    print(f"Running {variant_name}")
    result = run_variant(dataset, config)
    save_variant_result(output_dir, variant_name, result)
    return result


def load_saved_summary(config: dict[str, Any]) -> pd.DataFrame:
    summary_path = prepare_output_dir(config) / "variant_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")
    return pd.read_csv(summary_path)
""",
        """def save_variant_result(output_dir: Path, variant_name: str, result: dict[str, Any]) -> None:
    paths = artifact_layout(output_dir)
    result["score_df"].to_csv(paths["results_dir"] / "best_row.csv", index=False)
    result["val_scores_df"].to_csv(paths["evaluation_dir"] / "val_scores.csv", index=False)
    result["test_scores_df"].to_csv(paths["evaluation_dir"] / "test_scores.csv", index=False)
    result["threshold_sweep_df"].to_csv(paths["evaluation_dir"] / "threshold_sweep.csv", index=False)
    result["defect_breakdown_df"].to_csv(paths["evaluation_dir"] / "defect_breakdown.csv", index=False)
    with (paths["results_dir"] / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result["summary"], handle, indent=2)
    torch.save(result["checkpoint"], paths["checkpoints_dir"] / "best_model.pt")
    pd.DataFrame([result["summary"]]).to_csv(paths["results_dir"] / "variant_summary.csv", index=False)
    with (paths["results_dir"] / "config_snapshot.json").open("w", encoding="utf-8") as handle:
        json.dump(result["summary"]["config"], handle, indent=2)
    print(f"Saved outputs for {variant_name} to {output_dir}")


def run_and_save_variant(dataset: dict[str, np.ndarray], config: dict[str, Any]) -> dict[str, Any]:
    output_dir = prepare_output_dir(config)
    variant_name = str(config["run"]["variant_name"])
    print(f"Running {variant_name}")
    result = run_variant(dataset, config)
    save_variant_result(output_dir, variant_name, result)
    return result


def load_saved_summary(config: dict[str, Any]) -> pd.DataFrame:
    summary_path = artifact_layout(prepare_output_dir(config))["results_dir"] / "variant_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")
    return pd.read_csv(summary_path)
""",
    )
    replace_in_notebook(
        """summary_df = load_saved_summary(CONFIG)
variant_name = str(CONFIG["run"]["variant_name"])
print(f"Best-row CSV: {output_dir / f'{variant_name}_best_row.csv'}")
print(f"Validation-score CSV: {output_dir / f'{variant_name}_val_scores.csv'}")
print(f"Test-score CSV: {output_dir / f'{variant_name}_test_scores.csv'}")
print(f"Threshold-sweep CSV: {output_dir / f'{variant_name}_threshold_sweep.csv'}")
print(f"Defect-breakdown CSV: {output_dir / f'{variant_name}_defect_breakdown.csv'}")
print(f"Checkpoint: {output_dir / f'{variant_name}_best_model.pt'}")
display(summary_df)
""",
        """summary_df = load_saved_summary(CONFIG)
paths = artifact_layout(output_dir)
print(f"Best-row CSV: {paths['results_dir'] / 'best_row.csv'}")
print(f"Validation-score CSV: {paths['evaluation_dir'] / 'val_scores.csv'}")
print(f"Test-score CSV: {paths['evaluation_dir'] / 'test_scores.csv'}")
print(f"Threshold-sweep CSV: {paths['evaluation_dir'] / 'threshold_sweep.csv'}")
print(f"Defect-breakdown CSV: {paths['evaluation_dir'] / 'defect_breakdown.csv'}")
print(f"Checkpoint: {paths['checkpoints_dir'] / 'best_model.pt'}")
display(summary_df)
""",
    )
    replace_in_notebook(
        """    holdout_output_dir = output_dir / "holdout70k_3p5k"
    holdout_output_dir.mkdir(parents=True, exist_ok=True)
    holdout_val_scores_df.to_csv(holdout_output_dir / "val_scores.csv", index=False)
    holdout_test_scores_df.to_csv(holdout_output_dir / "test_scores.csv", index=False)
    holdout_threshold_sweep_df.to_csv(holdout_output_dir / "threshold_sweep.csv", index=False)
    holdout_defect_breakdown_df.to_csv(holdout_output_dir / "defect_breakdown.csv", index=False)
    pd.DataFrame([holdout_summary]).to_csv(holdout_output_dir / "summary.csv", index=False)
    (holdout_output_dir / "summary.json").write_text(json.dumps(holdout_summary, indent=2), encoding="utf-8")
""",
        """    holdout_output_dir = artifact_layout(output_dir)["holdout_dir"]
    holdout_eval_dir = artifact_layout(output_dir)["holdout_eval_dir"]
    holdout_val_scores_df.to_csv(holdout_eval_dir / "val_scores.csv", index=False)
    holdout_test_scores_df.to_csv(holdout_eval_dir / "test_scores.csv", index=False)
    holdout_threshold_sweep_df.to_csv(holdout_eval_dir / "threshold_sweep.csv", index=False)
    holdout_defect_breakdown_df.to_csv(holdout_eval_dir / "defect_breakdown.csv", index=False)
    pd.DataFrame([holdout_summary]).to_csv(holdout_output_dir / "summary.csv", index=False)
    (holdout_output_dir / "summary.json").write_text(json.dumps(holdout_summary, indent=2), encoding="utf-8")
""",
    )
    replace_in_notebook('evaluation_dir = output_dir / "evaluation"', 'evaluation_dir = artifact_layout(output_dir)["umap_reference_dir"]')
    replace_in_notebook('joint_fit_dir = output_dir / "evaluation" / "joint_fit"', 'joint_fit_dir = artifact_layout(output_dir)["umap_joint_dir"]')

    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def patch_umap_followup_notebook() -> None:
    notebook = json.loads(UMAP_FOLLOWUP_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    notebook["cells"][0]["source"] = """# EfficientNet-B1 PatchCore UMAP Follow-up (`x240`)

This notebook performs additional local UMAP inspection for the EfficientNet-B1 `x240` run after the main checkpoint and score artifacts have been created.

What it covers:

- reuse of the local EfficientNet-B1 one-layer checkpoint and saved score artifacts
- secondary `70k / 3.5k` holdout inspection
- UMAP-based geometry diagnostics for the saved benchmark run
- additional follow-up visual analysis beyond the main experiment notebook
""".splitlines(keepends=True)
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    UMAP_FOLLOWUP_NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def patch_readmes() -> None:
    README_PATH.write_text(
        """# EfficientNet-B1 PatchCore (`x240`, one-layer main run)

This branch contains the local EfficientNet-B1 one-layer PatchCore follow-up at the backbone's native `x240` scale.

The main notebook trains the benchmark run, saves the fitted checkpoint, evaluates the standard `50k / 5%` protocol, reevaluates the same checkpoint on the `70k / 3.5k` holdout, and exports UMAP diagnostics from the saved feature manifold.

## Files

- `notebook.ipynb`
  Canonical local training and evaluation workflow for the one-layer EfficientNet-B1 run.
- `artifacts/patchcore_efficientnet_b1_one_layer/checkpoints/`
  Saved model checkpoint for the benchmark run.
- `artifacts/patchcore_efficientnet_b1_one_layer/results/`
  Benchmark summaries, score CSVs, holdout evaluation files, UMAP exports, and config snapshots.
- `artifacts/patchcore_efficientnet_b1_one_layer/plots/`
  Benchmark and UMAP figures regenerated from the saved local artifacts.
""",
        encoding="utf-8",
    )
    X240_README_PATH.write_text(
        """# EfficientNet-B1 PatchCore (`x240`)

This resolution folder groups the EfficientNet-B1 PatchCore branches that share the `x240` input pipeline.

## Why `x240`

EfficientNet-B1 is closest to its pretrained operating scale at `x240`, so these branches test whether a larger input resolution improves feature-memory anomaly detection on wafer maps.

## Branches

- `main/`
  Local one-layer benchmark run with checkpoint, benchmark metrics, holdout evaluation, and UMAP exports.
- `one_layer/`
  Imported single-layer source notebook awaiting a full local training run.
- `layer3_5/`
  Imported two-stage follow-up notebook awaiting a full local training run.
- `layer3_5_no_defect_tuning/`
  Imported two-stage follow-up without defect-aware tuning.
- `umap_followup/`
  Review-oriented UMAP branch that builds on the saved artifacts from `main/`.
""",
        encoding="utf-8",
    )
    UMAP_FOLLOWUP_README_PATH.write_text(
        """# EfficientNet-B1 PatchCore UMAP Follow-up (`x240`)

This branch is a UMAP-oriented follow-up for the local EfficientNet-B1 `x240` main run.

It is meant for additional manifold inspection after the main checkpoint and benchmark results have already been generated in `main/`.
""",
        encoding="utf-8",
    )


def main() -> None:
    paths = reorganize_bundle()
    generate_plots(paths)
    patch_notebook()
    patch_umap_followup_notebook()
    patch_readmes()


if __name__ == "__main__":
    main()
