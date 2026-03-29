from __future__ import annotations

import json
import shutil
import textwrap
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


REPO_ROOT = Path(__file__).resolve().parents[2]


DETAILED_EXPERIMENTS = [
    {
        "slug": "multilayer",
        "title": "PatchCore Review Notebook (WideResNet50-2 Multilayer, x224)",
        "short_name": "WRN50-2 Multilayer PatchCore",
        "folder": REPO_ROOT / "experiments" / "anomaly_detection" / "patchcore" / "wideresnet50" / "x224" / "multilayer",
        "artifact_name": "patchcore-wideresnet50-multilayer",
        "metadata_csv": "data/processed/x224/wm811k/metadata_50k_5pct.csv",
        "dataset_notebook": "data/dataset/x224/benchmark_50k_5pct/notebook.ipynb",
        "dataset_config": "data/dataset/x224/benchmark_50k_5pct/data_config.toml",
        "family_label": "WideResNet50-2 multilayer features (`layer2 + layer3`)",
        "color_f1": "#264653",
        "color_auroc": "#2a9d8f",
        "color_val": "#4d908e",
        "color_normal": "#577590",
        "color_anomaly": "#e76f51",
        "color_defect": "#8ab17d",
    },
    {
        "slug": "layer2",
        "title": "PatchCore Review Notebook (WideResNet50-2 layer2, x224)",
        "short_name": "WRN50-2 layer2 PatchCore",
        "folder": REPO_ROOT / "experiments" / "anomaly_detection" / "patchcore" / "wideresnet50" / "x224" / "layer2",
        "artifact_name": "patchcore-wideresnet50-layer2",
        "metadata_csv": "data/processed/x224/wm811k/metadata_50k_5pct.csv",
        "dataset_notebook": "data/dataset/x224/benchmark_50k_5pct/notebook.ipynb",
        "dataset_config": "data/dataset/x224/benchmark_50k_5pct/data_config.toml",
        "family_label": "WideResNet50-2 single-layer features (`layer2`)",
        "color_f1": "#355070",
        "color_auroc": "#6d597a",
        "color_val": "#4d908e",
        "color_normal": "#577590",
        "color_anomaly": "#f3722c",
        "color_defect": "#90be6d",
    },
    {
        "slug": "layer3",
        "title": "PatchCore Review Notebook (WideResNet50-2 layer3, x224)",
        "short_name": "WRN50-2 layer3 PatchCore",
        "folder": REPO_ROOT / "experiments" / "anomaly_detection" / "patchcore" / "wideresnet50" / "x224" / "layer3",
        "artifact_name": "patchcore-wideresnet50-layer3",
        "metadata_csv": "data/processed/x224/wm811k/metadata_50k_5pct.csv",
        "dataset_notebook": "data/dataset/x224/benchmark_50k_5pct/notebook.ipynb",
        "dataset_config": "data/dataset/x224/benchmark_50k_5pct/data_config.toml",
        "family_label": "WideResNet50-2 single-layer features (`layer3`)",
        "color_f1": "#3d405b",
        "color_auroc": "#81b29a",
        "color_val": "#4d908e",
        "color_normal": "#577590",
        "color_anomaly": "#e07a5f",
        "color_defect": "#81b29a",
    },
]


WEIGHTED_EXPERIMENT = {
    "title": "PatchCore Review Notebook (WideResNet50-2 Weighted Sweep, x224)",
    "short_name": "WRN50-2 Weighted PatchCore",
    "folder": REPO_ROOT / "experiments" / "anomaly_detection" / "patchcore" / "wideresnet50" / "x224" / "weighted",
    "artifact_name": "patchcore-wideresnet50-weighted",
    "dataset_notebook": "data/dataset/x224/benchmark_50k_5pct/notebook.ipynb",
    "dataset_config": "data/dataset/x224/benchmark_50k_5pct/data_config.toml",
    "color_f1": "#264653",
    "color_auroc": "#2a9d8f",
}


def move_if_present(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    shutil.move(str(source), str(destination))


def write_missing_checkpoint(path: Path, note: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    path.write_text(note, encoding="utf-8")


def reorganize_detailed_artifacts(experiment: dict) -> dict:
    artifact_root = experiment["folder"] / "artifacts" / experiment["artifact_name"]
    results_dir = artifact_root / "results"
    plots_dir = artifact_root / "plots"
    checkpoints_dir = artifact_root / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)

    move_if_present(artifact_root / "patchcore_sweep_results.csv", results_dir / "patchcore_sweep_results.csv")
    move_if_present(artifact_root / "patchcore_sweep_summary.json", results_dir / "patchcore_sweep_summary.json")
    move_if_present(artifact_root / "config.json", results_dir / "config.json")

    sweep_summary = json.loads((results_dir / "patchcore_sweep_summary.json").read_text(encoding="utf-8"))
    selected_variant_name = str(sweep_summary["best_variant"]["name"])

    for variant_root in sorted(artifact_root.iterdir()):
        if not variant_root.is_dir() or variant_root.name in {"results", "plots", "checkpoints"}:
            continue

        variant_results_dir = variant_root / "results"
        variant_eval_dir = variant_results_dir / "evaluation"
        variant_plots_dir = variant_root / "plots"
        variant_checkpoints_dir = variant_root / "checkpoints"
        variant_results_dir.mkdir(exist_ok=True)
        variant_eval_dir.mkdir(exist_ok=True)
        variant_plots_dir.mkdir(exist_ok=True)
        variant_checkpoints_dir.mkdir(exist_ok=True)

        move_if_present(variant_root / "summary.json", variant_results_dir / "summary.json")
        move_if_present(variant_root / "summary (1).json", variant_results_dir / "summary.json")
        move_if_present(variant_root / "val_scores.csv", variant_eval_dir / "val_scores.csv")
        move_if_present(variant_root / "test_scores.csv", variant_eval_dir / "test_scores.csv")
        move_if_present(variant_root / "test_scores (1).csv", variant_eval_dir / "test_scores.csv")
        move_if_present(variant_root / "threshold_sweep.csv", variant_eval_dir / "threshold_sweep.csv")
        move_if_present(variant_root / "selected_defect_breakdown.csv", variant_eval_dir / "selected_defect_breakdown.csv")

        write_missing_checkpoint(
            variant_checkpoints_dir / "MISSING_CHECKPOINT.txt",
            "No checkpoint was checked in for this WRN x224 PatchCore variant. The notebook can still review the saved CSV artifacts and regenerate plots.",
        )

    write_missing_checkpoint(
        checkpoints_dir / "MISSING_CHECKPOINT.txt",
        "No canonical checkpoint was checked in for this WRN x224 PatchCore branch. The curated notebook operates in results-review mode using saved CSV artifacts.",
    )
    return {"artifact_root": artifact_root, "selected_variant_name": selected_variant_name}


def reorganize_weighted_artifacts(experiment: dict) -> None:
    artifact_root = experiment["folder"] / "artifacts" / experiment["artifact_name"]
    results_dir = artifact_root / "results"
    plots_dir = artifact_root / "plots"
    checkpoints_dir = artifact_root / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)

    move_if_present(artifact_root / "patchcore_sweep_results.csv", results_dir / "patchcore_sweep_results.csv")
    move_if_present(artifact_root / "patchcore_sweep_summary.json", results_dir / "patchcore_sweep_summary.json")
    move_if_present(artifact_root / "config.json", results_dir / "config.json")

    write_missing_checkpoint(
        checkpoints_dir / "MISSING_CHECKPOINT.txt",
        "This weighted WRN x224 sweep only has the aggregate sweep table checked in. No per-variant checkpoints or score CSVs were available in the repo snapshot.",
    )


def build_detailed_notebook(experiment: dict) -> nbformat.NotebookNode:
    folder_rel = experiment["folder"].relative_to(REPO_ROOT).as_posix()
    artifact_root_rel = f"{folder_rel}/artifacts/{experiment['artifact_name']}"
    metadata_csv = experiment["metadata_csv"]
    title = experiment["title"]
    short_name = experiment["short_name"]

    cells = [
        new_markdown_cell(
            textwrap.dedent(
                f"""\
                # {title}

                This notebook is the curated review notebook for the saved `x224` {short_name} sweep.

                Default behavior:
                - load the checked-in sweep CSVs and per-variant score files
                - recompute confusion matrices and defect analysis from the local `x224` metadata
                - save regenerated figures into each variant folder without retraining
                """
            )
        ),
        new_markdown_cell("## Imports and Paths\n\nThis cell loads the shared evaluation helpers and resolves the local artifact locations used by the review notebook."),
        new_code_cell(
            textwrap.dedent(
                f"""\
                from pathlib import Path
                import json
                import sys

                import matplotlib.pyplot as plt
                import numpy as np
                import pandas as pd
                from IPython.display import display

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

                from wafer_defect.evaluation import summarize_threshold_metrics

                ARTIFACT_ROOT = REPO_ROOT / "{artifact_root_rel}"
                RESULTS_DIR = ARTIFACT_ROOT / "results"
                PLOTS_DIR = ARTIFACT_ROOT / "plots"
                METADATA_PATH = REPO_ROOT / "{metadata_csv}"
                SELECTED_VARIANT_NAME = None
                RENDER_ALL_CACHED_VARIANTS = True
                VARIANTS_TO_RENDER: list[str] = []
                VARIANT_COLOR_VAL = "{experiment['color_val']}"
                VARIANT_COLOR_NORMAL = "{experiment['color_normal']}"
                VARIANT_COLOR_ANOMALY = "{experiment['color_anomaly']}"
                VARIANT_COLOR_DEFECT = "{experiment['color_defect']}"
                """
            )
        ),
        new_markdown_cell("## Metadata and Sweep Loading\n\nThis cell loads the saved sweep table, selects the best variant by default, and attaches the local `x224` benchmark metadata for downstream analysis."),
        new_code_cell(
            textwrap.dedent(
                """\
                metadata = pd.read_csv(METADATA_PATH)
                test_metadata = metadata[metadata["split"] == "test"].reset_index(drop=True)

                sweep_results_df = pd.read_csv(RESULTS_DIR / "patchcore_sweep_results.csv")
                sweep_summary = json.loads((RESULTS_DIR / "patchcore_sweep_summary.json").read_text(encoding="utf-8"))
                selected_variant_name = str(SELECTED_VARIANT_NAME or sweep_summary["best_variant"]["name"])

                display(metadata["split"].value_counts().rename_axis("split").to_frame("count"))
                display(sweep_results_df)
                print(f"Selected variant: {selected_variant_name}")
                """
            )
        ),
        new_markdown_cell("## Variant Loaders\n\nThese helpers normalize the per-variant artifact layout, recompute threshold metrics from the saved score CSVs, and render plots back into the variant folders."),
        new_code_cell(
            textwrap.dedent(
                """\
                def load_variant_outputs(variant_name: str) -> dict[str, object]:
                    variant_root = ARTIFACT_ROOT / variant_name
                    summary_path = variant_root / "results" / "summary.json"
                    if not summary_path.exists():
                        raise FileNotFoundError(f"Missing summary for {variant_name}: {summary_path}")

                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    val_scores_df = pd.read_csv(variant_root / "results" / "evaluation" / "val_scores.csv")
                    test_scores_df = pd.read_csv(variant_root / "results" / "evaluation" / "test_scores.csv")
                    threshold_sweep_df = pd.read_csv(variant_root / "results" / "evaluation" / "threshold_sweep.csv")

                    threshold = float(summary["threshold"])
                    metrics = summarize_threshold_metrics(
                        test_scores_df["is_anomaly"].to_numpy(),
                        test_scores_df["score"].to_numpy(),
                        threshold,
                    )
                    best_sweep = threshold_sweep_df.sort_values("f1", ascending=False).iloc[0].to_dict()
                    defect_breakdown_path = variant_root / "results" / "evaluation" / "selected_defect_breakdown.csv"
                    defect_breakdown_df = pd.read_csv(defect_breakdown_path) if defect_breakdown_path.exists() else None
                    return {
                        "summary": summary,
                        "val_scores_df": val_scores_df,
                        "test_scores_df": test_scores_df,
                        "threshold_sweep_df": threshold_sweep_df,
                        "metrics": metrics,
                        "best_sweep": best_sweep,
                        "variant_root": variant_root,
                        "defect_breakdown_df": defect_breakdown_df,
                    }


                def compute_failure_tables(test_metadata: pd.DataFrame, test_scores_df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                    analysis_df = test_metadata.copy()
                    analysis_df["score"] = test_scores_df.reset_index(drop=True)["score"]
                    analysis_df["predicted_anomaly"] = (analysis_df["score"] > threshold).astype(int)
                    analysis_df["error_type"] = "tn"
                    analysis_df.loc[(analysis_df["is_anomaly"] == 0) & (analysis_df["predicted_anomaly"] == 1), "error_type"] = "fp"
                    analysis_df.loc[(analysis_df["is_anomaly"] == 1) & (analysis_df["predicted_anomaly"] == 0), "error_type"] = "fn"
                    analysis_df.loc[(analysis_df["is_anomaly"] == 1) & (analysis_df["predicted_anomaly"] == 1), "error_type"] = "tp"

                    error_summary_df = (
                        analysis_df.groupby("error_type")
                        .agg(count=("error_type", "size"), mean_score=("score", "mean"))
                        .reindex(["tp", "fn", "fp", "tn"])
                    )

                    defect_recall_df = (
                        analysis_df[analysis_df["is_anomaly"] == 1]
                        .groupby("defect_type")
                        .agg(count=("defect_type", "size"), detected=("predicted_anomaly", "sum"), mean_score=("score", "mean"))
                        .sort_values(["detected", "count"], ascending=[False, False])
                    )
                    defect_recall_df["recall"] = defect_recall_df["detected"] / defect_recall_df["count"]
                    return analysis_df, error_summary_df, defect_recall_df


                def render_variant_artifacts(variant_name: str, payload: dict[str, object]) -> dict[str, str]:
                    summary = payload["summary"]
                    threshold = float(summary["threshold"])
                    val_scores_df = payload["val_scores_df"]
                    test_scores_df = payload["test_scores_df"]
                    threshold_sweep_df = payload["threshold_sweep_df"]
                    metrics = payload["metrics"]
                    best_sweep = payload["best_sweep"]
                    variant_root = payload["variant_root"]
                    variant_plots_dir = variant_root / "plots"
                    variant_eval_dir = variant_root / "results" / "evaluation"
                    variant_plots_dir.mkdir(exist_ok=True)
                    variant_eval_dir.mkdir(parents=True, exist_ok=True)

                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    axes[0].hist(val_scores_df["score"], bins=30, alpha=0.85, color=VARIANT_COLOR_VAL)
                    axes[0].axvline(threshold, color="red", linestyle="--", label=f"threshold={threshold:.4f}")
                    axes[0].set_title(f"Validation Normal Score Distribution\\n{variant_name}")
                    axes[0].legend()

                    axes[1].hist(test_scores_df[test_scores_df["is_anomaly"] == 0]["score"], bins=30, alpha=0.7, label="normal", color=VARIANT_COLOR_NORMAL)
                    axes[1].hist(test_scores_df[test_scores_df["is_anomaly"] == 1]["score"], bins=30, alpha=0.7, label="anomaly", color=VARIANT_COLOR_ANOMALY)
                    axes[1].axvline(threshold, color="red", linestyle="--", label=f"threshold={threshold:.4f}")
                    axes[1].set_title(f"Test Score Distribution\\n{variant_name}")
                    axes[1].legend()
                    plt.tight_layout()
                    fig.savefig(variant_plots_dir / "score_distribution.png", dpi=200, bbox_inches="tight")
                    plt.close(fig)

                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(threshold_sweep_df["threshold"], threshold_sweep_df["precision"], label="precision")
                    ax.plot(threshold_sweep_df["threshold"], threshold_sweep_df["recall"], label="recall")
                    ax.plot(threshold_sweep_df["threshold"], threshold_sweep_df["f1"], label="f1")
                    ax.axvline(threshold, color="red", linestyle="--", label=f"validation threshold = {threshold:.4f}")
                    ax.axvline(best_sweep["threshold"], color="green", linestyle=":", label=f"best sweep threshold = {best_sweep['threshold']:.4f}")
                    ax.set_title(f"Threshold Sweep on Test Split\\n{variant_name}")
                    ax.set_xlabel("Anomaly-score threshold")
                    ax.set_ylabel("Metric value")
                    ax.legend()
                    plt.tight_layout()
                    fig.savefig(variant_plots_dir / "threshold_sweep.png", dpi=200, bbox_inches="tight")
                    plt.close(fig)

                    cm_array = np.asarray(metrics["confusion_matrix"], dtype=float)
                    fig, ax = plt.subplots(figsize=(5, 4))
                    im = ax.imshow(cm_array, cmap="Blues")
                    ax.set_xticks([0, 1], labels=["pred_normal", "pred_anomaly"])
                    ax.set_yticks([0, 1], labels=["true_normal", "true_anomaly"])
                    ax.set_title(f"Confusion Matrix\\n{variant_name}")
                    for row_idx in range(cm_array.shape[0]):
                        for col_idx in range(cm_array.shape[1]):
                            ax.text(col_idx, row_idx, int(cm_array[row_idx, col_idx]), ha="center", va="center", color="black")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    plt.tight_layout()
                    fig.savefig(variant_plots_dir / "confusion_matrix.png", dpi=200, bbox_inches="tight")
                    plt.close(fig)

                    analysis_df, error_summary_df, defect_recall_df = compute_failure_tables(test_metadata, test_scores_df, threshold)
                    analysis_df.to_csv(variant_eval_dir / "analysis_with_predictions.csv", index=False)
                    error_summary_df.reset_index().to_csv(variant_eval_dir / "error_summary.csv", index=False)
                    defect_recall_df.reset_index().to_csv(variant_eval_dir / "defect_recall.csv", index=False)

                    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                    axes[0].bar(error_summary_df.index.astype(str), error_summary_df["count"], color=VARIANT_COLOR_ANOMALY)
                    axes[0].set_title(f"Prediction Outcome Counts\\n{variant_name}")
                    axes[0].set_ylabel("count")
                    top_defects_df = defect_recall_df.head(10).reset_index()
                    axes[1].barh(top_defects_df["defect_type"], top_defects_df["recall"], color=VARIANT_COLOR_DEFECT)
                    axes[1].set_xlim(0.0, 1.0)
                    axes[1].set_title("Top Defect-Type Recall")
                    axes[1].set_xlabel("recall")
                    axes[1].invert_yaxis()
                    plt.tight_layout()
                    fig.savefig(variant_plots_dir / "defect_breakdown.png", dpi=200, bbox_inches="tight")
                    plt.close(fig)

                    return {"plots_dir": str(variant_plots_dir), "evaluation_dir": str(variant_eval_dir)}
                """
            )
        ),
        new_markdown_cell("## Selected Variant Review\n\nThis cell loads the selected variant, shows the key metrics, and saves the main review plots into the branch-level `plots/` folder."),
        new_code_cell(
            textwrap.dedent(
                f"""\
                selected_variant = load_variant_outputs(selected_variant_name)
                summary = selected_variant["summary"]
                val_scores_df = selected_variant["val_scores_df"]
                test_scores_df = selected_variant["test_scores_df"]
                threshold_sweep_df = selected_variant["threshold_sweep_df"]
                metrics = selected_variant["metrics"]
                best_sweep = selected_variant["best_sweep"]
                threshold = float(summary["threshold"])

                metrics_df = pd.DataFrame(
                    [
                        {{"metric": "precision", "value": metrics["precision"]}},
                        {{"metric": "recall", "value": metrics["recall"]}},
                        {{"metric": "f1", "value": metrics["f1"]}},
                        {{"metric": "auroc", "value": metrics["auroc"]}},
                        {{"metric": "auprc", "value": metrics["auprc"]}},
                        {{"metric": "threshold", "value": threshold}},
                    ]
                )
                confusion_df = pd.DataFrame(
                    metrics["confusion_matrix"],
                    index=["true_normal", "true_anomaly"],
                    columns=["pred_normal", "pred_anomaly"],
                )
                display(metrics_df)
                display(confusion_df)

                plot_df = sweep_results_df.copy().sort_values(["f1", "auroc"], ascending=False).reset_index(drop=True)
                plot_df["label"] = plot_df["name"]
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                axes[0].barh(plot_df["label"], plot_df["f1"], color="{experiment['color_f1']}")
                axes[0].set_title("{experiment['short_name']}: F1")
                axes[0].invert_yaxis()
                axes[1].barh(plot_df["label"], plot_df["auroc"], color="{experiment['color_auroc']}")
                axes[1].set_title("{experiment['short_name']}: AUROC")
                axes[1].invert_yaxis()
                plt.tight_layout()
                fig.savefig(PLOTS_DIR / "variant_comparison_metrics.png", dpi=200, bbox_inches="tight")
                plt.show()
                plt.close(fig)

                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                axes[0].hist(val_scores_df["score"], bins=30, alpha=0.85, color=VARIANT_COLOR_VAL)
                axes[0].axvline(threshold, color="red", linestyle="--", label=f"threshold={{threshold:.4f}}")
                axes[0].set_title(f"Validation Normal Score Distribution\\n{{selected_variant_name}}")
                axes[0].legend()
                axes[1].hist(test_scores_df[test_scores_df["is_anomaly"] == 0]["score"], bins=30, alpha=0.7, label="normal", color=VARIANT_COLOR_NORMAL)
                axes[1].hist(test_scores_df[test_scores_df["is_anomaly"] == 1]["score"], bins=30, alpha=0.7, label="anomaly", color=VARIANT_COLOR_ANOMALY)
                axes[1].axvline(threshold, color="red", linestyle="--", label=f"threshold={{threshold:.4f}}")
                axes[1].set_title(f"Test Score Distribution\\n{{selected_variant_name}}")
                axes[1].legend()
                plt.tight_layout()
                fig.savefig(PLOTS_DIR / "score_distribution.png", dpi=200, bbox_inches="tight")
                plt.show()
                plt.close(fig)

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(threshold_sweep_df["threshold"], threshold_sweep_df["precision"], label="precision")
                ax.plot(threshold_sweep_df["threshold"], threshold_sweep_df["recall"], label="recall")
                ax.plot(threshold_sweep_df["threshold"], threshold_sweep_df["f1"], label="f1")
                ax.axvline(threshold, color="red", linestyle="--", label=f"validation threshold = {{threshold:.4f}}")
                ax.axvline(best_sweep["threshold"], color="green", linestyle=":", label=f"best sweep threshold = {{best_sweep['threshold']:.4f}}")
                ax.set_title(f"Threshold Sweep on Test Split\\n{{selected_variant_name}}")
                ax.legend()
                plt.tight_layout()
                fig.savefig(PLOTS_DIR / "threshold_sweep.png", dpi=200, bbox_inches="tight")
                plt.show()
                plt.close(fig)
                """
            )
        ),
        new_markdown_cell("## Failure Analysis and Cached Variant Rendering\n\nThis section computes the selected variant’s defect-level behavior and then regenerates the same outputs for every saved variant folder from cached CSVs."),
        new_code_cell(
            textwrap.dedent(
                """\
                analysis_df, error_summary_df, defect_recall_df = compute_failure_tables(
                    test_metadata,
                    test_scores_df,
                    threshold,
                )
                analysis_df.to_csv(RESULTS_DIR / "selected_analysis_with_predictions.csv", index=False)
                error_summary_df.reset_index().to_csv(RESULTS_DIR / "selected_error_summary.csv", index=False)
                defect_recall_df.reset_index().to_csv(RESULTS_DIR / "selected_defect_recall.csv", index=False)

                display(error_summary_df)
                display(defect_recall_df)

                fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                axes[0].bar(error_summary_df.index.astype(str), error_summary_df["count"], color=VARIANT_COLOR_ANOMALY)
                axes[0].set_title(f"Prediction Outcome Counts\\n{selected_variant_name}")
                axes[0].set_ylabel("count")
                top_defects_df = defect_recall_df.head(10).reset_index()
                axes[1].barh(top_defects_df["defect_type"], top_defects_df["recall"], color=VARIANT_COLOR_DEFECT)
                axes[1].set_xlim(0.0, 1.0)
                axes[1].set_title("Top Defect-Type Recall")
                axes[1].set_xlabel("recall")
                axes[1].invert_yaxis()
                plt.tight_layout()
                fig.savefig(PLOTS_DIR / "defect_breakdown.png", dpi=200, bbox_inches="tight")
                plt.show()
                plt.close(fig)

                variant_names = sweep_results_df["name"].astype(str).tolist() if RENDER_ALL_CACHED_VARIANTS else []
                variant_names.extend([str(name) for name in VARIANTS_TO_RENDER])
                variant_names.append(selected_variant_name)
                ordered_variant_names = []
                seen = set()
                for name in variant_names:
                    if name not in seen:
                        ordered_variant_names.append(name)
                        seen.add(name)

                rendered_rows = []
                for variant_name in ordered_variant_names:
                    payload = load_variant_outputs(variant_name)
                    render_info = render_variant_artifacts(variant_name, payload)
                    rendered_rows.append(
                        {
                            "variant_name": variant_name,
                            "plots_dir": render_info["plots_dir"],
                            "evaluation_dir": render_info["evaluation_dir"],
                        }
                    )

                rendered_variants_df = pd.DataFrame(rendered_rows)
                display(rendered_variants_df)
                """
            )
        ),
        new_markdown_cell("## Saved Outputs\n\nThis cell prints the final artifact locations for the curated review notebook."),
        new_code_cell(
            textwrap.dedent(
                """\
                saved_outputs = {
                    "artifact_root": str(ARTIFACT_ROOT),
                    "results_dir": str(RESULTS_DIR),
                    "plots_dir": str(PLOTS_DIR),
                    "selected_variant_name": selected_variant_name,
                    "rendered_variants": rendered_variants_df["variant_name"].tolist(),
                }
                saved_outputs
                """
            )
        ),
    ]

    notebook = new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata["language_info"] = {"name": "python", "version": "3.x"}
    return notebook


def build_weighted_notebook(experiment: dict) -> nbformat.NotebookNode:
    folder_rel = experiment["folder"].relative_to(REPO_ROOT).as_posix()
    artifact_root_rel = f"{folder_rel}/artifacts/{experiment['artifact_name']}"
    cells = [
        new_markdown_cell(
            textwrap.dedent(
                f"""\
                # {experiment["title"]}

                This is the curated review notebook for the saved weighted WRN `x224` sweep.

                This branch only has the aggregate sweep table checked in, so the notebook focuses on comparing the saved configurations rather than replaying per-variant score distributions.
                """
            )
        ),
        new_markdown_cell(
            textwrap.dedent(
                f"""\
                ## Submission Context

                - Dataset notebook: `{experiment["dataset_notebook"]}`
                - Dataset config: `{experiment["dataset_config"]}`
                - Artifact root: `{artifact_root_rel}`
                - Mode: sweep-table review only
                - Checkpoint status: no per-variant checkpoints or score CSVs were checked in
                """
            )
        ),
        new_markdown_cell("## Imports and Sweep Loading\n\nThis cell loads the saved weighted sweep table and the best saved row from the checked-in summary."),
        new_code_cell(
            textwrap.dedent(
                f"""\
                from pathlib import Path
                import json
                import sys

                import matplotlib.pyplot as plt
                import pandas as pd
                from IPython.display import display

                cwd = Path.cwd().resolve()
                candidate_roots = [cwd, *cwd.parents]
                REPO_ROOT = None
                for candidate in candidate_roots:
                    if (candidate / "src" / "wafer_defect").exists() and (candidate / "configs").exists():
                        REPO_ROOT = candidate
                        break

                if REPO_ROOT is None:
                    raise RuntimeError("Could not locate repo root containing src/wafer_defect and configs/")

                ARTIFACT_ROOT = REPO_ROOT / "{artifact_root_rel}"
                RESULTS_DIR = ARTIFACT_ROOT / "results"
                PLOTS_DIR = ARTIFACT_ROOT / "plots"
                sweep_results_df = pd.read_csv(RESULTS_DIR / "patchcore_sweep_results.csv")
                sweep_summary = json.loads((RESULTS_DIR / "patchcore_sweep_summary.json").read_text(encoding="utf-8"))
                best_variant = sweep_summary["best_variant"]
                display(sweep_results_df.head())
                pd.Series(best_variant)
                """
            )
        ),
        new_markdown_cell("## Weighted Sweep Comparison\n\nThese plots compare the saved weight combinations and top-k ratios using the checked-in sweep table."),
        new_code_cell(
            textwrap.dedent(
                f"""\
                top_plot_df = sweep_results_df.sort_values(["f1", "auroc"], ascending=False).head(15).copy()
                top_plot_df["label"] = top_plot_df["name"]

                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                axes[0].barh(top_plot_df["label"], top_plot_df["f1"], color="{experiment['color_f1']}")
                axes[0].set_title("Top Weighted Variants by F1")
                axes[0].invert_yaxis()
                axes[1].barh(top_plot_df["label"], top_plot_df["auroc"], color="{experiment['color_auroc']}")
                axes[1].set_title("Top Weighted Variants by AUROC")
                axes[1].invert_yaxis()
                plt.tight_layout()
                fig.savefig(PLOTS_DIR / "weighted_top_variants.png", dpi=200, bbox_inches="tight")
                plt.show()
                plt.close(fig)

                best_by_weight_df = (
                    sweep_results_df.sort_values(["weight_name", "f1", "auroc"], ascending=[True, False, False])
                    .groupby("weight_name", as_index=False)
                    .first()
                    .sort_values("f1", ascending=False)
                )
                best_by_weight_df.to_csv(RESULTS_DIR / "best_by_weight_name.csv", index=False)
                display(best_by_weight_df)

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.scatter(sweep_results_df["recall"], sweep_results_df["precision"], c=sweep_results_df["f1"], cmap="viridis", s=70)
                for _, row in best_by_weight_df.iterrows():
                    ax.text(row["recall"], row["precision"], row["weight_name"], fontsize=8)
                ax.set_xlabel("recall")
                ax.set_ylabel("precision")
                ax.set_title("Weighted Sweep Precision-Recall Tradeoff")
                plt.tight_layout()
                fig.savefig(PLOTS_DIR / "weighted_precision_recall_scatter.png", dpi=200, bbox_inches="tight")
                plt.show()
                plt.close(fig)
                """
            )
        ),
        new_markdown_cell("## Saved Outputs\n\nThis cell summarizes the saved outputs for the weighted sweep review notebook."),
        new_code_cell(
            textwrap.dedent(
                """\
                saved_outputs = {
                    "artifact_root": str(ARTIFACT_ROOT),
                    "results_dir": str(RESULTS_DIR),
                    "plots_dir": str(PLOTS_DIR),
                    "best_variant_name": best_variant["name"],
                }
                saved_outputs
                """
            )
        ),
    ]
    notebook = new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata["language_info"] = {"name": "python", "version": "3.x"}
    return notebook


def write_readme(experiment: dict, selected_variant_name: str | None, review_mode: str) -> None:
    folder = experiment["folder"]
    artifact_root = f"{folder.relative_to(REPO_ROOT).as_posix()}/artifacts/{experiment['artifact_name']}"
    if review_mode == "weighted":
        text = f"""# {experiment['short_name']} (`x224`)\n\nThis branch is curated as a sweep-review notebook. Only the aggregate sweep table was checked in, so the notebook focuses on ranking and comparing the saved weighted configurations.\n\nKey files:\n- notebook: `{folder.relative_to(REPO_ROOT).as_posix()}/notebook.ipynb`\n- artifact root: `{artifact_root}`\n\nSaved outputs:\n- `results/`: saved sweep tables and best-by-weight CSVs\n- `plots/`: sweep comparison figures\n- `checkpoints/`: note explaining that checkpoints were not checked in\n"""
    else:
        text = f"""# {experiment['short_name']} (`x224`)\n\nThis branch is curated as a results-review notebook built from saved PatchCore sweep artifacts. The notebook does not retrain; it reloads the saved per-variant score CSVs, recomputes defect analysis using the local `x224` metadata, and repopulates each variant folder with plots.\n\nKey files:\n- notebook: `{folder.relative_to(REPO_ROOT).as_posix()}/notebook.ipynb`\n- train config: `{folder.relative_to(REPO_ROOT).as_posix()}/train_config.toml`\n- artifact root: `{artifact_root}`\n\nArtifact layout:\n- `results/`: branch-level sweep tables and selected-variant review outputs\n- `plots/`: branch-level plots for the selected variant\n- `{selected_variant_name}/`: per-variant results/evaluation/plots generated from cached CSVs\n- `checkpoints/`: note explaining that checkpoints were not checked in\n"""
    (folder / "README.md").write_text(text + "\n", encoding="utf-8")


def update_family_readme() -> None:
    path = REPO_ROOT / "experiments" / "anomaly_detection" / "patchcore" / "README.md"
    content = path.read_text(encoding="utf-8")
    addition = "\n## WRN x224 Status\n\nThe curated `wideresnet50/x224` notebooks now run in review mode from checked-in sweep artifacts:\n\n- `multilayer/`, `layer2/`, and `layer3/` reload saved per-variant score CSVs, recompute analysis from the local `x224` metadata, and regenerate per-variant plots\n- `weighted/` reloads the aggregate weighted sweep table only because no per-variant score files or checkpoints were checked in\n"
    if "## WRN x224 Status" not in content:
        path.write_text(content.rstrip() + addition, encoding="utf-8")


def main() -> None:
    for experiment in DETAILED_EXPERIMENTS:
        state = reorganize_detailed_artifacts(experiment)
        nbformat.write(build_detailed_notebook(experiment), experiment["folder"] / "notebook.ipynb")
        write_readme(experiment, state["selected_variant_name"], "detailed")

    reorganize_weighted_artifacts(WEIGHTED_EXPERIMENT)
    nbformat.write(build_weighted_notebook(WEIGHTED_EXPERIMENT), WEIGHTED_EXPERIMENT["folder"] / "notebook.ipynb")
    write_readme(WEIGHTED_EXPERIMENT, None, "weighted")
    update_family_readme()
    print("Curated WRN x224 PatchCore review notebooks.")


if __name__ == "__main__":
    main()
