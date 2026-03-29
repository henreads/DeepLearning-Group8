from __future__ import annotations

import json
import shutil
import textwrap
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


REPO_ROOT = Path(__file__).resolve().parents[2]
FOLDER = REPO_ROOT / "experiments" / "anomaly_detection" / "patchcore" / "wideresnet50" / "x224" / "multilayer_umap_followup"
ARTIFACTS_DIR = FOLDER / "artifacts"
LEGACY_ROOT = ARTIFACTS_DIR / "18A2-patchcore-wideresnet50-multilayer-umap"
DUPLICATE_ROOT = ARTIFACTS_DIR / "umaps" / "18A2-patchcore-wideresnet50-multilayer-umap"
CANONICAL_ROOT = ARTIFACTS_DIR / "patchcore-wideresnet50-multilayer-umap"
DATASET_NOTEBOOK = "data/dataset/x224/benchmark_50k_5pct/notebook.ipynb"
DATASET_CONFIG = "data/dataset/x224/benchmark_50k_5pct/data_config.toml"
METADATA_CSV = "data/processed/x224/wm811k/metadata_50k_5pct.csv"


def move_if_present(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    shutil.move(str(source), str(destination))


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def merge_duplicate_artifacts() -> None:
    if LEGACY_ROOT.exists() and not CANONICAL_ROOT.exists():
        shutil.move(str(LEGACY_ROOT), str(CANONICAL_ROOT))

    if DUPLICATE_ROOT.exists():
        extracted_dir = CANONICAL_ROOT / "results" / "extracted_notebook_outputs"
        extracted_dir.mkdir(parents=True, exist_ok=True)
        for image_path in sorted((DUPLICATE_ROOT / "extracted_notebook_outputs").glob("*")):
            move_if_present(image_path, extracted_dir / image_path.name)
        shutil.rmtree(DUPLICATE_ROOT)
        duplicate_parent = ARTIFACTS_DIR / "umaps"
        if duplicate_parent.exists() and not any(duplicate_parent.iterdir()):
            duplicate_parent.rmdir()


def reorganize_artifacts() -> str:
    merge_duplicate_artifacts()

    results_dir = CANONICAL_ROOT / "results"
    plots_dir = CANONICAL_ROOT / "plots"
    checkpoints_dir = CANONICAL_ROOT / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)

    source_roots = [CANONICAL_ROOT]
    if LEGACY_ROOT.exists():
        source_roots.append(LEGACY_ROOT)

    for source_root in source_roots:
        move_if_present(source_root / "config.json", results_dir / "config.json")
        move_if_present(source_root / "patchcore_sweep_results.csv", results_dir / "patchcore_sweep_results.csv")
        move_if_present(source_root / "patchcore_sweep_summary.json", results_dir / "patchcore_sweep_summary.json")
        move_if_present(source_root / "selected_checkpoint.json", results_dir / "selected_checkpoint.json")
        move_if_present(source_root / "patchcore_sweep_barplots.png", plots_dir / "patchcore_sweep_barplots.png")

    sweep_summary = json.loads((results_dir / "patchcore_sweep_summary.json").read_text(encoding="utf-8"))
    selected_variant_name = str(sweep_summary["best_variant"]["name"])

    variant_names = set()
    for source_root in source_roots:
        for variant_root in source_root.iterdir():
            if not variant_root.is_dir() or variant_root.name in {"results", "plots", "checkpoints", "processed"}:
                continue
            variant_names.add(variant_root.name)

    for variant_name in sorted(variant_names):
        variant_root = CANONICAL_ROOT / variant_name
        variant_results_dir = variant_root / "results"
        variant_eval_dir = variant_results_dir / "evaluation"
        variant_umap_dir = variant_results_dir / "umap"
        variant_plots_dir = variant_root / "plots"
        variant_checkpoints_dir = variant_root / "checkpoints"
        variant_root.mkdir(exist_ok=True)
        variant_results_dir.mkdir(exist_ok=True)
        variant_eval_dir.mkdir(exist_ok=True)
        variant_umap_dir.mkdir(exist_ok=True)
        variant_plots_dir.mkdir(exist_ok=True)
        variant_checkpoints_dir.mkdir(exist_ok=True)

        for source_root in source_roots:
            source_variant_root = source_root / variant_name
            if not source_variant_root.exists():
                continue
            move_if_present(source_variant_root / "summary.json", variant_results_dir / "summary.json")
            move_if_present(source_variant_root / "val_scores.csv", variant_eval_dir / "val_scores.csv")
            move_if_present(source_variant_root / "test_scores.csv", variant_eval_dir / "test_scores.csv")
            move_if_present(source_variant_root / "threshold_sweep.csv", variant_eval_dir / "threshold_sweep.csv")
            move_if_present(source_variant_root / "val_embeddings.npy", variant_umap_dir / "val_embeddings.npy")
            move_if_present(source_variant_root / "test_embeddings.npy", variant_umap_dir / "test_embeddings.npy")
            move_if_present(source_variant_root / "val_labels.npy", variant_umap_dir / "val_labels.npy")
            move_if_present(source_variant_root / "test_labels.npy", variant_umap_dir / "test_labels.npy")
            move_if_present(source_variant_root / "val_scores.npy", variant_umap_dir / "val_scores.npy")
            move_if_present(source_variant_root / "test_scores.npy", variant_umap_dir / "test_scores.npy")
            move_if_present(source_variant_root / "best_model.pt", variant_checkpoints_dir / "best_model.pt")
            legacy_plots_dir = source_variant_root / "plots"
            if legacy_plots_dir.exists():
                for plot_path in legacy_plots_dir.iterdir():
                    if plot_path.is_file():
                        move_if_present(plot_path, variant_plots_dir / plot_path.name)

        if not (variant_checkpoints_dir / "best_model.pt").exists():
            write_text(
                variant_checkpoints_dir / "MISSING_CHECKPOINT.txt",
                "No checkpoint was checked in for this variant. The review notebook can still load the saved CSV artifacts.",
            )

    write_text(
        checkpoints_dir / "README.txt",
        f"The selected saved checkpoint lives under {selected_variant_name}/checkpoints/best_model.pt. This branch-level folder is only a navigation note.",
    )
    if LEGACY_ROOT.exists():
        shutil.rmtree(LEGACY_ROOT)
    return selected_variant_name


def write_train_config() -> None:
    content = """[run]
output_dir = "experiments/anomaly_detection/patchcore/wideresnet50/x224/multilayer_umap_followup/artifacts/patchcore-wideresnet50-multilayer-umap"
seed = 42
secondary_holdout_mode = false

[data]
metadata_csv = "data/processed/x224/wm811k/metadata_50k_5pct.csv"
image_size = 224
batch_size = 128
num_workers = 0

[split]
mode = "report_50k_5pct"

[training]
device = "auto"

[model]
type = "patchcore"
backbone_type = "wideresnet50_2"
teacher_layers = ["layer2", "layer3"]
pretrained = true
freeze_backbone = true
backbone_input_size = 224
normalize_imagenet = true
query_chunk_size = 1024
memory_chunk_size = 4096

[scoring]
threshold_quantile = 0.95

[[sweep_variants]]
name = "topk_mb50k_r010_x224"
memory_bank_size = 600000
reduction = "topk_mean"
topk_ratio = 0.10

[[sweep_variants]]
name = "topk_mb50k_r005_x224"
memory_bank_size = 600000
reduction = "topk_mean"
topk_ratio = 0.05
"""
    write_text(FOLDER / "train_config.toml", content)


def build_notebook(selected_variant_name: str) -> nbformat.NotebookNode:
    artifact_root_rel = CANONICAL_ROOT.relative_to(REPO_ROOT).as_posix()
    folder_rel = FOLDER.relative_to(REPO_ROOT).as_posix()
    cells = [
        new_markdown_cell(
            textwrap.dedent(
                """\
                # PatchCore Review Notebook (WideResNet50-2 Multilayer UMAP Follow-up, x224)

                This notebook is the curated review notebook for the saved `x224` WideResNet50-2 multilayer PatchCore UMAP follow-up.

                Default behavior:
                - load the checked-in sweep CSVs and per-variant artifacts
                - recompute confusion matrices and defect analysis from the local `x224` metadata
                - display and save the selected variant's review plots
                - surface the saved embedding artifacts and checked-in UMAP visualization without retraining
                """
            )
        ),
        new_markdown_cell(
            textwrap.dedent(
                f"""\
                ## Submission Context

                - Dataset notebook: `{DATASET_NOTEBOOK}`
                - Dataset config: `{DATASET_CONFIG}`
                - Experiment config: `{folder_rel}/train_config.toml`
                - Artifact root: `{artifact_root_rel}`
                - Default selected variant: `{selected_variant_name}`
                - Mode: artifact-first review with optional cached variant rendering
                """
            )
        ),
        new_markdown_cell("## Imports and Paths\n\nThis cell resolves the repo root, loads the shared evaluation helper, and defines the canonical paths used by the review notebook."),
        new_code_cell(
            textwrap.dedent(
                f"""\
                from pathlib import Path
                import json
                import shutil
                import sys

                import matplotlib.pyplot as plt
                import numpy as np
                import pandas as pd
                from IPython.display import Image, display

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
                METADATA_PATH = REPO_ROOT / "{METADATA_CSV}"
                SELECTED_VARIANT_NAME = "{selected_variant_name}"
                RENDER_ALL_SAVED_VARIANTS = True
                VARIANTS_TO_RENDER = []
                SHOW_LEGACY_UMAP_IMAGE = True
                """
            )
        ),
        new_markdown_cell("## Metadata and Sweep Loading\n\nThis cell loads the saved sweep table, summary, and local metadata used for defect-level analysis."),
        new_code_cell(
            textwrap.dedent(
                """\
                metadata = pd.read_csv(METADATA_PATH)
                test_metadata = metadata[metadata["split"] == "test"].reset_index(drop=True)

                sweep_results_df = pd.read_csv(RESULTS_DIR / "patchcore_sweep_results.csv")
                sweep_summary = json.loads((RESULTS_DIR / "patchcore_sweep_summary.json").read_text(encoding="utf-8"))
                selected_checkpoint_meta = json.loads((RESULTS_DIR / "selected_checkpoint.json").read_text(encoding="utf-8"))
                selected_variant_name = str(SELECTED_VARIANT_NAME or sweep_summary["best_variant"]["name"])

                display(metadata["split"].value_counts().rename_axis("split").to_frame("count"))
                display(sweep_results_df)
                pd.Series(selected_checkpoint_meta)
                """
            )
        ),
        new_markdown_cell("## Variant Loaders\n\nThese helpers normalize the reorganized artifact layout, recompute metrics from cached CSVs, and regenerate per-variant plots without retraining."),
        new_code_cell(
            textwrap.dedent(
                """\
                def load_variant_outputs(variant_name: str) -> dict[str, object]:
                    variant_root = ARTIFACT_ROOT / variant_name
                    summary_path = variant_root / "results" / "summary.json"
                    if not summary_path.exists():
                        raise FileNotFoundError(f"Missing summary for {variant_name}: {summary_path}")

                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    variant_eval_dir = variant_root / "results" / "evaluation"
                    variant_umap_dir = variant_root / "results" / "umap"
                    variant_plots_dir = variant_root / "plots"
                    variant_checkpoints_dir = variant_root / "checkpoints"

                    val_scores_df = pd.read_csv(variant_eval_dir / "val_scores.csv")
                    test_scores_df = pd.read_csv(variant_eval_dir / "test_scores.csv")
                    threshold_sweep_df = pd.read_csv(variant_eval_dir / "threshold_sweep.csv")

                    threshold = float(summary["threshold"])
                    metrics = summarize_threshold_metrics(
                        test_scores_df["is_anomaly"].to_numpy(),
                        test_scores_df["score"].to_numpy(),
                        threshold,
                    )
                    best_sweep = threshold_sweep_df.sort_values("f1", ascending=False).iloc[0].to_dict()

                    return {
                        "summary": summary,
                        "threshold": threshold,
                        "val_scores_df": val_scores_df,
                        "test_scores_df": test_scores_df,
                        "threshold_sweep_df": threshold_sweep_df,
                        "metrics": metrics,
                        "best_sweep": best_sweep,
                        "variant_root": variant_root,
                        "variant_eval_dir": variant_eval_dir,
                        "variant_umap_dir": variant_umap_dir,
                        "variant_plots_dir": variant_plots_dir,
                        "variant_checkpoints_dir": variant_checkpoints_dir,
                    }


                def compute_failure_tables(test_metadata: pd.DataFrame, test_scores_df: pd.DataFrame, threshold: float):
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
                    threshold = float(payload["threshold"])
                    val_scores_df = payload["val_scores_df"]
                    test_scores_df = payload["test_scores_df"]
                    threshold_sweep_df = payload["threshold_sweep_df"]
                    metrics = payload["metrics"]
                    best_sweep = payload["best_sweep"]
                    variant_plots_dir = payload["variant_plots_dir"]
                    variant_eval_dir = payload["variant_eval_dir"]
                    variant_plots_dir.mkdir(exist_ok=True)
                    variant_eval_dir.mkdir(parents=True, exist_ok=True)

                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    axes[0].hist(val_scores_df["score"], bins=30, alpha=0.85, color="#577590")
                    axes[0].axvline(threshold, color="red", linestyle="--", label=f"threshold={threshold:.4f}")
                    axes[0].set_title(f"Validation Normal Score Distribution\\n{variant_name}")
                    axes[0].legend()

                    axes[1].hist(test_scores_df[test_scores_df["is_anomaly"] == 0]["score"], bins=30, alpha=0.7, label="normal", color="#4d908e")
                    axes[1].hist(test_scores_df[test_scores_df["is_anomaly"] == 1]["score"], bins=30, alpha=0.7, label="anomaly", color="#f3722c")
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
                    axes[0].bar(error_summary_df.index.astype(str), error_summary_df["count"], color="#e76f51")
                    axes[0].set_title(f"Prediction Outcome Counts\\n{variant_name}")
                    axes[0].set_ylabel("count")
                    top_defects_df = defect_recall_df.head(10).reset_index()
                    axes[1].barh(top_defects_df["defect_type"], top_defects_df["recall"], color="#8ab17d")
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
    ]
    cells.extend(
        [
            new_markdown_cell("## Selected Variant Review\n\nThis cell loads the selected variant, shows the key metrics, and regenerates the main branch-level plots."),
            new_code_cell(
                textwrap.dedent(
                    """\
                    selected_variant = load_variant_outputs(selected_variant_name)
                    summary = selected_variant["summary"]
                    threshold = float(selected_variant["threshold"])
                    val_scores_df = selected_variant["val_scores_df"]
                    test_scores_df = selected_variant["test_scores_df"]
                    threshold_sweep_df = selected_variant["threshold_sweep_df"]
                    metrics = selected_variant["metrics"]
                    best_sweep = selected_variant["best_sweep"]

                    metrics_df = pd.DataFrame(
                        [
                            {"metric": "precision", "value": metrics["precision"]},
                            {"metric": "recall", "value": metrics["recall"]},
                            {"metric": "f1", "value": metrics["f1"]},
                            {"metric": "auroc", "value": metrics["auroc"]},
                            {"metric": "auprc", "value": metrics["auprc"]},
                            {"metric": "threshold", "value": threshold},
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
                    axes[0].barh(plot_df["label"], plot_df["f1"], color="#264653")
                    axes[0].set_title("WRN50-2 Multilayer UMAP Follow-up: F1")
                    axes[0].invert_yaxis()
                    axes[1].barh(plot_df["label"], plot_df["auroc"], color="#2a9d8f")
                    axes[1].set_title("WRN50-2 Multilayer UMAP Follow-up: AUROC")
                    axes[1].invert_yaxis()
                    plt.tight_layout()
                    fig.savefig(PLOTS_DIR / "variant_comparison_metrics.png", dpi=200, bbox_inches="tight")
                    plt.show()
                    plt.close(fig)

                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    axes[0].hist(val_scores_df["score"], bins=30, alpha=0.85, color="#577590")
                    axes[0].axvline(threshold, color="red", linestyle="--", label=f"threshold={threshold:.4f}")
                    axes[0].set_title(f"Validation Normal Score Distribution\\n{selected_variant_name}")
                    axes[0].legend()
                    axes[1].hist(test_scores_df[test_scores_df["is_anomaly"] == 0]["score"], bins=30, alpha=0.7, label="normal", color="#4d908e")
                    axes[1].hist(test_scores_df[test_scores_df["is_anomaly"] == 1]["score"], bins=30, alpha=0.7, label="anomaly", color="#f3722c")
                    axes[1].axvline(threshold, color="red", linestyle="--", label=f"threshold={threshold:.4f}")
                    axes[1].set_title(f"Test Score Distribution\\n{selected_variant_name}")
                    axes[1].legend()
                    plt.tight_layout()
                    fig.savefig(PLOTS_DIR / "score_distribution.png", dpi=200, bbox_inches="tight")
                    plt.show()
                    plt.close(fig)

                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(threshold_sweep_df["threshold"], threshold_sweep_df["precision"], label="precision")
                    ax.plot(threshold_sweep_df["threshold"], threshold_sweep_df["recall"], label="recall")
                    ax.plot(threshold_sweep_df["threshold"], threshold_sweep_df["f1"], label="f1")
                    ax.axvline(threshold, color="red", linestyle="--", label=f"validation threshold = {threshold:.4f}")
                    ax.axvline(best_sweep["threshold"], color="green", linestyle=":", label=f"best sweep threshold = {best_sweep['threshold']:.4f}")
                    ax.set_title(f"Threshold Sweep on Test Split\\n{selected_variant_name}")
                    ax.legend()
                    plt.tight_layout()
                    fig.savefig(PLOTS_DIR / "threshold_sweep.png", dpi=200, bbox_inches="tight")
                    plt.show()
                    plt.close(fig)

                    cm_array = np.asarray(metrics["confusion_matrix"], dtype=float)
                    fig, ax = plt.subplots(figsize=(5, 4))
                    im = ax.imshow(cm_array, cmap="Blues")
                    ax.set_xticks([0, 1], labels=["pred_normal", "pred_anomaly"])
                    ax.set_yticks([0, 1], labels=["true_normal", "true_anomaly"])
                    ax.set_title(f"Confusion Matrix\\n{selected_variant_name}")
                    for row_idx in range(cm_array.shape[0]):
                        for col_idx in range(cm_array.shape[1]):
                            ax.text(col_idx, row_idx, int(cm_array[row_idx, col_idx]), ha="center", va="center", color="black")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    plt.tight_layout()
                    fig.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=200, bbox_inches="tight")
                    plt.show()
                    plt.close(fig)
                    """
                )
            ),
            new_markdown_cell("## Failure Analysis\n\nThis cell computes the selected variant's error breakdown and saves the summary CSVs used in the report."),
            new_code_cell(
                textwrap.dedent(
                    """\
                    analysis_df, error_summary_df, defect_recall_df = compute_failure_tables(test_metadata, test_scores_df, threshold)
                    analysis_df.to_csv(RESULTS_DIR / "selected_analysis_with_predictions.csv", index=False)
                    error_summary_df.reset_index().to_csv(RESULTS_DIR / "selected_error_summary.csv", index=False)
                    defect_recall_df.reset_index().to_csv(RESULTS_DIR / "selected_defect_recall.csv", index=False)

                    display(error_summary_df)
                    display(defect_recall_df)

                    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                    axes[0].bar(error_summary_df.index.astype(str), error_summary_df["count"], color="#e76f51")
                    axes[0].set_title(f"Prediction Outcome Counts\\n{selected_variant_name}")
                    axes[0].set_ylabel("count")
                    top_defects_df = defect_recall_df.head(10).reset_index()
                    axes[1].barh(top_defects_df["defect_type"], top_defects_df["recall"], color="#8ab17d")
                    axes[1].set_xlim(0.0, 1.0)
                    axes[1].set_title("Top Defect-Type Recall")
                    axes[1].set_xlabel("recall")
                    axes[1].invert_yaxis()
                    plt.tight_layout()
                    fig.savefig(PLOTS_DIR / "defect_breakdown.png", dpi=200, bbox_inches="tight")
                    plt.show()
                    plt.close(fig)
                    """
                )
            ),
            new_markdown_cell("## UMAP Assets\n\nThis cell inventories the saved embedding arrays and displays the checked-in UMAP plot for the selected variant. It keeps the branch runnable without requiring a fresh embedding extraction pass."),
            new_code_cell(
                textwrap.dedent(
                    """\
                    selected_umap_dir = selected_variant["variant_umap_dir"]
                    selected_plots_dir = selected_variant["variant_plots_dir"]
                    umap_inventory = {}
                    for name in [
                        "val_embeddings.npy",
                        "test_embeddings.npy",
                        "val_labels.npy",
                        "test_labels.npy",
                        "val_scores.npy",
                        "test_scores.npy",
                    ]:
                        path = selected_umap_dir / name
                        umap_inventory[name] = {"exists": path.exists(), "path": str(path)}
                        if path.exists():
                            umap_inventory[name]["shape"] = tuple(np.load(path, mmap_mode="r").shape)

                    with open(RESULTS_DIR / "selected_umap_inventory.json", "w", encoding="utf-8") as handle:
                        json.dump(umap_inventory, handle, indent=2)

                    display(pd.DataFrame.from_dict(umap_inventory, orient="index"))

                    legacy_umap_plot = selected_plots_dir / "umap_by_split.png"
                    if SHOW_LEGACY_UMAP_IMAGE and legacy_umap_plot.exists():
                        shutil.copy2(legacy_umap_plot, PLOTS_DIR / "selected_variant_umap_by_split.png")
                        display(Image(filename=str(legacy_umap_plot)))
                    else:
                        print("No checked-in UMAP plot found for the selected variant.")
                    """
                )
            ),
            new_markdown_cell("## Cached Variant Rendering\n\nThis section regenerates the standardized evaluation plots for each saved variant from cached CSVs without retraining."),
            new_code_cell(
                textwrap.dedent(
                    """\
                    variant_names = sweep_results_df["name"].astype(str).tolist() if RENDER_ALL_SAVED_VARIANTS else []
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
                                "checkpoint_present": (payload["variant_checkpoints_dir"] / "best_model.pt").exists(),
                            }
                        )

                    rendered_variants_df = pd.DataFrame(rendered_rows)
                    display(rendered_variants_df)
                    """
                )
            ),
            new_markdown_cell("## Saved Outputs\n\nThis cell summarizes the branch-level outputs created or refreshed by the review notebook."),
            new_code_cell(
                textwrap.dedent(
                    """\
                    saved_outputs = {
                        "artifact_root": str(ARTIFACT_ROOT),
                        "results_dir": str(RESULTS_DIR),
                        "plots_dir": str(PLOTS_DIR),
                        "selected_variant_name": selected_variant_name,
                        "selected_variant_checkpoint": str(selected_variant["variant_checkpoints_dir"] / "best_model.pt"),
                        "rendered_variants": rendered_variants_df["variant_name"].tolist(),
                    }
                    saved_outputs
                    """
                )
            ),
        ]
    )
    notebook = new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata["language_info"] = {"name": "python", "version": "3.x"}
    return notebook


def write_readme(selected_variant_name: str) -> None:
    text = f"""# WRN50-2 Multilayer PatchCore UMAP Follow-up (`x224`)

This branch is curated as a results-review notebook built from the saved WideResNet50-2 multilayer PatchCore UMAP artifacts. The notebook does not retrain by default. It reloads the saved sweep tables, per-variant score CSVs, the selected variant checkpoint, and the saved embedding arrays for the selected UMAP run.

Key files:
- notebook: `experiments/anomaly_detection/patchcore/wideresnet50/x224/multilayer_umap_followup/notebook.ipynb`
- train config: `experiments/anomaly_detection/patchcore/wideresnet50/x224/multilayer_umap_followup/train_config.toml`
- artifact root: `experiments/anomaly_detection/patchcore/wideresnet50/x224/multilayer_umap_followup/artifacts/patchcore-wideresnet50-multilayer-umap`

Artifact layout:
- `results/`: sweep tables, selected-variant review outputs, and the extracted legacy notebook images
- `plots/`: branch-level figures regenerated by the review notebook
- `{selected_variant_name}/checkpoints/best_model.pt`: selected saved checkpoint
- `{selected_variant_name}/results/evaluation/`: saved score CSVs and threshold sweep
- `{selected_variant_name}/results/umap/`: saved embedding, label, and score arrays for the UMAP follow-up
- `{selected_variant_name}/plots/`: selected variant plots, including the checked-in `umap_by_split.png`

Cleanup note:
- The duplicate `artifacts/umaps/...` copy was folded into the canonical artifact root, and only the extra extracted notebook-output images were kept.
"""
    write_text(FOLDER / "README.md", text)


def update_family_readme() -> None:
    path = REPO_ROOT / "experiments" / "anomaly_detection" / "patchcore" / "README.md"
    content = path.read_text(encoding="utf-8")
    old = "- `multilayer/`, `layer2/`, and `layer3/` reload saved per-variant score CSVs, recompute analysis from the local `x224` metadata, and regenerate per-variant plots"
    new = "- `multilayer/`, `multilayer_umap_followup/`, `layer2/`, and `layer3/` reload saved per-variant score CSVs, recompute analysis from the local `x224` metadata, and regenerate per-variant plots"
    if old in content and new not in content:
        content = content.replace(old, new)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    selected_variant_name = reorganize_artifacts()
    write_train_config()
    nbformat.write(build_notebook(selected_variant_name), FOLDER / "notebook.ipynb")
    write_readme(selected_variant_name)
    update_family_readme()
    print("Curated WRN x224 multilayer UMAP follow-up branch.")


if __name__ == "__main__":
    main()
