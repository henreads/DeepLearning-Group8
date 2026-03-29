from __future__ import annotations

import json
import shutil
import textwrap
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


REPO_ROOT = Path(__file__).resolve().parents[2]


EFF_EXPERIMENTS = [
    {
        "title": "PatchCore Review Notebook (EfficientNet-B0, x64)",
        "short_name": "EfficientNet-B0 PatchCore",
        "folder": REPO_ROOT / "experiments" / "anomaly_detection" / "patchcore" / "efficientnet_b0" / "x64" / "main",
        "artifact_rel": "experiments/anomaly_detection/patchcore/efficientnet_b0/x64/main/artifacts/patchcore_efficientnet_b0",
        "metadata_csv": "data/processed/x64/wm811k/metadata_50k_5pct.csv",
        "dataset_notebook": "data/dataset/x64/benchmark_50k_5pct/notebook.ipynb",
        "dataset_config": "data/dataset/x64/benchmark_50k_5pct/data_config.toml",
        "summary_file": "effnet_b0_patchcore_summary.json",
        "variant_summary_file": "variant_summary.csv",
        "best_row_file": "effnet_b0_patchcore_best_row.csv",
        "val_scores_file": "effnet_b0_patchcore_val_scores.csv",
        "test_scores_file": "effnet_b0_patchcore_test_scores.csv",
        "threshold_sweep_file": "effnet_b0_patchcore_threshold_sweep.csv",
        "color_main": "#355070",
        "color_aux": "#90be6d",
        "color_defect": "#90be6d",
    },
    {
        "title": "PatchCore Review Notebook (EfficientNet-B0, x224)",
        "short_name": "EfficientNet-B0 PatchCore",
        "folder": REPO_ROOT / "experiments" / "anomaly_detection" / "patchcore" / "efficientnet_b0" / "x224" / "main",
        "artifact_rel": "experiments/anomaly_detection/patchcore/efficientnet_b0/x224/main/artifacts/patchcore_efficientnet_b0_5pct",
        "metadata_csv": "data/processed/x224/wm811k/metadata_50k_5pct.csv",
        "dataset_notebook": "data/dataset/x224/benchmark_50k_5pct/notebook.ipynb",
        "dataset_config": "data/dataset/x224/benchmark_50k_5pct/data_config.toml",
        "summary_file": "effnet_b0_patchcore_summary.json",
        "variant_summary_file": "variant_summary.csv",
        "best_row_file": "effnet_b0_patchcore_best_row.csv",
        "val_scores_file": "effnet_b0_patchcore_val_scores.csv",
        "test_scores_file": "effnet_b0_patchcore_test_scores.csv",
        "threshold_sweep_file": "effnet_b0_patchcore_threshold_sweep.csv",
        "color_main": "#264653",
        "color_aux": "#2a9d8f",
        "color_defect": "#8ab17d",
    },
]


VIT_EXPERIMENT = {
    "title": "PatchCore Review Notebook (ViT-B/16, x224)",
    "short_name": "ViT-B/16 PatchCore",
    "folder": REPO_ROOT / "experiments" / "anomaly_detection" / "patchcore" / "vit_b16" / "x224" / "main",
    "artifact_rel": "experiments/anomaly_detection/patchcore/vit_b16/x224/main/artifacts/patchcore_vit_b16_5pct/main_5pct",
    "metadata_csv": "data/processed/x224/wm811k/metadata_50k_5pct.csv",
    "dataset_notebook": "data/dataset/x224/benchmark_50k_5pct/notebook.ipynb",
    "dataset_config": "data/dataset/x224/benchmark_50k_5pct/data_config.toml",
    "color_main": "#3d405b",
    "color_aux": "#81b29a",
    "color_defect": "#81b29a",
}


def move_if_present(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    shutil.move(str(source), str(destination))


def reorganize_eff_artifacts(experiment: dict) -> None:
    artifact_root = REPO_ROOT / experiment["artifact_rel"]
    results_dir = artifact_root / "results"
    eval_dir = results_dir / "evaluation"
    plots_dir = artifact_root / "plots"
    checkpoints_dir = artifact_root / "checkpoints"
    for directory in [results_dir, eval_dir, plots_dir, checkpoints_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    move_if_present(artifact_root / experiment["summary_file"], results_dir / "summary.json")
    move_if_present(artifact_root / experiment["variant_summary_file"], results_dir / "variant_summary.csv")
    move_if_present(artifact_root / experiment["best_row_file"], results_dir / "best_row.csv")
    move_if_present(artifact_root / experiment["val_scores_file"], eval_dir / "val_scores.csv")
    move_if_present(artifact_root / experiment["test_scores_file"], eval_dir / "test_scores.csv")
    move_if_present(artifact_root / "effnet_b0_patchcore_test_scores (1).csv", eval_dir / "test_scores.csv")
    move_if_present(artifact_root / experiment["threshold_sweep_file"], eval_dir / "threshold_sweep.csv")
    move_if_present(artifact_root / "effnet_b0_patchcore_defect_breakdown.csv", eval_dir / "saved_defect_breakdown.csv")
    move_if_present(artifact_root / "patchcore_efficientnet_b0.json", results_dir / "config_snapshot.json")
    move_if_present(artifact_root / "config_snapshot.json", results_dir / "config_snapshot.json")

    missing_ckpt = checkpoints_dir / "MISSING_CHECKPOINT.txt"
    if not missing_ckpt.exists():
        missing_ckpt.write_text(
            "No EfficientNet-B0 PatchCore checkpoint was checked in for this branch. The curated notebook operates in results-review mode using the saved CSV artifacts.",
            encoding="utf-8",
        )


def reorganize_vit_artifact_root(artifact_root: Path) -> None:
    results_dir = artifact_root / "results"
    eval_dir = results_dir / "evaluation"
    umap_dir = results_dir / "umap"
    plots_dir = artifact_root / "plots"
    checkpoints_dir = artifact_root / "checkpoints"
    for directory in [results_dir, eval_dir, umap_dir, plots_dir, checkpoints_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    move_if_present(artifact_root / "summary.json", results_dir / "summary.json")
    move_if_present(artifact_root / "config.json", results_dir / "config.json")
    move_if_present(artifact_root / "evaluation_metrics.json", eval_dir / "evaluation_metrics.json")
    move_if_present(artifact_root / "val_scores.csv", eval_dir / "val_scores.csv")
    move_if_present(artifact_root / "test_scores.csv", eval_dir / "test_scores.csv")
    move_if_present(artifact_root / "threshold_sweep.csv", eval_dir / "threshold_sweep.csv")
    move_if_present(artifact_root / "defect_breakdown.csv", eval_dir / "saved_defect_breakdown.csv")
    move_if_present(artifact_root / "umap_summary.json", umap_dir / "umap_summary.json")
    move_if_present(artifact_root / "umap_test_embeddings.csv", umap_dir / "umap_test_embeddings.csv")
    move_if_present(artifact_root / "umap_knn_threshold_sweep.csv", umap_dir / "umap_knn_threshold_sweep.csv")

    move_if_present(artifact_root / "patchcore_vit_b16_model.pt", checkpoints_dir / "best_model.pt")
    move_if_present(artifact_root / "tune_score_dist.png", plots_dir / "score_distribution.png")
    move_if_present(artifact_root / "test_eval.png", plots_dir / "test_evaluation.png")
    move_if_present(artifact_root / "threshold_sweep_metrics.png", plots_dir / "threshold_sweep.png")
    move_if_present(artifact_root / "umap_test_embeddings.png", plots_dir / "umap_test_embeddings.png")
    move_if_present(artifact_root / "umap_by_score.png", plots_dir / "umap_by_score.png")


def reorganize_vit_artifacts(experiment: dict) -> None:
    artifact_root = REPO_ROOT / experiment["artifact_rel"]
    reorganize_vit_artifact_root(artifact_root)

    holdout_root = artifact_root.parent / "holdout70k_3p5k"
    if holdout_root.exists():
        reorganize_vit_artifact_root(holdout_root)


def build_eff_notebook(experiment: dict) -> nbformat.NotebookNode:
    artifact_rel = experiment["artifact_rel"]
    cells = [
        new_markdown_cell(
            textwrap.dedent(
                f"""\
                # {experiment["title"]}

                This curated notebook reviews the checked-in EfficientNet-B0 PatchCore results without retraining.
                """
            )
        ),
        new_markdown_cell(
            textwrap.dedent(
                f"""\
                ## Submission Context

                - Dataset notebook: `{experiment["dataset_notebook"]}`
                - Dataset config: `{experiment["dataset_config"]}`
                - Metadata CSV: `{experiment["metadata_csv"]}`
                - Artifact root: `{artifact_rel}`
                - Mode: results review from saved CSV artifacts
                - Checkpoint status: no checked-in checkpoint for this branch
                """
            )
        ),
        new_markdown_cell("## Imports and Saved Results\n\nThis cell loads the saved summary, score CSVs, and local metadata needed for review plots and defect analysis."),
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

                ARTIFACT_ROOT = REPO_ROOT / "{artifact_rel}"
                RESULTS_DIR = ARTIFACT_ROOT / "results"
                EVAL_DIR = RESULTS_DIR / "evaluation"
                PLOTS_DIR = ARTIFACT_ROOT / "plots"
                METADATA_PATH = REPO_ROOT / "{experiment["metadata_csv"]}"

                metadata = pd.read_csv(METADATA_PATH)
                test_metadata = metadata[metadata["split"] == "test"].reset_index(drop=True)
                summary = json.loads((RESULTS_DIR / "summary.json").read_text(encoding="utf-8"))
                variant_summary_df = pd.read_csv(RESULTS_DIR / "variant_summary.csv")
                best_row_df = pd.read_csv(RESULTS_DIR / "best_row.csv")
                val_scores_df = pd.read_csv(EVAL_DIR / "val_scores.csv")
                test_scores_df = pd.read_csv(EVAL_DIR / "test_scores.csv")
                threshold_sweep_df = pd.read_csv(EVAL_DIR / "threshold_sweep.csv")
                threshold = float(summary["threshold"])
                metrics = summarize_threshold_metrics(test_scores_df["is_anomaly"].to_numpy(), test_scores_df["score"].to_numpy(), threshold)

                display(variant_summary_df)
                display(best_row_df)
                """
            )
        ),
        new_markdown_cell("## Metrics and Plots\n\nThese cells recreate the main evaluation figures from the saved score CSVs and store them under `plots/`."),
        new_code_cell(
            textwrap.dedent(
                f"""\
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
                confusion_df = pd.DataFrame(metrics["confusion_matrix"], index=["true_normal", "true_anomaly"], columns=["pred_normal", "pred_anomaly"])
                display(metrics_df)
                display(confusion_df)

                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                axes[0].hist(val_scores_df["score"], bins=30, alpha=0.85, color="{experiment['color_main']}")
                axes[0].axvline(threshold, color="red", linestyle="--")
                axes[0].set_title("Validation Normal Score Distribution")
                axes[1].hist(test_scores_df[test_scores_df["is_anomaly"] == 0]["score"], bins=30, alpha=0.7, label="normal", color="#577590")
                axes[1].hist(test_scores_df[test_scores_df["is_anomaly"] == 1]["score"], bins=30, alpha=0.7, label="anomaly", color="#e76f51")
                axes[1].axvline(threshold, color="red", linestyle="--")
                axes[1].set_title("Test Score Distribution")
                axes[1].legend()
                plt.tight_layout()
                fig.savefig(PLOTS_DIR / "score_distribution.png", dpi=200, bbox_inches="tight")
                plt.show()
                plt.close(fig)

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(threshold_sweep_df.iloc[:, 0], threshold_sweep_df["precision"], label="precision")
                ax.plot(threshold_sweep_df.iloc[:, 0], threshold_sweep_df["recall"], label="recall")
                ax.plot(threshold_sweep_df.iloc[:, 0], threshold_sweep_df["f1"], label="f1")
                ax.set_title("Threshold Sweep")
                ax.legend()
                plt.tight_layout()
                fig.savefig(PLOTS_DIR / "threshold_sweep.png", dpi=200, bbox_inches="tight")
                plt.show()
                plt.close(fig)
                """
            )
        ),
        new_markdown_cell("## Defect Analysis\n\nThis cell recomputes failure analysis from the saved test scores and local metadata, then saves the resulting tables and plot."),
        new_code_cell(
            textwrap.dedent(
                f"""\
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

                analysis_df.to_csv(EVAL_DIR / "analysis_with_predictions.csv", index=False)
                error_summary_df.reset_index().to_csv(EVAL_DIR / "error_summary.csv", index=False)
                defect_recall_df.reset_index().to_csv(EVAL_DIR / "defect_recall.csv", index=False)

                top_defects_df = defect_recall_df.head(10).reset_index()
                fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                axes[0].bar(error_summary_df.index.astype(str), error_summary_df["count"], color="#e76f51")
                axes[0].set_title("Prediction Outcome Counts")
                axes[1].barh(top_defects_df["defect_type"], top_defects_df["recall"], color="{experiment['color_defect']}")
                axes[1].set_xlim(0.0, 1.0)
                axes[1].invert_yaxis()
                axes[1].set_title("Top Defect-Type Recall")
                plt.tight_layout()
                fig.savefig(PLOTS_DIR / "defect_breakdown.png", dpi=200, bbox_inches="tight")
                plt.show()
                plt.close(fig)

                display(error_summary_df)
                display(defect_recall_df)
                """
            )
        ),
    ]
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "version": "3.x"}
    return nb


def build_vit_notebook(experiment: dict) -> nbformat.NotebookNode:
    artifact_rel = experiment["artifact_rel"]
    cells = [
        new_markdown_cell(
            f"# {experiment['title']}\n\nThis curated notebook reviews the checked-in ViT PatchCore run from local artifacts."
        ),
        new_markdown_cell(
            f"## Submission Context\n\n- Dataset notebook: `{experiment['dataset_notebook']}`\n- Dataset config: `{experiment['dataset_config']}`\n- Metadata CSV: `{experiment['metadata_csv']}`\n- Artifact root: `{artifact_rel}`\n- Mode: artifact-first review with checked-in checkpoint"
        ),
        new_markdown_cell("## Imports and Saved Outputs\n\nThis cell loads the checked-in summary, evaluation files, checkpoint path, and saved UMAP outputs."),
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

                ARTIFACT_ROOT = REPO_ROOT / "{artifact_rel}"
                RESULTS_DIR = ARTIFACT_ROOT / "results"
                EVAL_DIR = RESULTS_DIR / "evaluation"
                UMAP_DIR = RESULTS_DIR / "umap"
                PLOTS_DIR = ARTIFACT_ROOT / "plots"
                CHECKPOINTS_DIR = ARTIFACT_ROOT / "checkpoints"
                METADATA_PATH = REPO_ROOT / "{experiment['metadata_csv']}"

                metadata = pd.read_csv(METADATA_PATH)
                test_metadata = metadata[metadata["split"] == "test"].reset_index(drop=True)
                summary = json.loads((RESULTS_DIR / "summary.json").read_text(encoding="utf-8"))
                evaluation_metrics = json.loads((EVAL_DIR / "evaluation_metrics.json").read_text(encoding="utf-8"))
                val_scores_df = pd.read_csv(EVAL_DIR / "val_scores.csv")
                test_scores_df = pd.read_csv(EVAL_DIR / "test_scores.csv")
                threshold_sweep_df = pd.read_csv(EVAL_DIR / "threshold_sweep.csv")
                threshold = float(summary["threshold_raw"])
                display(pd.Series(summary))
                display(pd.Series(evaluation_metrics))
                """
            )
        ),
        new_markdown_cell("## Metrics and Plots\n\nThese cells surface the saved ViT evaluation metrics and regenerate the key plots into the standardized `plots/` folder."),
        new_code_cell(
            textwrap.dedent(
                f"""\
                metrics_df = pd.DataFrame(
                    [
                        {{"metric": "precision", "value": evaluation_metrics["anomaly_precision"]}},
                        {{"metric": "recall", "value": evaluation_metrics["anomaly_recall"]}},
                        {{"metric": "f1", "value": evaluation_metrics["anomaly_f1"]}},
                        {{"metric": "roc_auc_z", "value": evaluation_metrics["roc_auc_z"]}},
                        {{"metric": "avg_precision_z", "value": evaluation_metrics["avg_precision_z"]}},
                        {{"metric": "threshold_raw", "value": summary["threshold_raw"]}},
                    ]
                )
                confusion_df = pd.DataFrame(evaluation_metrics["confusion_matrix"], index=["true_normal", "true_anomaly"], columns=["pred_normal", "pred_anomaly"])
                display(metrics_df)
                display(confusion_df)

                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                axes[0].hist(val_scores_df["score"], bins=30, alpha=0.85, color="{experiment['color_main']}")
                axes[0].axvline(threshold, color="red", linestyle="--")
                axes[0].set_title("Validation Score Distribution")
                axes[1].hist(test_scores_df[test_scores_df["is_anomaly"] == 0]["score"], bins=30, alpha=0.7, label="normal", color="#577590")
                axes[1].hist(test_scores_df[test_scores_df["is_anomaly"] == 1]["score"], bins=30, alpha=0.7, label="anomaly", color="#e76f51")
                axes[1].axvline(threshold, color="red", linestyle="--")
                axes[1].set_title("Test Score Distribution")
                axes[1].legend()
                plt.tight_layout()
                fig.savefig(PLOTS_DIR / "score_distribution_review.png", dpi=200, bbox_inches="tight")
                plt.show()
                plt.close(fig)

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(threshold_sweep_df["percentile"], threshold_sweep_df["precision"], label="precision")
                ax.plot(threshold_sweep_df["percentile"], threshold_sweep_df["recall"], label="recall")
                ax.plot(threshold_sweep_df["percentile"], threshold_sweep_df["f1"], label="f1")
                ax.set_title("Threshold Sweep by Percentile")
                ax.legend()
                plt.tight_layout()
                fig.savefig(PLOTS_DIR / "threshold_sweep_review.png", dpi=200, bbox_inches="tight")
                plt.show()
                plt.close(fig)
                """
            )
        ),
        new_markdown_cell("## Defect and UMAP Review\n\nThis cell surfaces the saved defect breakdown and confirms the checked-in checkpoint and UMAP outputs."),
        new_code_cell(
            textwrap.dedent(
                f"""\
                defect_breakdown_df = pd.read_csv(EVAL_DIR / "saved_defect_breakdown.csv")
                display(defect_breakdown_df)

                fig, ax = plt.subplots(figsize=(10, 5))
                plot_df = defect_breakdown_df.sort_values("recall", ascending=False)
                ax.barh(plot_df["failure_label"], plot_df["recall"], color="{experiment['color_defect']}")
                ax.set_xlim(0.0, 1.0)
                ax.invert_yaxis()
                ax.set_title("Saved Defect Recall by Failure Label")
                plt.tight_layout()
                fig.savefig(PLOTS_DIR / "defect_breakdown_review.png", dpi=200, bbox_inches="tight")
                plt.show()
                plt.close(fig)

                saved_outputs = {{
                    "checkpoint": str(CHECKPOINTS_DIR / "best_model.pt"),
                    "umap_summary": str(UMAP_DIR / "umap_summary.json"),
                    "umap_csv": str(UMAP_DIR / "umap_test_embeddings.csv"),
                    "plots_dir": str(PLOTS_DIR),
                }}
                saved_outputs
                """
            )
        ),
    ]
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "version": "3.x"}
    return nb


def write_readme(path: Path, text: str) -> None:
    path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    for experiment in EFF_EXPERIMENTS:
        reorganize_eff_artifacts(experiment)
        nbformat.write(build_eff_notebook(experiment), experiment["folder"] / "notebook.ipynb")
        write_readme(
            experiment["folder"] / "README.md",
            f"# {experiment['short_name']}\n\nThis branch is curated as a results-review notebook. It reloads the saved summary and score CSVs, recreates the main plots, and recomputes defect analysis from the local benchmark metadata.\n",
        )

    reorganize_vit_artifacts(VIT_EXPERIMENT)
    nbformat.write(build_vit_notebook(VIT_EXPERIMENT), VIT_EXPERIMENT["folder"] / "notebook.ipynb")
    write_readme(
        VIT_EXPERIMENT["folder"] / "README.md",
        "# ViT-B/16 PatchCore\n\nThis branch is curated as an artifact-first review notebook with a checked-in model checkpoint, saved evaluation CSVs, and saved UMAP outputs.\n",
    )
    holdout_readme = REPO_ROOT / "experiments" / "anomaly_detection" / "patchcore" / "vit_b16" / "x224" / "main" / "artifacts" / "patchcore_vit_b16_5pct" / "holdout70k_3p5k" / "README.md"
    write_readme(
        holdout_readme,
        "# ViT-B/16 PatchCore Holdout 70k/3.5k\n\nThis artifact bundle stores the holdout `70k / 3.5k` ViT PatchCore evaluation. It now follows the same `checkpoints/`, `results/`, and `plots/` layout as the curated main run.\n",
    )
    print("Curated EfficientNet/ViT PatchCore review notebooks.")


if __name__ == "__main__":
    main()
