from __future__ import annotations

import json
import shutil
import textwrap
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


REPO_ROOT = Path(__file__).resolve().parents[2]


EXPERIMENTS = [
    {
        "slug": "ae_bn",
        "title": "PatchCore Sweep Notebook (Autoencoder BatchNorm Backbone)",
        "short_name": "AE+BN PatchCore",
        "backbone_label": "autoencoder BatchNorm encoder features",
        "folder": REPO_ROOT / "experiments" / "anomaly_detection" / "patchcore" / "ae_bn" / "x64" / "main",
        "artifact_name": "patchcore_ae_bn",
        "config_path": "experiments/anomaly_detection/patchcore/ae_bn/x64/main/train_config.toml",
        "dataset_notebook": "data/dataset/x64/benchmark_50k_5pct/notebook.ipynb",
        "dataset_config": "data/dataset/x64/benchmark_50k_5pct/data_config.toml",
        "plot_prefix": "PatchCore AE+BN Sweep",
        "color_f1": "#355070",
        "color_auroc": "#6d597a",
        "color_val": "#4d908e",
        "color_normal": "#577590",
        "color_anomaly": "#f3722c",
        "color_defect": "#90be6d",
    },
    {
        "slug": "resnet18",
        "title": "PatchCore Sweep Notebook (Pretrained ResNet18 Backbone)",
        "short_name": "ResNet18 PatchCore",
        "backbone_label": "frozen ImageNet-pretrained ResNet18 features",
        "folder": REPO_ROOT / "experiments" / "anomaly_detection" / "patchcore" / "resnet18" / "x64" / "main",
        "artifact_name": "patchcore_resnet18",
        "config_path": "experiments/anomaly_detection/patchcore/resnet18/x64/main/train_config.toml",
        "dataset_notebook": "data/dataset/x64/benchmark_50k_5pct/notebook.ipynb",
        "dataset_config": "data/dataset/x64/benchmark_50k_5pct/data_config.toml",
        "plot_prefix": "PatchCore ResNet18 Sweep",
        "color_f1": "#277da1",
        "color_auroc": "#90be6d",
        "color_val": "#4d908e",
        "color_normal": "#577590",
        "color_anomaly": "#f3722c",
        "color_defect": "#43aa8b",
    },
    {
        "slug": "resnet50",
        "title": "PatchCore Sweep Notebook (Pretrained ResNet50 Backbone)",
        "short_name": "ResNet50 PatchCore",
        "backbone_label": "frozen ImageNet-pretrained ResNet50 features",
        "folder": REPO_ROOT / "experiments" / "anomaly_detection" / "patchcore" / "resnet50" / "x64" / "main",
        "artifact_name": "patchcore_resnet50",
        "config_path": "experiments/anomaly_detection/patchcore/resnet50/x64/main/train_config.toml",
        "dataset_notebook": "data/dataset/x64/benchmark_50k_5pct/notebook.ipynb",
        "dataset_config": "data/dataset/x64/benchmark_50k_5pct/data_config.toml",
        "plot_prefix": "PatchCore ResNet50 Sweep",
        "color_f1": "#264653",
        "color_auroc": "#2a9d8f",
        "color_val": "#4d908e",
        "color_normal": "#577590",
        "color_anomaly": "#e76f51",
        "color_defect": "#8ab17d",
    },
]


def move_if_present(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    shutil.move(str(source), str(destination))


def copy_if_missing(source: Path, destination: Path) -> None:
    if not source.exists() or destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def resolve_json_path(root: Path, relative_paths: list[str]) -> Path | None:
    for relative_path in relative_paths:
        candidate = root / relative_path
        if candidate.exists():
            return candidate
    return None


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def reorganize_artifacts(experiment: dict) -> dict:
    artifact_root = experiment["folder"] / "artifacts" / experiment["artifact_name"]
    artifact_root.mkdir(parents=True, exist_ok=True)

    checkpoints_dir = artifact_root / "checkpoints"
    results_dir = artifact_root / "results"
    evaluation_dir = results_dir / "evaluation"
    plots_dir = artifact_root / "plots"
    checkpoints_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    evaluation_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    move_if_present(artifact_root / "best_model.pt", checkpoints_dir / "best_model.pt")
    move_if_present(artifact_root / "last_model.pt", checkpoints_dir / "last_model.pt")
    move_if_present(artifact_root / "summary.json", results_dir / "summary.json")
    move_if_present(artifact_root / "patchcore_sweep_results.csv", results_dir / "patchcore_sweep_results.csv")
    move_if_present(artifact_root / "patchcore_sweep_summary.json", results_dir / "patchcore_sweep_summary.json")
    move_if_present(artifact_root / "evaluation" / "summary.json", evaluation_dir / "summary.json")
    move_if_present(artifact_root / "evaluation" / "val_scores.csv", evaluation_dir / "val_scores.csv")
    move_if_present(artifact_root / "evaluation" / "test_scores.csv", evaluation_dir / "test_scores.csv")
    move_if_present(artifact_root / "evaluation" / "threshold_sweep.csv", evaluation_dir / "threshold_sweep.csv")

    sweep_summary_path = resolve_json_path(
        artifact_root,
        ["results/patchcore_sweep_summary.json", "patchcore_sweep_summary.json"],
    )
    if sweep_summary_path is None:
        raise FileNotFoundError(f"Could not find sweep summary for {artifact_root}")

    sweep_summary = load_json(sweep_summary_path)
    selected_variant_name = str(sweep_summary["selected_variant_name"])

    variant_names = {selected_variant_name}
    for result in sweep_summary.get("results", []):
        name = result.get("name")
        if name:
            variant_names.add(str(name))

    for variant_name in sorted(variant_names):
        variant_root = artifact_root / variant_name
        if not variant_root.exists():
            continue

        variant_checkpoints_dir = variant_root / "checkpoints"
        variant_results_dir = variant_root / "results"
        variant_evaluation_dir = variant_results_dir / "evaluation"
        variant_checkpoints_dir.mkdir(exist_ok=True)
        variant_results_dir.mkdir(exist_ok=True)
        variant_evaluation_dir.mkdir(exist_ok=True)

        move_if_present(variant_root / "best_model.pt", variant_checkpoints_dir / "best_model.pt")
        move_if_present(variant_root / "last_model.pt", variant_checkpoints_dir / "last_model.pt")
        move_if_present(variant_root / "summary.json", variant_results_dir / "summary.json")
        move_if_present(variant_root / "evaluation" / "val_scores.csv", variant_evaluation_dir / "val_scores.csv")
        move_if_present(variant_root / "evaluation" / "test_scores.csv", variant_evaluation_dir / "test_scores.csv")
        move_if_present(variant_root / "evaluation" / "threshold_sweep.csv", variant_evaluation_dir / "threshold_sweep.csv")
        move_if_present(variant_root / "evaluation" / "summary.json", variant_evaluation_dir / "summary.json")

    selected_root = artifact_root / selected_variant_name
    copy_if_missing(selected_root / "checkpoints" / "best_model.pt", checkpoints_dir / "best_model.pt")
    copy_if_missing(selected_root / "checkpoints" / "last_model.pt", checkpoints_dir / "last_model.pt")
    copy_if_missing(selected_root / "results" / "summary.json", results_dir / "summary.json")
    copy_if_missing(selected_root / "results" / "evaluation" / "val_scores.csv", evaluation_dir / "val_scores.csv")
    copy_if_missing(selected_root / "results" / "evaluation" / "test_scores.csv", evaluation_dir / "test_scores.csv")
    copy_if_missing(selected_root / "results" / "evaluation" / "threshold_sweep.csv", evaluation_dir / "threshold_sweep.csv")

    selected_summary_path = selected_root / "results" / "summary.json"
    if selected_summary_path.exists() and not (evaluation_dir / "summary.json").exists():
        selected_summary = load_json(selected_summary_path)
        (evaluation_dir / "summary.json").write_text(json.dumps(selected_summary, indent=2), encoding="utf-8")

    return {
        "artifact_root": artifact_root,
        "selected_variant_name": selected_variant_name,
    }


def build_notebook(experiment: dict) -> nbformat.NotebookNode:
    title = experiment["title"]
    short_name = experiment["short_name"]
    config_path = experiment["config_path"]
    dataset_notebook = experiment["dataset_notebook"]
    dataset_config = experiment["dataset_config"]
    artifact_root = f"{Path(config_path).parent.as_posix()}/artifacts/{experiment['artifact_name']}"
    plot_prefix = experiment["plot_prefix"]

    cells = [
        new_markdown_cell(
            textwrap.dedent(
                f"""\
                # {title}

                This notebook is the curated submission entry point for the `x64` {short_name} sweep.

                Default behavior:
                - reuse the saved sweep summary, selected checkpoint, and saved evaluation CSVs when they already exist
                - rerun the full PatchCore sweep only if `FORCE_RERUN_SWEEP = True`
                - display the main figures in the notebook and save them into the experiment-local artifact folder
                """
            )
        ),
        new_markdown_cell(
            textwrap.dedent(
                f"""\
                ## Submission Context

                - Dataset notebook: `{dataset_notebook}`
                - Dataset config: `{dataset_config}`
                - Experiment config: `{config_path}`
                - Artifact root: `{artifact_root}`
                - Canonical checkpoint path: `{artifact_root}/checkpoints/best_model.pt`
                - Default mode: artifact-first evaluation and visualization
                """
            )
        ),
        new_markdown_cell("## Imports and Repo Discovery\n\nThis cell resolves the repository root, imports the shared PatchCore utilities, and prepares the plotting/data libraries used throughout the notebook."),
        new_code_cell(
            textwrap.dedent(
                """\
                from pathlib import Path
                import copy
                import json
                import random
                import shutil
                import sys

                import matplotlib.pyplot as plt
                import numpy as np
                import pandas as pd
                import torch
                from IPython.display import display
                from torch.utils.data import DataLoader

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
                from wafer_defect.evaluation import summarize_threshold_metrics, sweep_threshold_metrics
                from wafer_defect.models.patchcore import PatchCoreModel
                from wafer_defect.training.patchcore import build_memory_subset, collect_memory_bank
                """
            )
        ),
        new_markdown_cell("## Configuration and Artifact Policy\n\nThis cell loads the local experiment config and defines the small PatchCore sweep. `FORCE_RERUN_SWEEP` stays `False` by default so a grader can run the notebook without retraining."),
        new_code_cell(
            textwrap.dedent(
                f"""\
                CONFIG_PATH = REPO_ROOT / "{config_path}"
                config = load_toml(CONFIG_PATH)

                PATCHCORE_SWEEP = [
                    {{"name": "mean_mb10k", "memory_bank_size": 10_000, "reduction": "mean", "topk_ratio": 0.10}},
                    {{"name": "mean_mb50k", "memory_bank_size": 50_000, "reduction": "mean", "topk_ratio": 0.10}},
                    {{"name": "topk_mb10k_r005", "memory_bank_size": 10_000, "reduction": "topk_mean", "topk_ratio": 0.05}},
                    {{"name": "topk_mb50k_r005", "memory_bank_size": 50_000, "reduction": "topk_mean", "topk_ratio": 0.05}},
                    {{"name": "topk_mb50k_r010", "memory_bank_size": 50_000, "reduction": "topk_mean", "topk_ratio": 0.10}},
                    {{"name": "max_mb50k", "memory_bank_size": 50_000, "reduction": "max", "topk_ratio": 0.10}},
                ]

                FORCE_RERUN_SWEEP = False
                SELECTED_VARIANT_NAME = None
                AUTO_SELECT_METRIC = "f1"
                RENDER_ALL_SAVED_VARIANTS = True
                VARIANTS_TO_RENDER: list[str] = []

                artifact_root = REPO_ROOT / config["run"]["output_dir"]
                checkpoints_dir = artifact_root / "checkpoints"
                results_dir = artifact_root / "results"
                evaluation_dir = results_dir / "evaluation"
                plots_dir = artifact_root / "plots"
                VARIANT_COLOR_VAL = "{experiment['color_val']}"
                VARIANT_COLOR_NORMAL = "{experiment['color_normal']}"
                VARIANT_COLOR_ANOMALY = "{experiment['color_anomaly']}"
                VARIANT_COLOR_DEFECT = "{experiment['color_defect']}"

                for directory in [artifact_root, checkpoints_dir, results_dir, evaluation_dir, plots_dir]:
                    directory.mkdir(parents=True, exist_ok=True)

                config
                """
            )
        ),
        new_markdown_cell("## Runtime Setup\n\nThis cell fixes random seeds and resolves the compute device so reruns stay consistent."),
        new_code_cell(
            textwrap.dedent(
                """\
                def set_seed(seed: int) -> None:
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)


                def resolve_device(device_name: str) -> torch.device:
                    if device_name == "auto":
                        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    return torch.device(device_name)


                seed = int(config["run"]["seed"])
                set_seed(seed)
                device = resolve_device(config["training"].get("device", "auto"))
                device
                """
            )
        ),
        new_markdown_cell("## Dataset Loading\n\nThis cell loads the shared processed `64x64` benchmark split and prepares dataloaders for validation and test scoring."),
        new_code_cell(
            textwrap.dedent(
                """\
                image_size = int(config["data"].get("image_size", 64))
                batch_size = int(config["data"].get("batch_size", 64))
                num_workers = int(config["data"].get("num_workers", 0))
                metadata_path = REPO_ROOT / config["data"]["metadata_csv"]
                metadata = pd.read_csv(metadata_path)

                display(metadata.head())
                display(metadata["split"].value_counts().rename_axis("split").to_frame("count"))
                display(metadata["is_anomaly"].value_counts().rename_axis("is_anomaly").to_frame("count"))

                train_dataset = WaferMapDataset(metadata_path, split="train", image_size=image_size)
                val_dataset = WaferMapDataset(metadata_path, split="val", image_size=image_size)
                test_dataset = WaferMapDataset(metadata_path, split="test", image_size=image_size)

                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

                print(f"train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
                """
            )
        ),
        new_markdown_cell("## Model and Sweep Helpers\n\nThese helpers build the PatchCore model, reuse cached memory banks during reruns, and load previously saved variant outputs when retraining is not requested."),
        new_code_cell(
            textwrap.dedent(
                """\
                base_model_config = config["model"]
                memory_bank_cache: dict[int, dict[str, object]] = {}


                def build_patchcore_model(*, reduction: str, topk_ratio: float) -> PatchCoreModel:
                    return PatchCoreModel(
                        image_size=image_size,
                        backbone_type=str(base_model_config.get("backbone_type", "resnet18")),
                        use_batchnorm=bool(base_model_config.get("use_batchnorm", True)),
                        pretrained=bool(base_model_config.get("pretrained", True)),
                        freeze_backbone=bool(base_model_config.get("freeze_backbone", True)),
                        backbone_input_size=int(base_model_config.get("backbone_input_size", 224)),
                        normalize_imagenet=bool(base_model_config.get("normalize_imagenet", True)),
                        reduction=str(reduction),
                        topk_ratio=float(topk_ratio),
                        query_chunk_size=int(base_model_config.get("query_chunk_size", 2048)),
                        memory_chunk_size=int(base_model_config.get("memory_chunk_size", 8192)),
                    ).to(device)


                def get_memory_bank_info(memory_bank_size: int) -> dict[str, object]:
                    if memory_bank_size not in memory_bank_cache:
                        temp_model = build_patchcore_model(reduction="mean", topk_ratio=0.10)
                        memory_subset = build_memory_subset(
                            train_dataset,
                            memory_bank_size=memory_bank_size,
                            patches_per_image=temp_model.patches_per_image,
                            seed=seed,
                        )
                        memory_loader = DataLoader(memory_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                        memory_bank = collect_memory_bank(
                            model=temp_model,
                            dataloader=memory_loader,
                            device=device,
                            target_size=memory_bank_size,
                            seed=seed,
                        )
                        memory_bank_cache[memory_bank_size] = {
                            "memory_bank": memory_bank.cpu(),
                            "memory_subset_images": len(memory_subset),
                            "patches_per_image": int(temp_model.patches_per_image),
                            "feature_dim": int(temp_model.feature_dim),
                        }
                        del temp_model
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    return memory_bank_cache[memory_bank_size]


                def collect_scores(model: PatchCoreModel, dataloader: DataLoader) -> pd.DataFrame:
                    rows = []
                    model.eval()
                    with torch.inference_mode():
                        for inputs, labels in dataloader:
                            inputs = inputs.to(device)
                            scores = model(inputs).cpu().numpy()
                            labels = labels.cpu().numpy()
                            for score, label in zip(scores, labels):
                                rows.append({"score": float(score), "is_anomaly": int(label)})
                    return pd.DataFrame(rows)


                def first_existing_path(*candidates: Path) -> Path | None:
                    for candidate in candidates:
                        if candidate.exists():
                            return candidate
                    return None


                def load_json_file(path: Path) -> dict:
                    return json.loads(path.read_text(encoding="utf-8"))


                def load_variant_outputs(variant_name: str) -> dict[str, object]:
                    run_output_dir = artifact_root / variant_name
                    summary_path = first_existing_path(
                        run_output_dir / "results" / "summary.json",
                        run_output_dir / "summary.json",
                    )
                    val_scores_path = first_existing_path(
                        run_output_dir / "results" / "evaluation" / "val_scores.csv",
                        run_output_dir / "evaluation" / "val_scores.csv",
                    )
                    test_scores_path = first_existing_path(
                        run_output_dir / "results" / "evaluation" / "test_scores.csv",
                        run_output_dir / "evaluation" / "test_scores.csv",
                    )
                    threshold_sweep_path = first_existing_path(
                        run_output_dir / "results" / "evaluation" / "threshold_sweep.csv",
                        run_output_dir / "evaluation" / "threshold_sweep.csv",
                    )

                    if not all([summary_path, val_scores_path, test_scores_path, threshold_sweep_path]):
                        missing = {
                            "summary_path": summary_path,
                            "val_scores_path": val_scores_path,
                            "test_scores_path": test_scores_path,
                            "threshold_sweep_path": threshold_sweep_path,
                        }
                        raise FileNotFoundError(f"Missing cached files for variant {variant_name}: {missing}")

                    summary = load_json_file(summary_path)
                    val_scores_df = pd.read_csv(val_scores_path)
                    test_scores_df = pd.read_csv(test_scores_path)
                    threshold_sweep_df = pd.read_csv(threshold_sweep_path)

                    threshold = float(summary["threshold"])
                    labels = test_scores_df["is_anomaly"].to_numpy()
                    scores = test_scores_df["score"].to_numpy()
                    metrics = summarize_threshold_metrics(labels, scores, threshold)
                    best_sweep = threshold_sweep_df.sort_values("f1", ascending=False).iloc[0].to_dict()

                    return {
                        "summary": summary,
                        "val_scores_df": val_scores_df,
                        "test_scores_df": test_scores_df,
                        "threshold_sweep_df": threshold_sweep_df,
                        "metrics": metrics,
                        "best_sweep": best_sweep,
                        "output_dir": run_output_dir,
                    }


                def compute_failure_tables(test_metadata: pd.DataFrame, test_scores_df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                    analysis_df = test_metadata.reset_index(drop=True).copy()
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


                def render_variant_artifacts(variant_name: str, variant_payload: dict[str, object]) -> dict[str, str]:
                    summary = variant_payload["summary"]
                    val_scores_df = variant_payload["val_scores_df"]
                    test_scores_df = variant_payload["test_scores_df"]
                    threshold_sweep_df = variant_payload["threshold_sweep_df"]
                    metrics = variant_payload["metrics"]
                    best_sweep = variant_payload["best_sweep"]
                    threshold = float(summary["threshold"])

                    variant_root = variant_payload["output_dir"]
                    variant_plots_dir = variant_root / "plots"
                    variant_results_dir = variant_root / "results"
                    variant_evaluation_dir = variant_results_dir / "evaluation"
                    variant_plots_dir.mkdir(parents=True, exist_ok=True)
                    variant_evaluation_dir.mkdir(parents=True, exist_ok=True)

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

                    analysis_df, error_summary_df, defect_recall_df = compute_failure_tables(
                        test_dataset.metadata,
                        test_scores_df,
                        threshold,
                    )
                    analysis_df.to_csv(variant_evaluation_dir / "analysis_with_predictions.csv", index=False)
                    error_summary_df.reset_index().to_csv(variant_evaluation_dir / "error_summary.csv", index=False)
                    defect_recall_df.reset_index().to_csv(variant_evaluation_dir / "defect_recall.csv", index=False)

                    top_defects_df = defect_recall_df.head(10).reset_index()
                    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                    axes[0].bar(error_summary_df.index.astype(str), error_summary_df["count"], color=VARIANT_COLOR_ANOMALY)
                    axes[0].set_title(f"Prediction Outcome Counts\\n{variant_name}")
                    axes[0].set_ylabel("count")

                    axes[1].barh(top_defects_df["defect_type"], top_defects_df["recall"], color=VARIANT_COLOR_DEFECT)
                    axes[1].set_xlim(0.0, 1.0)
                    axes[1].set_title("Top Defect-Type Recall")
                    axes[1].set_xlabel("recall")
                    axes[1].invert_yaxis()

                    plt.tight_layout()
                    fig.savefig(variant_plots_dir / "defect_breakdown.png", dpi=200, bbox_inches="tight")
                    plt.close(fig)

                    return {
                        "plots_dir": str(variant_plots_dir),
                        "evaluation_dir": str(variant_evaluation_dir),
                    }


                def resolve_variant_names_to_render(sweep_results_df: pd.DataFrame, selected_variant_name: str) -> list[str]:
                    names = []
                    if RENDER_ALL_SAVED_VARIANTS:
                        names.extend(sweep_results_df["name"].astype(str).tolist())
                    names.extend([str(name) for name in VARIANTS_TO_RENDER])
                    names.append(str(selected_variant_name))
                    ordered = []
                    seen = set()
                    for name in names:
                        if name not in seen:
                            ordered.append(name)
                            seen.add(name)
                    return ordered


                print(
                    {
                        "sweep_variants": [variant["name"] for variant in PATCHCORE_SWEEP],
                        "artifact_root": str(artifact_root),
                        "backbone_type": str(base_model_config.get("backbone_type", "resnet18")),
                        "default_mode": "artifact reuse" if not FORCE_RERUN_SWEEP else "full sweep rerun",
                    }
                )
                """
            )
        ),
        new_markdown_cell("## Load Cached Sweep Results or Rerun the Sweep\n\nThis cell is the heart of the notebook. It loads the saved sweep and selected variant by default, but it can rerun the whole PatchCore sweep and refresh the artifacts if you explicitly set `FORCE_RERUN_SWEEP = True`."),
        new_code_cell(
            textwrap.dedent(
                """\
                sweep_results_path = first_existing_path(
                    results_dir / "patchcore_sweep_results.csv",
                    artifact_root / "patchcore_sweep_results.csv",
                )
                sweep_summary_path = first_existing_path(
                    results_dir / "patchcore_sweep_summary.json",
                    artifact_root / "patchcore_sweep_summary.json",
                )

                use_cached_outputs = (
                    not FORCE_RERUN_SWEEP
                    and sweep_results_path is not None
                    and sweep_summary_path is not None
                )

                variant_outputs = {}
                ranking_metrics = [AUTO_SELECT_METRIC, *[metric for metric in ["f1", "auroc", "auprc"] if metric != AUTO_SELECT_METRIC]]

                if use_cached_outputs:
                    sweep_results_df = pd.read_csv(sweep_results_path).sort_values(ranking_metrics, ascending=False).reset_index(drop=True)
                    sweep_summary = load_json_file(sweep_summary_path)
                    selected_variant_name = str(
                        SELECTED_VARIANT_NAME
                        or sweep_summary.get("selected_variant_name")
                        or sweep_results_df.iloc[0]["name"]
                    )
                    selected_variant = load_variant_outputs(selected_variant_name)
                    variant_outputs[selected_variant_name] = selected_variant
                    print(f"Loaded cached PatchCore sweep results from {sweep_results_path}")
                else:
                    sweep_rows = []
                    for variant in PATCHCORE_SWEEP:
                        variant_name = str(variant["name"])
                        run_output_dir = artifact_root / variant_name
                        variant_checkpoints_dir = run_output_dir / "checkpoints"
                        variant_results_dir = run_output_dir / "results"
                        variant_evaluation_dir = variant_results_dir / "evaluation"

                        for directory in [run_output_dir, variant_checkpoints_dir, variant_results_dir, variant_evaluation_dir]:
                            directory.mkdir(parents=True, exist_ok=True)

                        print(f"\\n=== PatchCore variant: {variant_name} ===")
                        model = build_patchcore_model(reduction=variant["reduction"], topk_ratio=variant["topk_ratio"])
                        memory_info = get_memory_bank_info(int(variant["memory_bank_size"]))
                        model.set_memory_bank(memory_info["memory_bank"].to(device))

                        run_config = copy.deepcopy(config)
                        run_config["run"]["output_dir"] = run_output_dir.relative_to(REPO_ROOT).as_posix()
                        run_config["model"]["memory_bank_size"] = int(variant["memory_bank_size"])
                        run_config["model"]["reduction"] = str(variant["reduction"])
                        run_config["model"]["topk_ratio"] = float(variant["topk_ratio"])

                        checkpoint = {
                            "model_state_dict": model.state_dict(),
                            "config": run_config,
                            "memory_bank_size": int(model.memory_bank.shape[0]),
                            "feature_dim": int(model.feature_dim),
                            "patches_per_image": int(model.patches_per_image),
                            "backbone_type": str(base_model_config.get("backbone_type", "resnet18")),
                        }
                        torch.save(checkpoint, variant_checkpoints_dir / "best_model.pt")
                        torch.save(checkpoint, variant_checkpoints_dir / "last_model.pt")

                        val_scores_df = collect_scores(model, val_loader)
                        test_scores_df = collect_scores(model, test_loader)

                        val_normal_scores = val_scores_df.loc[val_scores_df["is_anomaly"] == 0, "score"]
                        threshold = float(val_normal_scores.quantile(0.95))
                        labels = test_scores_df["is_anomaly"].to_numpy()
                        scores = test_scores_df["score"].to_numpy()

                        metrics = summarize_threshold_metrics(labels, scores, threshold)
                        threshold_sweep_df, best_sweep = sweep_threshold_metrics(labels, scores)

                        summary = {
                            "name": variant_name,
                            "memory_bank_size": int(model.memory_bank.shape[0]),
                            "memory_subset_images": int(memory_info["memory_subset_images"]),
                            "patches_per_image": int(memory_info["patches_per_image"]),
                            "feature_dim": int(memory_info["feature_dim"]),
                            "reduction": str(variant["reduction"]),
                            "topk_ratio": float(variant["topk_ratio"]),
                            "threshold": threshold,
                            "precision": float(metrics["precision"]),
                            "recall": float(metrics["recall"]),
                            "f1": float(metrics["f1"]),
                            "auroc": float(metrics["auroc"]),
                            "auprc": float(metrics["auprc"]),
                            "best_sweep_threshold": float(best_sweep["threshold"]),
                            "best_sweep_precision": float(best_sweep["precision"]),
                            "best_sweep_recall": float(best_sweep["recall"]),
                            "best_sweep_f1": float(best_sweep["f1"]),
                            "predicted_anomalies": int(metrics["predicted_anomalies"]),
                            "output_dir": run_output_dir.relative_to(REPO_ROOT).as_posix(),
                        }

                        val_scores_df.to_csv(variant_evaluation_dir / "val_scores.csv", index=False)
                        test_scores_df.to_csv(variant_evaluation_dir / "test_scores.csv", index=False)
                        threshold_sweep_df.to_csv(variant_evaluation_dir / "threshold_sweep.csv", index=False)
                        (variant_results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

                        sweep_rows.append(summary)
                        variant_outputs[variant_name] = {
                            "summary": summary,
                            "val_scores_df": val_scores_df,
                            "test_scores_df": test_scores_df,
                            "threshold_sweep_df": threshold_sweep_df,
                            "metrics": metrics,
                            "best_sweep": best_sweep,
                            "output_dir": run_output_dir,
                        }

                        del model
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    sweep_results_df = pd.DataFrame(sweep_rows).sort_values(ranking_metrics, ascending=False).reset_index(drop=True)
                    sweep_results_df.to_csv(results_dir / "patchcore_sweep_results.csv", index=False)

                    selected_variant_name = str(SELECTED_VARIANT_NAME or sweep_results_df.iloc[0]["name"])
                    selected_row = sweep_results_df.loc[sweep_results_df["name"] == selected_variant_name].iloc[0].to_dict()
                    sweep_summary = {
                        "selected_variant_name": selected_variant_name,
                        "auto_select_metric": AUTO_SELECT_METRIC,
                        "ranking_metrics": ranking_metrics,
                        "selected_row": selected_row,
                        "results": sweep_rows,
                    }
                    (results_dir / "patchcore_sweep_summary.json").write_text(json.dumps(sweep_summary, indent=2), encoding="utf-8")

                    selected_variant = variant_outputs[selected_variant_name]
                    shutil.copy2(selected_variant["output_dir"] / "checkpoints" / "best_model.pt", checkpoints_dir / "best_model.pt")
                    shutil.copy2(selected_variant["output_dir"] / "checkpoints" / "last_model.pt", checkpoints_dir / "last_model.pt")
                    shutil.copy2(selected_variant["output_dir"] / "results" / "summary.json", results_dir / "summary.json")
                    selected_variant["val_scores_df"].to_csv(evaluation_dir / "val_scores.csv", index=False)
                    selected_variant["test_scores_df"].to_csv(evaluation_dir / "test_scores.csv", index=False)
                    selected_variant["threshold_sweep_df"].to_csv(evaluation_dir / "threshold_sweep.csv", index=False)
                    (evaluation_dir / "summary.json").write_text(json.dumps(selected_variant["summary"], indent=2), encoding="utf-8")

                    print(f"Finished rerunning PatchCore sweep. Selected variant: {selected_variant_name}")

                display(sweep_results_df)
                """
            )
        ),
        new_markdown_cell("## Selected Variant Metrics\n\nThis cell loads the selected variant into a compact metrics table and confusion matrix so the notebook immediately shows the main benchmark numbers."),
        new_code_cell(
            textwrap.dedent(
                """\
                selected_variant = variant_outputs[selected_variant_name]
                output_dir = selected_variant["output_dir"]
                summary = selected_variant["summary"]
                val_scores_df = selected_variant["val_scores_df"]
                test_scores_df = selected_variant["test_scores_df"]
                threshold_sweep_df = selected_variant["threshold_sweep_df"]
                metrics = selected_variant["metrics"]
                best_sweep = selected_variant["best_sweep"]
                threshold = float(summary["threshold"])
                confusion_df = pd.DataFrame(
                    metrics["confusion_matrix"],
                    index=["true_normal", "true_anomaly"],
                    columns=["pred_normal", "pred_anomaly"],
                )

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

                display(metrics_df)
                display(confusion_df)
                print(f"Selected variant: {selected_variant_name}")
                print(f"Canonical checkpoint: {checkpoints_dir / 'best_model.pt'}")
                print(f"Best sweep threshold: {best_sweep['threshold']:.6f} | precision={best_sweep['precision']:.4f}, recall={best_sweep['recall']:.4f}, f1={best_sweep['f1']:.4f}")
                """
            )
        ),
        new_markdown_cell("## Visualizations\n\nThese plots summarize the sweep ranking, score distributions, threshold sensitivity, and confusion matrix. Each figure is displayed inline and saved under `plots/`."),
        new_code_cell(
            textwrap.dedent(
                f"""\
                plot_df = sweep_results_df.copy()
                plot_df["label"] = (
                    plot_df["name"]
                    + "\\n"
                    + plot_df["reduction"]
                    + ", mb="
                    + plot_df["memory_bank_size"].astype(str)
                )

                fig, axes = plt.subplots(1, 2, figsize=(16, 5))
                axes[0].barh(plot_df["label"], plot_df["f1"], color="{experiment['color_f1']}")
                axes[0].set_title("{plot_prefix}: Validation-Threshold F1")
                axes[0].set_xlabel("F1")
                axes[0].invert_yaxis()

                axes[1].barh(plot_df["label"], plot_df["auroc"], color="{experiment['color_auroc']}")
                axes[1].set_title("{plot_prefix}: AUROC")
                axes[1].set_xlabel("AUROC")
                axes[1].invert_yaxis()

                plt.tight_layout()
                fig.savefig(plots_dir / "variant_comparison_metrics.png", dpi=200, bbox_inches="tight")
                plt.show()
                plt.close(fig)

                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                axes[0].hist(val_scores_df["score"], bins=30, alpha=0.85, color="{experiment['color_val']}")
                axes[0].axvline(threshold, color="red", linestyle="--", label=f"threshold={{threshold:.4f}}")
                axes[0].set_title(f"Validation Normal Score Distribution\\n{{selected_variant_name}}")
                axes[0].legend()

                axes[1].hist(test_scores_df[test_scores_df["is_anomaly"] == 0]["score"], bins=30, alpha=0.7, label="normal", color="{experiment['color_normal']}")
                axes[1].hist(test_scores_df[test_scores_df["is_anomaly"] == 1]["score"], bins=30, alpha=0.7, label="anomaly", color="{experiment['color_anomaly']}")
                axes[1].axvline(threshold, color="red", linestyle="--", label=f"threshold={{threshold:.4f}}")
                axes[1].set_title(f"Test Score Distribution\\n{{selected_variant_name}}")
                axes[1].legend()

                plt.tight_layout()
                fig.savefig(plots_dir / "score_distribution.png", dpi=200, bbox_inches="tight")
                plt.show()
                plt.close(fig)

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(threshold_sweep_df["threshold"], threshold_sweep_df["precision"], label="precision")
                ax.plot(threshold_sweep_df["threshold"], threshold_sweep_df["recall"], label="recall")
                ax.plot(threshold_sweep_df["threshold"], threshold_sweep_df["f1"], label="f1")
                ax.axvline(threshold, color="red", linestyle="--", label=f"validation threshold = {{threshold:.4f}}")
                ax.axvline(best_sweep["threshold"], color="green", linestyle=":", label=f"best sweep threshold = {{best_sweep['threshold']:.4f}}")
                ax.set_title(f"Threshold Sweep on Test Split\\n{{selected_variant_name}}")
                ax.set_xlabel("Anomaly-score threshold")
                ax.set_ylabel("Metric value")
                ax.legend()

                plt.tight_layout()
                fig.savefig(plots_dir / "threshold_sweep.png", dpi=200, bbox_inches="tight")
                plt.show()
                plt.close(fig)

                cm_array = np.asarray(metrics["confusion_matrix"], dtype=float)
                fig, ax = plt.subplots(figsize=(5, 4))
                im = ax.imshow(cm_array, cmap="Blues")
                ax.set_xticks([0, 1], labels=["pred_normal", "pred_anomaly"])
                ax.set_yticks([0, 1], labels=["true_normal", "true_anomaly"])
                ax.set_title(f"Confusion Matrix\\n{{selected_variant_name}}")
                for row_idx in range(cm_array.shape[0]):
                    for col_idx in range(cm_array.shape[1]):
                        ax.text(col_idx, row_idx, int(cm_array[row_idx, col_idx]), ha="center", va="center", color="black")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                plt.tight_layout()
                fig.savefig(plots_dir / "confusion_matrix.png", dpi=200, bbox_inches="tight")
                plt.show()
                plt.close(fig)
                """
            )
        ),
        new_markdown_cell("## Failure Analysis Tables\n\nThis cell attaches the selected PatchCore scores to the test metadata, saves the analysis tables into `results/evaluation/`, and surfaces the main false-positive and false-negative patterns."),
        new_code_cell(
            textwrap.dedent(
                """\
                analysis_df = test_dataset.metadata.reset_index(drop=True).copy()
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

                analysis_df.to_csv(evaluation_dir / "analysis_with_predictions.csv", index=False)
                error_summary_df.reset_index().to_csv(evaluation_dir / "error_summary.csv", index=False)
                defect_recall_df.reset_index().to_csv(evaluation_dir / "defect_recall.csv", index=False)

                print(f"Failure analysis variant: {selected_variant_name}")
                display(error_summary_df)
                display(defect_recall_df)
                analysis_df.head()
                """
            )
        ),
        new_markdown_cell("## Failure Analysis Plot\n\nThis final figure turns the saved failure tables into a quick visual summary for the report and for notebook readers."),
        new_code_cell(
            textwrap.dedent(
                f"""\
                top_defects_df = defect_recall_df.head(10).reset_index()

                fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                axes[0].bar(error_summary_df.index.astype(str), error_summary_df["count"], color="{experiment['color_anomaly']}")
                axes[0].set_title(f"Prediction Outcome Counts\\n{{selected_variant_name}}")
                axes[0].set_ylabel("count")

                axes[1].barh(top_defects_df["defect_type"], top_defects_df["recall"], color="{experiment['color_defect']}")
                axes[1].set_xlim(0.0, 1.0)
                axes[1].set_title("Top Defect-Type Recall")
                axes[1].set_xlabel("recall")
                axes[1].invert_yaxis()

                plt.tight_layout()
                fig.savefig(plots_dir / "defect_breakdown.png", dpi=200, bbox_inches="tight")
                plt.show()
                plt.close(fig)
                """
            )
        ),
        new_markdown_cell("## Cached Variant Rendering\n\nThis section can populate each saved variant folder with its own plots and failure-analysis CSVs using the cached score files. By default it renders all saved sweep variants without retraining."),
        new_code_cell(
            textwrap.dedent(
                """\
                variant_names_to_render = resolve_variant_names_to_render(sweep_results_df, selected_variant_name)
                rendered_variant_rows = []

                for variant_name in variant_names_to_render:
                    if variant_name not in variant_outputs:
                        variant_outputs[variant_name] = load_variant_outputs(variant_name)
                    render_info = render_variant_artifacts(variant_name, variant_outputs[variant_name])
                    rendered_variant_rows.append(
                        {
                            "variant_name": variant_name,
                            "plots_dir": render_info["plots_dir"],
                            "evaluation_dir": render_info["evaluation_dir"],
                        }
                    )

                rendered_variants_df = pd.DataFrame(rendered_variant_rows)
                display(rendered_variants_df)
                """
            )
        ),
        new_markdown_cell("## Saved Outputs\n\nThis cell prints the final artifact locations so the notebook doubles as reproducibility documentation."),
        new_code_cell(
            textwrap.dedent(
                """\
                saved_outputs = {
                    "checkpoint_dir": str(checkpoints_dir),
                    "results_dir": str(results_dir),
                    "evaluation_dir": str(evaluation_dir),
                    "plots_dir": str(plots_dir),
                    "selected_variant_name": selected_variant_name,
                    "rendered_variants": rendered_variants_df["variant_name"].tolist(),
                }
                saved_outputs
                """
            )
        ),
    ]

    notebook = new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3.x"}
    return notebook


def write_readme(experiment: dict, selected_variant_name: str) -> None:
    folder = experiment["folder"]
    readme_path = folder / "README.md"
    artifact_root = f"{folder.relative_to(REPO_ROOT).as_posix()}/artifacts/{experiment['artifact_name']}"
    readme_path.write_text(
        textwrap.dedent(
            f"""\
            # {experiment["short_name"]} (`x64`)

            This is the curated submission-facing PatchCore sweep notebook for the `64x64` benchmark split.

            What this notebook does:
            - loads the processed `50k / 5%` benchmark dataset
            - reuses the saved PatchCore sweep artifacts by default
            - loads the selected local checkpoint from `checkpoints/best_model.pt`
            - regenerates the main plots and saves them into `plots/`
            - saves failure-analysis CSVs into `results/evaluation/`
            - can repopulate each saved variant folder with cached plots and failure-analysis outputs

            Key files:
            - notebook: `{folder.relative_to(REPO_ROOT).as_posix()}/notebook.ipynb`
            - train config: `{folder.relative_to(REPO_ROOT).as_posix()}/train_config.toml`
            - data config: `{folder.relative_to(REPO_ROOT).as_posix()}/data_config.toml`
            - artifact root: `{artifact_root}`

            Artifact layout:
            - `checkpoints/`: canonical selected checkpoint for this run
            - `results/`: sweep summary, selected-run summary, and evaluation CSVs
            - `plots/`: saved figures recreated by the notebook
            - per-variant folders such as `{selected_variant_name}/`: saved checkpoints and evaluation files for each sweep option

            Default behavior:
            - open the notebook and run top to bottom
            - it will reuse cached artifacts unless `FORCE_RERUN_SWEEP = True`
            - it will render per-variant plots from cached CSVs unless you turn `RENDER_ALL_SAVED_VARIANTS` off
            """
        ),
        encoding="utf-8",
    )


def update_family_readme() -> None:
    path = REPO_ROOT / "experiments" / "anomaly_detection" / "patchcore" / "README.md"
    content = path.read_text(encoding="utf-8")
    if "## Artifact Layout" in content:
        return
    content = content.rstrip() + "\n\n## Artifact Layout\n\nFor the curated `x64/main` notebooks (`ae_bn`, `resnet18`, and `resnet50`), the artifact folders now follow the same submission-friendly structure used in the other families:\n\n- `checkpoints/`: the canonical selected checkpoint for the notebook\n- `results/`: sweep summaries, evaluation CSVs, and failure-analysis tables\n- `plots/`: figures regenerated by the notebook\n- per-variant subfolders: the saved checkpoints and evaluation files for each PatchCore sweep option\n"
    path.write_text(content, encoding="utf-8")


def main() -> None:
    for experiment in EXPERIMENTS:
        state = reorganize_artifacts(experiment)
        notebook = build_notebook(experiment)
        nbformat.write(notebook, experiment["folder"] / "notebook.ipynb")
        write_readme(experiment, state["selected_variant_name"])

    update_family_readme()
    print("Curated PatchCore x64 main notebooks and reorganized artifacts.")


if __name__ == "__main__":
    main()
