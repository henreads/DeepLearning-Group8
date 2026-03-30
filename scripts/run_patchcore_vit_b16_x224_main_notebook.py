"""Run the PatchCore ViT-B/16 x224 source notebook headlessly and normalize outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NOTEBOOK_PATH = Path("experiments/anomaly_detection/patchcore/vit_b16/x224/one_layer_no_defect_tuning/notebook.ipynb")
ARTIFACT_OUTPUT_DIR = Path(
    "experiments/anomaly_detection/patchcore/vit_b16/x224/main/artifacts/patchcore_vit_b16_5pct/main_5pct"
)
CODE_CELL_INDICES = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]


def resolve_repo_root() -> Path:
    script_path = Path(__file__).resolve()
    candidates = [script_path.parent, *script_path.parents, Path.cwd().resolve(), *Path.cwd().resolve().parents]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / NOTEBOOK_PATH).exists() and (candidate / "experiments").exists() and (candidate / "scripts").exists():
            return candidate
    raise FileNotFoundError("Could not locate repo root for the PatchCore ViT-B/16 x224 runner.")


def display(obj: object) -> None:
    print(obj, flush=True)


def _summarize_threshold_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    labels = labels.astype(int)
    predictions = predictions.astype(int)
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    tp = int(((labels == 1) & (predictions == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "predicted_anomalies": int(tp + fp),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def _build_threshold_sweep(
    tune_normal_scores_z: np.ndarray,
    scores_z: np.ndarray,
    labels: np.ndarray,
    mu: float,
    std: float,
    percentile_min: float,
    percentile_max: float,
    percentile_steps: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for percentile in np.linspace(percentile_min, percentile_max, percentile_steps):
        threshold_z = float(np.percentile(tune_normal_scores_z, percentile))
        threshold_raw = float(mu + threshold_z * std)
        predictions = (scores_z > threshold_z).astype(int)
        metrics = _summarize_threshold_metrics(labels, predictions)
        rows.append(
            {
                "percentile": float(percentile),
                "threshold_z": threshold_z,
                "threshold_raw": threshold_raw,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "predicted_anomalies": metrics["predicted_anomalies"],
            }
        )
    return pd.DataFrame(rows)


def _plot_score_distribution(scores_df: pd.DataFrame, threshold_z: float, plot_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(
        scores_df.loc[scores_df["is_anomaly"] == 0, "score_z"],
        bins=40,
        alpha=0.65,
        label="normal",
        color="#8d99ae",
        density=True,
    )
    ax.hist(
        scores_df.loc[scores_df["is_anomaly"] == 1, "score_z"],
        bins=40,
        alpha=0.55,
        label="anomaly",
        color="#e76f51",
        density=True,
    )
    ax.axvline(threshold_z, color="#264653", linestyle="--", linewidth=2, label=f"threshold_z={threshold_z:.4f}")
    ax.set_title("ViT-B/16 PatchCore Score Distribution")
    ax.set_xlabel("anomaly score (z)")
    ax.set_ylabel("density")
    ax.legend()
    plt.tight_layout()
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_threshold_sweep(threshold_sweep_df: pd.DataFrame, plot_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.0))
    ax.plot(threshold_sweep_df["percentile"], threshold_sweep_df["precision"], label="precision")
    ax.plot(threshold_sweep_df["percentile"], threshold_sweep_df["recall"], label="recall")
    ax.plot(threshold_sweep_df["percentile"], threshold_sweep_df["f1"], label="f1")
    ax.set_title("Threshold Sweep by Percentile")
    ax.legend()
    plt.tight_layout()
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_defect_breakdown(defect_df: pd.DataFrame, plot_path: Path) -> None:
    if defect_df.empty:
        return
    plot_df = defect_df.sort_values(["recall", "count"], ascending=[False, False]).head(12)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.barh(plot_df["failure_label"], plot_df["recall"], color="#81b29a")
    ax.set_xlim(0.0, 1.0)
    ax.invert_yaxis()
    ax.set_title("Defect Recall")
    plt.tight_layout()
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _standardize_outputs(globals_dict: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    checkpoints_dir = output_dir / "checkpoints"
    results_dir = output_dir / "results"
    evaluation_dir = results_dir / "evaluation"
    umap_dir = results_dir / "umap"
    plots_dir = output_dir / "plots"
    for path in [checkpoints_dir, results_dir, evaluation_dir, umap_dir, plots_dir]:
        path.mkdir(parents=True, exist_ok=True)

    threshold_z = float(globals_dict["threshold_z"])
    threshold_raw = float(globals_dict["threshold_raw"])
    mu = float(globals_dict["mu"])
    std = float(globals_dict["std"])
    roc_auc = float(globals_dict["roc_auc"])
    cm = np.array(globals_dict["cm"], dtype=int)
    y_true = np.asarray(globals_dict["y_true"]).astype(int)
    scores_z = np.asarray(globals_dict["scores"], dtype=float)
    tune_normal_scores_z = np.asarray(globals_dict["tune_normal_scores_z"], dtype=float)
    tmp: pd.DataFrame = globals_dict["tmp"].copy()

    val_scores_df = pd.DataFrame(
        {
            "score_z": tune_normal_scores_z,
            "score_raw": mu + tune_normal_scores_z * std,
            "is_anomaly": np.zeros_like(tune_normal_scores_z, dtype=int),
        }
    )
    test_scores_df = pd.DataFrame(
        {
            "score_z": scores_z,
            "score_raw": mu + scores_z * std,
            "is_anomaly": y_true,
        }
    )
    predictions = (scores_z > threshold_z).astype(int)
    metrics = _summarize_threshold_metrics(y_true, predictions)
    defect_breakdown_df = (
        tmp.groupby("failure_label")
        .agg(count=("detected", "count"), detected=("detected", "sum"), recall=("detected", "mean"), mean_score=("score", "mean"))
        .reset_index()
        .sort_values(["recall", "count"], ascending=[False, False])
    )

    threshold_sweep_df = _build_threshold_sweep(
        tune_normal_scores_z=tune_normal_scores_z,
        scores_z=scores_z,
        labels=y_true,
        mu=mu,
        std=std,
        percentile_min=float(globals_dict["THRESHOLD_PERCENTILE_MIN"]),
        percentile_max=float(globals_dict["THRESHOLD_PERCENTILE_MAX"]),
        percentile_steps=int(globals_dict["THRESHOLD_PERCENTILE_STEPS"]),
    )

    checkpoint_src = Path(globals_dict["MODEL_EXPORT_PATH"])
    if checkpoint_src.exists() and checkpoint_src.resolve() != (checkpoints_dir / "best_model.pt").resolve():
        checkpoint_src.replace(checkpoints_dir / "best_model.pt")
    metrics_src = Path(globals_dict["METRICS_EXPORT_PATH"])
    if metrics_src.exists() and metrics_src.resolve() != (evaluation_dir / "evaluation_metrics.json").resolve():
        metrics_src.replace(evaluation_dir / "evaluation_metrics.json")

    summary = {
        "name": "vit_b16_one_layer_patchcore_x224",
        "threshold_z": threshold_z,
        "threshold_raw": threshold_raw,
        "train_score_mu": mu,
        "train_score_std": std,
        "roc_auc_z": roc_auc,
        "checkpoint": str(checkpoints_dir / "best_model.pt"),
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    val_scores_df.to_csv(evaluation_dir / "val_scores.csv", index=False)
    test_scores_df.to_csv(evaluation_dir / "test_scores.csv", index=False)
    threshold_sweep_df.to_csv(evaluation_dir / "threshold_sweep.csv", index=False)
    defect_breakdown_df.to_csv(evaluation_dir / "defect_breakdown.csv", index=False)
    pd.DataFrame(cm, index=["true_normal", "true_anomaly"], columns=["pred_normal", "pred_anomaly"]).to_csv(
        evaluation_dir / "confusion_matrix.csv"
    )
    (evaluation_dir / "summary.json").write_text(
        json.dumps(
            {
                "threshold_z": threshold_z,
                "threshold_raw": threshold_raw,
                "metrics_at_validation_threshold": metrics,
                "roc_auc_z": roc_auc,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    raw_umap_csv = results_dir / "umap_test_embeddings.csv"
    if raw_umap_csv.exists():
        raw_umap_csv.replace(umap_dir / "umap_test_embeddings.csv")
        umap_df = pd.read_csv(umap_dir / "umap_test_embeddings.csv")
        umap_summary = {
            "rows": int(len(umap_df)),
            "normal_rows": int((umap_df["label"] == 0).sum()) if "label" in umap_df else None,
            "anomaly_rows": int((umap_df["label"] == 1).sum()) if "label" in umap_df else None,
        }
        (umap_dir / "umap_summary.json").write_text(json.dumps(umap_summary, indent=2), encoding="utf-8")

    _plot_score_distribution(test_scores_df, threshold_z, plots_dir / "score_distribution.png")
    _plot_threshold_sweep(threshold_sweep_df, plots_dir / "threshold_sweep.png")
    _plot_defect_breakdown(defect_breakdown_df, plots_dir / "defect_breakdown.png")

    return {
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoints_dir / "best_model.pt"),
        "summary": str(results_dir / "summary.json"),
    }


def execute_notebook(
    notebook_path: Path,
    *,
    raw_pickle: str,
    output_dir: str,
    num_workers: int,
    batch_size: int,
) -> dict[str, Any]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    globals_dict: dict[str, Any] = {"__name__": "__main__", "display": display}
    print(f"[patchcore-vit-b16-x224-main] executing {len(CODE_CELL_INDICES)} code cells from {notebook_path}", flush=True)

    for step, cell_index in enumerate(CODE_CELL_INDICES, start=1):
        cell = notebook["cells"][cell_index]
        source = "".join(cell.get("source", []))
        print(
            f"[patchcore-vit-b16-x224-main] starting code cell {step}/{len(CODE_CELL_INDICES)} "
            f"(notebook index {cell_index})",
            flush=True,
        )
        code = compile(source, f"{notebook_path.name}::cell_{cell_index}", "exec")
        exec(code, globals_dict)

        if cell_index == 6:
            output_dir_path = Path(output_dir).resolve()
            checkpoints_dir = output_dir_path / "checkpoints"
            plots_dir = output_dir_path / "plots"
            results_dir = output_dir_path / "results"
            for path in [checkpoints_dir, plots_dir, results_dir]:
                path.mkdir(parents=True, exist_ok=True)
            globals_dict["DATA_PATH"] = raw_pickle
            globals_dict["ARTIFACT_DIR"] = str(output_dir_path)
            globals_dict["CHECKPOINTS_DIR"] = str(checkpoints_dir)
            globals_dict["PLOTS_DIR"] = str(plots_dir)
            globals_dict["RESULTS_DIR"] = str(results_dir)
            globals_dict["MODEL_EXPORT_PATH"] = str(checkpoints_dir / "best_model.pt")
            globals_dict["METRICS_EXPORT_PATH"] = str(results_dir / "evaluation_metrics.json")
            globals_dict["NUM_WORKERS"] = int(num_workers)
            globals_dict["BATCH_SIZE"] = int(batch_size)

        print(f"[patchcore-vit-b16-x224-main] finished code cell {step}/{len(CODE_CELL_INDICES)}", flush=True)

    output_dir_path = Path(output_dir).resolve()
    manifest = _standardize_outputs(globals_dict, output_dir_path)
    (output_dir_path / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-pickle", default="/root/project/data/raw/LSWMD.pkl")
    parser.add_argument("--output-dir", default=str(ARTIFACT_OUTPUT_DIR))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    repo_root = resolve_repo_root()
    os.chdir(repo_root)
    os.environ.setdefault("MPLBACKEND", "Agg")
    print(
        f"[patchcore-vit-b16-x224-main] repo root: {repo_root}; raw_pickle={args.raw_pickle}; "
        f"output_dir={args.output_dir}; num_workers={args.num_workers}; batch_size={args.batch_size}",
        flush=True,
    )
    manifest = execute_notebook(
        (repo_root / NOTEBOOK_PATH).resolve(),
        raw_pickle=args.raw_pickle,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
