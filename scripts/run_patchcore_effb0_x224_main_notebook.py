"""Run the PatchCore EfficientNet-B0 x224 source notebook headlessly and normalize outputs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NOTEBOOK_PATH = Path("experiments/archive/anomaly_50k_duplicates/21_patchcore_efficientnet_b0_5pct.ipynb")
ARTIFACT_OUTPUT_DIR = Path(
    "experiments/anomaly_detection/patchcore/efficientnet_b0/x224/main/artifacts/patchcore_efficientnet_b0_5pct"
)
CODE_CELL_INDICES = list(range(1, 12))


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
    raise FileNotFoundError("Could not locate repo root for the PatchCore EfficientNet-B0 x224 runner.")


def display(obj: object) -> None:
    print(obj, flush=True)


def _plot_score_distribution(test_scores_df: pd.DataFrame, threshold: float, plot_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(
        test_scores_df.loc[test_scores_df["is_anomaly"] == 0, "score"],
        bins=40,
        alpha=0.65,
        label="normal",
        color="#8d99ae",
        density=True,
    )
    ax.hist(
        test_scores_df.loc[test_scores_df["is_anomaly"] == 1, "score"],
        bins=40,
        alpha=0.55,
        label="anomaly",
        color="#e76f51",
        density=True,
    )
    ax.axvline(threshold, color="#264653", linestyle="--", linewidth=2, label=f"threshold={threshold:.4f}")
    ax.set_title("EfficientNet-B0 PatchCore Score Distribution")
    ax.set_xlabel("anomaly score")
    ax.set_ylabel("density")
    ax.legend()
    plt.tight_layout()
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_threshold_sweep(threshold_sweep_df: pd.DataFrame, plot_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.0))
    threshold_col = threshold_sweep_df.columns[0]
    ax.plot(threshold_sweep_df[threshold_col], threshold_sweep_df["precision"], label="precision")
    ax.plot(threshold_sweep_df[threshold_col], threshold_sweep_df["recall"], label="recall")
    ax.plot(threshold_sweep_df[threshold_col], threshold_sweep_df["f1"], label="f1")
    ax.set_title("Threshold Sweep")
    ax.legend()
    plt.tight_layout()
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_defect_breakdown(defect_df: pd.DataFrame, plot_path: Path) -> None:
    if defect_df.empty:
        return
    plot_df = defect_df.sort_values(["recall", "count"], ascending=[False, False]).head(12)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.barh(plot_df["defect_type"], plot_df["recall"], color="#2a9d8f")
    ax.set_xlim(0.0, 1.0)
    ax.invert_yaxis()
    ax.set_title("Defect Recall")
    plt.tight_layout()
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _standardize_outputs(globals_dict: dict[str, Any], output_dir: Path, repo_root: Path) -> dict[str, Any]:
    checkpoints_dir = output_dir / "checkpoints"
    results_dir = output_dir / "results"
    evaluation_dir = results_dir / "evaluation"
    plots_dir = output_dir / "plots"
    for path in [checkpoints_dir, results_dir, evaluation_dir, plots_dir]:
        path.mkdir(parents=True, exist_ok=True)

    root_summary = output_dir / "summary.json"
    root_ckpt = output_dir / "best_model.pt"
    eval_src_dir = output_dir / "evaluation"

    summary = globals_dict["summary"]
    metrics = globals_dict["metrics"]
    threshold = float(globals_dict["threshold"])
    val_scores_df: pd.DataFrame = globals_dict["val_scores_df"]
    test_scores_df: pd.DataFrame = globals_dict["test_scores_df"]
    threshold_sweep_df: pd.DataFrame = globals_dict["threshold_sweep_df"]
    test_analysis_df: pd.DataFrame = globals_dict["test_analysis_df"]

    defect_breakdown_df = (
        test_analysis_df[test_analysis_df["is_anomaly"] == 1]
        .groupby("defect_type")
        .agg(
            count=("is_anomaly", "size"),
            detected=("predicted_anomaly", "sum"),
            recall=("predicted_anomaly", "mean"),
            mean_score=("score", "mean"),
        )
        .reset_index()
        .sort_values(["recall", "count"], ascending=[False, False])
    )
    confusion_df = pd.DataFrame(
        np.array(metrics["confusion_matrix"], dtype=int),
        index=["true_normal", "true_anomaly"],
        columns=["pred_normal", "pred_anomaly"],
    )

    if root_ckpt.exists():
        shutil.copy2(root_ckpt, checkpoints_dir / "best_model.pt")
    if root_summary.exists():
        shutil.copy2(root_summary, results_dir / "summary.json")
    else:
        (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if eval_src_dir.exists():
        for filename in ["val_scores.csv", "test_scores.csv", "threshold_sweep.csv"]:
            src = eval_src_dir / filename
            if src.exists():
                shutil.copy2(src, evaluation_dir / filename)

    val_scores_df.to_csv(evaluation_dir / "val_scores.csv", index=False)
    test_scores_df.to_csv(evaluation_dir / "test_scores.csv", index=False)
    threshold_sweep_df.to_csv(evaluation_dir / "threshold_sweep.csv", index=False)
    defect_breakdown_df.to_csv(evaluation_dir / "defect_breakdown.csv", index=False)
    confusion_df.to_csv(evaluation_dir / "confusion_matrix.csv")
    (evaluation_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    config_snapshot = dict(summary.get("config", {}))
    config_snapshot["metadata_csv"] = str(Path(globals_dict["METADATA_PATH"]).resolve().relative_to(repo_root).as_posix())
    (results_dir / "config_snapshot.json").write_text(json.dumps(config_snapshot, indent=2), encoding="utf-8")

    _plot_score_distribution(test_scores_df, threshold, plots_dir / "score_distribution.png")
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
    metadata_path: str,
    output_dir: str,
    num_workers: int,
    batch_size: int,
) -> dict[str, Any]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    globals_dict: dict[str, Any] = {"__name__": "__main__", "display": display}
    repo_root = Path.cwd().resolve()
    print(f"[patchcore-effb0-x224-main] executing {len(CODE_CELL_INDICES)} code cells from {notebook_path}", flush=True)

    for step, cell_index in enumerate(CODE_CELL_INDICES, start=1):
        cell = notebook["cells"][cell_index]
        source = "".join(cell.get("source", []))
        if cell_index == 1:
            source = source.replace(
                """cwd = Path.cwd().resolve()
candidate_roots = [cwd, *cwd.parents]
REPO_ROOT = None
for candidate in candidate_roots:
    if (candidate / "src" / "wafer_defect").exists() and (candidate / "configs").exists():
        REPO_ROOT = candidate
        break

if REPO_ROOT is None:
    raise RuntimeError("Could not locate repo root containing src/wafer_defect and configs/")
""",
                f'REPO_ROOT = Path(r"{repo_root.as_posix()}")\n',
            )
        print(
            f"[patchcore-effb0-x224-main] starting code cell {step}/{len(CODE_CELL_INDICES)} "
            f"(notebook index {cell_index})",
            flush=True,
        )
        code = compile(source, f"{notebook_path.name}::cell_{cell_index}", "exec")
        exec(code, globals_dict)

        if cell_index == 2:
            globals_dict["IMAGE_SIZE"] = 224
            globals_dict["MODEL_INPUT_SIZE"] = 224
            globals_dict["METADATA_PATH"] = Path(metadata_path)
            globals_dict["NUM_WORKERS"] = int(num_workers)
            globals_dict["BATCH_SIZE"] = int(batch_size)
            globals_dict["OUTPUT_DIR"] = Path(output_dir).resolve()
            globals_dict["OUTPUT_DIR"].mkdir(parents=True, exist_ok=True)

        print(f"[patchcore-effb0-x224-main] finished code cell {step}/{len(CODE_CELL_INDICES)}", flush=True)

    output_dir_path = Path(globals_dict["OUTPUT_DIR"]).resolve()
    repo_root = Path(globals_dict["REPO_ROOT"]).resolve()
    manifest = _standardize_outputs(globals_dict, output_dir_path, repo_root)
    (output_dir_path / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-path", default="/root/project/data/processed/x224/wm811k/metadata_50k_5pct.csv")
    parser.add_argument("--output-dir", default=str(ARTIFACT_OUTPUT_DIR))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    repo_root = resolve_repo_root()
    os.chdir(repo_root)
    os.environ.setdefault("MPLBACKEND", "Agg")
    print(
        f"[patchcore-effb0-x224-main] repo root: {repo_root}; metadata_path={args.metadata_path}; "
        f"output_dir={args.output_dir}; num_workers={args.num_workers}; batch_size={args.batch_size}",
        flush=True,
    )
    manifest = execute_notebook(
        (repo_root / NOTEBOOK_PATH).resolve(),
        metadata_path=args.metadata_path,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
