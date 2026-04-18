from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def resolve_repo_root(start: Path | None = None) -> Path:
    start_path = (start or Path(__file__).resolve()).resolve()
    for candidate in [start_path, *start_path.parents]:
        if (candidate / "experiments").exists() and (candidate / "scripts").exists():
            return candidate
    raise FileNotFoundError("Could not locate repo root for holdout evaluation.")


def exec_notebook_code_cells(
    notebook_path: Path,
    cell_indices: list[int],
    globals_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    scope: dict[str, Any] = {"__name__": "__main__"}
    if globals_dict:
        scope.update(globals_dict)

    for index in cell_indices:
        cell = notebook["cells"][index]
        if cell.get("cell_type") != "code":
            raise ValueError(f"Notebook cell {index} is not code.")
        code = compile("".join(cell.get("source", [])), f"{notebook_path.name}::cell_{index}", "exec")
        exec(code, scope)
    return scope


def build_defect_breakdown(metadata_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    defect_rows = metadata_df.loc[metadata_df["is_anomaly"] == 1].copy()
    if defect_rows.empty:
        return pd.DataFrame(
            columns=["defect_type", "count", "detected", "missed", "recall", "mean_score", "median_score"]
        )

    defect_rows["predicted_anomaly"] = (defect_rows["score"] > threshold).astype(int)
    breakdown = (
        defect_rows.groupby("defect_type")
        .agg(
            count=("defect_type", "size"),
            detected=("predicted_anomaly", "sum"),
            mean_score=("score", "mean"),
            median_score=("score", "median"),
        )
        .reset_index()
    )
    breakdown["detected"] = breakdown["detected"].astype(int)
    breakdown["missed"] = breakdown["count"] - breakdown["detected"]
    breakdown["recall"] = breakdown["detected"] / breakdown["count"]
    return breakdown.sort_values(["recall", "count", "defect_type"], ascending=[True, False, True]).reset_index(
        drop=True
    )


def write_confusion_csv(path: Path, confusion_matrix: list[list[int]]) -> None:
    frame = pd.DataFrame(
        confusion_matrix,
        index=["true_normal", "true_anomaly"],
        columns=["pred_normal", "pred_anomaly"],
    )
    frame.to_csv(path)


def save_threshold_sweep_plot(sweep_df: pd.DataFrame, output_path: Path, *, title: str) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(sweep_df["threshold"], sweep_df["f1"], color="#1d3557", linewidth=2)
    axes[0].set_title(f"{title}: F1")
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("F1")
    axes[0].grid(alpha=0.25, linestyle="--")

    axes[1].plot(sweep_df["threshold"], sweep_df["precision"], label="precision", color="#2a9d8f")
    axes[1].plot(sweep_df["threshold"], sweep_df["recall"], label="recall", color="#e76f51")
    axes[1].set_title(f"{title}: Precision / Recall")
    axes[1].set_xlabel("Threshold")
    axes[1].grid(alpha=0.25, linestyle="--")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_defect_breakdown_plot(defect_df: pd.DataFrame, output_path: Path, *, title: str) -> None:
    if defect_df.empty:
        return

    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.bar(defect_df["defect_type"], defect_df["recall"], color="#457b9d")
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Recall")
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", alpha=0.25, linestyle="--")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def to_repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)
