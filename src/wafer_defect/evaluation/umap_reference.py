"""Helpers for reference-fit UMAP analysis and UMAP-space KNN thresholding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from wafer_defect.evaluation.reconstruction_metrics import (
    summarize_threshold_metrics,
    sweep_threshold_metrics,
)


def fit_reference_umap(
    train_normal_features: np.ndarray,
    *,
    umap_module: Any,
    random_state: int = 42,
    pca_components: int = 50,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
) -> tuple[PCA | None, Any]:
    """Fit PCA + UMAP on train-normal reference features only."""
    if train_normal_features.ndim != 2:
        raise ValueError(f"Expected 2D train-normal features, got shape={train_normal_features.shape}")
    if len(train_normal_features) < 2:
        raise ValueError("Need at least two train-normal points to fit reference UMAP.")

    pca = None
    umap_input = train_normal_features
    if train_normal_features.shape[1] > pca_components:
        n_components = min(pca_components, train_normal_features.shape[0] - 1, train_normal_features.shape[1])
        if n_components >= 2:
            pca = PCA(n_components=n_components, random_state=random_state)
            umap_input = pca.fit_transform(train_normal_features)

    reducer = umap_module.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        transform_seed=random_state,
    )
    reducer.fit(umap_input)
    return pca, reducer


def transform_reference_umap(features: np.ndarray, pca: PCA | None, reducer: Any) -> np.ndarray:
    """Project features into a pre-fitted PCA + UMAP reference space."""
    transformed = features if pca is None else pca.transform(features)
    return reducer.transform(transformed)


def knn_reference_scores(reference_points: np.ndarray, query_points: np.ndarray, k: int = 15) -> np.ndarray:
    """Mean Euclidean distance to the k nearest normal-reference points."""
    if len(reference_points) == 0:
        raise ValueError("Reference point set is empty.")
    k = max(1, min(k, len(reference_points)))
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nbrs.fit(reference_points)
    distances, _ = nbrs.kneighbors(query_points, return_distance=True)
    return distances.mean(axis=1).astype(np.float32)


def knn_reference_scores_leave_one_out(reference_points: np.ndarray, k: int = 15) -> np.ndarray:
    """Reference-set self-distances with leave-one-out KNN."""
    if len(reference_points) <= 1:
        return np.zeros(len(reference_points), dtype=np.float32)
    k = max(1, min(k, len(reference_points) - 1))
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(reference_points)
    distances, _ = nbrs.kneighbors(reference_points, return_distance=True)
    return distances[:, 1:].mean(axis=1).astype(np.float32)


def sample_indices(indices: np.ndarray, max_points: int | None, rng: np.random.Generator) -> np.ndarray:
    """Sample indices without replacement while preserving deterministic seeds."""
    if max_points is None or len(indices) <= max_points:
        return np.asarray(indices, dtype=np.int64)
    return np.asarray(rng.choice(indices, size=max_points, replace=False), dtype=np.int64)


def build_reference_umap_dataframe(
    *,
    train_points: np.ndarray,
    val_points: np.ndarray,
    val_labels: np.ndarray,
    test_points: np.ndarray,
    test_labels: np.ndarray,
    val_model_scores: np.ndarray | None = None,
    test_model_scores: np.ndarray | None = None,
    val_umap_knn_scores: np.ndarray | None = None,
    test_umap_knn_scores: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build a unified exported dataframe for reference-fit UMAP outputs."""
    rows: list[pd.DataFrame] = []
    rows.append(
        pd.DataFrame(
            {
                "umap_1": train_points[:, 0],
                "umap_2": train_points[:, 1],
                "split_label": "train_reference",
                "is_anomaly": 0,
                "model_score": np.nan,
                "umap_knn_score": np.nan,
            }
        )
    )
    rows.append(
        pd.DataFrame(
            {
                "umap_1": val_points[:, 0],
                "umap_2": val_points[:, 1],
                "split_label": np.where(val_labels == 0, "val_normal", "val_anomaly"),
                "is_anomaly": val_labels.astype(int),
                "model_score": np.nan if val_model_scores is None else val_model_scores.astype(float),
                "umap_knn_score": np.nan if val_umap_knn_scores is None else val_umap_knn_scores.astype(float),
            }
        )
    )
    rows.append(
        pd.DataFrame(
            {
                "umap_1": test_points[:, 0],
                "umap_2": test_points[:, 1],
                "split_label": np.where(test_labels == 0, "test_normal", "test_anomaly"),
                "is_anomaly": test_labels.astype(int),
                "model_score": np.nan if test_model_scores is None else test_model_scores.astype(float),
                "umap_knn_score": np.nan if test_umap_knn_scores is None else test_umap_knn_scores.astype(float),
            }
        )
    )
    return pd.concat(rows, ignore_index=True)


def plot_reference_umap(
    umap_df: pd.DataFrame,
    *,
    split_plot_path: str | Path,
    score_plot_path: str | Path | None = None,
    title_prefix: str = "Reference-Fit UMAP",
) -> None:
    """Save standard split-colored and KNN-score-colored UMAP plots."""
    split_plot_path = Path(split_plot_path)
    split_plot_path.parent.mkdir(parents=True, exist_ok=True)
    score_plot = Path(score_plot_path) if score_plot_path is not None else None
    if score_plot is not None:
        score_plot.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    style_map = {
        "train_reference": dict(s=8, alpha=0.20, label="train_reference", c="#b0b0b0"),
        "val_normal":      dict(s=10, alpha=0.25, label="val_normal",      c="#c0c0c0"),
        "val_anomaly":     dict(s=12, alpha=0.70, label="val_anomaly",     c="#e63946"),
        "test_normal":     dict(s=10, alpha=0.35, label="test_normal",     c="#aaaaaa"),
        "test_anomaly":    dict(s=14, alpha=0.75, label="test_anomaly",    c="#e63946"),
    }
    for split_name, group in umap_df.groupby("split_label"):
        style = style_map.get(split_name, dict(s=10, alpha=0.4, label=split_name))
        ax.scatter(group["umap_1"], group["umap_2"], linewidths=0, **style)
    ax.set_title(f"{title_prefix} by Split")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(split_plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    if score_plot is not None:
        fig, ax = plt.subplots(figsize=(9, 7))
        score_df = umap_df[umap_df["umap_knn_score"].notna()].copy()
        sc = ax.scatter(
            score_df["umap_1"],
            score_df["umap_2"],
            c=score_df["umap_knn_score"],
            cmap="viridis",
            s=10,
            alpha=0.7,
            linewidths=0,
        )
        ax.set_title(f"{title_prefix} by UMAP-KNN Score")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("UMAP-KNN Score")
        plt.tight_layout()
        fig.savefig(score_plot, dpi=220, bbox_inches="tight")
        plt.close(fig)


def export_reference_umap_bundle(
    *,
    output_dir: str | Path,
    umap_module: Any,
    train_normal_embeddings: np.ndarray,
    val_embeddings: np.ndarray,
    val_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    val_model_scores: np.ndarray | None = None,
    test_model_scores: np.ndarray | None = None,
    threshold_quantile: float = 0.95,
    random_state: int = 42,
    pca_components: int = 50,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    knn_k: int = 15,
    metric: str = "euclidean",
    max_train_reference: int | None = None,
    max_val_normal: int | None = None,
    max_test_normal: int | None = None,
    max_test_anomaly: int | None = None,
    title_prefix: str = "Reference-Fit UMAP",
    points_filename: str = "umap_points.csv",
    split_plot_filename: str = "umap_by_split.png",
    score_plot_filename: str = "umap_by_score.png",
    summary_filename: str = "umap_summary.json",
    sweep_filename: str = "umap_knn_threshold_sweep.csv",
) -> dict[str, Any]:
    """Fit a train-normal reference UMAP and export KNN-threshold artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_state)

    train_idx = sample_indices(np.arange(len(train_normal_embeddings)), max_train_reference, rng)
    val_normal_idx = sample_indices(np.where(val_labels == 0)[0], max_val_normal, rng)
    val_anomaly_idx = np.asarray(np.where(val_labels == 1)[0], dtype=np.int64)
    test_normal_idx = sample_indices(np.where(test_labels == 0)[0], max_test_normal, rng)
    test_anomaly_idx = sample_indices(np.where(test_labels == 1)[0], max_test_anomaly, rng)

    train_reference_embeddings = train_normal_embeddings[train_idx]
    pca, reducer = fit_reference_umap(
        train_reference_embeddings,
        umap_module=umap_module,
        random_state=random_state,
        pca_components=pca_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
    )

    train_points = transform_reference_umap(train_reference_embeddings, pca, reducer)
    val_points = transform_reference_umap(val_embeddings, pca, reducer)
    test_points = transform_reference_umap(test_embeddings, pca, reducer)

    reference_points = train_points
    val_umap_knn_scores = knn_reference_scores(reference_points, val_points, k=knn_k)
    test_umap_knn_scores = knn_reference_scores(reference_points, test_points, k=knn_k)

    deployed_threshold = float(pd.Series(val_umap_knn_scores[val_labels == 0]).quantile(threshold_quantile))
    umap_knn_metrics = summarize_threshold_metrics(test_labels, test_umap_knn_scores, deployed_threshold)
    umap_knn_sweep_df, best_sweep = sweep_threshold_metrics(test_labels, test_umap_knn_scores)

    points_df = build_reference_umap_dataframe(
        train_points=train_points,
        val_points=val_points,
        val_labels=val_labels,
        test_points=test_points,
        test_labels=test_labels,
        val_model_scores=val_model_scores,
        test_model_scores=test_model_scores,
        val_umap_knn_scores=val_umap_knn_scores,
        test_umap_knn_scores=test_umap_knn_scores,
    )
    umap_knn_sweep_df.to_csv(output_dir / sweep_filename, index=False)

    val_plot_idx = np.concatenate([val_normal_idx, val_anomaly_idx]) if len(val_anomaly_idx) else val_normal_idx
    test_plot_idx = (
        np.concatenate([test_normal_idx, test_anomaly_idx])
        if (len(test_normal_idx) + len(test_anomaly_idx))
        else np.empty(0, dtype=np.int64)
    )
    plot_df = build_reference_umap_dataframe(
        train_points=train_points,
        val_points=val_points[val_plot_idx],
        val_labels=val_labels[val_plot_idx],
        test_points=test_points[test_plot_idx],
        test_labels=test_labels[test_plot_idx],
        val_model_scores=None if val_model_scores is None else val_model_scores[val_plot_idx],
        test_model_scores=None if test_model_scores is None else test_model_scores[test_plot_idx],
        val_umap_knn_scores=val_umap_knn_scores[val_plot_idx],
        test_umap_knn_scores=test_umap_knn_scores[test_plot_idx],
    )
    plot_df.to_csv(output_dir / points_filename, index=False)
    plot_reference_umap(
        plot_df,
        split_plot_path=output_dir / split_plot_filename,
        score_plot_path=output_dir / score_plot_filename,
        title_prefix=title_prefix,
    )

    summary = {
        "threshold_quantile": float(threshold_quantile),
        "counts": {
            "train_reference": int(len(train_idx)),
            "val_total": int(len(val_embeddings)),
            "val_normal": int((val_labels == 0).sum()),
            "val_anomaly": int((val_labels == 1).sum()),
            "test_total": int(len(test_embeddings)),
            "test_normal": int((test_labels == 0).sum()),
            "test_anomaly": int((test_labels == 1).sum()),
        },
        "plot_counts": {
            "train_reference": int(len(train_idx)),
            "val_normal": int(len(val_normal_idx)),
            "val_anomaly": int(len(val_anomaly_idx)),
            "test_normal": int(len(test_normal_idx)),
            "test_anomaly": int(len(test_anomaly_idx)),
        },
        "umap_params": {
            "random_state": int(random_state),
            "pca_components": int(min(pca_components, train_reference_embeddings.shape[0], train_reference_embeddings.shape[1])),
            "n_neighbors": int(n_neighbors),
            "min_dist": float(min_dist),
            "metric": metric,
            "knn_k": int(knn_k),
        },
        "umap_knn_threshold": float(deployed_threshold),
        "umap_knn_metrics": umap_knn_metrics,
        "umap_knn_best_threshold_sweep": best_sweep,
        "outputs": {
            "points_csv": str((output_dir / points_filename).resolve()),
            "split_plot": str((output_dir / split_plot_filename).resolve()),
            "score_plot": str((output_dir / score_plot_filename).resolve()),
            "threshold_sweep_csv": str((output_dir / sweep_filename).resolve()),
        },
    }
    (output_dir / summary_filename).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "summary": summary,
        "points_df": plot_df,
        "full_points_df": points_df,
        "threshold_sweep_df": umap_knn_sweep_df,
        "val_umap_knn_scores": val_umap_knn_scores,
        "test_umap_knn_scores": test_umap_knn_scores,
        "train_reference_points": train_points,
        "val_points": val_points,
        "test_points": test_points,
    }
