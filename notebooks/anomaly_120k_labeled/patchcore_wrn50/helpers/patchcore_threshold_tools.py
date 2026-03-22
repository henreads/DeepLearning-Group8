"""Threshold and review-policy analysis helpers for PatchCore score bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_variant_artifacts(bundle_dir: str | Path, variant_name: str) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    variant_dir = Path(bundle_dir).resolve() / str(variant_name)
    summary = json.loads((variant_dir / "summary.json").read_text(encoding="utf-8"))
    val_scores_df = pd.read_csv(variant_dir / "val_scores.csv")
    test_scores_df = pd.read_csv(variant_dir / "test_scores.csv")
    return summary, val_scores_df, test_scores_df


def summarize_threshold_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predicted = scores >= float(threshold)

    tp = int(np.sum((predicted == 1) & (labels == 1)))
    fp = int(np.sum((predicted == 1) & (labels == 0)))
    tn = int(np.sum((predicted == 0) & (labels == 0)))
    fn = int(np.sum((predicted == 0) & (labels == 1)))

    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    false_positive_rate = 0.0 if fp + tn == 0 else fp / (fp + tn)

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(false_positive_rate),
        "predicted_anomalies": int(tp + fp),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def build_threshold_sweep(scores_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        scores_df.groupby(["score", "is_anomaly"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "neg_count", 1: "pos_count"})
        .reset_index()
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    if "neg_count" not in grouped.columns:
        grouped["neg_count"] = 0
    if "pos_count" not in grouped.columns:
        grouped["pos_count"] = 0

    total_pos = int(grouped["pos_count"].sum())
    total_neg = int(grouped["neg_count"].sum())

    grouped["tp"] = grouped["pos_count"].cumsum()
    grouped["fp"] = grouped["neg_count"].cumsum()
    grouped["fn"] = total_pos - grouped["tp"]
    grouped["tn"] = total_neg - grouped["fp"]
    grouped["precision"] = grouped["tp"] / (grouped["tp"] + grouped["fp"])
    grouped["precision"] = grouped["precision"].fillna(0.0)
    grouped["recall"] = grouped["tp"] / (grouped["tp"] + grouped["fn"])
    grouped["recall"] = grouped["recall"].fillna(0.0)
    grouped["f1"] = 2.0 * grouped["precision"] * grouped["recall"] / (grouped["precision"] + grouped["recall"])
    grouped["f1"] = grouped["f1"].fillna(0.0)
    grouped["false_positive_rate"] = grouped["fp"] / (grouped["fp"] + grouped["tn"])
    grouped["false_positive_rate"] = grouped["false_positive_rate"].fillna(0.0)
    grouped["predicted_anomalies"] = grouped["tp"] + grouped["fp"]

    columns = [
        "score",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "predicted_anomalies",
        "tp",
        "fp",
        "tn",
        "fn",
    ]
    return grouped[columns].rename(columns={"score": "threshold"}).sort_values("threshold").reset_index(drop=True)


def build_auto_normal_sweep(scores_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        scores_df.groupby(["score", "is_anomaly"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "neg_count", 1: "pos_count"})
        .reset_index()
        .sort_values("score", ascending=True)
        .reset_index(drop=True)
    )
    if "neg_count" not in grouped.columns:
        grouped["neg_count"] = 0
    if "pos_count" not in grouped.columns:
        grouped["pos_count"] = 0

    grouped["tn"] = grouped["neg_count"].cumsum()
    grouped["fn"] = grouped["pos_count"].cumsum()
    grouped["auto_normal_count"] = grouped["tn"] + grouped["fn"]
    grouped["auto_normal_anomaly_rate"] = grouped["fn"] / grouped["auto_normal_count"]
    grouped["auto_normal_anomaly_rate"] = grouped["auto_normal_anomaly_rate"].fillna(0.0)
    grouped["npv"] = grouped["tn"] / grouped["auto_normal_count"]
    grouped["npv"] = grouped["npv"].fillna(0.0)

    columns = ["score", "auto_normal_count", "auto_normal_anomaly_rate", "npv", "tn", "fn"]
    return grouped[columns].rename(columns={"score": "threshold"}).reset_index(drop=True)


def _select_best_row(
    sweep_df: pd.DataFrame,
    *,
    min_recall: float | None = None,
    max_false_positive_rate: float | None = None,
    min_precision: float | None = None,
    sort_columns: tuple[str, ...] = ("f1", "precision", "recall"),
) -> dict[str, Any]:
    eligible_df = sweep_df.copy()
    if min_recall is not None:
        eligible_df = eligible_df[eligible_df["recall"] >= float(min_recall)]
    if max_false_positive_rate is not None:
        eligible_df = eligible_df[eligible_df["false_positive_rate"] <= float(max_false_positive_rate)]
    if min_precision is not None:
        eligible_df = eligible_df[eligible_df["precision"] >= float(min_precision)]
    if eligible_df.empty:
        raise ValueError("No threshold satisfied the requested constraints.")
    return eligible_df.sort_values(list(sort_columns), ascending=False).iloc[0].to_dict()


def build_single_threshold_policy_table(
    val_scores_df: pd.DataFrame,
    test_scores_df: pd.DataFrame,
    *,
    current_threshold: float,
    min_recall: float = 0.70,
    max_false_positive_rate: float = 0.03,
) -> pd.DataFrame:
    val_sweep_df = build_threshold_sweep(val_scores_df)
    policies = [
        ("current_threshold", float(current_threshold)),
        ("validation_f1", float(_select_best_row(val_sweep_df)["threshold"])),
        (
            f"recall_floor_{min_recall:.2f}",
            float(
                _select_best_row(
                    val_sweep_df,
                    min_recall=min_recall,
                    sort_columns=("precision", "f1", "recall"),
                )["threshold"]
            ),
        ),
        (
            f"fp_cap_{max_false_positive_rate:.2%}",
            float(
                _select_best_row(
                    val_sweep_df,
                    max_false_positive_rate=max_false_positive_rate,
                    sort_columns=("recall", "f1", "precision"),
                )["threshold"]
            ),
        ),
    ]

    rows: list[dict[str, Any]] = []
    val_labels = val_scores_df["is_anomaly"].to_numpy()
    val_scores = val_scores_df["score"].to_numpy()
    test_labels = test_scores_df["is_anomaly"].to_numpy()
    test_scores = test_scores_df["score"].to_numpy()

    for policy_name, threshold in policies:
        val_metrics = summarize_threshold_metrics(val_labels, val_scores, threshold)
        test_metrics = summarize_threshold_metrics(test_labels, test_scores, threshold)
        rows.append(
            {
                "policy": policy_name,
                "threshold": float(threshold),
                "val_precision": float(val_metrics["precision"]),
                "val_recall": float(val_metrics["recall"]),
                "val_f1": float(val_metrics["f1"]),
                "val_false_positive_rate": float(val_metrics["false_positive_rate"]),
                "test_precision": float(test_metrics["precision"]),
                "test_recall": float(test_metrics["recall"]),
                "test_f1": float(test_metrics["f1"]),
                "test_false_positive_rate": float(test_metrics["false_positive_rate"]),
                "test_predicted_anomalies": int(test_metrics["predicted_anomalies"]),
                "test_tp": int(test_metrics["tp"]),
                "test_fp": int(test_metrics["fp"]),
                "test_tn": int(test_metrics["tn"]),
                "test_fn": int(test_metrics["fn"]),
            }
        )

    return pd.DataFrame(rows)


def summarize_review_band(scores_df: pd.DataFrame, low_threshold: float, high_threshold: float) -> dict[str, Any]:
    if float(low_threshold) >= float(high_threshold):
        raise ValueError("low_threshold must be strictly smaller than high_threshold.")

    labels = scores_df["is_anomaly"].to_numpy(dtype=int)
    scores = scores_df["score"].to_numpy(dtype=float)

    auto_normal = scores < float(low_threshold)
    auto_anomaly = scores >= float(high_threshold)
    review = ~(auto_normal | auto_anomaly)

    auto_tn = int(np.sum((labels == 0) & auto_normal))
    auto_fn = int(np.sum((labels == 1) & auto_normal))
    auto_tp = int(np.sum((labels == 1) & auto_anomaly))
    auto_fp = int(np.sum((labels == 0) & auto_anomaly))

    review_anomalies = int(np.sum((labels == 1) & review))
    review_normals = int(np.sum((labels == 0) & review))

    auto_normal_count = auto_tn + auto_fn
    auto_anomaly_count = auto_tp + auto_fp
    review_count = review_anomalies + review_normals
    resolved_count = auto_normal_count + auto_anomaly_count

    return {
        "low_threshold": float(low_threshold),
        "high_threshold": float(high_threshold),
        "auto_normal_count": int(auto_normal_count),
        "auto_normal_anomaly_rate": float(0.0 if auto_normal_count == 0 else auto_fn / auto_normal_count),
        "auto_normal_npv": float(0.0 if auto_normal_count == 0 else auto_tn / auto_normal_count),
        "auto_anomaly_count": int(auto_anomaly_count),
        "auto_anomaly_precision": float(0.0 if auto_anomaly_count == 0 else auto_tp / auto_anomaly_count),
        "auto_tn": int(auto_tn),
        "auto_fn": int(auto_fn),
        "auto_tp": int(auto_tp),
        "auto_fp": int(auto_fp),
        "review_count": int(review_count),
        "review_rate": float(0.0 if len(scores_df) == 0 else review_count / len(scores_df)),
        "review_anomalies": int(review_anomalies),
        "review_normals": int(review_normals),
        "resolved_count": int(resolved_count),
        "resolved_rate": float(0.0 if len(scores_df) == 0 else resolved_count / len(scores_df)),
        "automatic_error_count": int(auto_fn + auto_fp),
    }


def select_review_band_from_validation(
    val_scores_df: pd.DataFrame,
    *,
    max_auto_normal_anomaly_rate: float = 0.01,
    min_auto_anomaly_precision: float = 0.60,
) -> dict[str, Any]:
    low_sweep_df = build_auto_normal_sweep(val_scores_df)
    high_sweep_df = build_threshold_sweep(val_scores_df)

    low_candidates = low_sweep_df[
        low_sweep_df["auto_normal_anomaly_rate"] <= float(max_auto_normal_anomaly_rate)
    ]
    if low_candidates.empty:
        raise ValueError("No low threshold satisfied the requested auto-normal anomaly-rate constraint.")
    low_row = low_candidates.sort_values(["auto_normal_count", "threshold"], ascending=[False, False]).iloc[0]

    high_candidates = high_sweep_df[
        (high_sweep_df["precision"] >= float(min_auto_anomaly_precision))
        & (high_sweep_df["threshold"] > float(low_row["threshold"]))
    ]
    if high_candidates.empty:
        raise ValueError(
            "No high threshold satisfied the requested auto-anomaly precision constraint "
            "while remaining above the selected low threshold."
        )
    high_row = high_candidates.sort_values(["predicted_anomalies", "threshold"], ascending=[False, True]).iloc[0]

    return {
        "low_threshold": float(low_row["threshold"]),
        "high_threshold": float(high_row["threshold"]),
        "low_row": low_row.to_dict(),
        "high_row": high_row.to_dict(),
        "low_sweep_df": low_sweep_df,
        "high_sweep_df": high_sweep_df,
    }


def build_review_policy_summary(
    val_scores_df: pd.DataFrame,
    test_scores_df: pd.DataFrame,
    *,
    max_auto_normal_anomaly_rate: float = 0.01,
    min_auto_anomaly_precision: float = 0.60,
) -> pd.DataFrame:
    selection = select_review_band_from_validation(
        val_scores_df,
        max_auto_normal_anomaly_rate=max_auto_normal_anomaly_rate,
        min_auto_anomaly_precision=min_auto_anomaly_precision,
    )
    val_summary = summarize_review_band(
        val_scores_df,
        low_threshold=selection["low_threshold"],
        high_threshold=selection["high_threshold"],
    )
    test_summary = summarize_review_band(
        test_scores_df,
        low_threshold=selection["low_threshold"],
        high_threshold=selection["high_threshold"],
    )

    row = {
        "policy": "review_band",
        "low_threshold": float(selection["low_threshold"]),
        "high_threshold": float(selection["high_threshold"]),
        "target_max_auto_normal_anomaly_rate": float(max_auto_normal_anomaly_rate),
        "target_min_auto_anomaly_precision": float(min_auto_anomaly_precision),
    }
    for prefix, summary in [("val", val_summary), ("test", test_summary)]:
        for key, value in summary.items():
            if key in {"low_threshold", "high_threshold"}:
                continue
            row[f"{prefix}_{key}"] = value
    return pd.DataFrame([row])

