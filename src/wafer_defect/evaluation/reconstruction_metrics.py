"""Metric helpers for reconstruction-based anomaly detection evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def summarize_threshold_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predicted = (scores > threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "predicted_anomalies": int(predicted.sum()),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
    }


def sweep_threshold_metrics(labels: np.ndarray, scores: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any]]:
    precision_curve, recall_curve, thresholds = precision_recall_curve(labels, scores)

    if thresholds.size == 0:
        sweep_df = pd.DataFrame(
            [
                {
                    "threshold": float(scores[0]) if scores.size else 0.0,
                    "precision": float(precision_curve[0]),
                    "recall": float(recall_curve[0]),
                    "f1": 0.0,
                    "predicted_anomalies": int((scores > 0).sum()),
                }
            ]
        )
    else:
        sweep_df = pd.DataFrame(
            {
                "threshold": thresholds,
                "precision": precision_curve[:-1],
                "recall": recall_curve[:-1],
            }
        )
        sweep_df["f1"] = (
            2
            * sweep_df["precision"]
            * sweep_df["recall"]
            / (sweep_df["precision"] + sweep_df["recall"] + 1e-12)
        )
        sweep_df["predicted_anomalies"] = [(scores > threshold).sum() for threshold in sweep_df["threshold"]]

    best_row = sweep_df.loc[sweep_df["f1"].idxmax()]
    best_summary = {
        "threshold": float(best_row["threshold"]),
        "precision": float(best_row["precision"]),
        "recall": float(best_row["recall"]),
        "f1": float(best_row["f1"]),
        "predicted_anomalies": int(best_row["predicted_anomalies"]),
    }
    return sweep_df, best_summary
