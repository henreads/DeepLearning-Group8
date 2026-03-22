from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def confusion_matrix(labels: pd.Series, predicted: pd.Series) -> list[list[int]]:
    tn = int(((labels == 0) & (predicted == 0)).sum())
    fp = int(((labels == 0) & (predicted == 1)).sum())
    fn = int(((labels == 1) & (predicted == 0)).sum())
    tp = int(((labels == 1) & (predicted == 1)).sum())
    return [[tn, fp], [fn, tp]]


def load_threshold_and_confusion(summary_path: Path) -> tuple[float, list[list[int]] | None]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = summary.get("metrics_at_validation_threshold", summary)
    threshold = float(summary.get("threshold", metrics.get("threshold")))
    target_cm = metrics.get("confusion_matrix")
    return threshold, target_cm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--scores-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata_csv)
    test_meta = metadata.loc[metadata["split"] == "test", ["defect_type", "is_anomaly"]].reset_index(drop=True)
    scores = pd.read_csv(args.scores_csv)

    if len(scores) != len(test_meta):
        raise ValueError(f"Length mismatch: scores={len(scores)} metadata_test={len(test_meta)}")
    if not scores["is_anomaly"].reset_index(drop=True).equals(test_meta["is_anomaly"].astype(int)):
        raise ValueError("Saved test scores do not align with the shared test split ordering.")

    threshold, target_cm = load_threshold_and_confusion(Path(args.summary_json))
    pred_gt = (scores["score"] > threshold).astype(int)
    pred_ge = (scores["score"] >= threshold).astype(int)
    if target_cm == confusion_matrix(scores["is_anomaly"], pred_gt):
        predicted = pred_gt
    elif target_cm == confusion_matrix(scores["is_anomaly"], pred_ge):
        predicted = pred_ge
    else:
        predicted = pred_gt

    merged = test_meta.copy()
    merged["score"] = scores["score"].to_numpy()
    merged["predicted"] = predicted.to_numpy()
    anomalies = merged.loc[merged["is_anomaly"] == 1].copy()
    breakdown = (
        anomalies.groupby("defect_type", sort=False)
        .agg(
            count=("defect_type", "size"),
            detected=("predicted", "sum"),
            mean_score=("score", "mean"),
            median_score=("score", "median"),
        )
        .reset_index()
    )
    breakdown["detected"] = breakdown["detected"].astype(int)
    breakdown["missed"] = breakdown["count"] - breakdown["detected"]
    breakdown["recall"] = breakdown["detected"] / breakdown["count"]
    breakdown = breakdown.sort_values(["recall", "count", "defect_type"], ascending=[True, False, True]).reset_index(drop=True)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    breakdown.to_csv(output_path, index=False)
    print(f"Saved defect breakdown to {output_path}")


if __name__ == "__main__":
    main()
