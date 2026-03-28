"""Build a report-friendly table from holdout evaluation summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def infer_model_name(summary_path: Path, repo_root: Path) -> str:
    try:
        relative = summary_path.resolve().relative_to(repo_root.resolve())
        parts = list(relative.parts)
    except ValueError:
        parts = list(summary_path.parts)

    if "artifacts" in parts:
        artifacts_index = parts.index("artifacts")
        artifact_parts = parts[artifacts_index + 1 :]
        if len(artifact_parts) >= 3 and artifact_parts[-2].startswith("evaluation_"):
            return artifact_parts[-3]
        if len(artifact_parts) >= 2:
            return artifact_parts[-2]
    return summary_path.parent.name


def discover_summary_paths(repo_root: Path, inputs: list[str]) -> list[Path]:
    summary_paths: list[Path] = []
    for raw_input in inputs:
        candidate = Path(raw_input)
        if not candidate.is_absolute():
            candidate = (repo_root / candidate).resolve()

        if candidate.is_file():
            summary_paths.append(candidate)
            continue

        if candidate.is_dir():
            summary_paths.extend(sorted(candidate.rglob("summary.json")))
            continue

        raise FileNotFoundError(f"Could not resolve input path: {raw_input}")

    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in summary_paths:
        resolved = path.resolve()
        if resolved not in seen:
            unique_paths.append(resolved)
            seen.add(resolved)
    return unique_paths


def extract_row(summary_path: Path, repo_root: Path) -> dict[str, object]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if "metrics_at_validation_threshold" not in payload:
        raise ValueError(f"Unsupported summary format: {summary_path}")

    metrics = payload["metrics_at_validation_threshold"]
    best_sweep = payload["best_threshold_sweep"]
    counts = payload["counts"]

    confusion_matrix = metrics.get("confusion_matrix", [[None, None], [None, None]])
    tn, fp = confusion_matrix[0]
    fn, tp = confusion_matrix[1]

    test_normal = int(counts["test_normal"])
    test_anomaly = int(counts["test_anomaly"])
    normal_fpr = float(fp / test_normal) if test_normal else float("nan")
    defect_fnr = float(fn / test_anomaly) if test_anomaly else float("nan")

    metadata_csv = str(payload.get("metadata_csv", ""))
    split_tag = Path(metadata_csv).stem if metadata_csv else ""

    return {
        "model": infer_model_name(summary_path, repo_root),
        "model_type": payload.get("model_type", ""),
        "split_tag": split_tag,
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "auroc": float(metrics["auroc"]),
        "auprc": float(metrics["auprc"]),
        "threshold": float(payload["threshold"]),
        "best_sweep_f1": float(best_sweep["f1"]),
        "best_sweep_threshold": float(best_sweep["threshold"]),
        "normal_fpr": normal_fpr,
        "defect_fnr": defect_fnr,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_anomalies": int(metrics["predicted_anomalies"]),
        "test_normal": test_normal,
        "test_anomaly": test_anomaly,
        "checkpoint": str(payload.get("checkpoint", "")),
        "summary_json": str(summary_path),
    }


def build_markdown_table(df: pd.DataFrame) -> str:
    display_df = df[
        [
            "model",
            "f1",
            "auroc",
            "auprc",
            "precision",
            "recall",
            "normal_fpr",
            "tp",
            "fp",
            "fn",
            "tn",
            "best_sweep_f1",
        ]
    ].copy()

    for column in ["f1", "auroc", "auprc", "precision", "recall", "normal_fpr", "best_sweep_f1"]:
        display_df[column] = display_df[column].map(lambda value: f"{float(value):.3f}")

    headers = list(display_df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display_df.iterrows():
        values = [str(row[column]) for column in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Summary JSON file or directory to scan. Repeatable.",
    )
    parser.add_argument(
        "--output-csv",
        default="artifacts/x64/holdout70k_3p5k_reports/report_table.csv",
    )
    parser.add_argument(
        "--output-md",
        default="artifacts/x64/holdout70k_3p5k_reports/report_table.md",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    summary_paths = discover_summary_paths(repo_root, args.input)
    if not summary_paths:
        raise ValueError("No summary.json files found.")

    rows = [extract_row(summary_path, repo_root) for summary_path in summary_paths]
    report_df = pd.DataFrame(rows).sort_values(["f1", "auprc", "auroc"], ascending=False).reset_index(drop=True)

    output_csv = (repo_root / args.output_csv).resolve()
    output_md = (repo_root / args.output_md).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    report_df.to_csv(output_csv, index=False)
    output_md.write_text(build_markdown_table(report_df) + "\n", encoding="utf-8")

    print(f"Saved CSV report to {output_csv}")
    print(f"Saved Markdown report to {output_md}")
    print(report_df.to_string(index=False))


if __name__ == "__main__":
    main()
