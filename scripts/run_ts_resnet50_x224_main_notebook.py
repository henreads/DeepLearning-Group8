from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "anomaly_detection" / "teacher_student" / "resnet50" / "x224" / "main" / "train_config.toml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "anomaly_detection" / "teacher_student" / "resnet50" / "x224" / "main" / "artifacts" / "ts_resnet50_x224"
THRESHOLD_QUANTILE = 0.95
SCORE_SWEEP_WEIGHTS = [(1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (2.0, 1.0), (1.0, 2.0), (1.0, 0.5), (0.5, 1.0)]
SCORE_SWEEP_REDUCTIONS = [("mean", None), ("max", None), ("topk_mean", 0.01), ("topk_mean", 0.05), ("topk_mean", 0.1), ("topk_mean", 0.2)]


def _run(command: list[str]) -> None:
    print(" ".join(str(part) for part in command), flush=True)
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def _ensure_layout(output_dir: Path) -> tuple[Path, Path]:
    checkpoints_dir = output_dir / "checkpoints"
    results_dir = output_dir / "results"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return checkpoints_dir, results_dir


def _maybe_move(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


def _artifact_status(output_dir: Path) -> dict[str, bool]:
    checkpoints_dir = output_dir / "checkpoints"
    results_dir = output_dir / "results"
    return {
        "checkpoint_exists": (checkpoints_dir / "best_model.pt").exists(),
        "evaluation_summary_exists": (results_dir / "summary.json").exists(),
        "score_sweep_exists": (results_dir / "score_sweep_summary.csv").exists(),
    }


def _write_phase_manifest(output_dir: Path, phase: str) -> None:
    manifest = {"phase": phase, "artifact_dir": str(output_dir), **_artifact_status(output_dir)}
    (output_dir / f"{phase}_phase_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _write_run_manifest(output_dir: Path) -> None:
    manifest = {"phase": "sweep", "artifact_dir": str(output_dir), **_artifact_status(output_dir)}
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run_train(config_path: Path, output_dir: Path, fresh_train: bool) -> None:
    if fresh_train and output_dir.exists():
        shutil.rmtree(output_dir)
    checkpoints_dir, results_dir = _ensure_layout(output_dir)
    _run([sys.executable, "-u", str(REPO_ROOT / "scripts" / "train_ts_distillation.py"), "--config", str(config_path)])

    _maybe_move(output_dir / "best_model.pt", checkpoints_dir / "best_model.pt")
    _maybe_move(output_dir / "latest_checkpoint.pt", checkpoints_dir / "latest_checkpoint.pt")
    _maybe_move(output_dir / "last_model.pt", checkpoints_dir / "last_model.pt")
    _maybe_move(output_dir / "history.json", results_dir / "history.json")
    _maybe_move(output_dir / "summary.json", results_dir / "training_summary.json")

    for checkpoint in output_dir.glob("checkpoint_epoch_*.pt"):
        _maybe_move(checkpoint, checkpoints_dir / checkpoint.name)

    _write_phase_manifest(output_dir, "train")


def run_eval(config_path: Path, output_dir: Path) -> None:
    checkpoints_dir, results_dir = _ensure_layout(output_dir)
    checkpoint_path = checkpoints_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")

    _run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluate_reconstruction_model.py"),
            "--checkpoint",
            str(checkpoint_path),
            "--config",
            str(config_path),
            "--model-type",
            "ts_distillation",
            "--threshold-quantile",
            str(THRESHOLD_QUANTILE),
            "--output-dir",
            str(results_dir),
        ]
    )
    _write_phase_manifest(output_dir, "eval")


def run_sweep(config_path: Path, output_dir: Path) -> None:
    checkpoints_dir, results_dir = _ensure_layout(output_dir)
    checkpoint_path = checkpoints_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")

    sweep_tmp_dir = results_dir / "_score_sweep_tmp"
    if sweep_tmp_dir.exists():
        shutil.rmtree(sweep_tmp_dir)
    sweep_tmp_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str | None]] = []
    for student_weight, auto_weight in SCORE_SWEEP_WEIGHTS:
        for reduction, topk_ratio in SCORE_SWEEP_REDUCTIONS:
            variant_name = f"s{student_weight:g}_a{auto_weight:g}_{reduction}" + ("" if topk_ratio is None else f"_r{topk_ratio:.2f}")
            variant_dir = sweep_tmp_dir / variant_name
            _run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "evaluate_reconstruction_model.py"),
                    "--checkpoint",
                    str(checkpoint_path),
                    "--config",
                    str(config_path),
                    "--model-type",
                    "ts_distillation",
                    "--threshold-quantile",
                    str(THRESHOLD_QUANTILE),
                    "--output-dir",
                    str(variant_dir),
                    "--reduction",
                    reduction,
                    *(["--topk-ratio", str(topk_ratio)] if topk_ratio is not None else []),
                    "--score-student-weight",
                    str(student_weight),
                    "--score-autoencoder-weight",
                    str(auto_weight),
                ]
            )
            summary = json.loads((variant_dir / "summary.json").read_text(encoding="utf-8"))
            metrics = summary["metrics_at_validation_threshold"]
            best_sweep = summary["best_threshold_sweep"]
            rows.append(
                {
                    "name": variant_name,
                    "student_weight": student_weight,
                    "auto_weight": auto_weight,
                    "reduction": reduction,
                    "topk_ratio": topk_ratio,
                    "threshold": metrics["threshold"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "auroc": metrics["auroc"],
                    "auprc": metrics["auprc"],
                    "predicted_anomalies": metrics["predicted_anomalies"],
                    "best_sweep_f1": best_sweep["f1"],
                }
            )

    score_sweep_df = pd.DataFrame(rows).sort_values(["f1", "auprc", "auroc"], ascending=False).reset_index(drop=True)
    score_sweep_df.to_csv(results_dir / "score_sweep_summary.csv", index=False)

    best_row = score_sweep_df.iloc[0].to_dict()
    best_row["topk_ratio"] = None if pd.isna(best_row["topk_ratio"]) else float(best_row["topk_ratio"])
    (results_dir / "selected_score_variant.json").write_text(json.dumps(best_row, indent=2), encoding="utf-8")

    shutil.rmtree(sweep_tmp_dir, ignore_errors=True)
    _write_phase_manifest(output_dir, "sweep")
    _write_run_manifest(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--phase", choices=["train", "eval", "sweep"], required=True)
    parser.add_argument("--fresh-train", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()

    if args.phase == "train":
        run_train(config_path, output_dir, fresh_train=args.fresh_train)
    elif args.phase == "eval":
        run_eval(config_path, output_dir)
    else:
        run_sweep(config_path, output_dir)


if __name__ == "__main__":
    main()
