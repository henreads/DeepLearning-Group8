from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


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
    """Score sweep: load model once, collect maps in-memory, sweep all weight/reduction combos.

    ResNet50 at 224x224 is expensive — spawning 42 evaluate subprocesses would take ~30 min.
    Instead we collect normalised anomaly maps once and apply all combinations in Python.
    """
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from wafer_defect.config import load_toml
    from wafer_defect.data.wm811k import WaferMapDataset
    from wafer_defect.evaluation.reconstruction_metrics import summarize_threshold_metrics, sweep_threshold_metrics
    from wafer_defect.models.ts_distillation import build_ts_distillation_from_config
    from wafer_defect.scoring import spatial_max, spatial_mean, topk_spatial_mean

    checkpoints_dir, results_dir = _ensure_layout(output_dir)
    checkpoint_path = checkpoints_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")

    config = load_toml(config_path)
    image_size = int(config["data"].get("image_size", 224))
    batch_size = int(config["data"].get("batch_size", 256))
    num_workers = int(config["data"].get("num_workers", 0))
    metadata_path = REPO_ROOT / config["data"]["metadata_csv"]

    requested_device = str(config["training"].get("device", "auto"))
    device = torch.device("cuda" if requested_device == "auto" and torch.cuda.is_available() else requested_device)
    print(f"Device: {device}", flush=True)

    val_dataset = WaferMapDataset(metadata_path, split="val", image_size=image_size)
    test_dataset = WaferMapDataset(metadata_path, split="test", image_size=image_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_ts_distillation_from_config(config, image_size=image_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded checkpoint from {checkpoint_path}", flush=True)

    def collect_maps(loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        s_maps, a_maps, lbls = [], [], []
        with torch.inference_mode():
            for inputs, batch_labels in loader:
                inputs = inputs.to(device)
                s_map, a_map = model.raw_anomaly_maps(inputs)
                s_maps.append((s_map / model.student_map_scale.clamp_min(1e-6)).cpu())
                a_maps.append((a_map / model.autoencoder_map_scale.clamp_min(1e-6)).cpu())
                lbls.append(batch_labels.cpu())
        return torch.cat(s_maps), torch.cat(a_maps), torch.cat(lbls).numpy()

    def reduce_map(anomaly_map: torch.Tensor, reduction: str, topk_ratio: float | None) -> np.ndarray:
        if reduction == "mean":
            return spatial_mean(anomaly_map).numpy()
        if reduction == "max":
            return spatial_max(anomaly_map).numpy()
        return topk_spatial_mean(anomaly_map, topk_ratio=topk_ratio).numpy()

    print("Collecting val maps...", flush=True)
    val_s, val_a, val_labels = collect_maps(val_loader)
    print("Collecting test maps...", flush=True)
    test_s, test_a, test_labels = collect_maps(test_loader)

    rows: list[dict] = []
    n_combos = len(SCORE_SWEEP_WEIGHTS) * len(SCORE_SWEEP_REDUCTIONS)
    for i, (student_weight, auto_weight) in enumerate(SCORE_SWEEP_WEIGHTS):
        val_map = student_weight * val_s + auto_weight * val_a
        test_map = student_weight * test_s + auto_weight * test_a
        for reduction, topk_ratio in SCORE_SWEEP_REDUCTIONS:
            variant_name = f"s{student_weight:g}_a{auto_weight:g}_{reduction}" + ("" if topk_ratio is None else f"_r{topk_ratio:.2f}")
            val_scores = reduce_map(val_map, reduction, topk_ratio)
            test_scores = reduce_map(test_map, reduction, topk_ratio)
            threshold = float(np.quantile(val_scores[val_labels == 0], THRESHOLD_QUANTILE))
            metrics = summarize_threshold_metrics(test_labels, test_scores, threshold)
            _, best_sweep = sweep_threshold_metrics(test_labels, test_scores)
            rows.append({
                "name": variant_name,
                "student_weight": student_weight,
                "auto_weight": auto_weight,
                "reduction": reduction,
                "topk_ratio": topk_ratio,
                "threshold": threshold,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "predicted_anomalies": metrics["predicted_anomalies"],
                "best_sweep_f1": best_sweep["f1"],
            })
            print(f"[{len(rows)}/{n_combos}] {variant_name}  F1={metrics['f1']:.4f}  AUROC={metrics['auroc']:.4f}", flush=True)

    score_sweep_df = pd.DataFrame(rows).sort_values(["f1", "auprc", "auroc"], ascending=False).reset_index(drop=True)
    score_sweep_df.to_csv(results_dir / "score_sweep_summary.csv", index=False)

    best_row = score_sweep_df.iloc[0].to_dict()
    best_row["topk_ratio"] = None if pd.isna(best_row["topk_ratio"]) else float(best_row["topk_ratio"])
    (results_dir / "selected_score_variant.json").write_text(json.dumps(best_row, indent=2), encoding="utf-8")

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
