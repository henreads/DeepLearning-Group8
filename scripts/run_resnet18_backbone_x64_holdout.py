from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

SCRIPT_PATH = Path(__file__).resolve()
for candidate in [SCRIPT_PATH.parent, *SCRIPT_PATH.parents]:
    src_root = candidate / "src"
    if (src_root / "wafer_defect").exists():
        if str(src_root) not in sys.path:
            sys.path.insert(0, str(src_root))
        break

from wafer_defect.config import load_toml
from wafer_defect.data.wm811k import WaferMapDataset
from wafer_defect.evaluation.reconstruction_metrics import summarize_threshold_metrics, sweep_threshold_metrics

from holdout_eval_helpers import (
    build_defect_breakdown,
    resolve_repo_root,
    save_defect_breakdown_plot,
    save_threshold_sweep_plot,
    to_repo_relative,
    write_confusion_csv,
)


CONFIG_PATH = Path("experiments/anomaly_detection/backbone_embedding/resnet18/x64/baseline/train_config.toml")


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def collect_embeddings(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    embeddings: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for inputs, batch_labels in dataloader:
            batch_embeddings = model(inputs.to(device)).cpu().numpy().astype(np.float32)
            embeddings.append(batch_embeddings)
            labels.append(batch_labels.numpy())
    return np.concatenate(embeddings, axis=0), np.concatenate(labels, axis=0)


def l2_center_scores(embeddings: np.ndarray, center: np.ndarray) -> np.ndarray:
    return np.linalg.norm(embeddings - center[None, :], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--input-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    repo_root = resolve_repo_root()
    os.chdir(repo_root)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("WM811K_REPO_ROOT", str(repo_root))

    config = load_toml(repo_root / CONFIG_PATH)
    config["data"]["metadata_csv"] = args.metadata_path
    from wafer_defect.models.resnet import ResNetFeatureExtractor

    input_artifact_dir = (repo_root / args.input_artifact_dir).resolve()
    output_dir = (repo_root / args.output_dir)
    evaluation_dir = output_dir / "evaluation"
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device or config["training"].get("device", "auto"))
    image_size = int(config["data"].get("image_size", 64))
    batch_size = int(config["data"].get("batch_size", 64))
    num_workers = int(config["data"].get("num_workers", 0))
    metadata_path = (repo_root / args.metadata_path).resolve()

    train_dataset = WaferMapDataset(metadata_path, split="train", image_size=image_size)
    val_dataset = WaferMapDataset(metadata_path, split="val", image_size=image_size)
    test_dataset = WaferMapDataset(metadata_path, split="test", image_size=image_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = ResNetFeatureExtractor(
        backbone_name="resnet18",
        pretrained=bool(config["model"].get("pretrained", True)),
        input_size=int(config["model"].get("input_size", 224)),
        freeze_backbone=bool(config["model"].get("freeze_backbone", True)),
        normalize_imagenet=bool(config["model"].get("normalize_imagenet", True)),
    ).to(device)

    checkpoint_path = input_artifact_dir / "resnet18_backbone_baseline.pth"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])

    train_center_path = input_artifact_dir / "train_center.npy"
    if train_center_path.exists():
        train_center = np.load(train_center_path).astype(np.float32)
    else:
        train_embeddings, _ = collect_embeddings(model, train_loader, device)
        train_center = train_embeddings.mean(axis=0).astype(np.float32)

    val_embeddings, val_labels = collect_embeddings(model, val_loader, device)
    test_embeddings, test_labels = collect_embeddings(model, test_loader, device)
    val_scores = l2_center_scores(val_embeddings, train_center)
    test_scores = l2_center_scores(test_embeddings, train_center)

    val_scores_df = pd.DataFrame({"score": val_scores, "is_anomaly": val_labels.astype(int)})
    test_scores_df = pd.DataFrame({"score": test_scores, "is_anomaly": test_labels.astype(int)})

    threshold_quantile = float(config["scoring"].get("threshold_quantile", 0.95))
    threshold = float(val_scores_df.loc[val_scores_df["is_anomaly"] == 0, "score"].quantile(threshold_quantile))
    metrics = summarize_threshold_metrics(test_labels.astype(int), test_scores, threshold)
    threshold_sweep_df, best_sweep = sweep_threshold_metrics(test_labels.astype(int), test_scores)

    analysis_df = test_dataset.metadata.reset_index(drop=True).copy()
    analysis_df["score"] = test_scores_df["score"]
    analysis_df["predicted_anomaly"] = (analysis_df["score"] > threshold).astype(int)
    analysis_df["error_type"] = "tn"
    analysis_df.loc[(analysis_df["is_anomaly"] == 0) & (analysis_df["predicted_anomaly"] == 1), "error_type"] = "fp"
    analysis_df.loc[(analysis_df["is_anomaly"] == 1) & (analysis_df["predicted_anomaly"] == 0), "error_type"] = "fn"
    analysis_df.loc[(analysis_df["is_anomaly"] == 1) & (analysis_df["predicted_anomaly"] == 1), "error_type"] = "tp"
    defect_breakdown_df = build_defect_breakdown(analysis_df, threshold)

    val_scores_df.to_csv(evaluation_dir / "val_scores.csv", index=False)
    test_scores_df.to_csv(evaluation_dir / "test_scores.csv", index=False)
    threshold_sweep_df.to_csv(evaluation_dir / "threshold_sweep.csv", index=False)
    defect_breakdown_df.to_csv(evaluation_dir / "defect_breakdown.csv", index=False)
    analysis_df.to_csv(evaluation_dir / "failure_analysis.csv", index=False)
    np.save(output_dir / "train_center.npy", train_center)
    write_confusion_csv(evaluation_dir / "confusion_matrix.csv", metrics["confusion_matrix"])
    save_threshold_sweep_plot(
        threshold_sweep_df,
        plots_dir / "threshold_sweep.png",
        title="ResNet18 Backbone Holdout",
    )
    save_defect_breakdown_plot(
        defect_breakdown_df,
        plots_dir / "defect_breakdown.png",
        title="ResNet18 Backbone Holdout Defect Recall",
    )

    summary = {
        "experiment": "backbone_embedding_resnet18_x64_baseline",
        "protocol": "holdout70k_3p5k",
        "backbone": "resnet18",
        "checkpoint": to_repo_relative(checkpoint_path, repo_root) if checkpoint_path.exists() else "",
        "metadata_csv": to_repo_relative(metadata_path, repo_root),
        "threshold_quantile": threshold_quantile,
        "threshold": threshold,
        "train_center_norm": float(np.linalg.norm(train_center)),
        "metrics_at_validation_threshold": metrics,
        "best_threshold_sweep": best_sweep,
        "counts": {
            "val_normal": int((val_scores_df["is_anomaly"] == 0).sum()),
            "test_normal": int((test_scores_df["is_anomaly"] == 0).sum()),
            "test_anomaly": int((test_scores_df["is_anomaly"] == 1).sum()),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = {
        "output_dir": to_repo_relative(output_dir, repo_root),
        "summary_path": to_repo_relative(output_dir / "summary.json", repo_root),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
