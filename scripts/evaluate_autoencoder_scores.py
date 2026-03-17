from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from torch.utils.data import DataLoader

from wafer_defect.config import load_toml
from wafer_defect.data.wm811k import WaferMapDataset
from wafer_defect.evaluation import summarize_threshold_metrics, sweep_threshold_metrics
from wafer_defect.models.autoencoder import build_autoencoder_from_config
from wafer_defect.scoring import (
    absolute_error_map,
    masked_spatial_mean,
    pooled_error_map,
    spatial_max,
    spatial_mean,
    squared_error_map,
    topk_spatial_mean,
)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def infer_image_size(config: dict, checkpoint_path: Path) -> int:
    configured_size = config.get("data", {}).get("image_size")
    if configured_size is not None:
        return int(configured_size)

    for part in checkpoint_path.parts:
        if part.startswith("x") and part[1:].isdigit():
            return int(part[1:])
    return 64


def collect_score_frames(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    topk_ratio: float,
    foreground_threshold: float,
    pool_kernel_size: int,
) -> dict[str, pd.DataFrame]:
    rows: dict[str, list[dict[str, float]]] = {
        "mse_mean": [],
        "mae_mean": [],
        "max_abs": [],
        "topk_abs_mean": [],
        "foreground_mse": [],
        "foreground_mae": [],
        "pooled_mae_mean": [],
    }
    sample_index = 0
    model.eval()

    with torch.inference_mode():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            reconstructions = model(inputs)

            mse_map = squared_error_map(inputs, reconstructions)
            mae_map = absolute_error_map(inputs, reconstructions)
            foreground_mask = (inputs > foreground_threshold).to(dtype=inputs.dtype)
            pooled_mae = pooled_error_map(mae_map, kernel_size=pool_kernel_size)

            score_tensors = {
                "mse_mean": spatial_mean(mse_map),
                "mae_mean": spatial_mean(mae_map),
                "max_abs": spatial_max(mae_map),
                "topk_abs_mean": topk_spatial_mean(mae_map, topk_ratio=topk_ratio),
                "foreground_mse": masked_spatial_mean(mse_map, foreground_mask),
                "foreground_mae": masked_spatial_mean(mae_map, foreground_mask),
                "pooled_mae_mean": spatial_mean(pooled_mae),
            }

            for batch_pos, label in enumerate(labels.tolist()):
                for score_name, score_tensor in score_tensors.items():
                    rows[score_name].append(
                        {
                            "sample_index": sample_index + batch_pos,
                            "score": float(score_tensor[batch_pos].item()),
                            "is_anomaly": int(label),
                        }
                    )

            sample_index += inputs.shape[0]

    return {name: pd.DataFrame(score_rows) for name, score_rows in rows.items()}


def summarize_score(
    score_name: str,
    val_scores_df: pd.DataFrame,
    test_scores_df: pd.DataFrame,
    threshold_quantile: float,
) -> tuple[dict[str, object], pd.DataFrame]:
    val_normal_scores = val_scores_df.loc[val_scores_df["is_anomaly"] == 0, "score"]
    if val_normal_scores.empty:
        raise ValueError(f"No validation-normal scores found for score '{score_name}'.")

    threshold = float(val_normal_scores.quantile(threshold_quantile))
    labels = test_scores_df["is_anomaly"].to_numpy()
    scores = test_scores_df["score"].to_numpy()
    metrics = summarize_threshold_metrics(labels, scores, threshold)
    threshold_sweep_df, best_sweep = sweep_threshold_metrics(labels, scores)

    summary = {
        "score_name": score_name,
        "threshold_quantile": float(threshold_quantile),
        "threshold": threshold,
        "metrics_at_validation_threshold": metrics,
        "best_threshold_sweep": best_sweep,
    }
    return summary, threshold_sweep_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/train_autoencoder.toml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--topk-ratio", type=float, default=0.01)
    parser.add_argument("--foreground-threshold", type=float, default=0.0)
    parser.add_argument("--pool-kernel-size", type=int, default=5)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = load_toml(args.config) if args.config else checkpoint.get("config")
    if not config:
        raise ValueError("Could not load config from args or checkpoint.")

    image_size = infer_image_size(config, checkpoint_path)
    device = resolve_device(args.device or config["training"].get("device", "auto"))
    batch_size = args.batch_size or int(config["data"].get("batch_size", 64))

    model = build_autoencoder_from_config(config, image_size=image_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    val_dataset = WaferMapDataset(config["data"]["metadata_csv"], split="val", image_size=image_size)
    test_dataset = WaferMapDataset(config["data"]["metadata_csv"], split="test", image_size=image_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=int(config["data"].get("num_workers", 0)))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=int(config["data"].get("num_workers", 0)))

    val_frames = collect_score_frames(
        model,
        val_loader,
        device,
        topk_ratio=args.topk_ratio,
        foreground_threshold=args.foreground_threshold,
        pool_kernel_size=args.pool_kernel_size,
    )
    test_frames = collect_score_frames(
        model,
        test_loader,
        device,
        topk_ratio=args.topk_ratio,
        foreground_threshold=args.foreground_threshold,
        pool_kernel_size=args.pool_kernel_size,
    )

    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "score_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    score_summaries: list[dict[str, object]] = []
    for score_name in val_frames:
        score_dir = output_dir / score_name
        score_dir.mkdir(parents=True, exist_ok=True)

        val_df = val_frames[score_name]
        test_df = test_frames[score_name]
        summary, threshold_sweep_df = summarize_score(
            score_name=score_name,
            val_scores_df=val_df,
            test_scores_df=test_df,
            threshold_quantile=args.threshold_quantile,
        )

        val_df.to_csv(score_dir / "val_scores.csv", index=False)
        test_df.to_csv(score_dir / "test_scores.csv", index=False)
        threshold_sweep_df.to_csv(score_dir / "threshold_sweep.csv", index=False)
        with (score_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        metrics = summary["metrics_at_validation_threshold"]
        best_sweep = summary["best_threshold_sweep"]
        score_summaries.append(
            {
                "score_name": score_name,
                "threshold": summary["threshold"],
                "val_threshold_precision": metrics["precision"],
                "val_threshold_recall": metrics["recall"],
                "val_threshold_f1": metrics["f1"],
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "best_sweep_threshold": best_sweep["threshold"],
                "best_sweep_precision": best_sweep["precision"],
                "best_sweep_recall": best_sweep["recall"],
                "best_sweep_f1": best_sweep["f1"],
            }
        )

    score_summary_df = pd.DataFrame(score_summaries).sort_values(
        by=["val_threshold_f1", "auprc", "best_sweep_f1"],
        ascending=False,
    )
    score_summary_df.to_csv(output_dir / "score_summary.csv", index=False)

    summary_payload = {
        "checkpoint": str(checkpoint_path),
        "topk_ratio": float(args.topk_ratio),
        "foreground_threshold": float(args.foreground_threshold),
        "pool_kernel_size": int(args.pool_kernel_size),
        "threshold_quantile": float(args.threshold_quantile),
        "scores": score_summary_df.to_dict(orient="records"),
    }
    with (output_dir / "score_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    print(score_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
