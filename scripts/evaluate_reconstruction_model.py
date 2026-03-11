"""Evaluate saved autoencoder or VAE checkpoints on the shared anomaly protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from wafer_defect.data.wm811k import WaferMapDataset
from wafer_defect.evaluation.reconstruction_metrics import summarize_threshold_metrics, sweep_threshold_metrics
from wafer_defect.models.autoencoder import ConvAutoencoder
from wafer_defect.models.vae import ConvVariationalAutoencoder, VAEOutput
from wafer_defect.scoring import reconstruction_mse, vae_anomaly_score


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def infer_model_type(config: dict[str, Any], override: str) -> str:
    if override:
        return override.lower()
    model_type = str(config.get("model", {}).get("type", "autoencoder")).lower()
    if model_type not in {"autoencoder", "vae"}:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model_type


def infer_image_size(config: dict[str, Any], checkpoint_path: Path) -> int:
    configured_size = config.get("data", {}).get("image_size")
    if configured_size is not None:
        return int(configured_size)

    for part in checkpoint_path.parts:
        match = re.fullmatch(r"x(\d+)", part)
        if match:
            return int(match.group(1))

    return 64


def build_model(config: dict[str, Any], model_type: str, image_size: int) -> torch.nn.Module:
    latent_dim = int(config["model"]["latent_dim"])

    if model_type == "autoencoder":
        return ConvAutoencoder(latent_dim=latent_dim, image_size=image_size)
    if model_type == "vae":
        return ConvVariationalAutoencoder(latent_dim=latent_dim, image_size=image_size)
    raise ValueError(f"Unsupported model type: {model_type}")


def collect_scores(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    model_type: str,
    beta: float,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    model.eval()

    with torch.inference_mode():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)

            if model_type == "autoencoder":
                reconstructions = model(inputs)
                scores = reconstruction_mse(inputs, reconstructions)
            else:
                outputs = model(inputs)
                if not isinstance(outputs, VAEOutput):
                    raise TypeError("VAE model must return VAEOutput")
                scores = vae_anomaly_score(
                    inputs,
                    outputs.reconstruction,
                    outputs.mu,
                    outputs.logvar,
                    beta=beta,
                )

            for score, label in zip(scores.cpu().tolist(), labels.tolist()):
                rows.append({"score": float(score), "is_anomaly": int(label)})

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--model-type", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if args.config:
        from wafer_defect.config import load_toml

        config = load_toml(args.config)
    else:
        config = checkpoint.get("config")
        if not config:
            raise ValueError("Checkpoint does not include config. Pass --config explicitly.")

    model_type = infer_model_type(config, args.model_type)
    beta = float(config["model"].get("beta", 0.01))
    device = resolve_device(args.device or config["training"].get("device", "auto"))
    batch_size = args.batch_size or int(config["data"].get("batch_size", 64))
    image_size = infer_image_size(config, checkpoint_path)

    model = build_model(config, model_type, image_size=image_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    val_dataset = WaferMapDataset(config["data"]["metadata_csv"], split="val", image_size=image_size)
    test_dataset = WaferMapDataset(config["data"]["metadata_csv"], split="test", image_size=image_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=int(config["data"].get("num_workers", 0)))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=int(config["data"].get("num_workers", 0)))

    val_scores_df = collect_scores(model, val_loader, device, model_type, beta=beta)
    val_normal_scores = val_scores_df.loc[val_scores_df["is_anomaly"] == 0, "score"]
    if val_normal_scores.empty:
        raise ValueError("Validation split does not contain normal scores to derive a threshold.")
    threshold = float(val_normal_scores.quantile(args.threshold_quantile))

    test_scores_df = collect_scores(model, test_loader, device, model_type, beta=beta)
    labels = test_scores_df["is_anomaly"].to_numpy()
    scores = test_scores_df["score"].to_numpy()

    metrics = summarize_threshold_metrics(labels, scores, threshold)
    threshold_sweep_df, best_sweep = sweep_threshold_metrics(labels, scores)

    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    val_scores_df.to_csv(output_dir / "val_scores.csv", index=False)
    test_scores_df.to_csv(output_dir / "test_scores.csv", index=False)
    threshold_sweep_df.to_csv(output_dir / "threshold_sweep.csv", index=False)

    summary = {
        "model_type": model_type,
        "checkpoint": str(checkpoint_path),
        "threshold_quantile": float(args.threshold_quantile),
        "threshold": threshold,
        "metrics_at_validation_threshold": metrics,
        "best_threshold_sweep": best_sweep,
        "counts": {
            "val_normal": int((val_scores_df["is_anomaly"] == 0).sum()),
            "test_normal": int((test_scores_df["is_anomaly"] == 0).sum()),
            "test_anomaly": int((test_scores_df["is_anomaly"] == 1).sum()),
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
