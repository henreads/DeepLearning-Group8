"""Evaluate saved anomaly-model checkpoints on the shared anomaly protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any
from xml.parsers.expat import model

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from wafer_defect.data.wm811k import WaferMapDataset
from wafer_defect.evaluation.reconstruction_metrics import summarize_threshold_metrics, sweep_threshold_metrics
from wafer_defect.models.autoencoder import build_autoencoder_from_config
from wafer_defect.models.ts_distillation import build_ts_distillation_from_config
from wafer_defect.models.patchcore import PatchCoreModel
from wafer_defect.models.svdd import ConvDeepSVDD
from wafer_defect.models.vae import ConvVariationalAutoencoder, VAEOutput
from wafer_defect.scoring import reconstruction_mse, svdd_distance, vae_anomaly_score
from wafer_defect.models.resnet import ResNetFeatureExtractor


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def infer_model_type(config: dict[str, Any], override: str) -> str:
    if override:
        return override.lower()
    model_type = str(config.get("model", {}).get("type", "autoencoder")).lower()
    if model_type == "efficientad":
        return "ts_distillation"
    if model_type not in {
        "autoencoder",
        "vae",
        "svdd",
        "patchcore",
        "ts_distillation",
        "resnet18_backbone",
        "wideresnet50_backbone",
    }:
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
    if model_type == "autoencoder":
        return build_autoencoder_from_config(config, image_size=image_size)
    if model_type == "vae":
        latent_dim = int(config["model"]["latent_dim"])
        return ConvVariationalAutoencoder(latent_dim=latent_dim, image_size=image_size)
    if model_type == "svdd":
        latent_dim = int(config["model"]["latent_dim"])
        return ConvDeepSVDD(latent_dim=latent_dim, image_size=image_size)
    if model_type == "patchcore":
        return PatchCoreModel(
            image_size=image_size,
            backbone_type=str(config.get("model", {}).get("backbone_type", "conv")),
            use_batchnorm=bool(config.get("model", {}).get("use_batchnorm", True)),
            pretrained=bool(config.get("model", {}).get("pretrained", True)),
            freeze_backbone=bool(config.get("model", {}).get("freeze_backbone", True)),
            backbone_input_size=int(config.get("model", {}).get("backbone_input_size", 224)),
            normalize_imagenet=bool(config.get("model", {}).get("normalize_imagenet", True)),
            reduction=str(config.get("model", {}).get("reduction", "max")),
            topk_ratio=float(config.get("model", {}).get("topk_ratio", 0.1)),
            query_chunk_size=int(config.get("model", {}).get("query_chunk_size", 2048)),
            memory_chunk_size=int(config.get("model", {}).get("memory_chunk_size", 8192)),
        )
    if model_type == "ts_distillation":
        return build_ts_distillation_from_config(config, image_size=image_size)
    
    if model_type == "resnet18_backbone":
        return ResNetFeatureExtractor(
            backbone_name="resnet18",
            pretrained=bool(config.get("model", {}).get("pretrained", True)),
            input_size=int(config.get("model", {}).get("input_size", 224)),
            freeze_backbone=bool(config.get("model", {}).get("freeze_backbone", True)),
            normalize_imagenet=bool(config.get("model", {}).get("normalize_imagenet", True)),
        )

    if model_type == "wideresnet50_backbone":
        return ResNetFeatureExtractor(
            backbone_name="wide_resnet50_2",
            pretrained=bool(config.get("model", {}).get("pretrained", True)),
            input_size=int(config.get("model", {}).get("input_size", 224)),
            freeze_backbone=bool(config.get("model", {}).get("freeze_backbone", True)),
            normalize_imagenet=bool(config.get("model", {}).get("normalize_imagenet", True)),
        )
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
            elif model_type == "vae":
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
            elif model_type == "patchcore":
                scores = model(inputs)
            elif model_type == "ts_distillation":
                scores = model(inputs)
            else:
                embeddings = model(inputs)
                scores = svdd_distance(embeddings, model.center)

            for score, label in zip(scores.cpu().tolist(), labels.tolist()):
                rows.append({"score": float(score), "is_anomaly": int(label)})

    return pd.DataFrame(rows)


def collect_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    model.eval()

    with torch.inference_mode():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            embeddings = model(inputs)

            for embedding, label in zip(embeddings.cpu(), labels.tolist()):
                rows.append({
                    "embedding": embedding.numpy(),
                    "is_anomaly": int(label),
                })

    return pd.DataFrame(rows)


def collect_features(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    model_type: str,
) -> tuple[np.ndarray, np.ndarray]:

    import numpy as np

    features = []
    labels_all = []

    model.eval()

    with torch.inference_mode():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)

            if model_type == "autoencoder":
                recon = model(inputs)
                feat = (inputs - recon).view(inputs.size(0), -1)
            elif model_type == "vae":
                outputs = model(inputs)
                if not isinstance(outputs, VAEOutput):
                    raise TypeError("VAE model must return VAEOutput")
                feat = (inputs - outputs.reconstruction).view(inputs.size(0), -1)

            elif model_type == "ts_distillation":
                outputs = model(inputs)

                if isinstance(outputs, tuple):
                    feat = outputs[0]
                else:
                    feat = outputs

                feat = feat.view(feat.size(0), -1)

            elif model_type == "patchcore":
                feat = model.embed(inputs)  
                feat = feat.view(feat.size(0), -1)

            elif model_type == "vae":
                outputs = model(inputs)
                if not isinstance(outputs, VAEOutput):
                    raise TypeError("VAE model must return VAEOutput")
                feat = outputs.mu

            else:

                feat = model(inputs)
                feat = feat.view(feat.size(0), -1)

            features.append(feat.cpu().numpy())
            labels_all.extend(labels.tolist())

    return np.concatenate(features, axis=0), np.array(labels_all)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--metadata-csv", default="")
    parser.add_argument("--model-type", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--reduction", default="")
    parser.add_argument("--topk-ratio", type=float, default=-1.0)
    parser.add_argument("--score-student-weight", type=float, default=float("nan"))
    parser.add_argument("--score-autoencoder-weight", type=float, default=float("nan"))
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
    model_config = config.setdefault("model", {})
    data_config = config.setdefault("data", {})
    if args.metadata_csv:
        data_config["metadata_csv"] = args.metadata_csv
    if args.reduction:
        model_config["reduction"] = args.reduction
    if args.topk_ratio >= 0.0:
        model_config["topk_ratio"] = float(args.topk_ratio)
    if not torch.isnan(torch.tensor(args.score_student_weight)):
        model_config["score_student_weight"] = float(args.score_student_weight)
    if not torch.isnan(torch.tensor(args.score_autoencoder_weight)):
        model_config["score_autoencoder_weight"] = float(args.score_autoencoder_weight)

    beta = float(config["model"].get("beta", 0.01))
    device = resolve_device(args.device or config["training"].get("device", "auto"))
    batch_size = args.batch_size or int(data_config.get("batch_size", 64))
    image_size = infer_image_size(config, checkpoint_path)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(config, model_type, image_size=image_size)
    if model_type == "patchcore" and "memory_bank" in checkpoint["model_state_dict"]:
        model.set_memory_bank(checkpoint["model_state_dict"]["memory_bank"])
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    metadata_csv = data_config["metadata_csv"]
    num_workers = int(data_config.get("num_workers", 0))
    val_dataset = WaferMapDataset(metadata_csv, split="val", image_size=image_size)
    test_dataset = WaferMapDataset(metadata_csv, split="test", image_size=image_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if model_type in {"resnet18_backbone", "wideresnet50_backbone"}:
        val_embeddings_df = collect_embeddings(model, val_loader, device)
        val_normal_embeddings = val_embeddings_df.loc[val_embeddings_df["is_anomaly"] == 0, "embedding"]
        if val_normal_embeddings.empty:
            raise ValueError("Validation split does not contain normal embeddings to derive a threshold.")

        val_normal_matrix = torch.tensor(val_normal_embeddings.tolist(), dtype=torch.float32)
        center = val_normal_matrix.mean(dim=0)

        val_scores = torch.norm(val_normal_matrix - center.unsqueeze(0), dim=1).numpy()
        threshold = float(pd.Series(val_scores).quantile(args.threshold_quantile))

        val_scores_df = pd.DataFrame({
            "score": [
                float(torch.norm(torch.tensor(emb, dtype=torch.float32) - center, p=2).item())
                for emb in val_embeddings_df["embedding"]
            ],
            "is_anomaly": val_embeddings_df["is_anomaly"].astype(int),
        })

        test_embeddings_df = collect_embeddings(model, test_loader, device)
        test_scores_df = pd.DataFrame({
            "score": [
                float(torch.norm(torch.tensor(emb, dtype=torch.float32) - center, p=2).item())
                for emb in test_embeddings_df["embedding"]
            ],
            "is_anomaly": test_embeddings_df["is_anomaly"].astype(int),
        })
        test_features = np.stack(test_embeddings_df["embedding"].to_list(), axis=0)
        test_labels = test_embeddings_df["is_anomaly"].to_numpy(dtype=np.int64)
    else:
        val_scores_df = collect_scores(model, val_loader, device, model_type, beta=beta)
        val_normal_scores = val_scores_df.loc[val_scores_df["is_anomaly"] == 0, "score"]
        if val_normal_scores.empty:
            raise ValueError("Validation split does not contain normal scores to derive a threshold.")
        threshold = float(val_normal_scores.quantile(args.threshold_quantile))

        test_scores_df = collect_scores(model, test_loader, device, model_type, beta=beta)
        print("[INFO] Extracting features for UMAP...")

        test_features, test_labels = collect_features(
            model,
            test_loader,
            device,
            model_type,
        )

        print(f"[INFO] Saved features: {test_features.shape}")

    labels = test_scores_df["is_anomaly"].to_numpy()
    scores = test_scores_df["score"].to_numpy()

    metrics = summarize_threshold_metrics(labels, scores, threshold)
    threshold_sweep_df, best_sweep = sweep_threshold_metrics(labels, scores)

    val_scores_df.to_csv(output_dir / "val_scores.csv", index=False)
    test_scores_df.to_csv(output_dir / "test_scores.csv", index=False)
    threshold_sweep_df.to_csv(output_dir / "threshold_sweep.csv", index=False)
    np.save(output_dir / "test_features.npy", test_features)
    np.save(output_dir / "test_labels.npy", test_labels)

    summary = {
        "model_type": model_type,
        "checkpoint": str(checkpoint_path),
        "metadata_csv": str(metadata_csv),
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
