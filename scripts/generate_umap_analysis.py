from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

try:
    import umap
except ImportError as exc:  # pragma: no cover
    raise SystemExit("umap-learn is required. Use the project venv Python.") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wafer_defect.data.wm811k import WaferMapDataset
from wafer_defect.evaluation.reconstruction_metrics import summarize_threshold_metrics, sweep_threshold_metrics
from wafer_defect.models.autoencoder import build_autoencoder_from_config
from wafer_defect.models.patchcore import PatchCoreModel
from wafer_defect.models.resnet import ResNetFeatureExtractor
from wafer_defect.models.svdd import ConvDeepSVDD
from wafer_defect.models.ts_distillation import build_ts_distillation_from_config
from wafer_defect.models.vae import ConvVariationalAutoencoder, VAEOutput
from wafer_defect.scoring import (
    absolute_error_map,
    masked_spatial_mean,
    pooled_error_map,
    reconstruction_mse,
    spatial_max,
    spatial_mean,
    squared_error_map,
    svdd_distance,
    topk_spatial_mean,
    vae_anomaly_score,
)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def infer_model_type(config: dict[str, Any], checkpoint: dict[str, Any], override: str) -> str:
    if override:
        return override.lower()

    model_type = str(config.get("model", {}).get("type", "")).lower().strip()
    if not model_type:
        model_type = str(checkpoint.get("model_type", "")).lower().strip()
    if model_type == "efficientad":
        return "ts_distillation"
    if model_type == "wideresnet50_backbone":
        return "wideresnet50_backbone"
    if model_type not in {
        "autoencoder",
        "vae",
        "svdd",
        "patchcore",
        "ts_distillation",
        "resnet18_backbone",
        "wideresnet50_backbone",
    }:
        raise ValueError(f"Unsupported model type: {model_type!r}")
    return model_type


def infer_image_size(config: dict[str, Any], checkpoint_path: Path) -> int:
    configured_size = config.get("data", {}).get("image_size")
    if configured_size is not None:
        return int(configured_size)
    for part in checkpoint_path.parts:
        if part.startswith("x") and part[1:].isdigit():
            return int(part[1:])
    return 64


def build_model(config: dict[str, Any], model_type: str, image_size: int) -> torch.nn.Module:
    if model_type == "autoencoder":
        return build_autoencoder_from_config(config, image_size=image_size)
    if model_type == "vae":
        return ConvVariationalAutoencoder(
            latent_dim=int(config["model"]["latent_dim"]),
            image_size=image_size,
        )
    if model_type == "svdd":
        return ConvDeepSVDD(
            latent_dim=int(config["model"]["latent_dim"]),
            image_size=image_size,
        )
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


def autoencoder_score_tensor(
    inputs: torch.Tensor,
    reconstructions: torch.Tensor,
    score_name: str,
    topk_ratio: float,
    foreground_threshold: float,
    pool_kernel_size: int,
) -> torch.Tensor:
    mse_map = squared_error_map(inputs, reconstructions)
    mae_map = absolute_error_map(inputs, reconstructions)
    foreground_mask = (inputs > foreground_threshold).to(dtype=inputs.dtype)
    pooled_mae = pooled_error_map(mae_map, kernel_size=pool_kernel_size)

    score_map = {
        "mse_mean": spatial_mean(mse_map),
        "mae_mean": spatial_mean(mae_map),
        "max_abs": spatial_max(mae_map),
        "topk_abs_mean": topk_spatial_mean(mae_map, topk_ratio=topk_ratio),
        "foreground_mse": masked_spatial_mean(mse_map, foreground_mask),
        "foreground_mae": masked_spatial_mean(mae_map, foreground_mask),
        "pooled_mae_mean": spatial_mean(pooled_mae),
    }
    if score_name not in score_map:
        raise ValueError(f"Unsupported autoencoder score_name: {score_name}")
    return score_map[score_name]


def collect_split_features_scores(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    model_type: str,
    score_name: str,
    beta: float,
    topk_ratio: float,
    foreground_threshold: float,
    pool_kernel_size: int,
    center: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels_all: list[int] = []
    scores_all: list[float] = []
    model.eval()

    with torch.inference_mode():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)

            if model_type == "autoencoder":
                reconstructions = model(inputs)
                score_tensor = autoencoder_score_tensor(
                    inputs,
                    reconstructions,
                    score_name=score_name,
                    topk_ratio=topk_ratio,
                    foreground_threshold=foreground_threshold,
                    pool_kernel_size=pool_kernel_size,
                )
                feature_tensor = (inputs - reconstructions).flatten(start_dim=1)
            elif model_type == "vae":
                outputs = model(inputs)
                if not isinstance(outputs, VAEOutput):
                    raise TypeError("VAE model must return VAEOutput.")
                score_tensor = vae_anomaly_score(
                    inputs,
                    outputs.reconstruction,
                    outputs.mu,
                    outputs.logvar,
                    beta=beta,
                )
                feature_tensor = outputs.mu
            elif model_type == "svdd":
                feature_tensor = model(inputs)
                score_tensor = svdd_distance(feature_tensor, model.center)
            elif model_type == "ts_distillation":
                score_tensor = model(inputs)
                feature_tensor = model.normalized_anomaly_map(inputs).flatten(start_dim=1)
            elif model_type == "patchcore":
                score_tensor = model(inputs)
                feature_tensor = model.patch_embeddings(inputs).mean(dim=1)
            elif model_type in {"resnet18_backbone", "wideresnet50_backbone"}:
                feature_tensor = model(inputs)
                if center is None:
                    raise ValueError("Backbone baseline evaluation requires a center tensor.")
                score_tensor = torch.norm(feature_tensor - center.unsqueeze(0), dim=1)
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")

            features.append(feature_tensor.cpu().numpy())
            labels_all.extend(int(label) for label in labels.tolist())
            scores_all.extend(float(score) for score in score_tensor.cpu().tolist())

    return (
        np.concatenate(features, axis=0),
        np.asarray(labels_all, dtype=np.int64),
        np.asarray(scores_all, dtype=np.float32),
    )


def fit_umap_reference(
    val_normal_features: np.ndarray,
    random_state: int,
    pca_components: int,
    n_neighbors: int,
    min_dist: float,
) -> tuple[PCA | None, umap.UMAP, np.ndarray]:
    pca = None
    umap_input = val_normal_features
    if val_normal_features.shape[1] > pca_components:
        pca = PCA(n_components=min(pca_components, val_normal_features.shape[0], val_normal_features.shape[1]), random_state=random_state)
        umap_input = pca.fit_transform(val_normal_features)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state,
        transform_seed=random_state,
    )
    val_umap = reducer.fit_transform(umap_input)
    return pca, reducer, val_umap


def transform_features(
    features: np.ndarray,
    pca: PCA | None,
    reducer: umap.UMAP,
) -> np.ndarray:
    transformed = features if pca is None else pca.transform(features)
    return reducer.transform(transformed)


def knn_reference_scores(
    val_points: np.ndarray,
    query_points: np.ndarray,
    k: int,
) -> np.ndarray:
    k = max(1, min(k, len(val_points)))
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nbrs.fit(val_points)
    distances, _ = nbrs.kneighbors(query_points, return_distance=True)
    return distances.mean(axis=1)


def knn_reference_scores_leave_one_out(
    val_points: np.ndarray,
    k: int,
) -> np.ndarray:
    k = max(1, min(k, len(val_points) - 1 if len(val_points) > 1 else 1))
    if len(val_points) <= 1:
        return np.zeros(len(val_points), dtype=np.float32)
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(val_points)
    distances, _ = nbrs.kneighbors(val_points, return_distance=True)
    return distances[:, 1:].mean(axis=1)


def save_split_plot(df: pd.DataFrame, out_path: Path, title: str) -> None:
    colors = {
        "val_normal": "#1f77b4",
        "test_normal": "#2ca02c",
        "test_anomaly": "#d62728",
    }
    plt.figure(figsize=(9, 7))
    for group in ["val_normal", "test_normal", "test_anomaly"]:
        subset = df[df["group"] == group]
        if subset.empty:
            continue
        plt.scatter(subset["umap1"], subset["umap2"], s=10, alpha=0.55, label=group, c=colors[group])
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_score_plot(df: pd.DataFrame, out_path: Path, score_column: str, title: str) -> None:
    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(
        df["umap1"],
        df["umap2"],
        c=df[score_column],
        cmap="viridis",
        s=10,
        alpha=0.65,
    )
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title(title)
    cbar = plt.colorbar(scatter)
    cbar.set_label(score_column.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_histogram(
    df: pd.DataFrame,
    out_path: Path,
    score_column: str,
    threshold: float,
    title: str,
) -> None:
    plt.figure(figsize=(8, 5))
    for group, color in [("test_normal", "#2ca02c"), ("test_anomaly", "#d62728")]:
        subset = df[df["group"] == group]
        if subset.empty:
            continue
        plt.hist(subset[score_column], bins=50, alpha=0.55, label=group, color=color)
    plt.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"threshold={threshold:.4f}")
    plt.xlabel(score_column.replace("_", " ").title())
    plt.ylabel("Count")
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_threshold_sweep_plot(sweep_df: pd.DataFrame, threshold: float, out_path: Path, title: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(sweep_df["threshold"], sweep_df["precision"], label="precision")
    plt.plot(sweep_df["threshold"], sweep_df["recall"], label="recall")
    plt.plot(sweep_df["threshold"], sweep_df["f1"], label="f1")
    plt.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"val-95 threshold={threshold:.4f}")
    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--metadata-csv", default="")
    parser.add_argument("--model-type", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--score-name", default="mse_mean")
    parser.add_argument("--topk-ratio", type=float, default=0.01)
    parser.add_argument("--foreground-threshold", type=float, default=0.0)
    parser.add_argument("--pool-kernel-size", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--knn-k", type=int, default=15)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = REPO_ROOT / config_path
        import tomllib

        with config_path.open("rb") as handle:
            config = tomllib.load(handle)
    else:
        config = checkpoint.get("config", {})
    if not config:
        raise ValueError("Could not infer config from checkpoint. Pass --config explicitly.")

    if args.metadata_csv:
        config.setdefault("data", {})["metadata_csv"] = args.metadata_csv

    metadata_csv = str(config.get("data", {}).get("metadata_csv", ""))
    if metadata_csv.startswith("/root/artifacts/"):
        metadata_csv = metadata_csv.replace("/root/artifacts/", "artifacts/", 1)
    metadata_path = Path(metadata_csv)
    if not metadata_path.is_absolute():
        metadata_path = REPO_ROOT / metadata_path
    config.setdefault("data", {})["metadata_csv"] = str(metadata_path)

    model_type = infer_model_type(config, checkpoint, args.model_type)
    image_size = infer_image_size(config, checkpoint_path)
    device = resolve_device(args.device or config.get("training", {}).get("device", "auto"))
    batch_size = args.batch_size or int(config.get("data", {}).get("batch_size", 64))
    num_workers = int(config.get("data", {}).get("num_workers", 0))
    beta = float(config.get("model", {}).get("beta", 0.01))

    model = build_model(config, model_type=model_type, image_size=image_size)
    if model_type == "patchcore" and "memory_bank" in checkpoint["model_state_dict"]:
        model.set_memory_bank(checkpoint["model_state_dict"]["memory_bank"])
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    val_dataset = WaferMapDataset(str(metadata_path), split="val", image_size=image_size)
    test_dataset = WaferMapDataset(str(metadata_path), split="test", image_size=image_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    center = None
    if model_type in {"resnet18_backbone", "wideresnet50_backbone"}:
        val_embeddings, val_labels, _ = collect_split_features_scores(
            model,
            val_loader,
            device,
            model_type,
            args.score_name,
            beta,
            args.topk_ratio,
            args.foreground_threshold,
            args.pool_kernel_size,
            center=torch.zeros(1, device=device),
        )
        val_normal_embeddings = val_embeddings[val_labels == 0]
        center = torch.tensor(val_normal_embeddings.mean(axis=0), dtype=torch.float32, device=device)

    val_features, val_labels, val_scores = collect_split_features_scores(
        model,
        val_loader,
        device,
        model_type,
        args.score_name,
        beta,
        args.topk_ratio,
        args.foreground_threshold,
        args.pool_kernel_size,
        center=center,
    )
    test_features, test_labels, test_scores = collect_split_features_scores(
        model,
        test_loader,
        device,
        model_type,
        args.score_name,
        beta,
        args.topk_ratio,
        args.foreground_threshold,
        args.pool_kernel_size,
        center=center,
    )

    val_normal_mask = val_labels == 0
    val_normal_features = val_features[val_normal_mask]
    val_normal_scores = val_scores[val_normal_mask]
    model_threshold = float(pd.Series(val_normal_scores).quantile(args.threshold_quantile))
    original_metrics = summarize_threshold_metrics(test_labels, test_scores, model_threshold)
    original_sweep_df, original_best_sweep = sweep_threshold_metrics(test_labels, test_scores)

    pca, reducer, val_normal_points = fit_umap_reference(
        val_normal_features,
        random_state=args.random_state,
        pca_components=args.pca_components,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
    )
    val_points = transform_features(val_features, pca, reducer)
    test_points = transform_features(test_features, pca, reducer)

    val_umap_scores = knn_reference_scores_leave_one_out(val_points[val_normal_mask], k=args.knn_k)
    test_umap_scores = knn_reference_scores(val_points[val_normal_mask], test_points, k=args.knn_k)
    umap_threshold = float(pd.Series(val_umap_scores).quantile(args.threshold_quantile))
    umap_metrics = summarize_threshold_metrics(test_labels, test_umap_scores, umap_threshold)
    umap_sweep_df, umap_best_sweep = sweep_threshold_metrics(test_labels, test_umap_scores)

    val_df = pd.DataFrame(
        {
            "split": "val",
            "group": np.where(val_labels == 0, "val_normal", "val_anomaly"),
            "is_anomaly": val_labels.astype(int),
            "model_score": val_scores.astype(float),
        }
    )
    val_df["umap1"] = val_points[:, 0]
    val_df["umap2"] = val_points[:, 1]
    val_df["umap_knn_score"] = np.nan
    val_df.loc[val_normal_mask, "umap_knn_score"] = val_umap_scores
    if (~val_normal_mask).any():
        val_df.loc[~val_normal_mask, "umap_knn_score"] = knn_reference_scores(val_points[val_normal_mask], val_points[~val_normal_mask], k=args.knn_k)

    test_df = pd.DataFrame(
        {
            "split": "test",
            "group": np.where(test_labels == 0, "test_normal", "test_anomaly"),
            "is_anomaly": test_labels.astype(int),
            "model_score": test_scores.astype(float),
            "umap_knn_score": test_umap_scores.astype(float),
            "umap1": test_points[:, 0],
            "umap2": test_points[:, 1],
        }
    )
    combined_df = pd.concat([val_df, test_df], ignore_index=True)
    combined_df.to_csv(output_dir / "umap_points.csv", index=False)

    save_split_plot(
        combined_df[combined_df["group"].isin(["val_normal", "test_normal", "test_anomaly"])],
        plots_dir / "umap_by_split.png",
        title=f"{args.name}: UMAP by Split",
    )
    save_score_plot(
        combined_df[combined_df["group"].isin(["val_normal", "test_normal", "test_anomaly"])],
        plots_dir / "umap_by_score.png",
        score_column="model_score",
        title=f"{args.name}: UMAP by Model Score",
    )
    save_score_plot(
        combined_df[combined_df["group"].isin(["val_normal", "test_normal", "test_anomaly"])],
        plots_dir / "umap_by_umap_knn_score.png",
        score_column="umap_knn_score",
        title=f"{args.name}: UMAP by UMAP-kNN Score",
    )
    save_histogram(
        combined_df,
        plots_dir / "umap_knn_histogram.png",
        score_column="umap_knn_score",
        threshold=umap_threshold,
        title=f"{args.name}: UMAP-kNN Score Histogram",
    )
    save_threshold_sweep_plot(
        umap_sweep_df,
        threshold=umap_threshold,
        out_path=plots_dir / "umap_knn_threshold_sweep.png",
        title=f"{args.name}: UMAP-kNN Threshold Sweep",
    )

    _, nearest_val_dist = pairwise_distances_argmin_min(test_points, val_points[val_normal_mask], metric="euclidean")
    overlap_summary = {
        "test_to_val_normal_mean_distance": float(nearest_val_dist.mean()),
        "test_normal_to_val_normal_mean_distance": float(nearest_val_dist[test_labels == 0].mean()),
        "test_anomaly_to_val_normal_mean_distance": float(nearest_val_dist[test_labels == 1].mean()),
    }

    summary = {
        "name": args.name,
        "family": args.family,
        "model_type": model_type,
        "checkpoint": str(checkpoint_path),
        "metadata_csv": str(metadata_path),
        "image_size": image_size,
        "feature_representation": {
            "autoencoder": "flattened reconstruction residual",
            "vae": "latent mu vector",
            "svdd": "latent embedding",
            "ts_distillation": "flattened normalized anomaly map",
            "patchcore": "mean pooled normalized patch embeddings",
            "resnet18_backbone": "global embedding",
            "wideresnet50_backbone": "global embedding",
        }[model_type],
        "score_name": args.score_name,
        "threshold_quantile": float(args.threshold_quantile),
        "umap_params": {
            "random_state": int(args.random_state),
            "pca_components": int(args.pca_components),
            "n_neighbors": int(args.umap_neighbors),
            "min_dist": float(args.umap_min_dist),
            "knn_k": int(args.knn_k),
        },
        "counts": {
            "val_normal": int((val_labels == 0).sum()),
            "test_normal": int((test_labels == 0).sum()),
            "test_anomaly": int((test_labels == 1).sum()),
        },
        "original_score": {
            "threshold": model_threshold,
            "metrics_at_validation_threshold": original_metrics,
            "best_threshold_sweep": original_best_sweep,
        },
        "umap_knn_score": {
            "threshold": umap_threshold,
            "metrics_at_validation_threshold": umap_metrics,
            "best_threshold_sweep": umap_best_sweep,
        },
        "geometry_overlap": overlap_summary,
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    original_sweep_df.to_csv(output_dir / "original_threshold_sweep.csv", index=False)
    umap_sweep_df.to_csv(output_dir / "umap_knn_threshold_sweep.csv", index=False)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
