#!/usr/bin/env python3
"""Runner: EfficientNet-B1 one-layer PatchCore x240 main benchmark."""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import EfficientNet_B1_Weights, efficientnet_b1
from tqdm import tqdm


def _resolve_project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "src" / "wafer_defect").exists() and (candidate / "experiments").exists():
            return candidate
    return Path.cwd().resolve()


PROJECT_ROOT = _resolve_project_root()
_src = str(PROJECT_ROOT / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from wafer_defect.config import load_toml  # noqa: E402
from wafer_defect.data.wm811k import WaferMapDataset  # noqa: E402
from wafer_defect.evaluation.reconstruction_metrics import (  # noqa: E402
    summarize_threshold_metrics,
    sweep_threshold_metrics,
)
from wafer_defect.training.patchcore import build_memory_subset, collect_memory_bank  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


class OneHotWaferMapDataset(Dataset):
    """Wrap processed single-channel wafers into 3-channel state one-hots."""

    def __init__(self, base_dataset: WaferMapDataset, model_input_size: int) -> None:
        self.base_dataset = base_dataset
        self.model_input_size = int(model_input_size)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        wafer_tensor, label = self.base_dataset[index]
        states = torch.clamp(torch.round(wafer_tensor * 2.0), 0, 2).to(dtype=torch.long)
        one_hot = F.one_hot(states.squeeze(0), num_classes=3).permute(2, 0, 1).to(dtype=torch.float32)
        if one_hot.shape[-1] != self.model_input_size or one_hot.shape[-2] != self.model_input_size:
            one_hot = F.interpolate(
                one_hot.unsqueeze(0),
                size=(self.model_input_size, self.model_input_size),
                mode="nearest",
            ).squeeze(0)
        return one_hot, label


class EfficientNetB1OneLayerPatchCoreModel(nn.Module):
    def __init__(
        self,
        *,
        model_input_size: int,
        feature_idx: int,
        patch_embed_dim: int,
        reduction: str,
        topk_ratio: float,
        nn_k: int,
        query_chunk_size: int,
        amp_enabled: bool,
        pretrained: bool,
        freeze_backbone: bool,
    ) -> None:
        super().__init__()
        weights = EfficientNet_B1_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b1(weights=weights)
        self.features = backbone.features
        self.feature_idx = int(feature_idx)
        self.patch_embed_dim = int(patch_embed_dim)
        self.reduction = str(reduction)
        self.topk_ratio = float(topk_ratio)
        self.nn_k = int(nn_k)
        self.query_chunk_size = int(query_chunk_size)
        self.amp_enabled = bool(amp_enabled)

        with torch.inference_mode():
            dummy = torch.zeros(1, 3, model_input_size, model_input_size)
            x = dummy
            selected = None
            for index, block in enumerate(self.features):
                x = block(x)
                if index == self.feature_idx:
                    selected = x
                    break

        if selected is None:
            raise ValueError(f"Invalid EfficientNet-B1 feature index: {self.feature_idx}")
        if self.reduction not in {"max", "mean", "topk_mean"}:
            raise ValueError(f"Unsupported reduction: {self.reduction}")
        if self.reduction == "topk_mean" and not 0.0 < self.topk_ratio <= 1.0:
            raise ValueError(f"topk_ratio must be in (0, 1], got {self.topk_ratio}")

        in_dim = int(selected.shape[1])
        self.patch_grid = tuple(int(dim) for dim in selected.shape[-2:])
        self.feature_dim = self.patch_embed_dim
        self.proj = nn.Linear(in_dim, self.patch_embed_dim, bias=False)
        self.register_buffer("memory_bank", torch.empty(0, self.feature_dim))

        if freeze_backbone:
            for parameter in self.features.parameters():
                parameter.requires_grad_(False)
        for parameter in self.proj.parameters():
            parameter.requires_grad_(False)

    @property
    def patches_per_image(self) -> int:
        return int(self.patch_grid[0] * self.patch_grid[1])

    def forward_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        selected = None
        for index, block in enumerate(self.features):
            x = block(x)
            if index == self.feature_idx:
                selected = x
                break
        if selected is None:
            raise RuntimeError("Failed to collect EfficientNet-B1 feature map.")
        return selected

    def patch_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        amp_enabled = self.amp_enabled and x.device.type == "cuda"
        with torch.inference_mode():
            with torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=amp_enabled):
                feature_map = self.forward_feature_map(x)
                embeddings = feature_map.permute(0, 2, 3, 1).reshape(-1, feature_map.shape[1])
                embeddings = self.proj(embeddings)
            embeddings = F.normalize(embeddings.float(), p=2, dim=1)
        return embeddings.reshape(x.shape[0], self.patches_per_image, self.feature_dim)

    def set_memory_bank(self, memory_bank: torch.Tensor) -> None:
        if memory_bank.ndim != 2 or memory_bank.shape[1] != self.feature_dim:
            raise ValueError(
                f"memory_bank must have shape (N, {self.feature_dim}), got {tuple(memory_bank.shape)}"
            )
        normalized = F.normalize(memory_bank.to(dtype=torch.float32), p=2, dim=1)
        self.memory_bank = normalized.to(device=self.memory_bank.device, dtype=self.memory_bank.dtype)

    def nearest_patch_distances(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        if self.memory_bank.numel() == 0:
            raise ValueError("memory_bank is empty.")

        batch_size, patch_count, _ = patch_embeddings.shape
        flat_queries = patch_embeddings.reshape(-1, self.feature_dim)
        bank_t = self.memory_bank.t().contiguous()
        outputs: list[torch.Tensor] = []

        for start in range(0, flat_queries.shape[0], self.query_chunk_size):
            query_chunk = flat_queries[start : start + self.query_chunk_size]
            similarities = query_chunk @ bank_t
            k = min(self.nn_k, similarities.shape[1])
            best_sim = similarities.topk(k=k, dim=1).values
            distances = torch.sqrt(torch.clamp(2.0 - 2.0 * best_sim, min=0.0))
            outputs.append(distances.mean(dim=1))

        return torch.cat(outputs, dim=0).reshape(batch_size, patch_count)

    def reduce_patch_distances(self, patch_distances: torch.Tensor) -> torch.Tensor:
        if self.reduction == "max":
            return patch_distances.max(dim=1).values
        if self.reduction == "mean":
            return patch_distances.mean(dim=1)

        topk = max(1, int(math.ceil(patch_distances.shape[1] * self.topk_ratio)))
        topk = min(topk, patch_distances.shape[1])
        return torch.topk(patch_distances, k=topk, dim=1).values.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_embeddings = self.patch_embeddings(x)
        patch_distances = self.nearest_patch_distances(patch_embeddings)
        return self.reduce_patch_distances(patch_distances)


def ensure_processed_data(raw_pickle: Path, data_config_path: Path, metadata_csv: Path) -> None:
    if metadata_csv.exists():
        print(f"[data] Processed metadata already exists: {metadata_csv}", flush=True)
        return
    print(f"[data] Processed metadata not found. Generating from {raw_pickle} ...", flush=True)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "prepare_wm811k.py"),
        "--config",
        str(data_config_path),
    ]
    subprocess.run(command, check=True, cwd=str(PROJECT_ROOT))
    if not metadata_csv.exists():
        raise RuntimeError(f"prepare_wm811k.py ran but metadata CSV was not created: {metadata_csv}")


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
) -> DataLoader:
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def collect_scores(
    model: EfficientNetB1OneLayerPatchCoreModel,
    dataloader: DataLoader,
    device: torch.device,
    *,
    desc: str,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for inputs, labels in tqdm(dataloader, desc=desc, unit="batch", leave=False):
            scores = model(inputs.to(device)).cpu().numpy()
            for score, label in zip(scores.tolist(), labels.tolist()):
                rows.append({"score": float(score), "is_anomaly": int(label)})
    return pd.DataFrame(rows)


def save_score_distribution(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    threshold: float,
    out_path: Path,
    variant_name: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(val_df["score"], bins=40, alpha=0.85, color="#355070")
    axes[0].axvline(threshold, color="#d62828", linestyle="--", label=f"threshold={threshold:.4f}")
    axes[0].set_title(f"Val Normal Scores\n{variant_name}")
    axes[0].legend()
    axes[1].hist(
        test_df[test_df["is_anomaly"] == 0]["score"],
        bins=40,
        alpha=0.7,
        label="normal",
        color="#6d597a",
    )
    axes[1].hist(
        test_df[test_df["is_anomaly"] == 1]["score"],
        bins=40,
        alpha=0.7,
        label="anomaly",
        color="#e56b6f",
    )
    axes[1].axvline(threshold, color="#d62828", linestyle="--", label=f"threshold={threshold:.4f}")
    axes[1].set_title(f"Test Score Distribution\n{variant_name}")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_threshold_sweep_plot(
    sweep_df: pd.DataFrame,
    threshold: float,
    best_threshold: float,
    out_path: Path,
    variant_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sweep_df["threshold"], sweep_df["precision"], label="precision")
    ax.plot(sweep_df["threshold"], sweep_df["recall"], label="recall")
    ax.plot(sweep_df["threshold"], sweep_df["f1"], label="f1")
    ax.axvline(threshold, color="#d62828", linestyle="--", label=f"val threshold={threshold:.4f}")
    ax.axvline(best_threshold, color="#2a9d8f", linestyle=":", label=f"best sweep={best_threshold:.4f}")
    ax.set_title(f"Threshold Sweep\n{variant_name}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(metrics: dict[str, Any], out_path: Path, variant_name: str) -> None:
    confusion = np.array(metrics["confusion_matrix"], dtype=float)
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks([0, 1], labels=["pred_normal", "pred_anomaly"])
    ax.set_yticks([0, 1], labels=["true_normal", "true_anomaly"])
    ax.set_title(f"Confusion Matrix\n{variant_name}")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, int(confusion[row, col]), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_defect_breakdown(
    test_metadata: pd.DataFrame,
    test_df: pd.DataFrame,
    threshold: float,
    *,
    out_path: Path,
    csv_path: Path,
) -> None:
    df = test_metadata.copy().reset_index(drop=True)
    df["score"] = test_df["score"].values
    df["predicted"] = (df["score"] >= threshold).astype(int)
    defect_df = df[df["is_anomaly"] == 1].copy()
    recall_df = (
        defect_df.groupby("defect_type")
        .agg(count=("defect_type", "size"), detected=("predicted", "sum"), mean_score=("score", "mean"))
        .sort_values(["detected", "count"], ascending=[False, False])
    )
    recall_df["recall"] = recall_df["detected"] / recall_df["count"]
    recall_df.reset_index().to_csv(csv_path, index=False)

    top = recall_df.head(10).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["defect_type"], top["recall"], color="#8ab17d")
    ax.set_xlim(0.0, 1.0)
    ax.set_title("Top Defect-Type Recall")
    ax.set_xlabel("recall")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="EfficientNet-B1 one-layer PatchCore x240 runner")
    parser.add_argument("--raw-pickle", default=str(PROJECT_ROOT / "data" / "raw" / "LSWMD.pkl"))
    parser.add_argument(
        "--data-config",
        default=str(PROJECT_ROOT / "data" / "dataset" / "x240" / "benchmark_120k_5pct" / "data_config.toml"),
    )
    parser.add_argument(
        "--train-config",
        default=str(
            PROJECT_ROOT
            / "experiments"
            / "anomaly_detection"
            / "patchcore"
            / "efficientnet_b1"
            / "x240"
            / "main_120k"
            / "train_config.toml"
        ),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    raw_pickle = Path(args.raw_pickle)
    data_config_path = Path(args.data_config)
    train_config_path = Path(args.train_config)
    train_cfg = load_toml(train_config_path)

    output_dir = resolve_project_path(args.output_dir or str(train_cfg["run"]["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "results"
    plots_dir = output_dir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    seed = int(train_cfg["run"].get("seed", 42))
    set_seed(seed)

    data_section = train_cfg["data"]
    image_size = int(data_section["image_size"])
    batch_size = int(data_section["batch_size"])
    num_workers = int(args.num_workers) if args.num_workers is not None else int(data_section.get("num_workers", 0))
    persistent_workers = bool(data_section.get("persistent_workers", False))
    prefetch_factor = int(data_section.get("prefetch_factor", 2))
    metadata_csv = resolve_project_path(str(data_section["metadata_csv"]))

    ensure_processed_data(raw_pickle, data_config_path, metadata_csv)

    model_cfg = train_cfg["model"]
    threshold_quantile = float(train_cfg["scoring"]["threshold_quantile"])
    sweep_variants: list[dict[str, Any]] = list(train_cfg["sweep_variants"])
    device = resolve_device(str(train_cfg["training"].get("device", "auto")))
    pin_memory = device.type == "cuda"

    print(
        f"[setup] device={device}, image_size={image_size}, batch_size={batch_size}, workers={num_workers}",
        flush=True,
    )

    metadata = pd.read_csv(metadata_csv)
    test_metadata = metadata[metadata["split"] == "test"].reset_index(drop=True)

    model_input_size = int(model_cfg["backbone_input_size"])
    train_dataset = OneHotWaferMapDataset(WaferMapDataset(metadata_csv, split="train", image_size=image_size), model_input_size)
    val_dataset = OneHotWaferMapDataset(WaferMapDataset(metadata_csv, split="val", image_size=image_size), model_input_size)
    test_dataset = OneHotWaferMapDataset(WaferMapDataset(metadata_csv, split="test", image_size=image_size), model_input_size)

    val_loader = make_loader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    test_loader = make_loader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    run_config: dict[str, Any] = {
        "run": {"output_dir": str(output_dir), "seed": seed},
        "data": {
            "metadata_csv": str(metadata_csv),
            "image_size": image_size,
            "batch_size": batch_size,
            "num_workers": num_workers,
        },
        "model": {
            "backbone_type": str(model_cfg["backbone_type"]),
            "pretrained": bool(model_cfg.get("pretrained", True)),
            "freeze_backbone": bool(model_cfg.get("freeze_backbone", True)),
            "backbone_input_size": model_input_size,
            "feature_idx": int(model_cfg["feature_idx"]),
            "patch_embed_dim": int(model_cfg["patch_embed_dim"]),
            "nn_k": int(model_cfg["nn_k"]),
            "query_chunk_size": int(model_cfg["query_chunk_size"]),
            "amp": bool(model_cfg.get("amp", True)),
        },
        "scoring": {"threshold_quantile": threshold_quantile},
        "sweep_variants": sweep_variants,
        "split": {"mode": train_cfg.get("split", {}).get("mode", "normal_only_120k_5pct")},
    }
    (results_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    sweep_rows: list[dict[str, Any]] = []

    for variant in sweep_variants:
        variant_name = str(variant["name"])
        memory_bank_size = int(variant["memory_bank_size"])
        reduction = str(variant["reduction"])
        topk_ratio = float(variant.get("topk_ratio", 0.03))

        print(f"\n[variant] {variant_name}", flush=True)
        variant_dir = output_dir / variant_name
        variant_eval_dir = variant_dir / "results" / "evaluation"
        variant_plots_dir = variant_dir / "plots"
        variant_ckpt_dir = variant_dir / "checkpoints"
        for directory in [variant_eval_dir, variant_plots_dir, variant_ckpt_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        model = EfficientNetB1OneLayerPatchCoreModel(
            model_input_size=model_input_size,
            feature_idx=int(model_cfg["feature_idx"]),
            patch_embed_dim=int(model_cfg["patch_embed_dim"]),
            reduction=reduction,
            topk_ratio=topk_ratio,
            nn_k=int(model_cfg["nn_k"]),
            query_chunk_size=int(model_cfg["query_chunk_size"]),
            amp_enabled=bool(model_cfg.get("amp", True)),
            pretrained=bool(model_cfg.get("pretrained", True)),
            freeze_backbone=bool(model_cfg.get("freeze_backbone", True)),
        ).to(device).eval()

        print(f"  Building memory bank (target={memory_bank_size:,} patches) ...", flush=True)
        memory_subset = build_memory_subset(train_dataset, memory_bank_size, model.patches_per_image, seed)
        memory_loader = make_loader(
            memory_subset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
        memory_bank = collect_memory_bank(model, memory_loader, device, memory_bank_size, seed)
        model.set_memory_bank(memory_bank.to(device))
        print(f"  Memory bank: {memory_bank.shape[0]:,} patches x {memory_bank.shape[1]} dims", flush=True)

        print("  Scoring val ...", flush=True)
        val_df = collect_scores(model, val_loader, device, desc="  val scoring")
        print("  Scoring test ...", flush=True)
        test_df = collect_scores(model, test_loader, device, desc="  test scoring")

        val_normal_scores = val_df.loc[val_df["is_anomaly"] == 0, "score"].to_numpy()
        threshold = float(np.quantile(val_normal_scores, threshold_quantile))
        labels = test_df["is_anomaly"].to_numpy()
        scores = test_df["score"].to_numpy()
        metrics = summarize_threshold_metrics(labels, scores, threshold)
        sweep_df, best_sweep = sweep_threshold_metrics(labels, scores)

        print(
            f"  threshold={threshold:.4f}  P={metrics['precision']:.3f}  "
            f"R={metrics['recall']:.3f}  F1={metrics['f1']:.3f}  "
            f"AUROC={metrics['auroc']:.4f}",
            flush=True,
        )

        val_df.to_csv(variant_eval_dir / "val_scores.csv", index=False)
        test_df.to_csv(variant_eval_dir / "test_scores.csv", index=False)
        sweep_df.to_csv(variant_eval_dir / "threshold_sweep.csv", index=False)

        save_defect_breakdown(
            test_metadata,
            test_df,
            threshold,
            out_path=variant_plots_dir / "defect_breakdown.png",
            csv_path=variant_eval_dir / "defect_breakdown.csv",
        )
        save_score_distribution(val_df, test_df, threshold, variant_plots_dir / "score_distribution.png", variant_name)
        save_threshold_sweep_plot(
            sweep_df,
            threshold,
            float(best_sweep["threshold"]),
            variant_plots_dir / "threshold_sweep.png",
            variant_name,
        )
        save_confusion_matrix(metrics, variant_plots_dir / "confusion_matrix.png", variant_name)

        summary = {
            "name": variant_name,
            "memory_bank_size": memory_bank_size,
            "memory_subset_images": len(memory_subset),
            "patches_per_image": model.patches_per_image,
            "feature_dim": model.feature_dim,
            "reduction": reduction,
            "topk_ratio": topk_ratio,
            "threshold": float(threshold),
            "threshold_quantile": threshold_quantile,
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
            "auroc": float(metrics["auroc"]),
            "auprc": float(metrics["auprc"]),
            "best_sweep_threshold": float(best_sweep["threshold"]),
            "best_sweep_precision": float(best_sweep["precision"]),
            "best_sweep_recall": float(best_sweep["recall"]),
            "best_sweep_f1": float(best_sweep["f1"]),
            "predicted_anomalies": int(metrics["predicted_anomalies"]),
            "output_dir": str(variant_dir),
            "teacher_backbone": "efficientnet_b1",
            "feature_idx": int(model_cfg["feature_idx"]),
            "nn_k": int(model_cfg["nn_k"]),
        }
        (variant_dir / "results" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        checkpoint_path = variant_ckpt_dir / "best_model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": {
                    "backbone_type": str(model_cfg["backbone_type"]),
                    "feature_idx": int(model_cfg["feature_idx"]),
                    "patch_embed_dim": int(model_cfg["patch_embed_dim"]),
                    "reduction": reduction,
                    "topk_ratio": topk_ratio,
                    "nn_k": int(model_cfg["nn_k"]),
                    "query_chunk_size": int(model_cfg["query_chunk_size"]),
                    "amp": bool(model_cfg.get("amp", True)),
                    "feature_dim": model.feature_dim,
                    "memory_bank_shape": list(model.memory_bank.shape),
                },
                "summary": summary,
                "seed": seed,
            },
            checkpoint_path,
        )
        approx_gb = checkpoint_path.stat().st_size / 1e9
        print(f"  Checkpoint saved: {checkpoint_path} ({approx_gb:.2f} GB)", flush=True)

        sweep_rows.append(
            {
                "name": variant_name,
                "memory_bank_size": memory_bank_size,
                "memory_subset_images": len(memory_subset),
                "patches_per_image": model.patches_per_image,
                "feature_dim": model.feature_dim,
                "reduction": reduction,
                "topk_ratio": topk_ratio,
                "threshold": float(threshold),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "auroc": float(metrics["auroc"]),
                "auprc": float(metrics["auprc"]),
                "best_sweep_threshold": float(best_sweep["threshold"]),
                "best_sweep_precision": float(best_sweep["precision"]),
                "best_sweep_recall": float(best_sweep["recall"]),
                "best_sweep_f1": float(best_sweep["f1"]),
                "predicted_anomalies": int(metrics["predicted_anomalies"]),
                "output_dir": str(variant_dir),
            }
        )

        del model
        del memory_bank
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    sweep_results_df = pd.DataFrame(sweep_rows)
    sweep_results_df.to_csv(results_dir / "patchcore_sweep_results.csv", index=False)

    best_row = sweep_results_df.sort_values(["f1", "auroc"], ascending=False).iloc[0].to_dict()
    best_variant_name = str(best_row["name"])
    best_variant_dir = output_dir / best_variant_name
    best_checkpoint_path = best_variant_dir / "checkpoints" / "best_model.pt"

    sweep_summary = {
        "sweep_variants": sweep_results_df["name"].tolist(),
        "base_output_dir": str(output_dir),
        "teacher_backbone": "efficientnet_b1",
        "feature_idx": int(model_cfg["feature_idx"]),
        "best_variant": best_row,
    }
    (results_dir / "patchcore_sweep_summary.json").write_text(json.dumps(sweep_summary, indent=2), encoding="utf-8")

    selected_checkpoint_meta = {
        "variant_name": best_variant_name,
        "checkpoint_path": str(best_checkpoint_path),
        "checkpoint_name": str(best_checkpoint_path),
        "approx_size_gb": round(best_checkpoint_path.stat().st_size / 1e9, 3) if best_checkpoint_path.exists() else None,
        "memory_bank_shape": [int(best_row["memory_bank_size"]), int(best_row["feature_dim"])],
        "note": "Checkpoint includes the fitted PatchCore memory bank and the frozen projection state.",
    }
    (results_dir / "selected_checkpoint.json").write_text(json.dumps(selected_checkpoint_meta, indent=2), encoding="utf-8")

    plot_df = sweep_results_df.sort_values(["f1", "auroc"], ascending=False).reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh(plot_df["name"], plot_df["f1"], color="#264653")
    axes[0].set_title("EfficientNet-B1 PatchCore x240: F1")
    axes[0].invert_yaxis()
    axes[1].barh(plot_df["name"], plot_df["auroc"], color="#2a9d8f")
    axes[1].set_title("EfficientNet-B1 PatchCore x240: AUROC")
    axes[1].invert_yaxis()
    plt.tight_layout()
    fig.savefig(plots_dir / "sweep_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "status": "complete",
        "output_dir": str(output_dir),
        "best_variant": best_variant_name,
        "best_f1": float(best_row["f1"]),
        "best_auroc": float(best_row["auroc"]),
        "sweep_variants": sweep_results_df["name"].tolist(),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        f"\n[done] Best variant: {best_variant_name}  "
        f"F1={best_row['f1']:.4f}  AUROC={best_row['auroc']:.4f}",
        flush=True,
    )
    print(f"[done] Artifacts at: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
