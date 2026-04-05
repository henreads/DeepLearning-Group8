#!/usr/bin/env python3
"""Runner: WideResNet50-2 multilayer PatchCore x224 with UMAP embedding export.

Usage (called by the Modal app or locally):
    python scripts/run_patchcore_wrn50_x224_umap.py \\
        --raw-pickle data/raw/LSWMD.pkl \\
        --output-dir experiments/anomaly_detection/patchcore/wideresnet50/x224/multilayer_umap/artifacts/patchcore-wideresnet50-multilayer-umap \\
        --num-workers 4
"""

from __future__ import annotations

import argparse
import json
import math
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
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Resolve project root and configure import paths
# ---------------------------------------------------------------------------

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
from wafer_defect.evaluation.umap_reference import export_reference_umap_bundle  # noqa: E402


# ---------------------------------------------------------------------------
# Model definition: WideResNet50-2 multilayer PatchCore
# ---------------------------------------------------------------------------

class WideResNet50_2MultiLayerExtractor(nn.Module):
    """Frozen WRN50-2 feature extractor adapted to single-channel input."""

    def __init__(
        self,
        teacher_layers: list[str],
        pretrained: bool = True,
        input_size: int = 224,
        freeze_backbone: bool = True,
        normalize_imagenet: bool = True,
    ) -> None:
        super().__init__()
        self.teacher_layers = [str(layer).lower() for layer in teacher_layers]
        from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

        weights = Wide_ResNet50_2_Weights.DEFAULT if pretrained else None
        backbone = wide_resnet50_2(weights=weights)

        # Adapt conv1 from 3-channel to 1-channel by averaging input weights
        orig = backbone.conv1
        adapted = nn.Conv2d(
            1, orig.out_channels,
            kernel_size=orig.kernel_size,
            stride=orig.stride,
            padding=orig.padding,
            bias=False,
        )
        with torch.no_grad():
            adapted.weight.copy_(orig.weight.mean(dim=1, keepdim=True))
        backbone.conv1 = adapted

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.input_size = int(input_size)
        self.normalize_imagenet = bool(normalize_imagenet)
        self.register_buffer("image_mean", torch.tensor([0.4490], dtype=torch.float32).view(1, 1, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.2260], dtype=torch.float32).view(1, 1, 1, 1))
        self.feature_dims = {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}
        downsample_map = {"layer1": 4, "layer2": 8, "layer3": 16, "layer4": 32}
        self.output_spatial = max(
            1, self.input_size // max(downsample_map[layer] for layer in self.teacher_layers)
        )
        if freeze_backbone:
            for p in self.parameters():
                p.requires_grad = False

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        if self.normalize_imagenet:
            x = (x - self.image_mean) / self.image_std
        return x

    def forward_feature_maps(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        x = self.preprocess(x)
        x = self.stem(x)
        x = self.layer1(x)
        if "layer1" in self.teacher_layers:
            out["layer1"] = x
        x = self.layer2(x)
        if "layer2" in self.teacher_layers:
            out["layer2"] = x
        x = self.layer3(x)
        if "layer3" in self.teacher_layers:
            out["layer3"] = x
        x = self.layer4(x)
        if "layer4" in self.teacher_layers:
            out["layer4"] = x
        return out


class MultiLayerPatchCoreModel(nn.Module):
    """PatchCore with WRN50-2 multi-layer feature extraction."""

    def __init__(
        self,
        teacher_layers: list[str],
        reduction: str = "topk_mean",
        topk_ratio: float = 0.10,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        backbone_input_size: int = 224,
        normalize_imagenet: bool = True,
        query_chunk_size: int = 1024,
        memory_chunk_size: int = 4096,
    ) -> None:
        super().__init__()
        self.teacher_layers = [str(layer).lower() for layer in teacher_layers]
        self.reduction = str(reduction)
        self.topk_ratio = float(topk_ratio)
        self.query_chunk_size = int(query_chunk_size)
        self.memory_chunk_size = int(memory_chunk_size)
        self.teacher = WideResNet50_2MultiLayerExtractor(
            teacher_layers=self.teacher_layers,
            pretrained=pretrained,
            input_size=backbone_input_size,
            freeze_backbone=freeze_backbone,
            normalize_imagenet=normalize_imagenet,
        )
        self.feature_dim = int(sum(self.teacher.feature_dims[layer] for layer in self.teacher_layers))
        self.reduced_spatial = int(self.teacher.output_spatial)
        self.register_buffer("memory_bank", torch.empty(0, self.feature_dim))

    @property
    def patches_per_image(self) -> int:
        return self.reduced_spatial * self.reduced_spatial

    def patch_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        fmaps = self.teacher.forward_feature_maps(x)
        target = min(fm.shape[-1] for fm in fmaps.values())
        parts: list[torch.Tensor] = []
        for layer in self.teacher_layers:
            fm = fmaps[layer]
            if fm.shape[-1] != target or fm.shape[-2] != target:
                fm = F.interpolate(fm, size=(target, target), mode="bilinear", align_corners=False)
            parts.append(F.normalize(fm.permute(0, 2, 3, 1).reshape(x.shape[0], -1, fm.shape[1]), p=2, dim=-1))
        return F.normalize(torch.cat(parts, dim=-1), p=2, dim=-1)

    def set_memory_bank(self, memory_bank: torch.Tensor) -> None:
        normalized = F.normalize(memory_bank.float(), p=2, dim=1)
        self.memory_bank = normalized.to(device=self.memory_bank.device, dtype=self.memory_bank.dtype)

    def nearest_patch_distances(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        B, P, _ = patch_embeddings.shape
        flat = patch_embeddings.reshape(-1, self.feature_dim)
        mins: list[torch.Tensor] = []
        for qs in range(0, flat.shape[0], self.query_chunk_size):
            qc = flat[qs : qs + self.query_chunk_size]
            best: torch.Tensor | None = None
            for ms in range(0, self.memory_bank.shape[0], self.memory_chunk_size):
                mc = self.memory_bank[ms : ms + self.memory_chunk_size]
                cb = torch.cdist(qc, mc).min(dim=1).values
                best = cb if best is None else torch.minimum(best, cb)
            mins.append(best)
        return torch.cat(mins).reshape(B, P)

    def reduce_patch_distances(self, d: torch.Tensor) -> torch.Tensor:
        if self.reduction == "max":
            return d.max(dim=1).values
        if self.reduction == "mean":
            return d.mean(dim=1)
        k = max(1, int(math.ceil(d.shape[1] * self.topk_ratio)))
        return torch.topk(d, k=k, dim=1).values.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reduce_patch_distances(self.nearest_patch_distances(self.patch_embeddings(x)))


def load_model_state_ignoring_memory_bank(model: nn.Module, state_dict: dict[str, Any]) -> None:
    """Restore model weights while letting the runtime rebuild large memory-bank buffers."""
    filtered_state_dict = {key: value for key, value in state_dict.items() if key != "memory_bank"}
    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
    missing = [key for key in missing if key != "memory_bank"]
    unexpected = [key for key in unexpected if key != "memory_bank"]
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint restore mismatch after filtering memory_bank: "
            f"missing={missing}, unexpected={unexpected}"
        )


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def build_memory_subset(dataset: Dataset, memory_bank_size: int, patches_per_image: int, seed: int) -> Subset:
    image_count = min(len(dataset), max(1, math.ceil(memory_bank_size / patches_per_image)))
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(dataset), size=image_count, replace=False))
    return Subset(dataset, indices.tolist())


def collect_memory_bank(
    model: MultiLayerPatchCoreModel,
    dataloader: DataLoader,
    device: torch.device,
    target_size: int,
    seed: int,
) -> torch.Tensor:
    model.eval()
    batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for inputs, labels in tqdm(dataloader, desc="  memory bank", unit="batch", leave=False):
            normal_mask = (labels == 0)
            if not normal_mask.any():
                continue
            embs = model.patch_embeddings(inputs[normal_mask].to(device)).reshape(-1, model.feature_dim)
            batches.append(embs.cpu())
    bank = torch.cat(batches)
    if bank.shape[0] > target_size:
        g = torch.Generator().manual_seed(seed)
        keep = torch.randperm(bank.shape[0], generator=g)[:target_size]
        bank = bank[keep]
    return bank


def collect_scores(
    model: MultiLayerPatchCoreModel,
    dataloader: DataLoader,
    device: torch.device,
    desc: str = "  scoring",
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for inputs, labels in tqdm(dataloader, desc=desc, unit="batch", leave=False):
            scores = model(inputs.to(device)).cpu().numpy()
            for s, lb in zip(scores.tolist(), labels.tolist()):
                rows.append({"score": float(s), "is_anomaly": int(lb)})
    return pd.DataFrame(rows)


def collect_wafer_embeddings(
    model: MultiLayerPatchCoreModel,
    dataloader: DataLoader,
    device: torch.device,
    desc: str = "  embeddings",
) -> tuple[np.ndarray, np.ndarray]:
    """Mean-pool patch embeddings to one vector per wafer."""
    model.eval()
    all_embs: list[np.ndarray] = []
    all_labels: list[int] = []
    with torch.inference_mode():
        for inputs, labels in tqdm(dataloader, desc=desc, unit="batch", leave=False):
            embs = model.patch_embeddings(inputs.to(device))   # (B, P, D)
            all_embs.append(embs.mean(dim=1).cpu().numpy())     # (B, D)
            all_labels.extend(labels.tolist())
    return np.concatenate(all_embs).astype(np.float32), np.array(all_labels, dtype=np.int64)


def build_model_from_variant(
    *,
    teacher_layers: list[str],
    reduction: str,
    topk_ratio: float,
    pretrained: bool,
    freeze_backbone: bool,
    backbone_input_size: int,
    normalize_imagenet: bool,
    query_chunk_size: int,
    memory_chunk_size: int,
    device: torch.device,
) -> MultiLayerPatchCoreModel:
    return MultiLayerPatchCoreModel(
        teacher_layers=teacher_layers,
        reduction=reduction,
        topk_ratio=topk_ratio,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        backbone_input_size=backbone_input_size,
        normalize_imagenet=normalize_imagenet,
        query_chunk_size=query_chunk_size,
        memory_chunk_size=memory_chunk_size,
    ).to(device)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def save_score_distribution(val_df: pd.DataFrame, test_df: pd.DataFrame, threshold: float, out_path: Path, variant_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(val_df["score"], bins=40, alpha=0.85, color="#577590")
    axes[0].axvline(threshold, color="red", linestyle="--", label=f"threshold={threshold:.4f}")
    axes[0].set_title(f"Val Normal Scores\n{variant_name}")
    axes[0].legend()
    axes[1].hist(test_df[test_df["is_anomaly"] == 0]["score"], bins=40, alpha=0.7, label="normal", color="#4d908e")
    axes[1].hist(test_df[test_df["is_anomaly"] == 1]["score"], bins=40, alpha=0.7, label="anomaly", color="#f3722c")
    axes[1].axvline(threshold, color="red", linestyle="--", label=f"threshold={threshold:.4f}")
    axes[1].set_title(f"Test Score Distribution\n{variant_name}")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_threshold_sweep_plot(sweep_df: pd.DataFrame, threshold: float, best_thresh: float, out_path: Path, variant_name: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sweep_df["threshold"], sweep_df["precision"], label="precision")
    ax.plot(sweep_df["threshold"], sweep_df["recall"], label="recall")
    ax.plot(sweep_df["threshold"], sweep_df["f1"], label="f1")
    ax.axvline(threshold, color="red", linestyle="--", label=f"val threshold={threshold:.4f}")
    ax.axvline(best_thresh, color="green", linestyle=":", label=f"best sweep={best_thresh:.4f}")
    ax.set_title(f"Threshold Sweep (Test)\n{variant_name}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(metrics: dict[str, Any], out_path: Path, variant_name: str) -> None:
    cm = np.array(metrics["confusion_matrix"], dtype=float)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["pred_normal", "pred_anomaly"])
    ax.set_yticks([0, 1], labels=["true_normal", "true_anomaly"])
    ax.set_title(f"Confusion Matrix\n{variant_name}")
    for r in range(2):
        for c in range(2):
            ax.text(c, r, int(cm[r, c]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_defect_breakdown(test_metadata: pd.DataFrame, test_df: pd.DataFrame, threshold: float, out_path: Path, csv_path: Path) -> None:
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


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def ensure_processed_data(
    raw_pickle: Path,
    data_config_path: Path,
    metadata_csv: Path,
) -> None:
    if metadata_csv.exists():
        print(f"[data] Processed metadata already exists: {metadata_csv}", flush=True)
        return
    print(f"[data] Processed metadata not found. Generating from {raw_pickle} ...", flush=True)
    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "prepare_wm811k.py"),
        "--config", str(data_config_path),
    ]
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    if not metadata_csv.exists():
        raise RuntimeError(f"prepare_wm811k.py ran but metadata CSV was not created: {metadata_csv}")
    print(f"[data] Processed metadata created: {metadata_csv}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="WRN50-2 x224 PatchCore training + UMAP export")
    parser.add_argument("--raw-pickle", default=str(PROJECT_ROOT / "data" / "raw" / "LSWMD.pkl"))
    parser.add_argument(
        "--data-config",
        default=str(PROJECT_ROOT / "data" / "dataset" / "x224" / "benchmark_50k_5pct" / "data_config.toml"),
    )
    parser.add_argument(
        "--train-config",
        default=str(
            PROJECT_ROOT
            / "experiments"
            / "anomaly_detection"
            / "patchcore"
            / "wideresnet50"
            / "x224"
            / "multilayer_umap"
            / "train_config.toml"
        ),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--skip-umap", action="store_true", help="Skip UMAP export (faster, for debugging)")
    parser.add_argument("--umap-only", action="store_true", help="Reuse existing checkpoints/results and generate only UMAP artifacts")
    args = parser.parse_args()

    raw_pickle = Path(args.raw_pickle)
    data_config_path = Path(args.data_config)
    train_config_path = Path(args.train_config)
    train_cfg = load_toml(train_config_path)

    # Resolve output dir
    output_dir_str = args.output_dir or str(PROJECT_ROOT / train_cfg["run"]["output_dir"])
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "results"
    plots_dir = output_dir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    seed = int(train_cfg["run"].get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data config
    data_cfg = train_cfg["data"]
    image_size = int(data_cfg["image_size"])
    batch_size = int(data_cfg["batch_size"])
    num_workers = args.num_workers
    metadata_csv_rel = data_cfg["metadata_csv"]
    metadata_csv = PROJECT_ROOT / metadata_csv_rel

    ensure_processed_data(raw_pickle, data_config_path, metadata_csv)

    # Model config
    model_cfg = train_cfg["model"]
    teacher_layers: list[str] = [str(l) for l in model_cfg["teacher_layers"]]
    pretrained = bool(model_cfg.get("pretrained", True))
    freeze_backbone = bool(model_cfg.get("freeze_backbone", True))
    backbone_input_size = int(model_cfg.get("backbone_input_size", 224))
    normalize_imagenet = bool(model_cfg.get("normalize_imagenet", True))
    query_chunk_size = int(model_cfg.get("query_chunk_size", 1024))
    memory_chunk_size = int(model_cfg.get("memory_chunk_size", 4096))
    threshold_quantile = float(train_cfg["scoring"]["threshold_quantile"])
    sweep_variants: list[dict[str, Any]] = list(train_cfg["sweep_variants"])

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}, image_size={image_size}, workers={num_workers}", flush=True)

    # Load metadata for defect breakdown
    metadata = pd.read_csv(metadata_csv)
    test_metadata = metadata[metadata["split"] == "test"].reset_index(drop=True)

    # Datasets
    print("[data] Loading datasets ...", flush=True)
    train_dataset = WaferMapDataset(metadata_csv, split="train", image_size=image_size)
    val_dataset = WaferMapDataset(metadata_csv, split="val", image_size=image_size)
    test_dataset = WaferMapDataset(metadata_csv, split="test", image_size=image_size)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"[data] train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}", flush=True)

    # Save run config
    run_config: dict[str, Any] = {
        "run": {"output_dir": str(output_dir), "seed": seed},
        "data": {
            "image_size": image_size,
            "batch_size": batch_size,
            "num_workers": num_workers,
        },
        "model": {
            "backbone_type": "wideresnet50_2",
            "teacher_layers": teacher_layers,
            "pretrained": pretrained,
            "freeze_backbone": freeze_backbone,
            "backbone_input_size": backbone_input_size,
            "normalize_imagenet": normalize_imagenet,
            "query_chunk_size": query_chunk_size,
            "memory_chunk_size": memory_chunk_size,
        },
        "scoring": {"threshold_quantile": threshold_quantile},
        "sweep_variants": sweep_variants,
        "split": {"mode": train_cfg.get("split", {}).get("mode", "report_50k_5pct")},
    }
    (results_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    if args.umap_only and args.skip_umap:
        raise ValueError("--umap-only and --skip-umap cannot be used together.")

    # ---------------------------------------------------------------------------
    # Sweep variants
    # ---------------------------------------------------------------------------
    sweep_rows: list[dict[str, Any]] = []

    for variant in sweep_variants:
        variant_name = str(variant["name"])
        memory_bank_size = int(variant["memory_bank_size"])
        reduction = str(variant["reduction"])
        topk_ratio = float(variant.get("topk_ratio", 0.10))

        print(f"\n[variant] {variant_name}", flush=True)

        variant_dir = output_dir / variant_name
        variant_eval_dir = variant_dir / "results" / "evaluation"
        variant_umap_dir = variant_dir / "results" / "umap"
        variant_plots_dir = variant_dir / "plots"
        variant_ckpt_dir = variant_dir / "checkpoints"
        for d in [variant_eval_dir, variant_umap_dir, variant_plots_dir, variant_ckpt_dir]:
            d.mkdir(parents=True, exist_ok=True)

        if args.umap_only:
            ckpt_path = variant_ckpt_dir / "best_model.pt"
            val_scores_path = variant_eval_dir / "val_scores.csv"
            test_scores_path = variant_eval_dir / "test_scores.csv"
            summary_path = variant_dir / "results" / "summary.json"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"UMAP-only mode expected checkpoint at {ckpt_path}")
            if not val_scores_path.exists() or not test_scores_path.exists():
                raise FileNotFoundError(f"UMAP-only mode expected saved score CSVs under {variant_eval_dir}")
            if not summary_path.exists():
                raise FileNotFoundError(f"UMAP-only mode expected summary at {summary_path}")

            model = build_model_from_variant(
                teacher_layers=teacher_layers,
                reduction=reduction,
                topk_ratio=topk_ratio,
                pretrained=pretrained,
                freeze_backbone=freeze_backbone,
                backbone_input_size=backbone_input_size,
                normalize_imagenet=normalize_imagenet,
                query_chunk_size=query_chunk_size,
                memory_chunk_size=memory_chunk_size,
                device=device,
            )
            checkpoint = torch.load(ckpt_path, map_location=device)
            load_model_state_ignoring_memory_bank(model, checkpoint["model_state_dict"])
            val_df = pd.read_csv(val_scores_path)
            test_df = pd.read_csv(test_scores_path)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            threshold = float(summary["threshold"])
            print(f"  Reusing checkpoint and scores for UMAP: {ckpt_path}", flush=True)
        else:
            # Build model
            model = build_model_from_variant(
                teacher_layers=teacher_layers,
                reduction=reduction,
                topk_ratio=topk_ratio,
                pretrained=pretrained,
                freeze_backbone=freeze_backbone,
                backbone_input_size=backbone_input_size,
                normalize_imagenet=normalize_imagenet,
                query_chunk_size=query_chunk_size,
                memory_chunk_size=memory_chunk_size,
                device=device,
            )

            # Memory bank
            print(f"  Building memory bank (target={memory_bank_size:,} patches) ...", flush=True)
            memory_subset = build_memory_subset(train_dataset, memory_bank_size, model.patches_per_image, seed)
            mem_loader = DataLoader(memory_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
            bank = collect_memory_bank(model, mem_loader, device, memory_bank_size, seed)
            model.set_memory_bank(bank)
            print(f"  Memory bank: {bank.shape[0]:,} patches x {bank.shape[1]} dims", flush=True)

            # Scoring
            print("  Scoring val ...", flush=True)
            val_df = collect_scores(model, val_loader, device, desc="  val scoring")
            print("  Scoring test ...", flush=True)
            test_df = collect_scores(model, test_loader, device, desc="  test scoring")

            # Threshold
            val_normal_scores = val_df.loc[val_df["is_anomaly"] == 0, "score"]
            threshold = float(val_normal_scores.quantile(threshold_quantile))

            # Metrics
            test_labels = test_df["is_anomaly"].to_numpy()
            test_scores = test_df["score"].to_numpy()
            metrics = summarize_threshold_metrics(test_labels, test_scores, threshold)
            sweep_df, best_sweep = sweep_threshold_metrics(test_labels, test_scores)

            print(
                f"  threshold={threshold:.4f}  P={metrics['precision']:.3f}  "
                f"R={metrics['recall']:.3f}  F1={metrics['f1']:.3f}  "
                f"AUROC={metrics['auroc']:.4f}",
                flush=True,
            )

            # Save scores + sweep
            val_df.to_csv(variant_eval_dir / "val_scores.csv", index=False)
            test_df.to_csv(variant_eval_dir / "test_scores.csv", index=False)
            sweep_df.to_csv(variant_eval_dir / "threshold_sweep.csv", index=False)

            # Defect breakdown
            save_defect_breakdown(
                test_metadata,
                test_df,
                threshold,
                out_path=variant_plots_dir / "defect_breakdown.png",
                csv_path=variant_eval_dir / "selected_defect_breakdown.csv",
            )

            # Plots
            save_score_distribution(val_df, test_df, threshold, variant_plots_dir / "score_distribution.png", variant_name)
            save_threshold_sweep_plot(sweep_df, threshold, float(best_sweep["threshold"]), variant_plots_dir / "threshold_sweep.png", variant_name)
            save_confusion_matrix(metrics, variant_plots_dir / "confusion_matrix.png", variant_name)

            # Summary
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
                "teacher_backbone": "wideresnet50_2",
                "teacher_layers": teacher_layers,
            }
            (variant_dir / "results" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

            # Checkpoint
            ckpt_path = variant_ckpt_dir / "best_model.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "teacher_layers": teacher_layers,
                        "reduction": reduction,
                        "topk_ratio": topk_ratio,
                        "backbone_input_size": backbone_input_size,
                        "normalize_imagenet": normalize_imagenet,
                        "query_chunk_size": query_chunk_size,
                        "memory_chunk_size": memory_chunk_size,
                        "feature_dim": model.feature_dim,
                        "memory_bank_shape": list(model.memory_bank.shape),
                    },
                    "summary": summary,
                    "seed": seed,
                },
                ckpt_path,
            )
            approx_gb = ckpt_path.stat().st_size / 1e9
            print(f"  Checkpoint saved: {ckpt_path} ({approx_gb:.2f} GB)", flush=True)

        # ---------------------------------------------------------------------------
        # UMAP embedding export (best variant only or all — controlled by runner)
        # ---------------------------------------------------------------------------
        if not args.skip_umap:
            print("  Collecting train embeddings ...", flush=True)
            train_normal_indices = [
                i for i, row in enumerate(pd.read_csv(metadata_csv).itertuples())
                if row.split == "train" and row.is_anomaly == 0
            ]
            train_normal_subset = Subset(train_dataset, train_normal_indices)
            train_loader = DataLoader(
                train_normal_subset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True,
            )
            train_embs, _ = collect_wafer_embeddings(model, train_loader, device, desc="  train embeddings")
            train_labels = np.zeros(len(train_embs), dtype=np.int64)

            print("  Collecting val embeddings ...", flush=True)
            val_full_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
            val_embs, val_labels = collect_wafer_embeddings(model, val_full_loader, device, desc="  val embeddings")

            print("  Collecting test embeddings ...", flush=True)
            test_full_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
            test_embs, test_labels = collect_wafer_embeddings(model, test_full_loader, device, desc="  test embeddings")

            # Save raw embeddings
            np.save(variant_umap_dir / "train_embeddings.npy", train_embs)
            np.save(variant_umap_dir / "val_embeddings.npy", val_embs)
            np.save(variant_umap_dir / "val_labels.npy", val_labels)
            np.save(variant_umap_dir / "test_embeddings.npy", test_embs)
            np.save(variant_umap_dir / "test_labels.npy", test_labels)
            print(f"  Embeddings saved to {variant_umap_dir}", flush=True)

            # UMAP
            print("  Running UMAP ...", flush=True)
            try:
                import umap as umap_module

                val_model_scores = val_df["score"].to_numpy().astype(np.float32)
                test_model_scores = test_df["score"].to_numpy().astype(np.float32)

                umap_bundle = export_reference_umap_bundle(
                    output_dir=variant_umap_dir,
                    umap_module=umap_module,
                    train_normal_embeddings=train_embs,
                    val_embeddings=val_embs,
                    val_labels=val_labels,
                    test_embeddings=test_embs,
                    test_labels=test_labels,
                    val_model_scores=val_model_scores,
                    test_model_scores=test_model_scores,
                    pca_components=50,
                    n_neighbors=15,
                    min_dist=0.1,
                    knn_k=15,
                    metric="euclidean",
                    random_state=seed,
                    title_prefix=f"WRN50-2 PatchCore {variant_name}",
                    split_plot_filename="umap_by_split.png",
                    score_plot_filename="umap_by_score.png",
                    points_filename="umap_points.csv",
                    summary_filename="umap_summary.json",
                    sweep_filename="umap_knn_threshold_sweep.csv",
                )

                # Copy the split plot to the variant plots dir for the review notebook
                umap_split_png = variant_umap_dir / "umap_by_split.png"
                if umap_split_png.exists():
                    import shutil
                    shutil.copy2(umap_split_png, variant_plots_dir / "umap_by_split.png")
                print(f"  UMAP done. Summary: {umap_bundle['summary'].get('umap_knn_threshold', 'n/a')}", flush=True)

            except ImportError:
                print("  umap-learn not installed — skipping UMAP export.", flush=True)

        if not args.umap_only:
            # Collect row for sweep table
            sweep_rows.append({
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
            })

        # Release GPU memory before next variant
        del model
        if not args.umap_only:
            del bank
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---------------------------------------------------------------------------
    # Sweep-level summary
    # ---------------------------------------------------------------------------
    if args.umap_only:
        print(f"\n[done] UMAP-only refresh complete for {output_dir}", flush=True)
        return

    sweep_results_df = pd.DataFrame(sweep_rows)
    sweep_results_df.to_csv(results_dir / "patchcore_sweep_results.csv", index=False)

    best_row = sweep_results_df.sort_values(["f1", "auroc"], ascending=False).iloc[0].to_dict()
    best_variant_name = str(best_row["name"])
    best_variant_dir = output_dir / best_variant_name
    best_ckpt_path = best_variant_dir / "checkpoints" / "best_model.pt"

    sweep_summary: dict[str, Any] = {
        "sweep_variants": sweep_results_df["name"].tolist(),
        "base_output_dir": str(output_dir),
        "teacher_backbone": "wideresnet50_2",
        "teacher_layers": teacher_layers,
        "best_variant": best_row,
    }
    (results_dir / "patchcore_sweep_summary.json").write_text(json.dumps(sweep_summary, indent=2), encoding="utf-8")

    selected_checkpoint_meta: dict[str, Any] = {
        "variant_name": best_variant_name,
        "checkpoint_path": str(best_ckpt_path),
        "checkpoint_name": str(best_ckpt_path),
        "approx_size_gb": round(best_ckpt_path.stat().st_size / 1e9, 3) if best_ckpt_path.exists() else None,
        "memory_bank_shape": [int(best_row["memory_bank_size"]), int(best_row["feature_dim"])],
        "note": "Checkpoint includes the fitted PatchCore memory bank and backbone weights.",
    }
    (results_dir / "selected_checkpoint.json").write_text(json.dumps(selected_checkpoint_meta, indent=2), encoding="utf-8")

    # Sweep comparison plot
    plot_df = sweep_results_df.sort_values(["f1", "auroc"], ascending=False).reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh(plot_df["name"], plot_df["f1"], color="#264653")
    axes[0].set_title("WRN50-2 Multilayer PatchCore x224: F1")
    axes[0].invert_yaxis()
    axes[1].barh(plot_df["name"], plot_df["auroc"], color="#2a9d8f")
    axes[1].set_title("WRN50-2 Multilayer PatchCore x224: AUROC")
    axes[1].invert_yaxis()
    plt.tight_layout()
    fig.savefig(plots_dir / "sweep_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Run manifest
    manifest: dict[str, Any] = {
        "status": "complete",
        "output_dir": str(output_dir),
        "best_variant": best_variant_name,
        "best_f1": float(best_row["f1"]),
        "best_auroc": float(best_row["auroc"]),
        "sweep_variants": sweep_results_df["name"].tolist(),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n[done] Best variant: {best_variant_name}  F1={best_row['f1']:.4f}  AUROC={best_row['auroc']:.4f}", flush=True)
    print(f"[done] Artifacts at: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
