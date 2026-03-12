"""Fit a PatchCore-style memory bank on normal wafer features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from wafer_defect.config import load_toml
from wafer_defect.data.wm811k import WaferMapDataset
from wafer_defect.models.patchcore import PatchCoreModel
from wafer_defect.training.patchcore import build_memory_subset, collect_memory_bank


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training/train_patchcore.toml")
    args = parser.parse_args()

    config = load_toml(args.config)
    output_dir = Path(config["run"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["run"]["seed"])
    set_seed(seed)
    device = resolve_device(config["training"].get("device", "auto"))
    image_size = int(config["data"].get("image_size", 64))
    batch_size = int(config["data"].get("batch_size", 64))
    num_workers = int(config["data"].get("num_workers", 0))

    train_dataset = WaferMapDataset(config["data"]["metadata_csv"], split="train", image_size=image_size)
    model = PatchCoreModel(
        image_size=image_size,
        backbone_type=str(config["model"].get("backbone_type", "conv")),
        use_batchnorm=bool(config["model"].get("use_batchnorm", True)),
        pretrained=bool(config["model"].get("pretrained", True)),
        freeze_backbone=bool(config["model"].get("freeze_backbone", True)),
        backbone_input_size=int(config["model"].get("backbone_input_size", 224)),
        normalize_imagenet=bool(config["model"].get("normalize_imagenet", True)),
        reduction=str(config["model"].get("reduction", "max")),
        topk_ratio=float(config["model"].get("topk_ratio", 0.1)),
        query_chunk_size=int(config["model"].get("query_chunk_size", 2048)),
        memory_chunk_size=int(config["model"].get("memory_chunk_size", 8192)),
    ).to(device)

    backbone_type = str(config["model"].get("backbone_type", "conv"))
    backbone_checkpoint = str(config["model"].get("backbone_checkpoint", "")).strip()
    if backbone_type == "conv" and backbone_checkpoint:
        checkpoint_path = Path(backbone_checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path.cwd() / checkpoint_path
        backbone_config = model.load_backbone_from_autoencoder_checkpoint(checkpoint_path)
        print(f"Loaded PatchCore backbone from {checkpoint_path}")
    elif backbone_type == "conv":
        backbone_config = {}
        print("No backbone checkpoint provided. Using randomly initialized conv PatchCore backbone.")
    else:
        backbone_config = {}
        backbone_label = str(config["model"].get("backbone_type", "resnet18"))
        print(
            f"Using {backbone_label} PatchCore backbone "
            f"(pretrained={bool(config['model'].get('pretrained', True))}, "
            f"freeze_backbone={bool(config['model'].get('freeze_backbone', True))})."
        )

    memory_bank_size = int(config["model"].get("memory_bank_size", 50000))
    memory_subset = build_memory_subset(
        train_dataset,
        memory_bank_size=memory_bank_size,
        patches_per_image=model.patches_per_image,
        seed=seed,
    )
    memory_loader = DataLoader(
        memory_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    memory_bank = collect_memory_bank(
        model=model,
        dataloader=memory_loader,
        device=device,
        target_size=memory_bank_size,
        seed=seed,
    )
    model.set_memory_bank(memory_bank.to(device))

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "memory_bank_size": int(model.memory_bank.shape[0]),
        "feature_dim": int(model.feature_dim),
        "patches_per_image": int(model.patches_per_image),
        "backbone_type": backbone_type,
        "backbone_checkpoint": backbone_checkpoint,
        "backbone_config": backbone_config,
    }
    torch.save(checkpoint, output_dir / "best_model.pt")
    torch.save(checkpoint, output_dir / "last_model.pt")

    summary = {
        "memory_bank_size": int(model.memory_bank.shape[0]),
        "feature_dim": int(model.feature_dim),
        "patches_per_image": int(model.patches_per_image),
        "memory_subset_images": len(memory_subset),
        "backbone_type": backbone_type,
        "backbone_checkpoint": backbone_checkpoint,
        "reduction": model.reduction,
        "topk_ratio": float(model.topk_ratio),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
