"""Train the TS-ResNet teacher-student distillation model on normal wafers only."""

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
from wafer_defect.models.ts_distillation import build_ts_distillation_from_config
from wafer_defect.training.ts_distillation import estimate_ts_error_scales, run_ts_epoch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training/train_ts_resnet18.toml")
    args = parser.parse_args()

    config = load_toml(args.config)
    output_dir = Path(config["run"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(config["run"]["seed"]))
    device = resolve_device(config["training"]["device"])
    image_size = int(config["data"].get("image_size", 64))

    train_dataset = WaferMapDataset(config["data"]["metadata_csv"], split="train", image_size=image_size)
    val_dataset = WaferMapDataset(config["data"]["metadata_csv"], split="val", image_size=image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["data"]["num_workers"]),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
    )

    print(
        f"Training TS distillation | train={len(train_dataset)} | val={len(val_dataset)} "
        f"| batch_size={int(config['data']['batch_size'])} | output_dir={output_dir}",
        flush=True,
    )

    model = build_ts_distillation_from_config(config, image_size=image_size).to(device)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None
    patience = int(config["training"].get("early_stopping_patience", 0))
    min_delta = float(config["training"].get("early_stopping_min_delta", 0.0))
    checkpoint_every = int(config["training"].get("checkpoint_every", 5))
    resume_from = str(config["training"].get("resume_from", "")).strip()
    stale_epochs = 0
    start_epoch = 0

    if resume_from:
        resume_path = Path(resume_from)
        if not resume_path.is_absolute():
            resume_path = Path.cwd() / resume_path
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        best_epoch = int(checkpoint.get("best_epoch", best_epoch))
        stale_epochs = int(checkpoint.get("stale_epochs", stale_epochs))
        history = checkpoint.get("history", [])
        best_state_dict = clone_state_dict(model)
        print(f"Resumed from {resume_path} at epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, int(config["training"]["epochs"])):
        train_metrics = run_ts_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            desc=f"train:e{epoch + 1}",
        )
        val_metrics = run_ts_epoch(
            model,
            val_loader,
            device,
            desc=f"val:e{epoch + 1}",
        )
        record = {
            "epoch": epoch + 1,
            "train_loss": train_metrics.loss,
            "train_distillation_loss": train_metrics.distillation_loss,
            "train_autoencoder_loss": train_metrics.auxiliary_loss,
            "val_loss": val_metrics.loss,
            "val_distillation_loss": val_metrics.distillation_loss,
            "val_autoencoder_loss": val_metrics.auxiliary_loss,
        }
        history.append(record)
        print(
            "Epoch "
            f"{epoch + 1}/{int(config['training']['epochs'])} "
            f"| train_loss={train_metrics.loss:.6f} "
            f"| val_loss={val_metrics.loss:.6f} "
            f"| train_distill={train_metrics.distillation_loss:.6f} "
            f"| val_distill={val_metrics.distillation_loss:.6f} "
            f"| train_feat_ae={train_metrics.auxiliary_loss:.6f} "
            f"| val_feat_ae={val_metrics.auxiliary_loss:.6f} "
            f"| best_val={best_val_loss:.6f}",
            flush=True,
        )

        improved = (best_val_loss - val_metrics.loss) > min_delta
        if improved:
            best_val_loss = val_metrics.loss
            best_epoch = epoch + 1
            best_state_dict = clone_state_dict(model)
            stale_epochs = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": best_state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "stale_epochs": stale_epochs,
                    "history": history,
                },
                output_dir / "best_model.pt",
            )
        else:
            stale_epochs += 1

        latest_checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "stale_epochs": stale_epochs,
            "history": history,
        }
        torch.save(latest_checkpoint, output_dir / "latest_checkpoint.pt")

        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            torch.save(latest_checkpoint, output_dir / f"checkpoint_epoch_{epoch + 1}.pt")

        if patience > 0 and stale_epochs >= patience:
            print(
                f"Early stopping at epoch {epoch + 1}. "
                f"Best epoch: {best_epoch}, best val loss: {best_val_loss:.6f}"
            , flush=True)
            break

    if best_state_dict is None:
        best_state_dict = clone_state_dict(model)

    model.load_state_dict(best_state_dict)
    student_scale, autoencoder_scale = estimate_ts_error_scales(model, train_loader, device)
    model.set_error_scales(student_scale=student_scale, autoencoder_scale=autoencoder_scale)
    calibrated_best_state_dict = clone_state_dict(model)

    torch.save(
        {
            "epoch": best_epoch,
            "model_state_dict": calibrated_best_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "stale_epochs": stale_epochs,
            "history": history,
            "student_map_scale": student_scale,
            "autoencoder_map_scale": autoencoder_scale,
        },
        output_dir / "best_model.pt",
    )

    final_state_dict = clone_state_dict(model)
    torch.save(
        {
            "epoch": len(history),
            "model_state_dict": final_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "stale_epochs": stale_epochs,
            "history": history,
            "student_map_scale": student_scale,
            "autoencoder_map_scale": autoencoder_scale,
        },
        output_dir / "last_model.pt",
    )

    with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "epochs_ran": len(history),
        "resumed_from": resume_from,
        "student_map_scale": student_scale,
        "autoencoder_map_scale": autoencoder_scale,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(
        "Training complete "
        f"| best_epoch={best_epoch} "
        f"| best_val_loss={best_val_loss:.6f} "
        f"| student_scale={student_scale:.6f} "
        f"| autoencoder_scale={autoencoder_scale:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
