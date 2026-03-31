"""Train the convolutional VAE baseline on normal wafers only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from wafer_defect.config import load_toml
from wafer_defect.data.wm811k import WaferMapDataset
from wafer_defect.models.vae import ConvVariationalAutoencoder
from wafer_defect.training.vae import run_vae_epoch


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
    parser.add_argument("--config", default="configs/training/train_vae.toml")
    args = parser.parse_args()

    config = load_toml(args.config)
    output_dir = Path(config["run"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(config["run"]["seed"]))
    device = resolve_device(config["training"]["device"])
    image_size = int(config["data"].get("image_size", 64))
    beta = float(config["model"].get("beta", 0.01))

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

    model = ConvVariationalAutoencoder(
        latent_dim=int(config["model"]["latent_dim"]),
        image_size=image_size,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
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
        best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"Resumed from {resume_path} at epoch {start_epoch}")

    for epoch in tqdm(range(start_epoch, int(config["training"]["epochs"])), desc="Training epochs", total=int(config["training"]["epochs"]), initial=start_epoch):
        train_metrics = run_vae_epoch(model, train_loader, device, beta=beta, optimizer=optimizer)
        val_metrics = run_vae_epoch(model, val_loader, device, beta=beta)
        record = {
            "epoch": epoch + 1,
            "train_loss": train_metrics.loss,
            "train_reconstruction_loss": train_metrics.reconstruction_loss,
            "train_kl_loss": train_metrics.kl_loss,
            "val_loss": val_metrics.loss,
            "val_reconstruction_loss": val_metrics.reconstruction_loss,
            "val_kl_loss": val_metrics.kl_loss,
        }
        history.append(record)
        print(record)

        improved = (best_val_loss - val_metrics.loss) > min_delta
        if improved:
            best_val_loss = val_metrics.loss
            best_epoch = epoch + 1
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
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
            )
            break

    torch.save(
        {
            "epoch": len(history),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "stale_epochs": stale_epochs,
            "history": history,
        },
        output_dir / "last_model.pt",
    )
    with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "epochs_ran": len(history),
        "beta": beta,
        "resumed_from": resume_from,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
