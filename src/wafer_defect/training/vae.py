"""Training loop for variational autoencoder models."""
#src/wafer_defect/training/vae.py

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from wafer_defect.models.vae import VAEOutput
from wafer_defect.scoring import normalized_kl_divergence, reconstruction_mse
from wafer_defect.training.common import EpochMetrics


def run_vae_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    beta: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochMetrics:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_reconstruction_loss = 0.0
    total_kl_loss = 0.0
    total_items = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        normal_mask = labels == 0
        if not torch.any(normal_mask):
            continue

        normal_inputs = inputs[normal_mask]
        outputs = model(normal_inputs)
        if not isinstance(outputs, VAEOutput):
            raise TypeError("VAE model must return VAEOutput")

        reconstruction_loss = reconstruction_mse(normal_inputs, outputs.reconstruction).mean()
        kl_loss = normalized_kl_divergence(outputs.mu, outputs.logvar).mean()
        loss = reconstruction_loss + beta * kl_loss

        if is_training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = normal_inputs.shape[0]
        total_loss += loss.item() * batch_size
        total_reconstruction_loss += reconstruction_loss.item() * batch_size
        total_kl_loss += kl_loss.item() * batch_size
        total_items += batch_size

    denominator = max(total_items, 1)
    return EpochMetrics(
        loss=total_loss / denominator,
        reconstruction_loss=total_reconstruction_loss / denominator,
        kl_loss=total_kl_loss / denominator,
    )
