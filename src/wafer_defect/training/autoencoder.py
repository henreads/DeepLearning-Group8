"""Training loop for reconstruction-only autoencoder models."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from wafer_defect.scoring import reconstruction_mse
from wafer_defect.training.common import EpochMetrics


def run_autoencoder_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochMetrics:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_items = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        normal_mask = labels == 0
        if not torch.any(normal_mask):
            continue

        normal_inputs = inputs[normal_mask]
        reconstructions = model(normal_inputs)
        per_sample_loss = reconstruction_mse(normal_inputs, reconstructions)
        loss = per_sample_loss.mean()

        if is_training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = normal_inputs.shape[0]
        total_loss += loss.item() * batch_size
        total_items += batch_size

    average_loss = total_loss / max(total_items, 1)
    return EpochMetrics(loss=average_loss, reconstruction_loss=average_loss)
