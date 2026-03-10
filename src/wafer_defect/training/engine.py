from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class EpochMetrics:
    loss: float


def run_autoencoder_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochMetrics:
    criterion = nn.MSELoss()
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
        loss = criterion(reconstructions, normal_inputs)

        if is_training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = normal_inputs.shape[0]
        total_loss += loss.item() * batch_size
        total_items += batch_size

    return EpochMetrics(loss=total_loss / max(total_items, 1))

