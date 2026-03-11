"""Training loop and helpers for Deep SVDD models."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from wafer_defect.scoring import svdd_distance
from wafer_defect.training.common import EpochMetrics


def initialize_svdd_center(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    eps: float = 0.1,
) -> torch.Tensor:
    model.eval()
    total = None
    total_items = 0

    with torch.inference_mode():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            normal_mask = labels == 0
            if not torch.any(normal_mask):
                continue

            embeddings = model(inputs[normal_mask])
            batch_total = embeddings.sum(dim=0)
            total = batch_total if total is None else total + batch_total
            total_items += embeddings.shape[0]

    if total is None or total_items == 0:
        raise ValueError("Could not initialize SVDD center because no normal samples were found.")

    center = total / total_items
    near_zero_mask = center.abs() < eps
    center = center.clone()
    center[near_zero_mask & (center < 0)] = -eps
    center[near_zero_mask & (center >= 0)] = eps
    return center


def run_svdd_epoch(
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

        embeddings = model(inputs[normal_mask])
        per_sample_loss = svdd_distance(embeddings, model.center)
        loss = per_sample_loss.mean()

        if is_training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = embeddings.shape[0]
        total_loss += loss.item() * batch_size
        total_items += batch_size

    average_loss = total_loss / max(total_items, 1)
    return EpochMetrics(loss=average_loss, reconstruction_loss=average_loss)
