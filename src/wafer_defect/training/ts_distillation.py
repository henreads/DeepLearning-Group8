"""Training helpers for teacher-student distillation anomaly detection."""
#src/wafer_defect/training/ts_distillation.py

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from wafer_defect.training.common import EpochMetrics


def run_ts_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    desc: str | None = None,
) -> EpochMetrics:
    is_training = optimizer is not None
    model.train(is_training)
    if hasattr(model, "teacher"):
        model.teacher.eval()

    total_loss = 0.0
    total_student_loss = 0.0
    total_autoencoder_loss = 0.0
    total_items = 0

    iterator = dataloader
    if desc is not None:
        iterator = tqdm(dataloader, desc=desc, leave=False)

    for inputs, labels in iterator:
        inputs = inputs.to(device)
        labels = labels.to(device)

        normal_mask = labels == 0
        if not torch.any(normal_mask):
            continue

        normal_inputs = inputs[normal_mask]
        student_map, autoencoder_map = model.raw_anomaly_maps(normal_inputs)
        student_loss = student_map.mean()
        autoencoder_loss = autoencoder_map.mean()
        loss = model.student_weight * student_loss + model.autoencoder_weight * autoencoder_loss

        if is_training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = normal_inputs.shape[0]
        total_loss += loss.item() * batch_size
        total_student_loss += student_loss.item() * batch_size
        total_autoencoder_loss += autoencoder_loss.item() * batch_size
        total_items += batch_size
        if desc is not None:
            iterator.set_postfix(
                loss=f"{(total_loss / max(total_items, 1)):.4f}",
                distill=f"{(total_student_loss / max(total_items, 1)):.4f}",
                feat_ae=f"{(total_autoencoder_loss / max(total_items, 1)):.4f}",
            )

    average_loss = total_loss / max(total_items, 1)
    average_student_loss = total_student_loss / max(total_items, 1)
    average_autoencoder_loss = total_autoencoder_loss / max(total_items, 1)
    return EpochMetrics(
        loss=average_loss,
        reconstruction_loss=average_student_loss,
        kl_loss=average_autoencoder_loss,
        distillation_loss=average_student_loss,
        auxiliary_loss=average_autoencoder_loss,
    )


def estimate_ts_error_scales(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    total_student = 0.0
    total_autoencoder = 0.0
    total_items = 0

    with torch.inference_mode():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            normal_mask = labels == 0
            if not torch.any(normal_mask):
                continue

            student_map, autoencoder_map = model.raw_anomaly_maps(inputs[normal_mask])
            batch_size = int(normal_mask.sum().item())
            total_student += float(student_map.mean().item()) * batch_size
            total_autoencoder += float(autoencoder_map.mean().item()) * batch_size
            total_items += batch_size

    if total_items == 0:
        raise ValueError("Could not estimate TS distillation error scales because no normal samples were found.")

    student_scale = total_student / total_items
    autoencoder_scale = total_autoencoder / total_items
    return max(student_scale, 1e-6), max(autoencoder_scale, 1e-6)
