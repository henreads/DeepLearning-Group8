"""Training helpers for RD4AD anomaly detection."""
# src/wafer_defect/training/rd4ad.py

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from wafer_defect.models.rd4ad import rd4ad_loss
from wafer_defect.training.common import EpochMetrics


def run_rd4ad_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    *,
    progress_desc: str | None = None,
) -> EpochMetrics:
    """Run one training or validation epoch for RD4AD.

    Args:
        model: RD4ADModel instance.
        dataloader: DataLoader yielding (image, label) pairs.
        device: Target device.
        optimizer: If provided, runs a training step; otherwise validation only.
        scaler: AMP GradScaler for mixed-precision training. Optional.
        progress_desc: Label shown in the tqdm progress bar.

    Returns:
        EpochMetrics with loss fields populated.
    """
    is_training = optimizer is not None
    model.train(is_training)
    # Encoder must always stay in eval mode (it is frozen but has BN layers).
    if hasattr(model, "encoder"):
        model.encoder.eval()

    total_loss = 0.0
    total_loss1 = 0.0
    total_loss2 = 0.0
    total_loss3 = 0.0
    total_items = 0

    iterator = dataloader
    if progress_desc:
        iterator = tqdm(dataloader, desc=progress_desc, leave=False)

    use_amp = scaler is not None and is_training

    for inputs, labels in iterator:
        inputs = inputs.to(device)
        labels = labels.to(device)

        normal_mask = labels == 0
        if not torch.any(normal_mask):
            continue

        normal_inputs = inputs[normal_mask]

        if use_amp:
            with torch.autocast(device_type=device.type):
                f1, f2, f3 = model.encode(normal_inputs)
                g1, g2, g3 = model.decode(f3)
                loss, l1, l2, l3 = rd4ad_loss(f1, f2, f3, g1, g2, g3)
        else:
            f1, f2, f3 = model.encode(normal_inputs)
            g1, g2, g3 = model.decode(f3)
            loss, l1, l2, l3 = rd4ad_loss(f1, f2, f3, g1, g2, g3)

        if is_training:
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        batch_size = normal_inputs.shape[0]
        total_loss += loss.item() * batch_size
        total_loss1 += l1.item() * batch_size
        total_loss2 += l2.item() * batch_size
        total_loss3 += l3.item() * batch_size
        total_items += batch_size

        if progress_desc and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                loss=f"{loss.item():.4f}",
                l1=f"{l1.item():.4f}",
                l2=f"{l2.item():.4f}",
                l3=f"{l3.item():.4f}",
            )

    n = max(total_items, 1)
    return EpochMetrics(
        loss=total_loss / n,
        reconstruction_loss=total_loss1 / n,
        kl_loss=total_loss2 / n,
        distillation_loss=total_loss3 / n,
    )
