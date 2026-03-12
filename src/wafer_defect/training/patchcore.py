"""Helpers for fitting a PatchCore-style memory bank on normal wafers."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from wafer_defect.models.patchcore import PatchCoreModel


def sample_memory_indices(
    dataset_size: int,
    memory_bank_size: int,
    patches_per_image: int,
    seed: int,
) -> np.ndarray:
    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive")
    if memory_bank_size <= 0:
        raise ValueError("memory_bank_size must be positive")
    if patches_per_image <= 0:
        raise ValueError("patches_per_image must be positive")

    image_count = min(dataset_size, max(1, math.ceil(memory_bank_size / patches_per_image)))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(dataset_size, size=image_count, replace=False))


def build_memory_subset(
    dataset: Dataset,
    memory_bank_size: int,
    patches_per_image: int,
    seed: int,
) -> Subset:
    indices = sample_memory_indices(
        dataset_size=len(dataset),
        memory_bank_size=memory_bank_size,
        patches_per_image=patches_per_image,
        seed=seed,
    )
    return Subset(dataset, indices.tolist())


def collect_memory_bank(
    model: PatchCoreModel,
    dataloader: DataLoader,
    device: torch.device,
    target_size: int,
    seed: int,
) -> torch.Tensor:
    if target_size <= 0:
        raise ValueError("target_size must be positive")

    patch_batches: list[torch.Tensor] = []
    model.eval()

    with torch.inference_mode():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            normal_mask = labels == 0
            if not torch.any(normal_mask):
                continue

            embeddings = model.patch_embeddings(inputs[normal_mask]).reshape(-1, model.feature_dim)
            patch_batches.append(embeddings.cpu())

    if not patch_batches:
        raise ValueError("Could not build memory bank because no normal embeddings were collected.")

    memory_bank = torch.cat(patch_batches, dim=0)
    if memory_bank.shape[0] > target_size:
        generator = torch.Generator().manual_seed(seed)
        keep = torch.randperm(memory_bank.shape[0], generator=generator)[:target_size]
        memory_bank = memory_bank[keep]

    return memory_bank
