"""Per-sample scoring utilities shared by training and evaluation code."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def reconstruction_mse(inputs: torch.Tensor, reconstructions: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, inputs.ndim))
    return torch.mean((reconstructions - inputs).pow(2), dim=dims)


def normalized_kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def vae_anomaly_score(
    inputs: torch.Tensor,
    reconstructions: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    return reconstruction_mse(inputs, reconstructions) + beta * normalized_kl_divergence(mu, logvar)


def svdd_distance(embeddings: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    return torch.sum((embeddings - center.unsqueeze(0)).pow(2), dim=1)


def absolute_error_map(inputs: torch.Tensor, reconstructions: torch.Tensor) -> torch.Tensor:
    return torch.abs(reconstructions - inputs)


def squared_error_map(inputs: torch.Tensor, reconstructions: torch.Tensor) -> torch.Tensor:
    return (reconstructions - inputs).pow(2)


def spatial_mean(error_map: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, error_map.ndim))
    return error_map.mean(dim=dims)


def spatial_max(error_map: torch.Tensor) -> torch.Tensor:
    flattened = error_map.flatten(start_dim=1)
    return flattened.max(dim=1).values


def topk_spatial_mean(error_map: torch.Tensor, topk_ratio: float) -> torch.Tensor:
    if not 0.0 < topk_ratio <= 1.0:
        raise ValueError(f"topk_ratio must be in (0, 1], got {topk_ratio}")

    flattened = error_map.flatten(start_dim=1)
    topk = max(1, int(math.ceil(flattened.shape[1] * topk_ratio)))
    return torch.topk(flattened, k=topk, dim=1).values.mean(dim=1)


def masked_spatial_mean(error_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if error_map.shape != mask.shape:
        raise ValueError(f"mask shape {tuple(mask.shape)} must match error_map shape {tuple(error_map.shape)}")

    flattened_error = error_map.flatten(start_dim=1)
    flattened_mask = mask.flatten(start_dim=1).to(dtype=error_map.dtype)
    weighted_sum = (flattened_error * flattened_mask).sum(dim=1)
    denominator = flattened_mask.sum(dim=1).clamp_min(1.0)
    return weighted_sum / denominator


def pooled_error_map(error_map: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
    padding = kernel_size // 2
    return F.avg_pool2d(error_map, kernel_size=kernel_size, stride=1, padding=padding)
