"""Per-sample scoring utilities shared by training and evaluation code."""

from __future__ import annotations

import torch


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
