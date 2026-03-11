"""Convolutional Deep SVDD encoder for one-class anomaly detection."""

from __future__ import annotations

import torch
from torch import nn


class ConvDeepSVDD(nn.Module):
    def __init__(self, latent_dim: int = 128, image_size: int = 64) -> None:
        super().__init__()
        if image_size % 8 != 0:
            raise ValueError(f"image_size must be divisible by 8, got {image_size}")

        reduced_spatial = image_size // 8
        encoded_features = 64 * reduced_spatial * reduced_spatial

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Flatten(),
            nn.Linear(encoded_features, latent_dim, bias=False),
        )
        self.register_buffer("center", torch.zeros(latent_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def set_center(self, center: torch.Tensor) -> None:
        if center.ndim != 1 or center.shape[0] != self.center.shape[0]:
            raise ValueError(
                f"center must have shape ({self.center.shape[0]},), got {tuple(center.shape)}"
            )
        self.center.copy_(center.to(device=self.center.device, dtype=self.center.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)
