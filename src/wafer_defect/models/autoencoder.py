"""Convolutional autoencoder baseline used for reconstruction-based detection."""

from __future__ import annotations

import torch
from torch import nn


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 128, image_size: int = 64) -> None:
        super().__init__()
        if image_size % 8 != 0:
            raise ValueError(f"image_size must be divisible by 8, got {image_size}")

        reduced_spatial = image_size // 8
        encoded_features = 64 * reduced_spatial * reduced_spatial

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(encoded_features, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, encoded_features),
            nn.ReLU(),
            nn.Unflatten(1, (64, reduced_spatial, reduced_spatial)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        return self.decoder(latent)
