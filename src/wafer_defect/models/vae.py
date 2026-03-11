"""Convolutional variational autoencoder used for anomaly detection experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class VAEOutput:
    reconstruction: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor


class ConvVariationalAutoencoder(nn.Module):
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
        )
        self.fc_mu = nn.Linear(encoded_features, latent_dim)
        self.fc_logvar = nn.Linear(encoded_features, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, encoded_features)
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Unflatten(1, (64, reduced_spatial, reduced_spatial)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        return self.fc_mu(features), self.fc_logvar(features)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        decoded = self.decoder_input(z)
        return self.decoder(decoded)

    def forward(self, x: torch.Tensor) -> VAEOutput:
        mu, logvar = self.encode(x)
        latent = self.reparameterize(mu, logvar)
        reconstruction = self.decode(latent)
        return VAEOutput(reconstruction=reconstruction, mu=mu, logvar=logvar)
