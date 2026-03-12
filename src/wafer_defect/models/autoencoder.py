"""Convolutional autoencoder baseline used for reconstruction-based detection."""

from __future__ import annotations

import torch
from torch import nn


class ConvAutoencoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        image_size: int = 64,
        use_batchnorm: bool = False,
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        if image_size % 8 != 0:
            raise ValueError(f"image_size must be divisible by 8, got {image_size}")
        if dropout_prob < 0.0 or dropout_prob >= 1.0:
            raise ValueError(f"dropout_prob must be in [0, 1), got {dropout_prob}")

        reduced_spatial = image_size // 8
        encoded_features = 64 * reduced_spatial * reduced_spatial
        self.use_batchnorm = use_batchnorm
        self.dropout_prob = dropout_prob

        encoder_layers: list[nn.Module] = []
        encoder_specs = [(1, 16), (16, 32), (32, 64)]
        for in_channels, out_channels in encoder_specs:
            encoder_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
            if use_batchnorm:
                encoder_layers.append(nn.BatchNorm2d(out_channels))
            encoder_layers.append(nn.ReLU())
        encoder_layers.extend(
            [
                nn.Flatten(),
                nn.Linear(encoded_features, latent_dim),
                nn.ReLU(),
            ]
        )
        if dropout_prob > 0.0:
            encoder_layers.append(nn.Dropout(p=dropout_prob))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = [
            nn.Linear(latent_dim, encoded_features),
            nn.ReLU(),
            nn.Unflatten(1, (64, reduced_spatial, reduced_spatial)),
        ]
        decoder_specs = [(64, 32), (32, 16)]
        for in_channels, out_channels in decoder_specs:
            decoder_layers.append(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
            )
            if use_batchnorm:
                decoder_layers.append(nn.BatchNorm2d(out_channels))
            decoder_layers.append(nn.ReLU())
        decoder_layers.extend(
            [
                nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid(),
            ]
        )
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        return self.decoder(latent)
