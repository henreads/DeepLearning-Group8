"""Autoencoder architectures used for reconstruction-based anomaly detection."""

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


def _maybe_batchnorm(use_batchnorm: bool, channels: int) -> nn.Module:
    return nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()


class ResidualDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm1 = _maybe_batchnorm(use_batchnorm, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = _maybe_batchnorm(use_batchnorm, out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
        self.skip_norm = _maybe_batchnorm(use_batchnorm, out_channels)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip_norm(self.skip(x))
        out = self.activation(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.activation(out + identity)
        return out


class ResidualUpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = _maybe_batchnorm(use_batchnorm, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = _maybe_batchnorm(use_batchnorm, out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.skip_norm = _maybe_batchnorm(use_batchnorm, out_channels)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        upsampled = self.upsample(x)
        identity = self.skip_norm(self.skip(upsampled))
        out = self.activation(self.norm1(self.conv1(upsampled)))
        out = self.norm2(self.conv2(out))
        out = self.activation(out + identity)
        return out


class ResidualAutoencoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        image_size: int = 64,
        use_batchnorm: bool = True,
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        if image_size % 8 != 0:
            raise ValueError(f"image_size must be divisible by 8, got {image_size}")
        if dropout_prob < 0.0 or dropout_prob >= 1.0:
            raise ValueError(f"dropout_prob must be in [0, 1), got {dropout_prob}")

        reduced_spatial = image_size // 8
        bottleneck_channels = 128
        encoded_features = bottleneck_channels * reduced_spatial * reduced_spatial
        self.use_batchnorm = use_batchnorm
        self.dropout_prob = dropout_prob

        encoder_layers: list[nn.Module] = [
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            _maybe_batchnorm(use_batchnorm, 32),
            nn.ReLU(),
            ResidualDownBlock(32, 64, use_batchnorm=use_batchnorm),
            ResidualDownBlock(64, bottleneck_channels, use_batchnorm=use_batchnorm),
            nn.Flatten(),
            nn.Linear(encoded_features, latent_dim),
            nn.ReLU(),
        ]
        if dropout_prob > 0.0:
            encoder_layers.append(nn.Dropout(p=dropout_prob))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = [
            nn.Linear(latent_dim, encoded_features),
            nn.ReLU(),
            nn.Unflatten(1, (bottleneck_channels, reduced_spatial, reduced_spatial)),
            ResidualUpBlock(bottleneck_channels, 64, use_batchnorm=use_batchnorm),
            ResidualUpBlock(64, 32, use_batchnorm=use_batchnorm),
            ResidualUpBlock(32, 16, use_batchnorm=use_batchnorm),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        ]
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        return self.decoder(latent)


def build_autoencoder_from_config(config: dict, image_size: int) -> nn.Module:
    model_config = config.get("model", {})
    architecture = str(model_config.get("architecture", "baseline")).lower()
    common_kwargs = {
        "latent_dim": int(model_config["latent_dim"]),
        "image_size": image_size,
        "use_batchnorm": bool(model_config.get("use_batchnorm", False)),
        "dropout_prob": float(model_config.get("dropout_prob", 0.0)),
    }

    if architecture in {"baseline", "conv", "conv_autoencoder"}:
        return ConvAutoencoder(**common_kwargs)
    if architecture in {"residual", "resnet", "residual_autoencoder"}:
        return ResidualAutoencoder(**common_kwargs)
    raise ValueError(f"Unsupported autoencoder architecture: {architecture}")
