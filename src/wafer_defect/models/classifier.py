"""Residual CNN classifiers for WM-811K multiclass labeling."""

from __future__ import annotations

import torch
from torch import nn


def make_group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """Use GroupNorm so inference does not depend on running batch statistics."""
    num_groups = min(max_groups, num_channels)
    while num_channels % num_groups != 0 and num_groups > 1:
        num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = make_group_norm(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = make_group_norm(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                make_group_norm(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + residual
        return self.act(x)


class ResidualStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride=stride),
            ResidualBlock(out_channels, out_channels, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden_channels = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.net(self.pool(x))
        return x * scale


class EnhancedResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.0,
        se_reduction: int = 8,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = make_group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = make_group_norm(out_channels)
        self.attention = SqueezeExcitation(out_channels, reduction=se_reduction)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()
        self.act = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                make_group_norm(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x = self.attention(x)
        x = self.dropout(x)
        return self.act(x + residual)


class EnhancedResidualStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        dropout: float = 0.0,
        se_reduction: int = 8,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            EnhancedResidualBlock(
                in_channels,
                out_channels,
                stride=stride,
                dropout=dropout,
                se_reduction=se_reduction,
            ),
            EnhancedResidualBlock(
                out_channels,
                out_channels,
                stride=1,
                dropout=dropout,
                se_reduction=se_reduction,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DualPoolClassifierHead(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_features = self.avgpool(x)
        max_features = self.maxpool(x)
        return self.classifier(torch.cat([avg_features, max_features], dim=1))


class WaferClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        base_channels: int = 32,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        variant: str = "baseline",
        block_dropout: float = 0.0,
        se_reduction: int = 8,
    ) -> None:
        super().__init__()
        variant = variant.lower()
        if variant not in {"baseline", "enhanced"}:
            raise ValueError(f"Unsupported classifier variant: {variant}")

        self.variant = variant
        widths = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        ]
        self.stem = nn.Sequential(
            nn.Conv2d(1, widths[0], kernel_size=3, stride=1, padding=1, bias=False),
            make_group_norm(widths[0]),
            nn.ReLU(inplace=True),
        )

        if variant == "baseline":
            self.features = nn.Sequential(
                ResidualStage(widths[0], widths[0], stride=1),
                ResidualStage(widths[0], widths[1], stride=2),
                ResidualStage(widths[1], widths[2], stride=2),
                ResidualStage(widths[2], widths[3], stride=2),
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(widths[3], hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.features = nn.Sequential(
                EnhancedResidualStage(
                    widths[0],
                    widths[0],
                    stride=1,
                    dropout=block_dropout,
                    se_reduction=se_reduction,
                ),
                EnhancedResidualStage(
                    widths[0],
                    widths[1],
                    stride=2,
                    dropout=block_dropout,
                    se_reduction=se_reduction,
                ),
                EnhancedResidualStage(
                    widths[1],
                    widths[2],
                    stride=2,
                    dropout=block_dropout,
                    se_reduction=se_reduction,
                ),
                EnhancedResidualStage(
                    widths[2],
                    widths[3],
                    stride=2,
                    dropout=block_dropout,
                    se_reduction=se_reduction,
                ),
            )
            self.classifier = DualPoolClassifierHead(
                in_channels=widths[3],
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                dropout=dropout,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        return self.classifier(x)
