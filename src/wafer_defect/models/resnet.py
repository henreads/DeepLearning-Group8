"""Pretrained ResNet feature extractors for wafer-map experiments."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    Wide_ResNet50_2_Weights,
    resnet18,
    resnet50,
    wide_resnet50_2,
)


def _build_resnet_backbone(backbone_name: str, pretrained: bool) -> nn.Module:
    backbone_name = backbone_name.lower()
    if backbone_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        return resnet18(weights=weights)
    if backbone_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        return resnet50(weights=weights)
    if backbone_name in {"wideresnet50", "wide_resnet50_2", "wideresnet50_2"}:
        weights = Wide_ResNet50_2_Weights.DEFAULT if pretrained else None
        return wide_resnet50_2(weights=weights)
    raise ValueError(f"Unsupported ResNet backbone: {backbone_name}")


class ResNetFeatureExtractor(nn.Module):
    """Pretrained ResNet adapted for single-channel wafer maps."""

    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        input_size: int = 224,
        freeze_backbone: bool = True,
        normalize_imagenet: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name.lower()
        backbone = _build_resnet_backbone(self.backbone_name, pretrained=pretrained)

        # Adapt the RGB stem to grayscale by averaging pretrained RGB filters.
        original_conv = backbone.conv1
        adapted_conv = nn.Conv2d(
            1,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            adapted_conv.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))
        backbone.conv1 = adapted_conv

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.layers = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        )
        self.avgpool = backbone.avgpool
        self.embedding_dim = backbone.fc.in_features
        self.input_size = int(input_size)
        self.output_spatial = max(1, self.input_size // 32)
        self.normalize_imagenet = bool(normalize_imagenet)
        self.register_buffer("image_mean", torch.tensor([0.4490], dtype=torch.float32).view(1, 1, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.2260], dtype=torch.float32).view(1, 1, 1, 1))

        if freeze_backbone:
            for parameter in self.parameters():
                parameter.requires_grad = False

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        if self.normalize_imagenet:
            x = (x - self.image_mean) / self.image_std
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)
        x = self.stem(x)
        x = self.layers(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)
        x = self.stem(x)
        return self.layers(x)

    def forward_intermediate_feature_map(self, x: torch.Tensor, layer_name: str = "layer4") -> torch.Tensor:
        layer_name = layer_name.lower()
        if layer_name not in {"layer1", "layer2", "layer3", "layer4"}:
            raise ValueError(f"Unsupported ResNet feature layer: {layer_name}")

        x = self.preprocess(x)
        x = self.stem(x)
        x = self.layer1(x)
        if layer_name == "layer1":
            return x
        x = self.layer2(x)
        if layer_name == "layer2":
            return x
        x = self.layer3(x)
        if layer_name == "layer3":
            return x
        return self.layer4(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


class ResNet18FeatureExtractor(ResNetFeatureExtractor):
    """Pretrained ResNet18 adapted for single-channel wafer maps."""

    def __init__(
        self,
        pretrained: bool = True,
        input_size: int = 224,
        freeze_backbone: bool = True,
        normalize_imagenet: bool = True,
    ) -> None:
        super().__init__(
            backbone_name="resnet18",
            pretrained=pretrained,
            input_size=input_size,
            freeze_backbone=freeze_backbone,
            normalize_imagenet=normalize_imagenet,
        )
