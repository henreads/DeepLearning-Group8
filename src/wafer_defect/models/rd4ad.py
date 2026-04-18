"""RD4AD (Reverse Distillation for Anomaly Detection) model for wafer maps.

Architecture:
  - Encoder: frozen pretrained WideResNet50-2, extracts features at layer1,
    layer2, and layer3.
  - One-class embedding (OCE): a trainable BatchNorm bottleneck applied to
    the layer3 encoder output.
  - Decoder: three trainable blocks that progressively upsample from layer3
    scale back to layer1 scale, producing reconstructed feature maps at each
    resolution.
  - Loss: mean cosine distance between encoder and decoder features summed
    across all three scales.
  - Anomaly score: mean (1 - cosine_similarity) across scales, reduced to a
    per-wafer scalar via mean, max, or topk_mean.

Reference:
  Deng et al., "Anomaly Detection via Reverse Distillation from One-Class
  Embedding", CVPR 2022.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from wafer_defect.models.resnet import ResNetFeatureExtractor
from wafer_defect.scoring import spatial_max, spatial_mean, topk_spatial_mean

# WideResNet50-2 output channels per layer (same as ResNet50 output dims).
_CH = {"layer1": 256, "layer2": 512, "layer3": 1024}


def _encode_multilayer(
    encoder: ResNetFeatureExtractor,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single forward pass extracting layer1, layer2, and layer3 features."""
    x = encoder.preprocess(x)
    x = encoder.stem(x)
    f1 = encoder.layer1(x)
    f2 = encoder.layer2(f1)
    f3 = encoder.layer3(f2)
    return f1, f2, f3


class _DecoderBlock(nn.Module):
    """Conv-BN-ReLU block used inside the RD4AD decoder."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RD4ADDecoder(nn.Module):
    """Trainable decoder that mirrors the WideResNet50-2 encoder.

    Takes the frozen encoder's layer3 features as input and produces
    reconstructed feature maps at layer3, layer2, and layer1 scales via
    progressive 2x bilinear upsampling. Each output is compared to the
    corresponding encoder feature map using cosine similarity loss.

    At 224x224 input the spatial sizes are:
      layer3 -> 14x14   (1024 ch)
      layer2 -> 28x28   (512 ch)
      layer1 -> 56x56   (256 ch)
    """

    def __init__(self) -> None:
        super().__init__()
        ch1, ch2, ch3 = _CH["layer1"], _CH["layer2"], _CH["layer3"]

        # One-class embedding: trainable BN on the encoder's layer3 output.
        self.oce = nn.BatchNorm2d(ch3)

        # Reconstruct at layer3 scale.
        self.dec3 = _DecoderBlock(ch3, ch3)

        # Upsample 2x then reconstruct at layer2 scale.
        self.dec2 = _DecoderBlock(ch3, ch2)

        # Upsample 2x then reconstruct at layer1 scale.
        self.dec1 = _DecoderBlock(ch2, ch1)

    def forward(
        self,
        f3: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            f3: Encoder layer3 features, shape (B, 1024, H3, W3).

        Returns:
            (g1, g2, g3): Decoded features at layer1, layer2, layer3 scales.
        """
        x = self.oce(f3)
        g3 = self.dec3(x)
        g2 = self.dec2(
            F.interpolate(g3, scale_factor=2, mode="bilinear", align_corners=False)
        )
        g1 = self.dec1(
            F.interpolate(g2, scale_factor=2, mode="bilinear", align_corners=False)
        )
        return g1, g2, g3


class RD4ADModel(nn.Module):
    """Full RD4AD anomaly detection model.

    The encoder is always frozen. Only the decoder (including the OCE
    BatchNorm) has trainable parameters.
    """

    def __init__(
        self,
        pretrained: bool = True,
        reduction: str = "topk_mean",
        topk_ratio: float = 0.10,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "max", "topk_mean"}:
            raise ValueError(f"Unsupported reduction: {reduction!r}")
        if reduction == "topk_mean" and not 0.0 < topk_ratio <= 1.0:
            raise ValueError(f"topk_ratio must be in (0, 1], got {topk_ratio}")

        self.image_size = int(image_size)
        self.reduction = reduction
        self.topk_ratio = float(topk_ratio)

        self.encoder = ResNetFeatureExtractor(
            backbone_name="wide_resnet50_2",
            pretrained=pretrained,
            input_size=image_size,
            freeze_backbone=True,
            normalize_imagenet=True,
        )
        self.decoder = RD4ADDecoder()

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract frozen encoder features at layer1, layer2, layer3."""
        with torch.no_grad():
            return _encode_multilayer(self.encoder, x)

    def decode(
        self, f3: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run trainable decoder from layer3 features."""
        return self.decoder(f3)

    # ------------------------------------------------------------------
    # Anomaly scoring
    # ------------------------------------------------------------------

    def cosine_anomaly_map(
        self,
        x: torch.Tensor,
        target_size: int | None = None,
    ) -> torch.Tensor:
        """Per-pixel anomaly map as mean (1 - cosine_sim) across three scales.

        Args:
            x: Input batch, shape (B, 1, H, W).
            target_size: Spatial size of the output map. Defaults to image_size.

        Returns:
            Anomaly map of shape (B, 1, target_size, target_size).
        """
        target = target_size or self.image_size
        f1, f2, f3 = self.encode(x)
        g1, g2, g3 = self.decode(f3)

        maps = []
        for enc, dec in [(f1, g1), (f2, g2), (f3, g3)]:
            cos_sim = F.cosine_similarity(enc, dec, dim=1, eps=1e-8)  # (B, H, W)
            amap = (1.0 - cos_sim).unsqueeze(1)                       # (B, 1, H, W)
            amap = F.interpolate(
                amap, size=(target, target), mode="bilinear", align_corners=False
            )
            maps.append(amap)

        return torch.stack(maps, dim=0).mean(dim=0)  # (B, 1, target, target)

    def reduce_anomaly_map(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean":
            return spatial_mean(anomaly_map)
        if self.reduction == "max":
            return spatial_max(anomaly_map)
        return topk_spatial_mean(anomaly_map, topk_ratio=self.topk_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        amap = self.cosine_anomaly_map(x)
        return self.reduce_anomaly_map(amap)


# ------------------------------------------------------------------
# Loss
# ------------------------------------------------------------------


def cosine_loss(enc_feat: torch.Tensor, dec_feat: torch.Tensor) -> torch.Tensor:
    """Mean (1 - cosine_similarity) over batch and spatial dimensions."""
    cos_sim = F.cosine_similarity(enc_feat, dec_feat, dim=1, eps=1e-8)
    return (1.0 - cos_sim).mean()


def rd4ad_loss(
    f1: torch.Tensor,
    f2: torch.Tensor,
    f3: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
    g3: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Multi-scale cosine loss.

    Returns:
        (total, loss1, loss2, loss3) where total = mean of the three scale losses.
    """
    loss1 = cosine_loss(f1, g1)
    loss2 = cosine_loss(f2, g2)
    loss3 = cosine_loss(f3, g3)
    total = (loss1 + loss2 + loss3) / 3.0
    return total, loss1, loss2, loss3


# ------------------------------------------------------------------
# Config builder
# ------------------------------------------------------------------


def build_rd4ad_from_config(config: dict) -> RD4ADModel:
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    image_size = int(model_cfg.get("image_size", data_cfg.get("image_size", 224)))
    return RD4ADModel(
        pretrained=bool(model_cfg.get("pretrained", True)),
        reduction=str(model_cfg.get("reduction", "topk_mean")),
        topk_ratio=float(model_cfg.get("topk_ratio", 0.10)),
        image_size=image_size,
    )
