"""Teacher-student distillation anomaly detector for wafer maps."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from wafer_defect.models.resnet import ResNetFeatureExtractor
from wafer_defect.scoring import spatial_max, spatial_mean, topk_spatial_mean


def _teacher_feature_dim(backbone_name: str, layer_name: str) -> int:
    backbone_name = backbone_name.lower()
    layer_name = layer_name.lower()

    dims = {
        "resnet18": {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512},
        "resnet50": {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048},
        "vit_b16": {f"block{i}": 768 for i in range(12)},
        "vit_base_patch16_224": {f"block{i}": 768 for i in range(12)},
    }
    if backbone_name not in dims or layer_name not in dims[backbone_name]:
        raise ValueError(f"Unsupported teacher backbone/layer combination: {backbone_name} / {layer_name}")
    return dims[backbone_name][layer_name]


class TSStudent(nn.Module):
    def __init__(self, feature_dim: int, input_size: int = 224) -> None:
        super().__init__()
        if input_size % 8 != 0:
            raise ValueError(f"input_size must be divisible by 8, got {input_size}")

        self.input_size = input_size
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TeacherFeatureAutoencoder(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        bottleneck_dim = max(hidden_dim // 2, 16)
        self.encoder = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, bottleneck_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(bottleneck_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, feature_dim, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class TSDistillationModel(nn.Module):
    def __init__(
        self,
        image_size: int = 64,
        teacher_backbone: str = "resnet18",
        teacher_layer: str = "layer2",
        teacher_pretrained: bool = True,
        teacher_input_size: int = 224,
        normalize_teacher_input: bool = True,
        reduction: str = "topk_mean",
        topk_ratio: float = 0.01,
        feature_autoencoder_hidden_dim: int = 64,
        student_weight: float = 1.0,
        autoencoder_weight: float = 1.0,
        score_student_weight: float | None = None,
        score_autoencoder_weight: float | None = None,
    ) -> None:
        super().__init__()
        if reduction not in {"max", "mean", "topk_mean"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        if reduction == "topk_mean" and not 0.0 < topk_ratio <= 1.0:
            raise ValueError(f"topk_ratio must be in (0, 1], got {topk_ratio}")

        self.image_size = image_size
        self.teacher_backbone = teacher_backbone.lower()
        self.teacher_layer = teacher_layer.lower()
        self.teacher_input_size = int(teacher_input_size)
        self.reduction = reduction
        self.topk_ratio = float(topk_ratio)
        self.student_weight = float(student_weight)
        self.autoencoder_weight = float(autoencoder_weight)
        self.score_student_weight = (
            float(score_student_weight) if score_student_weight is not None else self.student_weight
        )
        self.score_autoencoder_weight = (
            float(score_autoencoder_weight) if score_autoencoder_weight is not None else self.autoencoder_weight
        )
        self.feature_dim = _teacher_feature_dim(self.teacher_backbone, self.teacher_layer)

        if self.teacher_backbone in {"vit_b16", "vit_base_patch16_224"}:
            from wafer_defect.models.vit import ViTFeatureExtractor

            self.teacher = ViTFeatureExtractor(
                backbone_name=self.teacher_backbone,
                pretrained=teacher_pretrained,
                input_size=self.teacher_input_size,
                freeze_backbone=True,
                normalize_imagenet=normalize_teacher_input,
            )
        else:
            self.teacher = ResNetFeatureExtractor(
                backbone_name=self.teacher_backbone,
                pretrained=teacher_pretrained,
                input_size=self.teacher_input_size,
                freeze_backbone=True,
                normalize_imagenet=normalize_teacher_input,
            )
        self.student = TSStudent(feature_dim=self.feature_dim, input_size=self.teacher_input_size)
        self.autoencoder = TeacherFeatureAutoencoder(
            feature_dim=self.feature_dim,
            hidden_dim=int(feature_autoencoder_hidden_dim),
        )

        self.register_buffer("student_map_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("autoencoder_map_scale", torch.tensor(1.0, dtype=torch.float32))

    def teacher_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.teacher.forward_intermediate_feature_map(x, layer_name=self.teacher_layer)

    def _student_input(self, x: torch.Tensor) -> torch.Tensor:
        return self.teacher.preprocess(x)

    def student_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return self.student(self._student_input(x))

    def autoencoder_feature_map(self, teacher_features: torch.Tensor) -> torch.Tensor:
        autoencoder_features = self.autoencoder(teacher_features)
        if autoencoder_features.shape[-2:] != teacher_features.shape[-2:]:
            autoencoder_features = F.interpolate(
                autoencoder_features,
                size=teacher_features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return autoencoder_features

    def raw_anomaly_maps(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        teacher_features = self.teacher_feature_map(x)
        student_features = self.student_feature_map(x)
        if student_features.shape[-2:] != teacher_features.shape[-2:]:
            # Allow layer sweeps across teacher stages with different spatial resolutions.
            student_features = F.interpolate(
                student_features,
                size=teacher_features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        autoencoder_features = self.autoencoder_feature_map(teacher_features)
        student_map = torch.mean((student_features - teacher_features).pow(2), dim=1, keepdim=True)
        autoencoder_map = torch.mean((autoencoder_features - teacher_features).pow(2), dim=1, keepdim=True)
        return student_map, autoencoder_map

    def normalized_anomaly_map(self, x: torch.Tensor) -> torch.Tensor:
        student_map, autoencoder_map = self.raw_anomaly_maps(x)
        student_scale = self.student_map_scale.clamp_min(1e-6)
        autoencoder_scale = self.autoencoder_map_scale.clamp_min(1e-6)
        combined = (
            self.score_student_weight * (student_map / student_scale)
            + self.score_autoencoder_weight * (autoencoder_map / autoencoder_scale)
        )
        return combined

    def set_error_scales(self, student_scale: float, autoencoder_scale: float) -> None:
        self.student_map_scale.fill_(max(float(student_scale), 1e-6))
        self.autoencoder_map_scale.fill_(max(float(autoencoder_scale), 1e-6))

    def reduce_anomaly_map(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean":
            return spatial_mean(anomaly_map)
        if self.reduction == "max":
            return spatial_max(anomaly_map)
        return topk_spatial_mean(anomaly_map, topk_ratio=self.topk_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        anomaly_map = self.normalized_anomaly_map(x)
        return self.reduce_anomaly_map(anomaly_map)


def build_ts_distillation_from_config(config: dict, image_size: int) -> TSDistillationModel:
    model_config = config.get("model", {})
    return TSDistillationModel(
        image_size=image_size,
        teacher_backbone=str(model_config.get("teacher_backbone", "resnet18")),
        teacher_layer=str(model_config.get("teacher_layer", "layer2")),
        teacher_pretrained=bool(model_config.get("teacher_pretrained", True)),
        teacher_input_size=int(model_config.get("teacher_input_size", 224)),
        normalize_teacher_input=bool(model_config.get("normalize_teacher_input", True)),
        reduction=str(model_config.get("reduction", "topk_mean")),
        topk_ratio=float(model_config.get("topk_ratio", 0.01)),
        feature_autoencoder_hidden_dim=int(model_config.get("feature_autoencoder_hidden_dim", 64)),
        student_weight=float(model_config.get("student_weight", 1.0)),
        autoencoder_weight=float(model_config.get("autoencoder_weight", 1.0)),
        score_student_weight=model_config.get("score_student_weight"),
        score_autoencoder_weight=model_config.get("score_autoencoder_weight"),
    )
