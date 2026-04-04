"""Pretrained ViT feature extractors for wafer-map experiments."""

from __future__ import annotations

import re

import timm
import torch
import torch.nn.functional as F
from torch import nn


def _parse_block_index(layer_name: str, num_blocks: int) -> int:
    match = re.fullmatch(r"block(\d+)", layer_name.lower())
    if match is None:
        raise ValueError(f"Unsupported ViT feature layer: {layer_name}")
    block_index = int(match.group(1))
    if block_index < 0 or block_index >= num_blocks:
        raise ValueError(f"ViT block index out of range: {layer_name}")
    return block_index


class ViTFeatureExtractor(nn.Module):
    """Pretrained timm ViT adapted for single-channel wafer maps."""

    def __init__(
        self,
        backbone_name: str = "vit_b16",
        pretrained: bool = True,
        input_size: int = 224,
        freeze_backbone: bool = True,
        normalize_imagenet: bool = True,
    ) -> None:
        super().__init__()
        backbone_name = backbone_name.lower()
        if backbone_name not in {"vit_b16", "vit_base_patch16_224"}:
            raise ValueError(f"Unsupported ViT backbone: {backbone_name}")

        self.backbone_name = backbone_name
        self.input_size = int(input_size)
        self.normalize_imagenet = bool(normalize_imagenet)

        self.vit = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=0)
        original_proj = self.vit.patch_embed.proj
        adapted_proj = nn.Conv2d(
            1,
            original_proj.out_channels,
            kernel_size=original_proj.kernel_size,
            stride=original_proj.stride,
            padding=original_proj.padding,
            bias=original_proj.bias is not None,
        )
        with torch.no_grad():
            adapted_proj.weight.copy_(original_proj.weight.mean(dim=1, keepdim=True))
            if original_proj.bias is not None and adapted_proj.bias is not None:
                adapted_proj.bias.copy_(original_proj.bias)
        self.vit.patch_embed.proj = adapted_proj

        self.embedding_dim = int(self.vit.embed_dim)
        self.patch_size = int(self.vit.patch_embed.patch_size[0])
        self.output_spatial = self.input_size // self.patch_size
        self.num_blocks = len(self.vit.blocks)

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

    def _tokens_after_block(self, x: torch.Tensor, block_index: int) -> torch.Tensor:
        x = self.preprocess(x)
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        x = self.vit.patch_drop(x)
        x = self.vit.norm_pre(x)
        for idx, block in enumerate(self.vit.blocks):
            x = block(x)
            if idx == block_index:
                return x
        raise RuntimeError(f"Could not collect ViT tokens for block {block_index}")

    def forward_intermediate_feature_map(self, x: torch.Tensor, layer_name: str = "block6") -> torch.Tensor:
        block_index = _parse_block_index(layer_name, self.num_blocks)
        tokens = self._tokens_after_block(x, block_index)
        patch_tokens = tokens[:, 1:, :]
        spatial_size = int(round(patch_tokens.shape[1] ** 0.5))
        if spatial_size * spatial_size != patch_tokens.shape[1]:
            raise ValueError(f"ViT patch token count is not square: {patch_tokens.shape[1]}")
        return patch_tokens.transpose(1, 2).reshape(x.shape[0], self.embedding_dim, spatial_size, spatial_size)

    def forward_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_intermediate_feature_map(x, layer_name=f"block{self.num_blocks - 1}")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = self.forward_feature_map(x)
        return feature_map.mean(dim=(-2, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)
