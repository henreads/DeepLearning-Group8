"""PatchCore-style local nearest-neighbor anomaly detector."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class PatchCoreModel(nn.Module):
    def __init__(
        self,
        image_size: int = 64,
        use_batchnorm: bool = True,
        reduction: str = "max",
        topk_ratio: float = 0.1,
        query_chunk_size: int = 2048,
        memory_chunk_size: int = 8192,
    ) -> None:
        super().__init__()
        if image_size % 8 != 0:
            raise ValueError(f"image_size must be divisible by 8, got {image_size}")
        if reduction not in {"max", "mean", "topk_mean"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        if reduction == "topk_mean" and not 0.0 < topk_ratio <= 1.0:
            raise ValueError(f"topk_ratio must be in (0, 1], got {topk_ratio}")

        self.image_size = image_size
        self.use_batchnorm = use_batchnorm
        self.reduction = reduction
        self.topk_ratio = topk_ratio
        self.query_chunk_size = query_chunk_size
        self.memory_chunk_size = memory_chunk_size

        feature_layers: list[nn.Module] = []
        channel_specs = [(1, 16), (16, 32), (32, 64)]
        for in_channels, out_channels in channel_specs:
            feature_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            )
            if use_batchnorm:
                feature_layers.append(nn.BatchNorm2d(out_channels))
            feature_layers.append(nn.ReLU())
        self.features = nn.Sequential(*feature_layers)
        self.feature_dim = 64
        self.reduced_spatial = image_size // 8
        self.register_buffer("memory_bank", torch.empty(0, self.feature_dim))

    @property
    def patches_per_image(self) -> int:
        return self.reduced_spatial * self.reduced_spatial

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def patch_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = self.feature_map(x)
        embeddings = feature_map.permute(0, 2, 3, 1).reshape(x.shape[0], -1, self.feature_dim)
        return F.normalize(embeddings, p=2, dim=-1)

    def set_memory_bank(self, memory_bank: torch.Tensor) -> None:
        if memory_bank.ndim != 2 or memory_bank.shape[1] != self.feature_dim:
            raise ValueError(
                f"memory_bank must have shape (N, {self.feature_dim}), got {tuple(memory_bank.shape)}"
            )
        normalized = F.normalize(memory_bank.to(dtype=torch.float32), p=2, dim=1)
        self.memory_bank = normalized.to(device=self.memory_bank.device, dtype=self.memory_bank.dtype)

    def nearest_patch_distances(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        if self.memory_bank.numel() == 0:
            raise ValueError("memory_bank is empty. Fit PatchCore before scoring.")

        batch_size, patch_count, _ = patch_embeddings.shape
        flat_queries = patch_embeddings.reshape(-1, self.feature_dim)
        all_mins: list[torch.Tensor] = []

        for query_start in range(0, flat_queries.shape[0], self.query_chunk_size):
            query_chunk = flat_queries[query_start : query_start + self.query_chunk_size]
            chunk_best = None
            for memory_start in range(0, self.memory_bank.shape[0], self.memory_chunk_size):
                memory_chunk = self.memory_bank[memory_start : memory_start + self.memory_chunk_size]
                distances = torch.cdist(query_chunk, memory_chunk)
                current_best = distances.min(dim=1).values
                chunk_best = current_best if chunk_best is None else torch.minimum(chunk_best, current_best)
            all_mins.append(chunk_best)

        return torch.cat(all_mins, dim=0).reshape(batch_size, patch_count)

    def reduce_patch_distances(self, patch_distances: torch.Tensor) -> torch.Tensor:
        if self.reduction == "max":
            return patch_distances.max(dim=1).values
        if self.reduction == "mean":
            return patch_distances.mean(dim=1)

        topk = max(1, int(math.ceil(patch_distances.shape[1] * self.topk_ratio)))
        return torch.topk(patch_distances, k=topk, dim=1).values.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_embeddings = self.patch_embeddings(x)
        patch_distances = self.nearest_patch_distances(patch_embeddings)
        return self.reduce_patch_distances(patch_distances)

    def load_backbone_from_autoencoder_checkpoint(self, checkpoint_path: str | Path) -> dict:
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
        checkpoint_config = checkpoint.get("config", {})
        state_dict = checkpoint["model_state_dict"]
        mapped_state_dict = {}
        feature_count = len(self.features)

        for key, value in state_dict.items():
            if not key.startswith("encoder."):
                continue

            parts = key.split(".")
            if len(parts) < 3:
                continue

            layer_index = int(parts[1])
            if layer_index >= feature_count:
                continue

            mapped_key = ".".join(["features", parts[1], *parts[2:]])
            mapped_state_dict[mapped_key] = value

        missing, unexpected = self.load_state_dict(mapped_state_dict, strict=False)
        unexpected = [key for key in unexpected if not key.startswith("memory_bank")]
        if unexpected:
            raise ValueError(f"Unexpected backbone keys when loading PatchCore backbone: {unexpected}")

        required_keys = {
            f"features.{name}"
            for name, _ in self.features.state_dict().items()
            if name.endswith("weight") or name.endswith("bias")
        }
        loaded_keys = set(mapped_state_dict)
        missing_required = sorted(key for key in required_keys if key not in loaded_keys)
        if missing_required:
            raise ValueError(f"Missing backbone keys when loading PatchCore backbone: {missing_required}")

        return checkpoint_config
