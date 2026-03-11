"""Shared training datatypes used by model-specific training loops."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EpochMetrics:
    loss: float
    reconstruction_loss: float
    kl_loss: float = 0.0
