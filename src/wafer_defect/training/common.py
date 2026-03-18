"""Shared training datatypes used by model-specific training loops."""
# src/wafer_defect/training/common.py

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EpochMetrics:
    loss: float
    reconstruction_loss: float
    kl_loss: float = 0.0
    distillation_loss: float = 0.0
    auxiliary_loss: float = 0.0
