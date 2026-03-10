from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class WaferMapDataset(Dataset):
    def __init__(self, metadata_csv: str | Path, split: str, image_size: int = 64) -> None:
        self.metadata_path = Path(metadata_csv).resolve()
        self.repo_root = self._find_repo_root(self.metadata_path)
        self.metadata = pd.read_csv(self.metadata_path)
        self.metadata = self.metadata[self.metadata["split"] == split].reset_index(drop=True)
        self.image_size = image_size

    @staticmethod
    def _find_repo_root(metadata_path: Path) -> Path:
        for candidate in [metadata_path.parent, *metadata_path.parents]:
            if (candidate / "src" / "wafer_defect").exists() and (candidate / "configs").exists():
                return candidate
        raise FileNotFoundError(
            f"Could not determine repo root from metadata path: {metadata_path}"
        )

    def __len__(self) -> int:
        return len(self.metadata)

    def _resolve_array_path(self, raw_path: str | Path) -> Path:
        array_path = Path(raw_path)
        if array_path.is_absolute():
            return array_path

        candidates = [self.repo_root / array_path]
        candidates.extend(parent / array_path for parent in self.metadata_path.parents)

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"Could not resolve array path '{raw_path}' from metadata '{self.metadata_path}'. "
            f"Tried: {[str(candidate) for candidate in candidates[:5]]}"
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.metadata.iloc[index]
        array_path = self._resolve_array_path(row["array_path"])
        wafer_map = np.load(array_path).astype(np.float32)
        tensor = torch.from_numpy(wafer_map).unsqueeze(0)
        label = torch.tensor(int(row["is_anomaly"]), dtype=torch.long)
        return tensor, label
