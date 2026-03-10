from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class WaferMapDataset(Dataset):
    def __init__(self, metadata_csv: str | Path, split: str, image_size: int = 64) -> None:
        self.metadata_path = Path(metadata_csv)
        self.repo_root = self.metadata_path.parent.parent.parent
        self.metadata = pd.read_csv(self.metadata_path)
        self.metadata = self.metadata[self.metadata["split"] == split].reset_index(drop=True)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.metadata.iloc[index]
        array_path = self.repo_root / row["array_path"]
        wafer_map = np.load(array_path).astype(np.float32)
        tensor = torch.from_numpy(wafer_map).unsqueeze(0)
        label = torch.tensor(int(row["is_anomaly"]), dtype=torch.long)
        return tensor, label
