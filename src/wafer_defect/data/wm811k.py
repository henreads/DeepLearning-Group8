"""Dataset loader for processed WM-811K wafer-map arrays and split metadata."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class WaferMapDataset(Dataset):
    def __init__(self, metadata_csv: str | Path, split: str, image_size: int = 64) -> None:
        self.metadata_path = self._resolve_metadata_path(Path(metadata_csv), image_size).resolve()
        self.repo_root = self._find_repo_root(self.metadata_path)
        self.metadata = pd.read_csv(self.metadata_path)
        self.metadata = self.metadata[self.metadata["split"] == split].reset_index(drop=True)
        self.image_size = image_size

    @staticmethod
    def _candidate_repo_roots() -> list[Path]:
        roots: list[Path] = []
        module_root = Path(__file__).resolve().parents[3]
        roots.append(module_root)

        cwd = Path.cwd().resolve()
        roots.extend([cwd, *cwd.parents])

        unique_roots: list[Path] = []
        seen: set[Path] = set()
        for root in roots:
            if root not in seen:
                unique_roots.append(root)
                seen.add(root)
        return unique_roots

    @staticmethod
    def _resolve_metadata_path(metadata_path: Path, image_size: int) -> Path:
        if metadata_path.exists():
            return metadata_path

        cwd_candidate = (Path.cwd() / metadata_path).resolve()
        if cwd_candidate.exists():
            return cwd_candidate

        matches: list[Path] = []
        for repo_root in WaferMapDataset._candidate_repo_roots():
            direct_candidate = (repo_root / metadata_path).resolve()
            if direct_candidate.exists():
                return direct_candidate

            processed_root = repo_root / "data" / "processed"
            if processed_root.exists():
                matches.extend(processed_root.rglob(metadata_path.name))

        unique_matches: list[Path] = []
        seen: set[Path] = set()
        for match in sorted(path.resolve() for path in matches):
            if match not in seen:
                unique_matches.append(match)
                seen.add(match)

        preferred_matches = [match for match in unique_matches if f"x{image_size}" in match.parts]
        if len(preferred_matches) == 1:
            return preferred_matches[0]
        if len(unique_matches) == 1:
            return unique_matches[0]

        raise FileNotFoundError(f"Could not resolve metadata path: {metadata_path}")

    @staticmethod
    def _find_repo_root(metadata_path: Path) -> Path:
        env_root = os.environ.get("WM811K_REPO_ROOT")
        if env_root:
            return Path(env_root).resolve()

        for candidate in WaferMapDataset._candidate_repo_roots():
            if (candidate / "data").exists() and (
                (candidate / "src" / "wafer_defect").exists() or (candidate / "experiments").exists()
            ):
                return candidate

        for candidate in [metadata_path.parent, *metadata_path.parents]:
            if (candidate / "data").exists() and (
                (candidate / "src" / "wafer_defect").exists() or (candidate / "experiments").exists()
            ):
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
