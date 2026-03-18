"""Helpers for supervised multiclass wafer-defect classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from wafer_defect.data.legacy_pickle import read_legacy_pickle, unwrap_legacy_value


NORMAL_CLASS = "none"
DEFAULT_CLASS_NAMES = [
    NORMAL_CLASS,
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
]


def augment_wafer_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Apply light train-time augmentation while preserving defect semantics."""
    rotation_k = int(torch.randint(0, 4, (1,)).item())
    if rotation_k:
        tensor = torch.rot90(tensor, k=rotation_k, dims=(1, 2))

    if torch.rand(1).item() < 0.2:
        tensor = tensor.transpose(1, 2)

    if torch.rand(1).item() < 0.5:
        tensor = torch.flip(tensor, dims=[2])
    if torch.rand(1).item() < 0.5:
        tensor = torch.flip(tensor, dims=[1])

    shift_y = int(torch.randint(-4, 5, (1,)).item())
    shift_x = int(torch.randint(-4, 5, (1,)).item())
    if shift_y != 0 or shift_x != 0:
        tensor = torch.roll(tensor, shifts=(shift_y, shift_x), dims=(1, 2))

    if torch.rand(1).item() < 0.2:
        mask_size = int(torch.randint(4, 9, (1,)).item())
        height, width = tensor.shape[1:]
        top = int(torch.randint(0, max(1, height - mask_size + 1), (1,)).item())
        left = int(torch.randint(0, max(1, width - mask_size + 1), (1,)).item())
        tensor[:, top : top + mask_size, left : left + mask_size] = 0.0

    if torch.rand(1).item() < 0.25:
        noise_scale = float(torch.empty(1).uniform_(0.01, 0.04).item())
        tensor = tensor + torch.randn_like(tensor) * noise_scale

    return tensor.clamp_(0.0, 1.0)


def normalize_map(wafer_map: np.ndarray, image_size: int) -> np.ndarray:
    wafer_map = np.asarray(wafer_map, dtype=np.float32)
    if wafer_map.ndim != 2:
        raise ValueError(f"Expected a 2D wafer map, got shape {wafer_map.shape}")

    wafer_map = wafer_map / 2.0
    tensor = torch.from_numpy(wafer_map).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=(image_size, image_size), mode="nearest")
    return resized.squeeze(0).squeeze(0).numpy()


def extract_failure_type(row: pd.Series, valid_class_names: set[str]) -> str | None:
    raw_value = unwrap_legacy_value(row.get("failureType", ""))
    failure_type = str(raw_value).strip()
    if not failure_type:
        return None
    if failure_type not in valid_class_names:
        return None
    return failure_type


def load_raw_wm811k(raw_pickle: str | Path) -> pd.DataFrame:
    df = read_legacy_pickle(Path(raw_pickle)).copy()
    return df.reset_index(drop=True)


def prepare_supervised_dataframe(
    raw_pickle: str | Path,
    class_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    class_names = class_names or DEFAULT_CLASS_NAMES
    valid_class_names = set(class_names)

    df = load_raw_wm811k(raw_pickle)
    df["failure_type"] = df.apply(extract_failure_type, axis=1, valid_class_names=valid_class_names)
    df["source_split"] = df["trianTestLabel"].map(unwrap_legacy_value)
    df["raw_index"] = np.arange(len(df))

    labeled_df = df[df["failure_type"].notna()].copy().reset_index(drop=True)
    unlabeled_df = df[df["failure_type"].isna()].copy().reset_index(drop=True)
    return labeled_df, unlabeled_df


def stratified_split_indices(
    labels: pd.Series,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = train_fraction + val_fraction + test_fraction
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")

    all_indices = np.arange(len(labels))
    train_indices, temp_indices = train_test_split(
        all_indices,
        train_size=train_fraction,
        random_state=random_seed,
        shuffle=True,
        stratify=labels,
    )

    temp_labels = labels.iloc[temp_indices]
    relative_val_fraction = val_fraction / (val_fraction + test_fraction)
    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=relative_val_fraction,
        random_state=random_seed,
        shuffle=True,
        stratify=temp_labels,
    )
    return train_indices, val_indices, test_indices


def build_labeled_metadata(
    labeled_df: pd.DataFrame,
    class_names: list[str],
    image_size: int,
    arrays_dir: str | Path,
    metadata_path: str | Path,
    repo_root: str | Path,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    random_seed: int,
    limit_per_class: int | None = None,
) -> pd.DataFrame:
    repo_root = Path(repo_root).resolve()
    arrays_dir = Path(arrays_dir)
    metadata_path = Path(metadata_path)

    class_to_index = {name: idx for idx, name in enumerate(class_names)}
    export_df = labeled_df.copy()

    if limit_per_class is not None:
        sampled_frames: list[pd.DataFrame] = []
        for class_name in class_names:
            class_rows = export_df[export_df["failure_type"] == class_name]
            sampled_frames.append(
                class_rows.sample(
                    n=min(limit_per_class, len(class_rows)),
                    random_state=random_seed,
                )
            )
        export_df = pd.concat(sampled_frames, ignore_index=True)

    train_idx, val_idx, test_idx = stratified_split_indices(
        export_df["failure_type"],
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        random_seed=random_seed,
    )

    export_df["split"] = ""
    export_df.loc[train_idx, "split"] = "train"
    export_df.loc[val_idx, "split"] = "val"
    export_df.loc[test_idx, "split"] = "test"
    export_df = export_df.reset_index(drop=True)

    arrays_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for row_index, row in export_df.iterrows():
        file_name = f"wafer_{row_index:07d}.npy"
        array_path = arrays_dir / file_name
        raw_map = np.asarray(row["waferMap"])
        wafer_map = normalize_map(raw_map, image_size=image_size)
        np.save(array_path, wafer_map)
        relative_array_path = array_path.resolve().relative_to(repo_root).as_posix()
        label_name = str(row["failure_type"])
        records.append(
            {
                "array_path": relative_array_path,
                "label_name": label_name,
                "label_index": class_to_index[label_name],
                "split": row["split"],
                "raw_index": int(row["raw_index"]),
                "source_split": str(row["source_split"] or "unlabeled"),
                "original_height": int(raw_map.shape[0]),
                "original_width": int(raw_map.shape[1]),
            }
        )

    metadata = pd.DataFrame(records)
    metadata.to_csv(metadata_path, index=False)

    class_info_path = metadata_path.with_suffix(".classes.json")
    class_info_path.write_text(
        json.dumps(
            {
                "class_names": class_names,
                "class_to_index": class_to_index,
                "image_size": image_size,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return metadata


class LabeledWaferDataset(Dataset):
    def __init__(self, metadata_csv: str | Path, split: str) -> None:
        self.metadata_path = Path(metadata_csv).resolve()
        self.repo_root = self._find_repo_root(self.metadata_path)
        self.metadata = pd.read_csv(self.metadata_path)
        self.metadata = self.metadata[self.metadata["split"] == split].reset_index(drop=True)
        self.apply_augmentation = split == "train"

    @staticmethod
    def _find_repo_root(metadata_path: Path) -> Path:
        for candidate in [metadata_path.parent, *metadata_path.parents]:
            if (candidate / "src" / "wafer_defect").exists() and (candidate / "configs").exists():
                return candidate
        raise FileNotFoundError(f"Could not determine repo root from {metadata_path}")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.metadata.iloc[index]
        array_path = self.repo_root / Path(row["array_path"])
        wafer_map = np.load(array_path).astype(np.float32)
        tensor = torch.from_numpy(wafer_map).unsqueeze(0)
        if self.apply_augmentation:
            tensor = augment_wafer_tensor(tensor)
        label = torch.tensor(int(row["label_index"]), dtype=torch.long)
        return tensor, label


class RawWaferInferenceDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_size: int) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.dataframe.iloc[index]
        wafer_map = normalize_map(np.asarray(row["waferMap"]), image_size=self.image_size)
        tensor = torch.from_numpy(wafer_map).unsqueeze(0)
        raw_index = torch.tensor(int(row["raw_index"]), dtype=torch.long)
        return tensor, raw_index
