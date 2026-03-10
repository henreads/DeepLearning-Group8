from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from wafer_defect.config import load_toml
from wafer_defect.data.legacy_pickle import read_legacy_pickle, unwrap_legacy_value


LABEL_NORMAL = "none"
LABEL_DEFECT = "pattern"


def normalize_map(wafer_map: np.ndarray, image_size: int) -> np.ndarray:
    wafer_map = np.asarray(wafer_map, dtype=np.float32)
    if wafer_map.ndim != 2:
        raise ValueError(f"Expected 2D wafer map, got shape {wafer_map.shape}")

    wafer_map = wafer_map / 2.0
    tensor = torch.from_numpy(wafer_map).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=(image_size, image_size), mode="nearest")
    return resized.squeeze(0).squeeze(0).numpy()


def infer_label_from_row(row: pd.Series) -> str | None:
    failure = unwrap_legacy_value(row.get("failureType", "")).lower()
    train_test = unwrap_legacy_value(row.get("trianTestLabel", "")).lower()

    if failure and failure != "none":
        return LABEL_DEFECT
    if failure == "none" or train_test == "training":
        return LABEL_NORMAL
    return None


def split_normals(normal_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    shuffled = normal_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    shuffled.loc[: train_end - 1, "split"] = "train"
    shuffled.loc[train_end: val_end - 1, "split"] = "val"
    shuffled.loc[val_end:, "split"] = "test"
    return shuffled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.toml")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    config = load_toml(args.config)
    dataset_cfg = config["dataset"]
    dev_cfg = config["dev_subset"]
    split_seed = config["splits"]["random_seed"]
    image_size = int(dataset_cfg["image_size"])

    raw_pickle = Path(dataset_cfg["raw_pickle"])
    processed_root = Path(dataset_cfg["processed_root"])
    arrays_dir = processed_root / "arrays"
    metadata_path = Path(dataset_cfg["dev_metadata_csv"] if args.dev else dataset_cfg["metadata_csv"])

    if not raw_pickle.exists():
        raise FileNotFoundError(f"Raw dataset file not found: {raw_pickle}")

    processed_root.mkdir(parents=True, exist_ok=True)
    arrays_dir.mkdir(parents=True, exist_ok=True)

    df = read_legacy_pickle(raw_pickle)
    df = df.copy()
    df["failureTypeText"] = df["failureType"].map(unwrap_legacy_value)
    df["trianTestLabelText"] = df["trianTestLabel"].map(unwrap_legacy_value)
    df["label"] = df.apply(infer_label_from_row, axis=1)
    df = df[df["label"].notna()].reset_index(drop=True)

    normal_df = df[df["label"] == LABEL_NORMAL].copy()
    defect_df = df[df["label"] == LABEL_DEFECT].copy()

    if args.dev:
        normal_df = normal_df.sample(n=min(dev_cfg["normal_count"], len(normal_df)), random_state=split_seed)
        defect_df = defect_df.sample(n=min(dev_cfg["defect_count"], len(defect_df)), random_state=split_seed)

    normal_df = split_normals(normal_df, split_seed)
    defect_df["split"] = "test"

    export_df = pd.concat([normal_df, defect_df], ignore_index=True)
    records: list[dict[str, object]] = []

    for row_index, row in export_df.iterrows():
        file_name = f"wafer_{row_index:07d}.npy"
        array_path = arrays_dir / file_name
        raw_map = np.asarray(row["waferMap"])
        wafer_map = normalize_map(raw_map, image_size=image_size)
        np.save(array_path, wafer_map)
        records.append(
            {
                "array_path": array_path.as_posix(),
                "label": row["label"],
                "defect_type": row["failureTypeText"] or "unlabeled",
                "is_anomaly": int(row["label"] == LABEL_DEFECT),
                "split": row["split"],
                "source_split": row["trianTestLabelText"] or "unlabeled",
                "original_height": int(raw_map.shape[0]),
                "original_width": int(raw_map.shape[1]),
            }
        )

    metadata = pd.DataFrame(records)
    metadata.to_csv(metadata_path, index=False)
    print(f"Saved {len(metadata)} rows to {metadata_path}")


if __name__ == "__main__":
    main()
