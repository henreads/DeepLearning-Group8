"""Build a secondary holdout metadata file on top of the existing 50k_5pct split.

This keeps the current 50k_5pct train/val rows unchanged so validation-threshold
evaluation remains directly comparable, while replacing the test split with a
larger disjoint holdout sampled from unused raw WM-811K rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from wafer_defect.data.legacy_pickle import read_legacy_pickle, unwrap_legacy_value


LABEL_NORMAL = "none"
LABEL_DEFECT = "pattern"
DEFAULT_SEED = 42


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
    if failure and failure != "none":
        return LABEL_DEFECT
    if failure == "none":
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


def load_raw_labeled_dataframe(raw_pickle: Path) -> pd.DataFrame:
    df = read_legacy_pickle(raw_pickle).copy()
    df["raw_index"] = np.arange(len(df), dtype=np.int64)
    df["failureTypeText"] = df["failureType"].map(unwrap_legacy_value)
    df["trianTestLabelText"] = df["trianTestLabel"].map(unwrap_legacy_value)
    df["label"] = df.apply(infer_label_from_row, axis=1)
    return df[df["label"].notna()].reset_index(drop=True)


def reconstruct_base_split(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    normal_df = df[df["label"] == LABEL_NORMAL].copy()
    defect_df = df[df["label"] == LABEL_DEFECT].copy()

    base_normals = normal_df.sample(n=50000, random_state=seed).copy()
    base_normals = split_normals(base_normals, seed)
    base_defects = defect_df.sample(n=250, random_state=seed).copy()
    base_defects["split"] = "test"
    return base_normals, base_defects


def build_export_records(
    rows: pd.DataFrame,
    *,
    arrays_dir: Path,
    processed_root: Path,
    image_size: int,
    file_offset: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    relative_arrays_root = processed_root / arrays_dir.name
    for local_index, (_, row) in enumerate(rows.iterrows(), start=file_offset):
        file_name = f"wafer_{local_index:07d}.npy"
        array_path = arrays_dir / file_name
        raw_map = np.asarray(row["waferMap"])
        wafer_map = normalize_map(raw_map, image_size=image_size)
        np.save(array_path, wafer_map)
        relative_array_path = (relative_arrays_root / file_name).as_posix()
        records.append(
            {
                "array_path": relative_array_path,
                "label": row["label"],
                "defect_type": row["failureTypeText"] or "unlabeled",
                "is_anomaly": int(row["label"] == LABEL_DEFECT),
                "split": "test",
                "source_split": row["trianTestLabelText"] or "unlabeled",
                "original_height": int(raw_map.shape[0]),
                "original_width": int(raw_map.shape[1]),
                "raw_index": int(row["raw_index"]),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-pickle", default="data/raw/LSWMD.pkl")
    parser.add_argument("--base-metadata", default="data/processed/x64/wm811k/metadata_50k_5pct.csv")
    parser.add_argument("--output-metadata", default="data/processed/x64/wm811k/metadata_50k_5pct_holdout70k_3p5k.csv")
    parser.add_argument("--test-normal-count", type=int, default=70000)
    parser.add_argument("--test-defect-count", type=int, default=3500)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    raw_pickle = (repo_root / args.raw_pickle).resolve()
    base_metadata_path = (repo_root / args.base_metadata).resolve()
    output_metadata_path = (repo_root / args.output_metadata).resolve()
    output_arrays_dir = output_metadata_path.parent / f"arrays_{output_metadata_path.stem.removeprefix('metadata_')}"

    if not raw_pickle.exists():
        raise FileNotFoundError(f"Raw dataset file not found: {raw_pickle}")
    if not base_metadata_path.exists():
        raise FileNotFoundError(f"Base metadata file not found: {base_metadata_path}")

    output_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    output_arrays_dir.mkdir(parents=True, exist_ok=True)

    base_metadata = pd.read_csv(base_metadata_path)
    base_train_val = base_metadata[base_metadata["split"].isin(["train", "val"])].copy()

    raw_df = load_raw_labeled_dataframe(raw_pickle)
    base_normals, base_defects = reconstruct_base_split(raw_df, args.seed)
    used_raw_indices = set(base_normals["raw_index"].astype(int).tolist())
    used_raw_indices.update(base_defects["raw_index"].astype(int).tolist())

    normal_pool = raw_df[
        (raw_df["label"] == LABEL_NORMAL) & (~raw_df["raw_index"].isin(used_raw_indices))
    ].copy()
    defect_pool = raw_df[
        (raw_df["label"] == LABEL_DEFECT) & (~raw_df["raw_index"].isin(used_raw_indices))
    ].copy()

    if args.test_normal_count > len(normal_pool):
        raise ValueError(
            f"Requested {args.test_normal_count} holdout normals, but only {len(normal_pool)} are available."
        )
    if args.test_defect_count > len(defect_pool):
        raise ValueError(
            f"Requested {args.test_defect_count} holdout defects, but only {len(defect_pool)} are available."
        )

    holdout_normals = normal_pool.sample(n=args.test_normal_count, random_state=args.seed).copy()
    holdout_defects = defect_pool.sample(n=args.test_defect_count, random_state=args.seed).copy()

    records = build_export_records(
        pd.concat([holdout_normals, holdout_defects], ignore_index=True),
        arrays_dir=output_arrays_dir,
        processed_root=Path(args.output_metadata).parent,
        image_size=args.image_size,
        file_offset=0,
    )
    holdout_metadata = pd.DataFrame(records)

    merged_metadata = pd.concat(
        [
            base_train_val,
            holdout_metadata.drop(columns=["raw_index"]),
        ],
        ignore_index=True,
    )
    merged_metadata.to_csv(output_metadata_path, index=False)

    test_summary = (
        holdout_metadata.groupby(["label", "defect_type"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["label", "count", "defect_type"], ascending=[True, False, True])
    )
    split_summary = (
        merged_metadata.groupby(["split", "is_anomaly"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["split", "is_anomaly"])
    )

    print(f"Saved metadata to {output_metadata_path}")
    print(f"Saved holdout arrays to {output_arrays_dir}")
    print("Split summary:")
    print(split_summary.to_string(index=False))
    print("Holdout defect breakdown:")
    print(test_summary[test_summary["label"] == LABEL_DEFECT].to_string(index=False))
    print("Holdout normal count:")
    print(int((holdout_metadata["label"] == LABEL_NORMAL).sum()))


if __name__ == "__main__":
    main()
