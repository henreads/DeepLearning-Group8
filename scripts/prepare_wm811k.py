"""Prepare processed WM-811K arrays and split metadata from the raw pickle."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from wafer_defect.config import load_toml
from wafer_defect.data.legacy_pickle import read_legacy_pickle, unwrap_legacy_value


LABEL_NORMAL = "none"
LABEL_DEFECT = "pattern"


def format_count_slug(count: int | None) -> str:
    if count is None:
        return "full"
    if count >= 1000 and count % 1000 == 0:
        return f"{count // 1000}k"
    return str(count)


def format_ratio_slug(ratio: float) -> str:
    percent = ratio * 100.0
    if math.isclose(percent, round(percent), rel_tol=0.0, abs_tol=1e-9):
        return f"{int(round(percent))}pct"

    text = f"{percent:.4f}".rstrip("0").rstrip(".")
    return f"{text.replace('.', 'p')}pct"


def build_variant_slug(args: argparse.Namespace, dev_cfg: dict, train_subset_cfg: dict) -> str:
    if args.dev:
        normal_count = int(dev_cfg["normal_count"])
        defect_count = int(dev_cfg["defect_count"])
        return f"dev_{format_count_slug(normal_count)}n_{format_count_slug(defect_count)}d"

    normal_count = args.normal_limit
    if normal_count is None and train_subset_cfg.get("normal_count"):
        normal_count = int(train_subset_cfg["normal_count"])

    count_slug = format_count_slug(normal_count)
    if bool(train_subset_cfg.get("use_all_defects_for_test", True)):
        return f"{count_slug}_all"

    ratio = float(train_subset_cfg.get("test_defect_fraction_of_test_normals", 1.0))
    return f"{count_slug}_{format_ratio_slug(ratio)}"


def default_output_paths(
    processed_root: Path,
    args: argparse.Namespace,
    dev_cfg: dict,
    train_subset_cfg: dict,
) -> tuple[Path, Path]:
    variant_slug = build_variant_slug(args, dev_cfg, train_subset_cfg)
    metadata_path = processed_root / f"metadata_{variant_slug}.csv"
    arrays_dir = processed_root / f"arrays_{variant_slug}"
    return metadata_path, arrays_dir


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


def sample_test_defects(
    defect_df: pd.DataFrame,
    normal_df: pd.DataFrame,
    train_subset_cfg: dict,
    seed: int,
) -> pd.DataFrame:
    use_all_defects = bool(train_subset_cfg.get("use_all_defects_for_test", True))
    if use_all_defects:
        sampled = defect_df.copy()
        sampled["split"] = "test"
        return sampled

    test_normal_count = int((normal_df["split"] == "test").sum())
    fraction = float(train_subset_cfg.get("test_defect_fraction_of_test_normals", 1.0))
    requested = max(1, int(round(test_normal_count * fraction)))
    sampled = defect_df.sample(n=min(requested, len(defect_df)), random_state=seed).copy()
    sampled["split"] = "test"
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/data.toml")
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--normal-limit", type=int, default=None)
    parser.add_argument("--metadata-path", default=None)
    args = parser.parse_args()

    config = load_toml(args.config)
    dataset_cfg = config["dataset"]
    dev_cfg = config["dev_subset"]
    train_subset_cfg = config.get("train_subset", {})
    split_seed = config["splits"]["random_seed"]
    image_size = int(dataset_cfg["image_size"])

    raw_pickle = Path(dataset_cfg["raw_pickle"])
    processed_root = Path(dataset_cfg["processed_root"])
    default_metadata_path, default_arrays_dir = default_output_paths(
        processed_root,
        args,
        dev_cfg,
        train_subset_cfg,
    )
    if args.metadata_path:
        metadata_path = Path(args.metadata_path)
        arrays_stem = metadata_path.stem
        if arrays_stem.startswith("metadata_"):
            arrays_stem = arrays_stem[len("metadata_") :]
        arrays_dir = metadata_path.parent / f"arrays_{arrays_stem}"
    else:
        metadata_path = default_metadata_path
        arrays_dir = default_arrays_dir

    if not raw_pickle.exists():
        raise FileNotFoundError(f"Raw dataset file not found: {raw_pickle}")

    processed_root.mkdir(parents=True, exist_ok=True)
    arrays_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

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
    elif args.normal_limit is not None:
        normal_df = normal_df.sample(n=min(args.normal_limit, len(normal_df)), random_state=split_seed)
    elif train_subset_cfg.get("normal_count"):
        normal_df = normal_df.sample(
            n=min(int(train_subset_cfg["normal_count"]), len(normal_df)),
            random_state=split_seed,
        )

    normal_df = split_normals(normal_df, split_seed)
    defect_df = sample_test_defects(defect_df, normal_df, train_subset_cfg, split_seed)

    export_df = pd.concat([normal_df, defect_df], ignore_index=True)
    records: list[dict[str, object]] = []
    repo_root = Path.cwd().resolve()

    for row_index, row in export_df.iterrows():
        file_name = f"wafer_{row_index:07d}.npy"
        array_path = arrays_dir / file_name
        raw_map = np.asarray(row["waferMap"])
        wafer_map = normalize_map(raw_map, image_size=image_size)
        np.save(array_path, wafer_map)
        relative_array_path = array_path.resolve().relative_to(repo_root).as_posix()
        records.append(
            {
                "array_path": relative_array_path,
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
    print(f"Using arrays directory: {arrays_dir}")
    print(f"Saved {len(metadata)} rows to {metadata_path}")


if __name__ == "__main__":
    main()
