"""Build a supervised multiclass dataset from labeled WM-811K rows."""

from __future__ import annotations

import argparse
from pathlib import Path

from wafer_defect.config import load_toml
from wafer_defect.classification.data import (
    DEFAULT_CLASS_NAMES,
    build_labeled_metadata,
    prepare_supervised_dataframe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/classifier/data_multiclass.toml")
    parser.add_argument("--limit-per-class", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_toml(args.config)
    dataset_cfg = config["dataset"]
    split_cfg = config["splits"]

    raw_pickle = Path(dataset_cfg["raw_pickle"])
    processed_root = Path(dataset_cfg["processed_root"])
    metadata_csv = Path(dataset_cfg["labeled_metadata_csv"])
    arrays_dir = Path(dataset_cfg["labeled_arrays_dir"])
    image_size = int(dataset_cfg["image_size"])
    class_names = list(dataset_cfg.get("class_names", DEFAULT_CLASS_NAMES))

    labeled_df, unlabeled_df = prepare_supervised_dataframe(
        raw_pickle=raw_pickle,
        class_names=class_names,
    )

    metadata = build_labeled_metadata(
        labeled_df=labeled_df,
        class_names=class_names,
        image_size=image_size,
        arrays_dir=arrays_dir,
        metadata_path=metadata_csv,
        repo_root=Path.cwd(),
        train_fraction=float(split_cfg["train_fraction"]),
        val_fraction=float(split_cfg["val_fraction"]),
        test_fraction=float(split_cfg["test_fraction"]),
        random_seed=int(split_cfg["random_seed"]),
        limit_per_class=args.limit_per_class,
    )

    processed_root.mkdir(parents=True, exist_ok=True)
    summary_path = processed_root / "dataset_summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"labeled_rows={len(labeled_df)}",
                f"unlabeled_rows={len(unlabeled_df)}",
                "",
                "class_counts:",
                metadata["label_name"].value_counts().sort_index().to_string(),
                "",
                "split_counts:",
                metadata["split"].value_counts().sort_index().to_string(),
            ]
        ),
        encoding="utf-8",
    )

    print(f"Saved labeled metadata to {metadata_csv}")
    print(f"Saved labeled arrays under {arrays_dir}")
    print(f"Detected {len(labeled_df)} labeled rows and {len(unlabeled_df)} unlabeled rows")


if __name__ == "__main__":
    main()
