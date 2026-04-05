"""Project labeled and pseudo-labeled WM-811K rows into classifier embedding UMAP space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from wafer_defect.classification.data import (
    DEFAULT_CLASS_NAMES,
    RawWaferInferenceDataset,
    prepare_supervised_dataframe,
)
from wafer_defect.classification.models import WaferClassifier
from wafer_defect.config import load_toml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/classifier/data_multiclass_all_80_10_10.toml")
    parser.add_argument(
        "--checkpoint",
        default="artifacts/multiclass_classifier_all_80_10_10_seed07/best_model.pt",
    )
    parser.add_argument(
        "--pseudo-label-csv",
        default="artifacts/multiclass_classifier_all_80_10_10_seed07_pseudolabels/unlabeled_predictions.seed07.pseudolabels.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/multiclass_classifier_all_80_10_10_seed07_pseudolabels/umap",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-labeled-per-class", type=int, default=500)
    parser.add_argument("--max-pseudo-per-class", type=int, default=1000)
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--random-seed", type=int, default=7)
    return parser.parse_args()


def load_umap_module():
    try:
        import umap
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "UMAP visualization requires the 'umap-learn' package. "
            "Install it first, for example with `pip install umap-learn`."
        ) from exc
    return umap


def load_classifier(checkpoint_path: Path, dataset_cfg: dict[str, object]) -> tuple[WaferClassifier, list[str]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    class_names = list(checkpoint.get("class_names", dataset_cfg.get("class_names", DEFAULT_CLASS_NAMES)))
    model_cfg = checkpoint["model_config"]
    model = WaferClassifier(
        num_classes=len(class_names),
        base_channels=int(model_cfg["base_channels"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        dropout=float(model_cfg["dropout"]),
        variant=str(model_cfg.get("variant", "baseline")),
        block_dropout=float(model_cfg.get("block_dropout", 0.0)),
        se_reduction=int(model_cfg.get("se_reduction", 8)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, class_names


def stratified_sample(
    dataframe: pd.DataFrame,
    group_column: str,
    max_per_group: int | None,
    random_seed: int,
) -> pd.DataFrame:
    if max_per_group is None or max_per_group <= 0:
        return dataframe.reset_index(drop=True).copy()

    sampled_frames: list[pd.DataFrame] = []
    for _, group_df in dataframe.groupby(group_column, sort=True):
        if len(group_df) <= max_per_group:
            sampled_frames.append(group_df.copy())
        else:
            sampled_frames.append(group_df.sample(n=max_per_group, random_state=random_seed))
    return pd.concat(sampled_frames, ignore_index=True).reset_index(drop=True)


def normalize_prediction_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    renamed = predictions.copy()
    if "pseudo_label" not in renamed.columns and "predicted_label" in renamed.columns:
        renamed = renamed.rename(columns={"predicted_label": "pseudo_label"})
    if "pseudo_label_confidence" not in renamed.columns and "confidence" in renamed.columns:
        renamed = renamed.rename(columns={"confidence": "pseudo_label_confidence"})
    return renamed


@torch.no_grad()
def collect_embeddings(
    model: WaferClassifier,
    dataframe: pd.DataFrame,
    image_size: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    dataset = RawWaferInferenceDataset(dataframe=dataframe, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    embedding_batches: list[np.ndarray] = []

    model = model.to(device)
    model.eval()

    for inputs, _ in loader:
        embeddings = model.extract_embedding(inputs.to(device))
        embedding_batches.append(embeddings.cpu().numpy())

    if not embedding_batches:
        raise ValueError("Could not extract embeddings because the dataframe is empty.")
    return np.concatenate(embedding_batches, axis=0)


def build_palette(class_names: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10")
    return {class_name: cmap(idx % cmap.N) for idx, class_name in enumerate(class_names)}


def plot_umap_projection(
    labeled_points: pd.DataFrame,
    pseudo_points: pd.DataFrame,
    class_names: list[str],
    min_confidence: float,
    figure_path: Path,
) -> None:
    palette = build_palette(class_names)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)

    for class_name in class_names:
        label_mask = labeled_points["true_label"] == class_name
        if label_mask.any():
            axes[0].scatter(
                labeled_points.loc[label_mask, "umap_x"],
                labeled_points.loc[label_mask, "umap_y"],
                s=16,
                alpha=0.75,
                color=palette[class_name],
                label=class_name,
            )
    axes[0].set_title("Labeled WM-811K Reference")
    axes[0].set_xlabel("UMAP-1")
    axes[0].set_ylabel("UMAP-2")
    axes[0].grid(alpha=0.2)
    axes[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    axes[1].scatter(
        labeled_points["umap_x"],
        labeled_points["umap_y"],
        s=8,
        alpha=0.18,
        color="#9ca3af",
        label="Labeled reference",
    )
    for class_name in class_names:
        label_mask = pseudo_points["pseudo_label"] == class_name
        if label_mask.any():
            point_sizes = 12.0 + 26.0 * pseudo_points.loc[label_mask, "pseudo_label_confidence"].to_numpy()
            axes[1].scatter(
                pseudo_points.loc[label_mask, "umap_x"],
                pseudo_points.loc[label_mask, "umap_y"],
                s=point_sizes,
                alpha=0.55,
                color=palette[class_name],
                label=class_name,
            )
    axes[1].set_title(f"Pseudo Labels Projected Into Labeled Space (confidence >= {min_confidence:.2f})")
    axes[1].set_xlabel("UMAP-1")
    axes[1].set_ylabel("UMAP-2")
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    umap = load_umap_module()

    config = load_toml(args.config)
    dataset_cfg = config["dataset"]
    checkpoint_path = Path(args.checkpoint)
    pseudo_label_csv = Path(args.pseudo_label_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, class_names = load_classifier(checkpoint_path, dataset_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labeled_df, unlabeled_df = prepare_supervised_dataframe(
        raw_pickle=dataset_cfg["raw_pickle"],
        class_names=class_names,
    )

    pseudo_predictions = normalize_prediction_columns(pd.read_csv(pseudo_label_csv))
    required_prediction_columns = {"raw_index", "pseudo_label", "pseudo_label_confidence"}
    missing_columns = required_prediction_columns.difference(pseudo_predictions.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise KeyError(f"Pseudo-label CSV is missing required columns: {missing_text}")

    pseudo_predictions = pseudo_predictions[pseudo_predictions["pseudo_label_confidence"] >= args.min_confidence].copy()
    if pseudo_predictions.empty:
        raise ValueError(
            f"No pseudo-labeled rows met the requested minimum confidence of {args.min_confidence:.2f}."
        )

    labeled_sample = stratified_sample(
        labeled_df.rename(columns={"failure_type": "true_label"}),
        group_column="true_label",
        max_per_group=args.max_labeled_per_class,
        random_seed=args.random_seed,
    )
    pseudo_prediction_sample = stratified_sample(
        pseudo_predictions,
        group_column="pseudo_label",
        max_per_group=args.max_pseudo_per_class,
        random_seed=args.random_seed,
    )
    pseudo_sample = unlabeled_df[["raw_index", "waferMap"]].merge(
        pseudo_prediction_sample,
        on="raw_index",
        how="inner",
        validate="one_to_one",
    )

    image_size = int(dataset_cfg["image_size"])
    labeled_embeddings = collect_embeddings(
        model=model,
        dataframe=labeled_sample[["raw_index", "waferMap"]].copy(),
        image_size=image_size,
        batch_size=args.batch_size,
        device=device,
    )
    pseudo_embeddings = collect_embeddings(
        model=model,
        dataframe=pseudo_sample[["raw_index", "waferMap"]].copy(),
        image_size=image_size,
        batch_size=args.batch_size,
        device=device,
    )

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.random_seed,
        transform_seed=args.random_seed,
    )
    labeled_coords = reducer.fit_transform(labeled_embeddings)
    pseudo_coords = reducer.transform(pseudo_embeddings)

    labeled_points = labeled_sample[
        ["raw_index", "true_label", "source_split", "original_height", "original_width"]
    ].copy()
    labeled_points["umap_x"] = labeled_coords[:, 0]
    labeled_points["umap_y"] = labeled_coords[:, 1]

    pseudo_points = pseudo_sample[
        [
            "raw_index",
            "pseudo_label",
            "pseudo_label_confidence",
            "second_choice_label",
            "second_choice_confidence",
            "accepted_for_pseudo_label",
        ]
    ].copy()
    pseudo_points["umap_x"] = pseudo_coords[:, 0]
    pseudo_points["umap_y"] = pseudo_coords[:, 1]

    labeled_csv = output_dir / "labeled_reference_umap.csv"
    pseudo_csv = output_dir / "pseudo_label_umap.csv"
    figure_path = output_dir / "seed07_pseudolabel_umap.png"
    summary_path = output_dir / "seed07_pseudolabel_umap.summary.json"

    labeled_points.to_csv(labeled_csv, index=False)
    pseudo_points.to_csv(pseudo_csv, index=False)
    plot_umap_projection(
        labeled_points=labeled_points,
        pseudo_points=pseudo_points,
        class_names=class_names,
        min_confidence=args.min_confidence,
        figure_path=figure_path,
    )

    summary = {
        "checkpoint": str(checkpoint_path),
        "pseudo_label_csv": str(pseudo_label_csv),
        "output_dir": str(output_dir),
        "device": str(device),
        "class_names": class_names,
        "min_confidence": float(args.min_confidence),
        "max_labeled_per_class": int(args.max_labeled_per_class),
        "max_pseudo_per_class": int(args.max_pseudo_per_class),
        "n_neighbors": int(args.n_neighbors),
        "min_dist": float(args.min_dist),
        "metric": args.metric,
        "labeled_points": int(len(labeled_points)),
        "pseudo_points": int(len(pseudo_points)),
        "labeled_distribution": labeled_points["true_label"].value_counts().sort_index().to_dict(),
        "pseudo_distribution": pseudo_points["pseudo_label"].value_counts().sort_index().to_dict(),
        "mean_pseudo_confidence": float(pseudo_points["pseudo_label_confidence"].mean()),
        "figure_path": str(figure_path),
        "labeled_csv": str(labeled_csv),
        "pseudo_csv": str(pseudo_csv),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved labeled reference UMAP points to {labeled_csv}")
    print(f"Saved pseudo-label UMAP points to {pseudo_csv}")
    print(f"Saved UMAP figure to {figure_path}")
    print(f"Saved UMAP summary to {summary_path}")


if __name__ == "__main__":
    main()
