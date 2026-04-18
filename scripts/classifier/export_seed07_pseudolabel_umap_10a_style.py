"""Project pseudo labels with the 10A-style PCA->UMAP recipe using classifier embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
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
        default="artifacts/multiclass_classifier_all_80_10_10_seed07_pseudolabels/umap_10a_style",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-labeled-normal", type=int, default=4000)
    parser.add_argument("--max-labeled-defect", type=int, default=4000)
    parser.add_argument("--max-pseudo", type=int, default=8000)
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.10)
    parser.add_argument("--metric", default="euclidean")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-pca-dim", type=int, default=50)
    return parser.parse_args()


def load_umap_module():
    try:
        import umap.umap_ as umap
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


def normalize_prediction_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    renamed = predictions.copy()
    if "pseudo_label" not in renamed.columns and "predicted_label" in renamed.columns:
        renamed = renamed.rename(columns={"predicted_label": "pseudo_label"})
    if "pseudo_label_confidence" not in renamed.columns and "confidence" in renamed.columns:
        renamed = renamed.rename(columns={"confidence": "pseudo_label_confidence"})
    return renamed


def sample_rows(dataframe: pd.DataFrame, max_rows: int | None, random_seed: int) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(dataframe) <= max_rows:
        return dataframe.copy().reset_index(drop=True)
    return dataframe.sample(n=max_rows, random_state=random_seed).reset_index(drop=True)


def extract_embedding(model: WaferClassifier, inputs: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "extract_embedding"):
        return model.extract_embedding(inputs)

    x = model.stem(inputs)
    x = model.features(x)
    if getattr(model, "variant", "baseline") == "baseline":
        x = model.classifier[0](x)
        x = model.classifier[1](x)
        x = model.classifier[2](x)
        return model.classifier[3](x)

    avg_features = model.classifier.avgpool(x)
    max_features = model.classifier.maxpool(x)
    x = torch.cat([avg_features, max_features], dim=1)
    x = model.classifier.classifier[0](x)
    x = model.classifier.classifier[1](x)
    return model.classifier.classifier[2](x)


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
        embeddings = extract_embedding(model, inputs.to(device))
        embedding_batches.append(embeddings.cpu().numpy())

    if not embedding_batches:
        raise ValueError("Could not extract embeddings because the dataframe is empty.")
    return np.concatenate(embedding_batches, axis=0)


def save_split_plot(umap_df: pd.DataFrame, figure_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    style_map = {
        "labeled_normal": dict(s=10, alpha=0.30, label="labeled_normal", color="#4d908e"),
        "labeled_defect": dict(s=12, alpha=0.50, label="labeled_defect", color="#f3722c"),
        "pseudo_unlabeled": dict(s=14, alpha=0.65, label="pseudo_unlabeled", color="#577590"),
    }

    for split_name, group in umap_df.groupby("split_label"):
        ax.scatter(group["umap_1"], group["umap_2"], **style_map[split_name])

    ax.set_title("10A-style UMAP of Seed07 Classifier Embeddings")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_score_plot(umap_df: pd.DataFrame, figure_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    score_mask = umap_df["split_label"] == "pseudo_unlabeled"

    ax.scatter(
        umap_df.loc[~score_mask, "umap_1"],
        umap_df.loc[~score_mask, "umap_2"],
        s=8,
        alpha=0.12,
        color="#9ca3af",
        label="labeled reference",
    )
    sc = ax.scatter(
        umap_df.loc[score_mask, "umap_1"],
        umap_df.loc[score_mask, "umap_2"],
        c=umap_df.loc[score_mask, "score"],
        cmap="viridis",
        s=18,
        alpha=0.85,
        label="pseudo_unlabeled",
    )
    fig.colorbar(sc, ax=ax, label="pseudo-label confidence")
    ax.set_title("10A-style UMAP Colored by Pseudo-Label Confidence")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_pseudo_label_plot(umap_df: pd.DataFrame, figure_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    score_mask = umap_df["split_label"] == "pseudo_unlabeled"

    ax.scatter(
        umap_df.loc[~score_mask, "umap_1"],
        umap_df.loc[~score_mask, "umap_2"],
        s=8,
        alpha=0.10,
        color="#d1d5db",
        label="labeled reference",
    )

    palette = plt.get_cmap("tab10")
    for idx, (pseudo_label, group) in enumerate(umap_df.loc[score_mask].groupby("pseudo_label", sort=True)):
        ax.scatter(
            group["umap_1"],
            group["umap_2"],
            s=18,
            alpha=0.70,
            label=str(pseudo_label),
            color=palette(idx % palette.N),
        )

    ax.set_title("10A-style UMAP with Pseudo Labels Highlighted")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
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

    labeled_df = labeled_df.rename(columns={"failure_type": "true_label"})
    labeled_normal = sample_rows(
        labeled_df[labeled_df["true_label"] == "none"],
        max_rows=args.max_labeled_normal,
        random_seed=args.random_seed,
    )
    labeled_defect = sample_rows(
        labeled_df[labeled_df["true_label"] != "none"],
        max_rows=args.max_labeled_defect,
        random_seed=args.random_seed,
    )
    pseudo_prediction_sample = sample_rows(
        pseudo_predictions,
        max_rows=args.max_pseudo,
        random_seed=args.random_seed,
    )
    pseudo_sample = unlabeled_df[["raw_index", "waferMap"]].merge(
        pseudo_prediction_sample,
        on="raw_index",
        how="inner",
        validate="one_to_one",
    )

    image_size = int(dataset_cfg["image_size"])
    normal_embeddings = collect_embeddings(
        model=model,
        dataframe=labeled_normal[["raw_index", "waferMap"]].copy(),
        image_size=image_size,
        batch_size=args.batch_size,
        device=device,
    )
    defect_embeddings = collect_embeddings(
        model=model,
        dataframe=labeled_defect[["raw_index", "waferMap"]].copy(),
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

    X = np.concatenate([normal_embeddings, defect_embeddings, pseudo_embeddings], axis=0)
    split_labels = (
        ["labeled_normal"] * len(normal_embeddings)
        + ["labeled_defect"] * len(defect_embeddings)
        + ["pseudo_unlabeled"] * len(pseudo_embeddings)
    )
    scores = np.concatenate(
        [
            np.full(len(normal_embeddings), np.nan, dtype=np.float32),
            np.full(len(defect_embeddings), np.nan, dtype=np.float32),
            pseudo_sample["pseudo_label_confidence"].to_numpy(dtype=np.float32),
        ],
        axis=0,
    )

    pca_dim = min(args.max_pca_dim, X.shape[0], X.shape[1])
    if pca_dim < 2:
        raise ValueError(f"Not enough points/features for UMAP: X.shape={X.shape}")
    X_reduced = PCA(n_components=pca_dim, random_state=args.random_seed).fit_transform(X)

    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_components=2,
        metric=args.metric,
        random_state=args.random_seed,
    )
    X_umap = reducer.fit_transform(X_reduced)

    umap_df = pd.DataFrame(
        {
            "umap_1": X_umap[:, 0],
            "umap_2": X_umap[:, 1],
            "split_label": split_labels,
            "score": scores,
        }
    )

    labeled_normal_meta = labeled_normal[["raw_index", "true_label"]].copy()
    labeled_normal_meta["pseudo_label"] = pd.NA
    labeled_normal_meta["pseudo_label_confidence"] = np.nan

    labeled_defect_meta = labeled_defect[["raw_index", "true_label"]].copy()
    labeled_defect_meta["pseudo_label"] = pd.NA
    labeled_defect_meta["pseudo_label_confidence"] = np.nan

    pseudo_meta = pseudo_sample[
        [
            "raw_index",
            "pseudo_label",
            "pseudo_label_confidence",
            "second_choice_label",
            "second_choice_confidence",
            "accepted_for_pseudo_label",
        ]
    ].copy()
    pseudo_meta["true_label"] = pd.NA

    meta_df = pd.concat([labeled_normal_meta, labeled_defect_meta, pseudo_meta], ignore_index=True)
    umap_df = pd.concat([umap_df, meta_df], axis=1)

    points_csv_path = output_dir / "embedding_umap_points_10a_style.csv"
    split_plot_path = output_dir / "umap_by_split_10a_style.png"
    score_plot_path = output_dir / "umap_by_score_10a_style.png"
    pseudo_label_plot_path = output_dir / "umap_by_pseudo_label_10a_style.png"
    summary_json_path = output_dir / "umap_10a_style.summary.json"

    umap_df.to_csv(points_csv_path, index=False)
    save_split_plot(umap_df=umap_df, figure_path=split_plot_path)
    save_score_plot(umap_df=umap_df, figure_path=score_plot_path)
    save_pseudo_label_plot(umap_df=umap_df, figure_path=pseudo_label_plot_path)

    summary = {
        "checkpoint": str(checkpoint_path),
        "pseudo_label_csv": str(pseudo_label_csv),
        "output_dir": str(output_dir),
        "device": str(device),
        "class_names": class_names,
        "min_confidence": float(args.min_confidence),
        "max_labeled_normal": int(args.max_labeled_normal),
        "max_labeled_defect": int(args.max_labeled_defect),
        "max_pseudo": int(args.max_pseudo),
        "n_neighbors": int(args.n_neighbors),
        "min_dist": float(args.min_dist),
        "metric": args.metric,
        "random_seed": int(args.random_seed),
        "pca_dim": int(pca_dim),
        "labeled_normal_points": int(len(normal_embeddings)),
        "labeled_defect_points": int(len(defect_embeddings)),
        "pseudo_points": int(len(pseudo_embeddings)),
        "pseudo_distribution": pseudo_sample["pseudo_label"].value_counts().sort_index().to_dict(),
        "points_csv_path": str(points_csv_path),
        "split_plot_path": str(split_plot_path),
        "score_plot_path": str(score_plot_path),
        "pseudo_label_plot_path": str(pseudo_label_plot_path),
    }
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved UMAP points to {points_csv_path}")
    print(f"Saved split plot to {split_plot_path}")
    print(f"Saved score plot to {score_plot_path}")
    print(f"Saved pseudo-label plot to {pseudo_label_plot_path}")
    print(f"Saved UMAP summary to {summary_json_path}")


if __name__ == "__main__":
    main()
