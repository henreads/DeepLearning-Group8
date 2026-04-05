"""Train the RD4AD model on normal wafers only and export review artifacts."""
# scripts/train_rd4ad.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import shutil
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from wafer_defect.config import load_toml
from wafer_defect.data.wm811k import WaferMapDataset
from wafer_defect.evaluation.reconstruction_metrics import (
    summarize_threshold_metrics,
    sweep_threshold_metrics,
)
from wafer_defect.evaluation.umap_reference import export_reference_umap_bundle
from wafer_defect.models.rd4ad import build_rd4ad_from_config
from wafer_defect.training.rd4ad import run_rd4ad_epoch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def safe_torch_load(path: Path, *, map_location: str | torch.device, label: str) -> Any | None:
    try:
        return torch.load(path, map_location=map_location)
    except Exception as exc:
        print(
            f"Warning: failed to load {label} checkpoint at {path}: {exc}. "
            "Ignoring this checkpoint and continuing.",
            flush=True,
        )
        return None


def list_resume_candidates(checkpoints_dir: Path) -> list[Path]:
    latest_checkpoint_path = checkpoints_dir / "latest_checkpoint.pt"
    epoch_checkpoints = sorted(
        checkpoints_dir.glob("checkpoint_epoch_*.pt"),
        key=lambda path: int(path.stem.split("_")[-1]),
        reverse=True,
    )
    candidates: list[Path] = []
    if latest_checkpoint_path.exists():
        candidates.append(latest_checkpoint_path)
    candidates.extend(epoch_checkpoints)
    return candidates


def load_checkpoint_if_available(
    *,
    checkpoints_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
) -> tuple[int, list[dict[str, Any]], int, float, int, dict[str, torch.Tensor] | None]:
    best_checkpoint_path = checkpoints_dir / "best_model.pt"
    resume_candidates = list_resume_candidates(checkpoints_dir)
    if not resume_candidates:
        if not best_checkpoint_path.exists():
            return 0, [], 0, float("inf"), 0, None

        best_checkpoint = safe_torch_load(best_checkpoint_path, map_location="cpu", label="best")
        if best_checkpoint is None:
            return 0, [], 0, float("inf"), 0, None

        best_state_dict = {
            key: value.detach().cpu().clone()
            for key, value in best_checkpoint["model_state_dict"].items()
        }
        model.load_state_dict(best_state_dict)
        print(
            f"Loaded best-model weights from {best_checkpoint_path} without optimizer state; "
            "training will restart from epoch 1.",
            flush=True,
        )
        return 0, [], 0, float("inf"), 0, best_state_dict

    resume_checkpoint_path: Path | None = None
    checkpoint: Any | None = None
    for candidate_path in resume_candidates:
        label = "latest" if candidate_path.name == "latest_checkpoint.pt" else candidate_path.stem
        checkpoint = safe_torch_load(candidate_path, map_location=device, label=label)
        if checkpoint is not None:
            resume_checkpoint_path = candidate_path
            break

    if checkpoint is None or resume_checkpoint_path is None:
        best_checkpoint = safe_torch_load(best_checkpoint_path, map_location="cpu", label="best")
        if best_checkpoint is None:
            return 0, [], 0, float("inf"), 0, None

        best_state_dict = {
            key: value.detach().cpu().clone()
            for key, value in best_checkpoint["model_state_dict"].items()
        }
        model.load_state_dict(best_state_dict)
        print(
            f"All resume checkpoints were unusable; loaded best-model weights from {best_checkpoint_path} "
            "without optimizer state. Training will restart from epoch 1.",
            flush=True,
        )
        return 0, [], 0, float("inf"), 0, best_state_dict

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    history = list(checkpoint.get("history", []))
    best_epoch = int(checkpoint.get("best_epoch", 0))
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    stale_epochs = int(checkpoint.get("stale_epochs", 0))
    start_epoch = int(checkpoint.get("epoch", 0))

    best_state_dict: dict[str, torch.Tensor] | None = None
    if best_checkpoint_path.exists():
        best_checkpoint = safe_torch_load(best_checkpoint_path, map_location="cpu", label="best")
        if best_checkpoint is not None:
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in best_checkpoint["model_state_dict"].items()
            }

    print(
        f"Resuming from checkpoint: {resume_checkpoint_path} "
        f"(next epoch: {start_epoch + 1}, best epoch: {best_epoch}, best val loss: {best_val_loss:.6f})",
        flush=True,
    )
    return start_epoch, history, best_epoch, best_val_loss, stale_epochs, best_state_dict


@torch.no_grad()
def infer_scores(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    desc: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for images, labels in tqdm(dataloader, desc=desc, leave=False):
        images = images.to(device)
        scores = model(images).detach().cpu().numpy()
        all_scores.append(scores)
        all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_scores)


@torch.no_grad()
def infer_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    desc: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_embeddings: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for images, labels in tqdm(dataloader, desc=desc, leave=False):
        images = images.to(device)
        _, _, f3 = model.encode(images)
        pooled = f3.mean(dim=(2, 3)).detach().cpu().numpy().astype(np.float32)
        all_embeddings.append(pooled)
        all_labels.append(labels.numpy())
    return np.concatenate(all_embeddings), np.concatenate(all_labels)


def save_score_distribution(
    val_labels: np.ndarray,
    val_scores: np.ndarray,
    test_labels: np.ndarray,
    test_scores: np.ndarray,
    threshold: float,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, labels, scores, split in [
        (axes[0], val_labels, val_scores, "Val"),
        (axes[1], test_labels, test_scores, "Test"),
    ]:
        ax.hist(scores[labels == 0], bins=80, alpha=0.6, label="Normal", density=True, color="steelblue")
        ax.hist(scores[labels == 1], bins=80, alpha=0.6, label="Defect", density=True, color="tomato")
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1.2, label=f"Threshold ({threshold:.4f})")
        ax.set_title(f"{split} score distribution")
        ax.set_xlabel("Anomaly score")
        ax.set_ylabel("Density")
        ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_threshold_sweep_plot(
    sweep_df: pd.DataFrame,
    threshold: float,
    best_sweep: dict[str, Any],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(sweep_df["threshold"], sweep_df["f1"], label="F1")
    ax.plot(sweep_df["threshold"], sweep_df["precision"], label="Precision", linestyle="--")
    ax.plot(sweep_df["threshold"], sweep_df["recall"], label="Recall", linestyle=":")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.2, label=f"Val threshold ({threshold:.4f})")
    ax.axvline(
        float(best_sweep["threshold"]),
        color="green",
        linestyle=":",
        linewidth=1.2,
        label=f"Oracle best F1 ({float(best_sweep['f1']):.4f})",
    )
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Test threshold sweep")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(test_metrics: dict[str, Any], out_path: Path) -> None:
    cm = np.array(test_metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    ax.set_xticks([0, 1], ["Pred Normal", "Pred Defect"])
    ax.set_yticks([0, 1], ["True Normal", "True Defect"])
    ax.set_title("Test confusion matrix")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_per_defect_recall(
    metadata_csv: str | Path,
    test_labels: np.ndarray,
    test_scores: np.ndarray,
    threshold: float,
    out_csv: Path,
    out_plot: Path,
    overall_recall: float,
) -> None:
    metadata = pd.read_csv(metadata_csv)
    test_meta = metadata[metadata["split"] == "test"].reset_index(drop=True).copy()
    if len(test_meta) != len(test_scores):
        return
    defect_label_col = "failureType" if "failureType" in test_meta.columns else ("defect_type" if "defect_type" in test_meta.columns else None)
    if defect_label_col is None:
        return
    test_meta["score"] = test_scores
    test_meta["predicted"] = (test_scores > threshold).astype(int)
    true_defects = test_meta[np.asarray(test_labels) == 1].copy()
    if true_defects.empty:
        return
    recall_by_defect = (
        true_defects
        .groupby(defect_label_col)
        .apply(lambda g: g["predicted"].mean())
        .rename("recall")
        .reset_index()
        .sort_values("recall", ascending=False)
    )
    recall_by_defect.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(recall_by_defect[defect_label_col].astype(str), recall_by_defect["recall"], color="steelblue")
    ax.axhline(overall_recall, color="red", linestyle="--", label=f"Overall recall ({overall_recall:.3f})")
    ax.set_xlabel("Defect type")
    ax.set_ylabel("Recall")
    ax.set_title("Per-defect recall (test set)")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(out_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to train_config.toml")
    args = parser.parse_args()

    config = load_toml(args.config)
    output_dir = Path(config["run"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    results_dir = output_dir / "results"
    plots_dir = output_dir / "plots"
    umap_dir = results_dir / "umap"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    umap_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(config["run"]["seed"]))
    device = resolve_device(config["training"]["device"])
    image_size = int(config["data"].get("image_size", 224))
    batch_size = int(config["data"]["batch_size"])
    num_workers = int(config["data"]["num_workers"])
    metadata_csv = config["data"]["metadata_csv"]

    train_dataset = WaferMapDataset(metadata_csv, split="train", image_size=image_size)
    val_dataset = WaferMapDataset(metadata_csv, split="val", image_size=image_size)
    test_dataset = WaferMapDataset(metadata_csv, split="test", image_size=image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_rd4ad_from_config(config).to(device)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    use_amp = bool(config["training"].get("use_amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    patience = int(config["training"].get("early_stopping_patience", 0))
    min_delta = float(config["training"].get("early_stopping_min_delta", 0.0))
    checkpoint_every = int(config["training"].get("checkpoint_every", 5))
    epochs = int(config["training"]["epochs"])

    start_epoch, history, best_epoch, best_val_loss, stale_epochs, best_state_dict = load_checkpoint_if_available(
        checkpoints_dir=checkpoints_dir,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
    )

    for epoch in range(start_epoch, epochs):
        train_metrics = run_rd4ad_epoch(
            model, train_loader, device, optimizer=optimizer, scaler=scaler,
            progress_desc=f"Train {epoch + 1}/{epochs}",
        )
        val_metrics = run_rd4ad_epoch(
            model, val_loader, device,
            progress_desc=f"Val {epoch + 1}/{epochs}",
        )

        record = {
            "epoch": epoch + 1,
            "train_loss": train_metrics.loss,
            "train_l1": train_metrics.reconstruction_loss,
            "train_l2": train_metrics.kl_loss,
            "train_l3": train_metrics.distillation_loss,
            "val_loss": val_metrics.loss,
            "val_l1": val_metrics.reconstruction_loss,
            "val_l2": val_metrics.kl_loss,
            "val_l3": val_metrics.distillation_loss,
        }
        history.append(record)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"| train={train_metrics.loss:.6f} "
            f"| val={val_metrics.loss:.6f} "
            f"| best_val={best_val_loss:.6f}",
            flush=True,
        )

        improved = (best_val_loss - val_metrics.loss) > min_delta
        if improved:
            best_val_loss = val_metrics.loss
            best_epoch = epoch + 1
            best_state_dict = clone_state_dict(model)
            stale_epochs = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": best_state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                    "config": config,
                    "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "stale_epochs": stale_epochs,
                "history": history,
                },
                checkpoints_dir / "best_model.pt",
            )
        else:
            stale_epochs += 1

        latest = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "config": config,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "stale_epochs": stale_epochs,
            "history": history,
        }
        torch.save(latest, checkpoints_dir / "latest_checkpoint.pt")

        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            torch.save(latest, checkpoints_dir / f"checkpoint_epoch_{epoch + 1}.pt")

        if patience > 0 and stale_epochs >= patience:
            print(
                f"Early stopping at epoch {epoch + 1}. "
                f"Best epoch: {best_epoch}, best val loss: {best_val_loss:.6f}",
                flush=True,
            )
            break

    if best_state_dict is None:
        best_state_dict = clone_state_dict(model)
        torch.save(
            {
                "epoch": len(history),
                "model_state_dict": best_state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "config": config,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "stale_epochs": stale_epochs,
                "history": history,
            },
            checkpoints_dir / "best_model.pt",
        )

    with (results_dir / "history.json").open("w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)

    training_summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "epochs_ran": len(history),
    }
    with (results_dir / "training_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(training_summary, fh, indent=2)

    # Evaluate with best checkpoint weights so the notebook can load review artifacts directly.
    model.load_state_dict(best_state_dict)
    model.eval()

    val_labels, val_scores = infer_scores(model, val_loader, device, desc="Infer val scores")
    test_labels, test_scores = infer_scores(model, test_loader, device, desc="Infer test scores")

    threshold_quantile = float(config["scoring"].get("threshold_quantile", 0.95))
    val_normal_scores = val_scores[val_labels == 0]
    threshold = float(np.quantile(val_normal_scores, threshold_quantile))

    val_metrics_summary = summarize_threshold_metrics(val_labels, val_scores, threshold)
    test_metrics_summary = summarize_threshold_metrics(test_labels, test_scores, threshold)
    sweep_df, best_sweep = sweep_threshold_metrics(test_labels, test_scores)

    pd.DataFrame({"label": val_labels, "score": val_scores}).to_csv(results_dir / "val_scores.csv", index=False)
    pd.DataFrame({"label": test_labels, "score": test_scores}).to_csv(results_dir / "test_scores.csv", index=False)
    sweep_df.to_csv(results_dir / "threshold_sweep.csv", index=False)

    evaluation_summary = {
        "threshold": threshold,
        "threshold_quantile": threshold_quantile,
        "val": val_metrics_summary,
        "test": test_metrics_summary,
        "best_sweep": best_sweep,
    }
    (results_dir / "summary.json").write_text(json.dumps(evaluation_summary, indent=2), encoding="utf-8")

    save_score_distribution(val_labels, val_scores, test_labels, test_scores, threshold, plots_dir / "score_distribution.png")
    save_threshold_sweep_plot(sweep_df, threshold, best_sweep, plots_dir / "threshold_sweep.png")
    save_confusion_matrix(test_metrics_summary, plots_dir / "confusion_matrix.png")
    save_per_defect_recall(
        metadata_csv,
        test_labels,
        test_scores,
        threshold,
        results_dir / "per_defect_recall.csv",
        plots_dir / "per_defect_recall.png",
        float(test_metrics_summary["recall"]),
    )

    # Save embeddings and UMAP artifacts so the notebook can load them without regeneration.
    train_embeddings, train_labels = infer_embeddings(model, train_eval_loader, device, desc="Infer train embeddings")
    val_embeddings, val_embedding_labels = infer_embeddings(model, val_loader, device, desc="Infer val embeddings")
    test_embeddings, test_embedding_labels = infer_embeddings(model, test_loader, device, desc="Infer test embeddings")
    np.save(umap_dir / "train_normal_embeddings.npy", train_embeddings)
    np.save(umap_dir / "train_normal_labels.npy", train_labels)
    np.save(umap_dir / "val_embeddings.npy", val_embeddings)
    np.save(umap_dir / "val_labels.npy", val_embedding_labels)
    np.save(umap_dir / "test_embeddings.npy", test_embeddings)
    np.save(umap_dir / "test_labels.npy", test_embedding_labels)

    try:
        import umap as umap_module

        export_reference_umap_bundle(
            output_dir=umap_dir,
            umap_module=umap_module,
            train_normal_embeddings=train_embeddings,
            val_embeddings=val_embeddings,
            val_labels=val_embedding_labels,
            test_embeddings=test_embeddings,
            test_labels=test_embedding_labels,
            val_model_scores=val_scores.astype(np.float32),
            test_model_scores=test_scores.astype(np.float32),
            threshold_quantile=threshold_quantile,
            random_state=int(config["run"].get("seed", 42)),
            pca_components=50,
            n_neighbors=15,
            min_dist=0.1,
            knn_k=15,
            metric="euclidean",
            title_prefix="RD4AD WRN50 x224",
            points_filename="umap_points.csv",
            split_plot_filename="umap_by_split.png",
            score_plot_filename="umap_by_score.png",
            summary_filename="umap_summary.json",
            sweep_filename="umap_knn_threshold_sweep.csv",
        )
        for filename in ["umap_by_split.png", "umap_by_score.png"]:
            src = umap_dir / filename
            if src.exists():
                shutil.copy2(src, plots_dir / filename)
    except ImportError:
        print("umap-learn is not installed; skipping UMAP export.", flush=True)

    print(
        f"Training complete | best_epoch={best_epoch} | best_val_loss={best_val_loss:.6f}",
        flush=True,
    )
    print(
        f"Evaluation complete | test_f1={test_metrics_summary['f1']:.4f} "
        f"| test_auroc={test_metrics_summary['auroc']:.4f} "
        f"| test_auprc={test_metrics_summary['auprc']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
