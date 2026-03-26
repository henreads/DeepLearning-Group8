"""Train a multiclass wafer-defect classifier from labeled metadata."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from wafer_defect.config import load_toml
from wafer_defect.classification.data import LabeledWaferDataset
from wafer_defect.classification.models import WaferClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training/classifier/train_multiclass_classifier.toml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=0)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_limit_dataset(dataset: LabeledWaferDataset, limit: int | None) -> LabeledWaferDataset:
    if limit is None or limit >= len(dataset):
        return dataset
    dataset.metadata = dataset.metadata.iloc[:limit].reset_index(drop=True)
    return dataset


def build_train_sampler(dataset: LabeledWaferDataset, num_classes: int) -> WeightedRandomSampler:
    class_counts = (
        dataset.metadata["label_index"].value_counts().reindex(range(num_classes), fill_value=0).sort_index()
    )
    nonzero_class_counts = class_counts[class_counts > 0]
    class_weights = pd.Series(0.0, index=class_counts.index, dtype=np.float64)
    class_weights.loc[nonzero_class_counts.index] = 1.0 / nonzero_class_counts
    sample_weights = dataset.metadata["label_index"].map(class_weights).to_numpy(dtype=np.float64)
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_class_weights(dataset: LabeledWaferDataset, num_classes: int) -> torch.Tensor:
    class_counts = (
        dataset.metadata["label_index"].value_counts().reindex(range(num_classes), fill_value=0).sort_index()
    )
    weights = np.ones(num_classes, dtype=np.float32)
    nonzero_mask = class_counts.to_numpy() > 0
    weights[nonzero_mask] = 1.0 / class_counts.to_numpy(dtype=np.float32)[nonzero_mask]
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32)


def select_metric(metrics: dict[str, float], metric_name: str) -> float:
    if metric_name not in metrics:
        raise KeyError(f"Unsupported checkpoint metric '{metric_name}'. Available metrics: {sorted(metrics)}")
    return float(metrics[metric_name])


def mixup_batch(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return inputs, targets, targets, 1.0

    lam = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(inputs.size(0), device=inputs.device)
    mixed_inputs = lam * inputs + (1.0 - lam) * inputs[permutation]
    return mixed_inputs, targets, targets[permutation], lam


def mixup_loss(
    criterion: nn.Module,
    logits: torch.Tensor,
    targets_a: torch.Tensor,
    targets_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    if lam >= 1.0:
        return criterion(logits, targets_a)
    return lam * criterion(logits, targets_a) + (1.0 - lam) * criterion(logits, targets_b)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    mixup_alpha: float = 0.0,
    grad_clip_norm: float | None = None,
    log_interval: int = 0,
    phase_name: str = "train",
    epoch_index: int | None = None,
    total_epochs: int | None = None,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    all_targets: list[int] = []
    all_predictions: list[int] = []
    samples_seen = 0
    total_batches = len(loader)

    for batch_index, (inputs, targets) in enumerate(loader, start=1):
        inputs = inputs.to(device)
        targets = targets.to(device)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            inputs, targets_a, targets_b, lam = mixup_batch(inputs, targets, alpha=mixup_alpha)
        else:
            targets_a = targets
            targets_b = targets
            lam = 1.0

        logits = model(inputs)
        loss = mixup_loss(criterion, logits, targets_a, targets_b, lam)

        if optimizer is not None:
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

        total_loss += float(loss.item()) * len(inputs)
        predictions = logits.argmax(dim=1)
        all_targets.extend(targets.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())
        samples_seen += len(inputs)

        if log_interval > 0 and is_training and (batch_index % log_interval == 0 or batch_index == total_batches):
            running_accuracy = accuracy_score(all_targets, all_predictions)
            epoch_label = "?"
            if epoch_index is not None and total_epochs is not None:
                epoch_label = f"{epoch_index:03d}/{total_epochs:03d}"
            print(
                f"[{phase_name}] epoch {epoch_label} | "
                f"batch {batch_index:04d}/{total_batches:04d} | "
                f"samples {samples_seen:06d}/{len(loader.dataset):06d} | "
                f"loss={total_loss / max(1, samples_seen):.4f} | "
                f"acc={running_accuracy:.4f}"
            )

    return {
        "loss": total_loss / max(1, len(loader.dataset)),
        "accuracy": accuracy_score(all_targets, all_predictions),
        "balanced_accuracy": balanced_accuracy_score(all_targets, all_predictions),
    }


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    return run_epoch(model, loader, criterion, device, optimizer=None)


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    model.eval()
    rows: list[dict[str, object]] = []
    all_targets: list[int] = []
    all_predictions: list[int] = []

    offset = 0
    for inputs, targets in loader:
        batch_size = len(inputs)
        logits = model(inputs.to(device))
        probabilities = torch.softmax(logits, dim=1)
        confidences, predictions = probabilities.max(dim=1)

        for batch_index in range(batch_size):
            target = int(targets[batch_index].item())
            prediction = int(predictions[batch_index].item())
            rows.append(
                {
                    "dataset_index": offset + batch_index,
                    "target_index": target,
                    "target_label": class_names[target],
                    "predicted_index": prediction,
                    "predicted_label": class_names[prediction],
                    "confidence": float(confidences[batch_index].item()),
                }
            )
            all_targets.append(target)
            all_predictions.append(prediction)
        offset += batch_size

    report = classification_report(
        all_targets,
        all_predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "accuracy": accuracy_score(all_targets, all_predictions),
        "balanced_accuracy": balanced_accuracy_score(all_targets, all_predictions),
        "classification_report": report,
    }
    return pd.DataFrame(rows), metrics


def main() -> None:
    args = parse_args()
    config = load_toml(args.config)
    training_cfg = config["training"]
    model_cfg = config["model"]

    seed = int(training_cfg["random_seed"])
    seed_everything(seed)

    metadata_csv = Path(training_cfg["metadata_csv"])
    output_dir = Path(training_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(metadata_csv)
    class_names = (
        metadata[["label_index", "label_name"]]
        .drop_duplicates()
        .sort_values("label_index")["label_name"]
        .tolist()
    )

    train_dataset = maybe_limit_dataset(LabeledWaferDataset(metadata_csv, split="train"), args.limit_train)
    val_dataset = maybe_limit_dataset(LabeledWaferDataset(metadata_csv, split="val"), args.limit_val)
    test_dataset = maybe_limit_dataset(LabeledWaferDataset(metadata_csv, split="test"), args.limit_test)

    batch_size = int(args.batch_size or training_cfg["batch_size"])
    num_workers = int(training_cfg["num_workers"])
    if os.name == "nt" and num_workers > 0:
        print("Windows environment detected; forcing num_workers=0 for DataLoader stability.")
        num_workers = 0
    use_weighted_sampler = bool(training_cfg.get("use_weighted_sampler", True))
    use_class_weights = bool(training_cfg.get("use_class_weights", False))
    checkpoint_metric = str(training_cfg.get("checkpoint_metric", "balanced_accuracy"))
    label_smoothing = float(training_cfg.get("label_smoothing", 0.0))
    mixup_alpha = float(training_cfg.get("mixup_alpha", 0.0))
    grad_clip_norm = float(training_cfg.get("grad_clip_norm", 0.0))
    log_interval = int(args.log_interval)

    train_sampler = build_train_sampler(train_dataset, len(class_names)) if use_weighted_sampler else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
    )
    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training config: {args.config}")
    print(f"Metadata CSV: {metadata_csv}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {device}")
    print(
        "Dataset sizes | "
        f"train={len(train_dataset)} | val={len(val_dataset)} | test={len(test_dataset)} | "
        f"classes={len(class_names)}"
    )
    print(
        "Run settings | "
        f"seed={seed} | epochs={int(args.epochs or training_cfg['epochs'])} | "
        f"batch_size={batch_size} | weighted_sampler={use_weighted_sampler} | "
        f"class_weights={use_class_weights} | checkpoint_metric={checkpoint_metric}"
    )
    print("Train class distribution:")
    print(
        train_dataset.metadata["label_name"]
        .value_counts()
        .sort_index()
        .to_string()
    )
    model = WaferClassifier(
        num_classes=len(class_names),
        base_channels=int(model_cfg["base_channels"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        dropout=float(model_cfg["dropout"]),
        variant=str(model_cfg.get("variant", "baseline")),
        block_dropout=float(model_cfg.get("block_dropout", 0.0)),
        se_reduction=int(model_cfg.get("se_reduction", 8)),
    ).to(device)

    class_weights = build_class_weights(train_dataset, len(class_names)).to(device) if use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(training_cfg.get("lr_decay_factor", 0.5)),
        patience=int(training_cfg.get("lr_patience", 5)),
    )

    epochs = int(args.epochs or training_cfg["epochs"])
    early_stopping_patience = int(training_cfg.get("early_stopping_patience", 12))
    history_rows: list[dict[str, float | int]] = []
    best_checkpoint_metric = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_checkpoint_path = output_dir / "best_model.pt"

    for epoch in range(1, epochs + 1):
        print(f"Starting epoch {epoch:03d}/{epochs:03d}")
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            mixup_alpha=mixup_alpha,
            grad_clip_norm=grad_clip_norm if grad_clip_norm > 0 else None,
            log_interval=log_interval,
            phase_name="train",
            epoch_index=epoch,
            total_epochs=epochs,
        )
        train_eval_metrics = evaluate_loader(model, train_eval_loader, criterion, device)
        val_metrics = run_epoch(model, val_loader, criterion, device, optimizer=None)
        current_checkpoint_metric = select_metric(val_metrics, checkpoint_metric)
        scheduler.step(current_checkpoint_metric)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "train_eval_loss": train_eval_metrics["loss"],
            "train_eval_accuracy": train_eval_metrics["accuracy"],
            "train_eval_balanced_accuracy": train_eval_metrics["balanced_accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history_rows.append(row)

        if current_checkpoint_metric > best_checkpoint_metric:
            best_checkpoint_metric = current_checkpoint_metric
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "model_config": {
                        "base_channels": int(model_cfg["base_channels"]),
                        "hidden_dim": int(model_cfg["hidden_dim"]),
                        "dropout": float(model_cfg["dropout"]),
                        "variant": str(model_cfg.get("variant", "baseline")),
                        "block_dropout": float(model_cfg.get("block_dropout", 0.0)),
                        "se_reduction": int(model_cfg.get("se_reduction", 8)),
                    },
                    "metadata_csv": str(metadata_csv),
                    "best_epoch": best_epoch,
                    "best_checkpoint_metric": best_checkpoint_metric,
                    "checkpoint_metric_name": checkpoint_metric,
                },
                best_checkpoint_path,
            )
            print(
                f"New best checkpoint at epoch {epoch:03d} | "
                f"{checkpoint_metric}={best_checkpoint_metric:.4f} | "
                f"saved to {best_checkpoint_path}"
            )
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:03d} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"train_eval_acc={train_eval_metrics['accuracy']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping triggered after {epoch} epochs with no {checkpoint_metric} improvement "
                f"for {early_stopping_patience} consecutive epochs."
            )
            break

    history = pd.DataFrame(history_rows)
    history.to_csv(output_dir / "history.csv", index=False)

    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    train_predictions, train_report = collect_predictions(model, train_eval_loader, device, class_names)
    val_predictions, val_report = collect_predictions(model, val_loader, device, class_names)
    test_predictions, test_report = collect_predictions(model, test_loader, device, class_names)

    train_predictions.to_csv(output_dir / "train_predictions.csv", index=False)
    val_predictions.to_csv(output_dir / "val_predictions.csv", index=False)
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    metrics = {
        "best_checkpoint_metric": best_checkpoint_metric,
        "checkpoint_metric_name": checkpoint_metric,
        "best_epoch": best_epoch,
        "train": train_report,
        "val": val_report,
        "test": test_report,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Best checkpoint saved to {best_checkpoint_path}")
    print(f"Checkpoint metric ({checkpoint_metric}): {best_checkpoint_metric:.4f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Train accuracy: {train_report['accuracy']:.4f}")
    print(f"Validation accuracy: {val_report['accuracy']:.4f}")
    print(f"Validation balanced accuracy: {val_report['balanced_accuracy']:.4f}")
    print(f"Test accuracy: {test_report['accuracy']:.4f}")
    print(f"Test balanced accuracy: {test_report['balanced_accuracy']:.4f}")


if __name__ == "__main__":
    main()
