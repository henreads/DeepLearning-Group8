"""Evaluate an ensemble classifier with richer multiclass and binary defect metrics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from wafer_defect.classification.data import LabeledWaferDataset
from wafer_defect.classification.ensemble import (
    SUPPORTED_STACKING_FEATURE_TYPES,
    SUPPORTED_STACKING_SELECTION_METRICS,
    StackingCombiner,
    average_probabilities,
    collect_model_outputs,
    fit_stacking_combiner,
    load_classifier_models,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--combiner", choices=("average", "stacking"), default="average")
    parser.add_argument(
        "--stacking-feature-types",
        nargs="+",
        default=list(SUPPORTED_STACKING_FEATURE_TYPES),
        choices=SUPPORTED_STACKING_FEATURE_TYPES,
    )
    parser.add_argument("--stacking-c-grid", nargs="+", type=float, default=[0.1, 0.3, 1.0, 3.0, 10.0])
    parser.add_argument("--stacking-cv-folds", type=int, default=5)
    parser.add_argument("--stacking-max-iter", type=int, default=2000)
    parser.add_argument("--stacking-random-seed", type=int, default=42)
    parser.add_argument(
        "--stacking-selection-metric",
        default="balanced_accuracy",
        choices=SUPPORTED_STACKING_SELECTION_METRICS,
    )
    parser.add_argument(
        "--stacking-class-weight-options",
        nargs="+",
        default=["none", "balanced"],
        choices=("none", "balanced"),
    )
    return parser.parse_args()


def parse_class_weight_options(raw_values: list[str]) -> list[str | None]:
    options: list[str | None] = []
    for value in raw_values:
        normalized = value.strip().lower()
        if normalized == "none":
            options.append(None)
        elif normalized == "balanced":
            options.append("balanced")
        else:
            raise ValueError(f"Unsupported class-weight option: {value}")
    return options


def sanitize_label_for_column(label: str) -> str:
    return label.replace(" ", "_").replace("/", "_")


def build_loader(dataset: LabeledWaferDataset, batch_size: int, num_workers: int, pin_memory: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


@torch.no_grad()
def collect_probabilities(
    models,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    combiner: StackingCombiner | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    offset = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        if combiner is None:
            combined_probabilities = average_probabilities(models, inputs).detach().cpu().numpy()
        else:
            logits, probabilities = collect_model_outputs(models, inputs)
            source_outputs = probabilities if combiner.feature_type == "probabilities" else logits
            combined_probabilities = combiner.predict_proba(source_outputs)

        confidences = combined_probabilities.max(axis=1)
        predictions = combined_probabilities.argmax(axis=1)
        target_values = targets.numpy()

        for batch_index in range(len(target_values)):
            target = int(target_values[batch_index])
            prediction = int(predictions[batch_index])
            row: dict[str, object] = {
                "dataset_index": offset + batch_index,
                "target_index": target,
                "target_label": class_names[target],
                "predicted_index": prediction,
                "predicted_label": class_names[prediction],
                "confidence": float(confidences[batch_index]),
            }
            for class_index, class_name in enumerate(class_names):
                row[f"prob__{sanitize_label_for_column(class_name)}"] = float(
                    combined_probabilities[batch_index, class_index]
                )
            rows.append(row)
        offset += len(target_values)

    return pd.DataFrame(rows)


@torch.no_grad()
def collect_labeled_model_outputs(
    models,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    probability_batches: list[np.ndarray] = []
    logit_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        logits, probabilities = collect_model_outputs(models, inputs)
        logit_batches.append(logits.detach().cpu().numpy().astype(np.float32, copy=False))
        probability_batches.append(probabilities.detach().cpu().numpy().astype(np.float32, copy=False))
        target_batches.append(targets.numpy().astype(np.int64, copy=False))

    return (
        {
            "logits": np.concatenate(logit_batches, axis=0),
            "probabilities": np.concatenate(probability_batches, axis=0),
        },
        np.concatenate(target_batches, axis=0),
    )


def join_dataset_metadata(predictions: pd.DataFrame, dataset: LabeledWaferDataset) -> pd.DataFrame:
    metadata_columns = [
        "array_path",
        "label_name",
        "label_index",
        "split",
        "raw_index",
        "source_split",
        "original_height",
        "original_width",
    ]
    metadata = dataset.metadata.reset_index(drop=True)[metadata_columns]
    return predictions.join(metadata, rsuffix="_meta")


def classification_report_dict(
    targets: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
) -> dict[str, object]:
    return classification_report(
        targets,
        predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )


def summarize_multiclass_metrics(predictions: pd.DataFrame, class_names: list[str]) -> dict[str, object]:
    targets = predictions["target_index"].to_numpy(dtype=np.int64)
    predicted = predictions["predicted_index"].to_numpy(dtype=np.int64)
    probability_columns = [f"prob__{sanitize_label_for_column(name)}" for name in class_names]
    probabilities = predictions[probability_columns].to_numpy(dtype=np.float64)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        targets,
        predicted,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        targets,
        predicted,
        average="weighted",
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy_score(targets, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predicted)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "classification_report": classification_report_dict(targets, predicted, class_names),
        "confusion_matrix": confusion_matrix(
            targets,
            predicted,
            labels=list(range(len(class_names))),
        ).tolist(),
    }

    try:
        metrics["macro_auroc_ovr"] = float(
            roc_auc_score(
                targets,
                probabilities,
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        metrics["macro_auroc_ovr"] = None

    try:
        one_hot_targets = np.eye(len(class_names), dtype=np.float64)[targets]
        metrics["macro_auprc_ovr"] = float(
            average_precision_score(
                one_hot_targets,
                probabilities,
                average="macro",
            )
        )
    except ValueError:
        metrics["macro_auprc_ovr"] = None

    return metrics


def select_best_f1_threshold(targets: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(targets, scores)
    if thresholds.size == 0:
        predictions = scores >= 0.5
        return {
            "threshold": 0.5,
            "precision": float(precision_score(targets, predictions, zero_division=0)),
            "recall": float(recall_score(targets, predictions, zero_division=0)),
            "f1": float(f1_score(targets, predictions, zero_division=0)),
        }

    f1_values = (2.0 * precision[:-1] * recall[:-1]) / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    best_index = int(np.nanargmax(f1_values))
    return {
        "threshold": float(thresholds[best_index]),
        "precision": float(precision[best_index]),
        "recall": float(recall[best_index]),
        "f1": float(f1_values[best_index]),
    }


def summarize_binary_defect_metrics(
    predictions: pd.DataFrame,
    threshold: float,
    none_class_name: str,
) -> dict[str, object]:
    targets = (predictions["target_label"] != none_class_name).to_numpy(dtype=np.int64)
    scores = predictions[f"prob__{sanitize_label_for_column(none_class_name)}"].rsub(1.0).to_numpy(dtype=np.float64)
    predicted_positive = scores >= threshold

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(targets, predicted_positive)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predicted_positive)),
        "precision": float(precision_score(targets, predicted_positive, zero_division=0)),
        "recall": float(recall_score(targets, predicted_positive, zero_division=0)),
        "f1": float(f1_score(targets, predicted_positive, zero_division=0)),
        "predicted_anomalies": int(predicted_positive.sum()),
        "true_anomalies": int(targets.sum()),
    }

    try:
        metrics["auroc"] = float(roc_auc_score(targets, scores))
    except ValueError:
        metrics["auroc"] = None

    try:
        metrics["auprc"] = float(average_precision_score(targets, scores))
    except ValueError:
        metrics["auprc"] = None

    best_sweep = select_best_f1_threshold(targets, scores)
    metrics["best_sweep_f1"] = float(best_sweep["f1"])
    metrics["best_sweep_threshold"] = float(best_sweep["threshold"])

    per_defect_recall: dict[str, object] = {}
    defect_rows = predictions[predictions["target_label"] != none_class_name]
    for class_name, class_rows in defect_rows.groupby("target_label"):
        class_scores = class_rows[f"prob__{sanitize_label_for_column(none_class_name)}"].rsub(1.0)
        detected = int((class_scores >= threshold).sum())
        support = int(len(class_rows))
        per_defect_recall[str(class_name)] = {
            "support": support,
            "detected": detected,
            "recall": float(detected / support) if support else 0.0,
        }
    metrics["per_defect_recall"] = per_defect_recall

    return metrics


def main() -> None:
    args = parse_args()

    metadata_csv = Path(args.metadata_csv)
    checkpoint_paths = [Path(path) for path in args.checkpoints]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_workers = int(args.num_workers)
    if os.name == "nt" and num_workers > 0:
        print("Windows environment detected; forcing num_workers=0 for DataLoader stability.")
        num_workers = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    models, class_names = load_classifier_models(checkpoint_paths, device)
    none_class_name = class_names[0]

    val_dataset = LabeledWaferDataset(metadata_csv, split="val")
    test_dataset = LabeledWaferDataset(metadata_csv, split="test")
    val_loader = build_loader(val_dataset, args.batch_size, num_workers, pin_memory)
    test_loader = build_loader(test_dataset, args.batch_size, num_workers, pin_memory)

    combiner: StackingCombiner | None = None
    stacking_selection: dict[str, object] | None = None
    combiner_path: Path | None = None
    if args.combiner == "stacking":
        val_outputs, val_targets = collect_labeled_model_outputs(models, val_loader, device)
        combiner, stacking_selection = fit_stacking_combiner(
            class_names=class_names,
            checkpoint_paths=checkpoint_paths,
            outputs_by_type=val_outputs,
            targets=val_targets,
            feature_types=list(args.stacking_feature_types),
            c_grid=[float(value) for value in args.stacking_c_grid],
            cv_folds=int(args.stacking_cv_folds),
            max_iter=int(args.stacking_max_iter),
            random_seed=int(args.stacking_random_seed),
            selection_metric=str(args.stacking_selection_metric),
            class_weight_options=parse_class_weight_options(list(args.stacking_class_weight_options)),
        )
        combiner_path = output_dir / "stacking_combiner.json"
        combiner.save(combiner_path)

    val_predictions = join_dataset_metadata(
        collect_probabilities(models, val_loader, device, class_names, combiner=combiner),
        val_dataset,
    )
    test_predictions = join_dataset_metadata(
        collect_probabilities(models, test_loader, device, class_names, combiner=combiner),
        test_dataset,
    )

    val_predictions["defect_probability"] = 1.0 - val_predictions[f"prob__{sanitize_label_for_column(none_class_name)}"]
    test_predictions["defect_probability"] = 1.0 - test_predictions[f"prob__{sanitize_label_for_column(none_class_name)}"]

    threshold_selection = select_best_f1_threshold(
        (val_predictions["target_label"] != none_class_name).to_numpy(dtype=np.int64),
        val_predictions["defect_probability"].to_numpy(dtype=np.float64),
    )

    val_multiclass_metrics = summarize_multiclass_metrics(val_predictions, class_names)
    test_multiclass_metrics = summarize_multiclass_metrics(test_predictions, class_names)
    val_binary_metrics = summarize_binary_defect_metrics(
        val_predictions,
        threshold=threshold_selection["threshold"],
        none_class_name=none_class_name,
    )
    test_binary_metrics = summarize_binary_defect_metrics(
        test_predictions,
        threshold=threshold_selection["threshold"],
        none_class_name=none_class_name,
    )

    val_predictions["predicted_anomaly_at_val_threshold"] = val_predictions["defect_probability"] >= threshold_selection["threshold"]
    test_predictions["predicted_anomaly_at_val_threshold"] = test_predictions["defect_probability"] >= threshold_selection["threshold"]

    val_predictions.to_csv(output_dir / "val_predictions.csv", index=False)
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    metrics: dict[str, object] = {
        "combiner": args.combiner,
        "ensemble_size": len(checkpoint_paths),
        "checkpoints": [str(path) for path in checkpoint_paths],
        "class_names": class_names,
        "none_class_name": none_class_name,
        "val_threshold_selection": {
            "metric": "binary_defect_f1",
            **threshold_selection,
        },
        "splits": {
            "val": {
                "multiclass": val_multiclass_metrics,
                "binary_defect_vs_none": val_binary_metrics,
            },
            "test": {
                "multiclass": test_multiclass_metrics,
                "binary_defect_vs_none": test_binary_metrics,
            },
        },
    }
    if combiner is not None and stacking_selection is not None and combiner_path is not None:
        metrics["stacking"] = {
            "combiner_json": str(combiner_path),
            "feature_type": combiner.feature_type,
            "best_C": combiner.C,
            "class_weight": combiner.class_weight or "none",
            "selection": stacking_selection,
        }

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved validation predictions to {output_dir / 'val_predictions.csv'}")
    print(f"Saved test predictions to {output_dir / 'test_predictions.csv'}")
    print(f"Validation macro F1: {val_multiclass_metrics['macro_f1']:.4f}")
    print(f"Test accuracy: {test_multiclass_metrics['accuracy']:.4f}")
    print(f"Test macro F1: {test_multiclass_metrics['macro_f1']:.4f}")
    print(f"Test weighted F1: {test_multiclass_metrics['weighted_f1']:.4f}")
    print(f"Test binary defect F1 at validation threshold: {test_binary_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
