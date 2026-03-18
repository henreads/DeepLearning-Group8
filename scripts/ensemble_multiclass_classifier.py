"""Evaluate and use an averaged or stacked multiclass classifier ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from torch.utils.data import DataLoader

from wafer_defect.config import load_toml
from wafer_defect.data.supervised import LabeledWaferDataset, RawWaferInferenceDataset, prepare_supervised_dataframe
from wafer_defect.ensemble import (
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
    parser.add_argument("--data-config", default="configs/data/data_multiclass_50k.toml")
    parser.add_argument("--metadata-csv", default="data/processed/x64/wm811k_multiclass_50k/metadata_labeled_50k.csv")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output-dir", default="artifacts/multiclass_classifier_50k_ensemble")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--min-confidence", type=float, default=0.98)
    parser.add_argument("--limit-unlabeled", type=int, default=None)
    parser.add_argument("--skip-unlabeled", action="store_true")
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


@torch.no_grad()
def collect_labeled_predictions(
    models,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    combiner: StackingCombiner | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    all_targets: list[int] = []
    all_predictions: list[int] = []

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

        batch_targets = targets.numpy()
        for batch_index in range(len(batch_targets)):
            target = int(batch_targets[batch_index])
            prediction = int(predictions[batch_index])
            rows.append(
                {
                    "dataset_index": offset + batch_index,
                    "target_index": target,
                    "target_label": class_names[target],
                    "predicted_index": prediction,
                    "predicted_label": class_names[prediction],
                    "confidence": float(confidences[batch_index]),
                }
            )
            all_targets.append(target)
            all_predictions.append(prediction)
        offset += len(batch_targets)

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


@torch.no_grad()
def collect_unlabeled_predictions(
    models,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    min_confidence: float,
    combiner: StackingCombiner | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for inputs, raw_indices in loader:
        inputs = inputs.to(device)
        if combiner is None:
            combined_probabilities = average_probabilities(models, inputs).detach().cpu().numpy()
        else:
            logits, probabilities = collect_model_outputs(models, inputs)
            source_outputs = probabilities if combiner.feature_type == "probabilities" else logits
            combined_probabilities = combiner.predict_proba(source_outputs)

        confidences = combined_probabilities.max(axis=1)
        predictions = combined_probabilities.argmax(axis=1)
        top2_indices = np.argsort(combined_probabilities, axis=1)[:, -2:][:, ::-1]

        raw_index_values = raw_indices.numpy()
        for batch_index in range(len(raw_index_values)):
            predicted_index = int(predictions[batch_index])
            second_index = int(top2_indices[batch_index, 1])
            confidence = float(confidences[batch_index])
            rows.append(
                {
                    "raw_index": int(raw_index_values[batch_index]),
                    "predicted_index": predicted_index,
                    "predicted_label": class_names[predicted_index],
                    "confidence": confidence,
                    "second_choice_label": class_names[second_index],
                    "second_choice_confidence": float(combined_probabilities[batch_index, second_index]),
                    "accepted_for_pseudo_label": confidence >= min_confidence,
                }
            )
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


def main() -> None:
    args = parse_args()
    data_config = load_toml(args.data_config)
    dataset_cfg = data_config["dataset"]

    metadata_csv = Path(args.metadata_csv)
    checkpoint_paths = [Path(path) for path in args.checkpoints]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, class_names = load_classifier_models(checkpoint_paths, device)
    pin_memory = device.type == "cuda"

    test_dataset = LabeledWaferDataset(metadata_csv, split="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    average_test_predictions, average_test_metrics = collect_labeled_predictions(
        models,
        test_loader,
        device,
        class_names,
    )

    combiner: StackingCombiner | None = None
    stacking_selection: dict[str, object] | None = None
    combiner_path: Path | None = None
    if args.combiner == "stacking":
        val_dataset = LabeledWaferDataset(metadata_csv, split="val")
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
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

    final_test_predictions = average_test_predictions
    final_test_metrics = average_test_metrics
    if combiner is not None:
        final_test_predictions, final_test_metrics = collect_labeled_predictions(
            models,
            test_loader,
            device,
            class_names,
            combiner=combiner,
        )

    final_test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    average_test_predictions.to_csv(output_dir / "test_predictions_average.csv", index=False)
    if combiner is not None:
        final_test_predictions.to_csv(output_dir / "test_predictions_stacking.csv", index=False)

    accepted_count = None
    if not args.skip_unlabeled:
        _, unlabeled_df = prepare_supervised_dataframe(
            raw_pickle=dataset_cfg["raw_pickle"],
            class_names=class_names,
        )
        if args.limit_unlabeled is not None:
            unlabeled_df = unlabeled_df.iloc[: args.limit_unlabeled].reset_index(drop=True)

        unlabeled_dataset = RawWaferInferenceDataset(unlabeled_df, image_size=int(dataset_cfg["image_size"]))
        unlabeled_loader = DataLoader(
            unlabeled_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        unlabeled_predictions = collect_unlabeled_predictions(
            models,
            unlabeled_loader,
            device,
            class_names,
            min_confidence=args.min_confidence,
            combiner=combiner,
        )
        unlabeled_predictions.to_csv(output_dir / "unlabeled_predictions.csv", index=False)
        accepted_count = int(unlabeled_predictions["accepted_for_pseudo_label"].sum())

    metrics = {
        "combiner": args.combiner,
        "ensemble_size": len(checkpoint_paths),
        "checkpoints": [str(path) for path in checkpoint_paths],
        "min_confidence": args.min_confidence,
        "skip_unlabeled": args.skip_unlabeled,
        "test": final_test_metrics,
        "average_test": average_test_metrics,
    }
    if accepted_count is not None:
        metrics["accepted_pseudo_labels"] = accepted_count
    if combiner is not None and stacking_selection is not None and combiner_path is not None:
        metrics["stacking"] = {
            "combiner_json": str(combiner_path),
            "feature_type": combiner.feature_type,
            "best_C": combiner.C,
            "class_weight": combiner.class_weight or "none",
            "selection": stacking_selection,
        }

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved ensemble test predictions to {output_dir / 'test_predictions.csv'}")
    print(f"Combiner: {args.combiner}")
    print(f"Average-ensemble test accuracy: {average_test_metrics['accuracy']:.4f}")
    print(f"Average-ensemble test balanced accuracy: {average_test_metrics['balanced_accuracy']:.4f}")
    if combiner is not None:
        print(f"Stacked-ensemble test accuracy: {final_test_metrics['accuracy']:.4f}")
        print(f"Stacked-ensemble test balanced accuracy: {final_test_metrics['balanced_accuracy']:.4f}")
        print(f"Saved stacking combiner to {combiner_path}")
    if accepted_count is not None:
        print(f"Saved ensemble unlabeled predictions to {output_dir / 'unlabeled_predictions.csv'}")
        print(f"Accepted {accepted_count} unlabeled rows with confidence >= {args.min_confidence:.2f}")
    else:
        print("Skipped unlabeled inference")


if __name__ == "__main__":
    main()
