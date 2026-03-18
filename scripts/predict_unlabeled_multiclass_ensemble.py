"""Predict unlabeled WM-811K rows with an averaged or stacked multiclass ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from wafer_defect.config import load_toml
from wafer_defect.data.supervised import RawWaferInferenceDataset, prepare_supervised_dataframe
from wafer_defect.ensemble import StackingCombiner, average_probabilities, collect_model_outputs, load_classifier_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/data_multiclass_50k.toml")
    parser.add_argument("--raw-pickle", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--checkpoints", nargs="+", default=None)
    parser.add_argument("--combiner-json", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-confidence", type=float, default=0.98)
    return parser.parse_args()


def load_manifest(manifest_path: str | Path) -> dict[str, object]:
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def resolve_checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    if args.checkpoints:
        return [Path(path) for path in args.checkpoints]

    if args.manifest is None:
        raise ValueError("Provide either --checkpoints or --manifest")

    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)
    manifest_root = manifest_path.parent
    checkpoints = manifest.get("checkpoints", [])
    if not checkpoints:
        raise ValueError(f"No checkpoints listed in manifest: {manifest_path}")
    return [(manifest_root / Path(path)).resolve() for path in checkpoints]


def resolve_combiner_path(args: argparse.Namespace) -> Path | None:
    if args.combiner_json:
        return Path(args.combiner_json)

    if args.manifest is None:
        return None

    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)
    combiner_json = manifest.get("combiner_json")
    if not combiner_json:
        return None
    return (manifest_path.parent / Path(str(combiner_json))).resolve()


def resolve_raw_pickle_path(raw_pickle: str | Path) -> Path:
    raw_pickle_path = Path(raw_pickle)
    if raw_pickle_path.exists():
        return raw_pickle_path

    hint = ""
    if "/kaggle/input/" in raw_pickle_path.as_posix():
        hint = " Mount the Kaggle dataset containing LSWMD.pkl and pass its real path, for example /kaggle/input/<dataset-slug>/LSWMD.pkl."
    raise FileNotFoundError(f"Raw pickle not found: {raw_pickle_path}.{hint}")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_toml(args.config)
    dataset_cfg = config["dataset"]
    raw_pickle = resolve_raw_pickle_path(args.raw_pickle or dataset_cfg["raw_pickle"])

    checkpoint_paths = resolve_checkpoint_paths(args)
    combiner_path = resolve_combiner_path(args)
    output_csv = Path(args.output_csv or "artifacts/multiclass_classifier_50k_ensemble/unlabeled_predictions.csv")
    summary_json = Path(args.summary_json or output_csv.with_suffix(".summary.json"))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, class_names = load_classifier_models(checkpoint_paths, device)

    combiner = None
    if combiner_path is not None:
        combiner = StackingCombiner.load(combiner_path)
        if combiner.class_names != class_names:
            raise ValueError(f"Combiner class-name mismatch for {combiner_path}")

    _, unlabeled_df = prepare_supervised_dataframe(
        raw_pickle=raw_pickle,
        class_names=class_names,
    )
    if args.limit is not None:
        unlabeled_df = unlabeled_df.iloc[: args.limit].reset_index(drop=True)

    dataset = RawWaferInferenceDataset(unlabeled_df, image_size=int(dataset_cfg["image_size"]))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

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
                    "accepted_for_pseudo_label": confidence >= args.min_confidence,
                }
            )

    predictions = pd.DataFrame(rows)
    predictions.to_csv(output_csv, index=False)
    accepted_count = int(predictions["accepted_for_pseudo_label"].sum())

    summary = {
        "combiner": "stacking" if combiner is not None else "average",
        "ensemble_size": len(checkpoint_paths),
        "checkpoints": [str(path) for path in checkpoint_paths],
        "combiner_json": str(combiner_path) if combiner_path is not None else None,
        "raw_pickle": str(raw_pickle),
        "num_rows": len(predictions),
        "accepted_pseudo_labels": accepted_count,
        "min_confidence": args.min_confidence,
        "class_distribution": predictions["predicted_label"].value_counts().sort_index().to_dict(),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved {len(rows)} ensemble predictions to {output_csv}")
    print(f"Saved summary to {summary_json}")
    print(f"Combiner: {summary['combiner']}")
    print(f"Accepted {accepted_count} rows with confidence >= {args.min_confidence:.2f}")


if __name__ == "__main__":
    main()
