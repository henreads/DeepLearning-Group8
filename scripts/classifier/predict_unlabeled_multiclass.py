"""Predict labels for unlabeled WM-811K rows using a trained classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from wafer_defect.config import load_toml
from wafer_defect.classification.data import DEFAULT_CLASS_NAMES, RawWaferInferenceDataset, prepare_supervised_dataframe
from wafer_defect.classification.models import WaferClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/classifier/data_multiclass.toml")
    parser.add_argument("--checkpoint", default="artifacts/multiclass_classifier/best_model.pt")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-confidence", type=float, default=0.95)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_toml(args.config)
    dataset_cfg = config["dataset"]

    checkpoint_path = Path(args.checkpoint)
    output_csv = Path(args.output_csv or dataset_cfg["unlabeled_predictions_csv"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    _, unlabeled_df = prepare_supervised_dataframe(
        raw_pickle=dataset_cfg["raw_pickle"],
        class_names=class_names,
    )
    if args.limit is not None:
        unlabeled_df = unlabeled_df.iloc[: args.limit].reset_index(drop=True)

    dataset = RawWaferInferenceDataset(unlabeled_df, image_size=int(dataset_cfg["image_size"]))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    rows: list[dict[str, object]] = []
    for inputs, raw_indices in loader:
        logits = model(inputs.to(device))
        probabilities = torch.softmax(logits, dim=1)
        confidences, predictions = probabilities.max(dim=1)
        top2_confidences, top2_indices = probabilities.topk(k=2, dim=1)

        for batch_index in range(len(inputs)):
            predicted_index = int(predictions[batch_index].item())
            second_index = int(top2_indices[batch_index, 1].item())
            rows.append(
                {
                    "raw_index": int(raw_indices[batch_index].item()),
                    "predicted_index": predicted_index,
                    "predicted_label": class_names[predicted_index],
                    "confidence": float(confidences[batch_index].item()),
                    "second_choice_label": class_names[second_index],
                    "second_choice_confidence": float(top2_confidences[batch_index, 1].item()),
                    "accepted_for_pseudo_label": float(confidences[batch_index].item()) >= args.min_confidence,
                }
            )

    predictions = pd.DataFrame(rows)
    predictions.to_csv(output_csv, index=False)
    accepted_count = int(predictions["accepted_for_pseudo_label"].sum())
    print(f"Saved {len(rows)} unlabeled predictions to {output_csv}")
    print(
        f"Accepted {accepted_count} rows with confidence >= {args.min_confidence:.2f} "
        f"for safer pseudo-labeling."
    )


if __name__ == "__main__":
    main()
