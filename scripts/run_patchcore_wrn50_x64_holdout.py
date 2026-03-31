from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

SCRIPT_PATH = Path(__file__).resolve()
for candidate in [SCRIPT_PATH.parent, *SCRIPT_PATH.parents]:
    src_root = candidate / "src"
    if (src_root / "wafer_defect").exists():
        if str(src_root) not in sys.path:
            sys.path.insert(0, str(src_root))
        break

from wafer_defect.evaluation.reconstruction_metrics import summarize_threshold_metrics, sweep_threshold_metrics

from holdout_eval_helpers import (
    build_defect_breakdown,
    exec_notebook_code_cells,
    resolve_repo_root,
    save_defect_breakdown_plot,
    save_threshold_sweep_plot,
    to_repo_relative,
    write_confusion_csv,
)


NOTEBOOK_PATH = Path("experiments/anomaly_detection/patchcore/wideresnet50/x64/main/notebook.ipynb")


def load_selected_variant(input_artifact_dir: Path, variant_name: str) -> tuple[str, Path]:
    if variant_name:
        checkpoint_path = input_artifact_dir / variant_name / variant_name / "patchcore_checkpoint.pt"
        if not checkpoint_path.exists():
            checkpoint_path = input_artifact_dir / variant_name / "patchcore_checkpoint.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"PatchCore checkpoint not found for variant {variant_name!r}: {checkpoint_path}")
        return variant_name, checkpoint_path

    best_checkpoint = input_artifact_dir / "best_variant_checkpoint.pt"
    if not best_checkpoint.exists():
        raise FileNotFoundError(f"Best-variant checkpoint not found: {best_checkpoint}")
    checkpoint = torch.load(best_checkpoint, map_location="cpu")
    return str(checkpoint["name"]), best_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--input-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant-name", default="")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    repo_root = resolve_repo_root()
    os.chdir(repo_root)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("WM811K_REPO_ROOT", str(repo_root))

    scope = exec_notebook_code_cells((repo_root / NOTEBOOK_PATH), [3])
    MultiLayerPatchCoreModel = scope["MultiLayerPatchCoreModel"]
    WaferArrayDataset = scope["WaferArrayDataset"]
    collect_scores = scope["collect_scores"]
    resolve_device = scope["resolve_device"]

    input_artifact_dir = (repo_root / args.input_artifact_dir).resolve()
    output_dir = (repo_root / args.output_dir)
    evaluation_dir = output_dir / "evaluation"
    plots_dir = output_dir / "plots"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    selected_variant, checkpoint_path = load_selected_variant(input_artifact_dir, args.variant_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    device = resolve_device(args.device)

    metadata_path = (repo_root / args.metadata_path).resolve()
    val_dataset = WaferArrayDataset(metadata_path, split="val")
    test_dataset = WaferArrayDataset(metadata_path, split="test")
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=args.num_workers)

    model = MultiLayerPatchCoreModel(
        image_size=int(checkpoint["image_size"]),
        teacher_layers=list(checkpoint["teacher_layers"]),
        memory_bank_size=int(checkpoint["memory_bank_size"]),
        reduction=str(checkpoint["reduction"]),
        topk_ratio=float(checkpoint["topk_ratio"]),
        pretrained=bool(checkpoint["pretrained"]),
        freeze_backbone=bool(checkpoint["freeze_backbone"]),
        backbone_input_size=int(checkpoint["backbone_input_size"]),
        normalize_imagenet=bool(checkpoint["normalize_imagenet"]),
        query_chunk_size=int(checkpoint["query_chunk_size"]),
        memory_chunk_size=int(checkpoint["memory_chunk_size"]),
    ).to(device)
    model.set_memory_bank(checkpoint["memory_bank"])

    val_scores_df = collect_scores(model, val_loader, device)
    test_scores_df = collect_scores(model, test_loader, device)

    threshold_quantile = float(checkpoint.get("threshold_quantile", 0.95))
    threshold = float(val_scores_df.loc[val_scores_df["is_anomaly"] == 0, "score"].quantile(threshold_quantile))
    labels = test_scores_df["is_anomaly"].to_numpy()
    scores = test_scores_df["score"].to_numpy()
    metrics = summarize_threshold_metrics(labels, scores, threshold)
    threshold_sweep_df, best_sweep = sweep_threshold_metrics(labels, scores)

    analysis_df = test_dataset.metadata.reset_index(drop=True).copy()
    analysis_df["score"] = test_scores_df.reset_index(drop=True)["score"]
    defect_breakdown_df = build_defect_breakdown(analysis_df, threshold)

    val_scores_df.to_csv(evaluation_dir / "val_scores.csv", index=False)
    test_scores_df.to_csv(evaluation_dir / "test_scores.csv", index=False)
    threshold_sweep_df.to_csv(evaluation_dir / "threshold_sweep.csv", index=False)
    defect_breakdown_df.to_csv(evaluation_dir / "defect_breakdown.csv", index=False)
    write_confusion_csv(evaluation_dir / "confusion_matrix.csv", metrics["confusion_matrix"])
    save_threshold_sweep_plot(
        threshold_sweep_df,
        plots_dir / "threshold_sweep.png",
        title=f"WRN50 PatchCore Holdout ({selected_variant})",
    )
    save_defect_breakdown_plot(
        defect_breakdown_df,
        plots_dir / "defect_breakdown.png",
        title=f"WRN50 PatchCore Holdout Defect Recall ({selected_variant})",
    )

    summary = {
        "experiment": "patchcore_wrn50_x64_main",
        "protocol": "holdout70k_3p5k",
        "selected_variant": selected_variant,
        "checkpoint": to_repo_relative(checkpoint_path, repo_root),
        "metadata_csv": to_repo_relative(metadata_path, repo_root),
        "threshold_quantile": threshold_quantile,
        "threshold": threshold,
        "metrics_at_validation_threshold": metrics,
        "best_threshold_sweep": best_sweep,
        "counts": {
            "val_normal": int((val_scores_df["is_anomaly"] == 0).sum()),
            "test_normal": int((test_scores_df["is_anomaly"] == 0).sum()),
            "test_anomaly": int((test_scores_df["is_anomaly"] == 1).sum()),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "selected_variant.json").write_text(
        json.dumps(
            {
                "selected_variant": selected_variant,
                "checkpoint": to_repo_relative(checkpoint_path, repo_root),
                "output_dir": to_repo_relative(output_dir, repo_root),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest = {
        "output_dir": to_repo_relative(output_dir, repo_root),
        "selected_variant": selected_variant,
        "summary_path": to_repo_relative(output_dir / "summary.json", repo_root),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
