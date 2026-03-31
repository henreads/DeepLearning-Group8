from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

SCRIPT_PATH = Path(__file__).resolve()
for candidate in [SCRIPT_PATH.parent, *SCRIPT_PATH.parents]:
    src_root = candidate / "src"
    if (src_root / "wafer_defect").exists():
        if str(src_root) not in sys.path:
            sys.path.insert(0, str(src_root))
        break

from wafer_defect.config import load_toml
from wafer_defect.data.wm811k import WaferMapDataset
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


NOTEBOOK_PATH = Path("experiments/anomaly_detection/fastflow/x64/main/notebook.ipynb")
CONFIG_PATH = Path("experiments/anomaly_detection/fastflow/x64/main/train_config.toml")


def to_numpy_scores(values: object) -> np.ndarray:
    if hasattr(values, "detach"):
        return values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float32)


def load_selected_variant(input_artifact_dir: Path, variant_name: str) -> tuple[dict, Path, dict]:
    summary_path = input_artifact_dir / "results" / "fastflow_variant_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"FastFlow summary CSV not found: {summary_path}")

    summary_df = pd.read_csv(summary_path)
    if summary_df.empty:
        raise ValueError(f"FastFlow summary CSV is empty: {summary_path}")

    if variant_name:
        selected = summary_df.loc[summary_df["variant"] == variant_name]
        if selected.empty:
            raise ValueError(f"Variant {variant_name!r} not found in {summary_path}")
        best_row = selected.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0].to_dict()
    else:
        best_row = summary_df.iloc[0].to_dict()

    selected_variant = str(best_row["variant"])
    checkpoint_path = input_artifact_dir / selected_variant / "checkpoints" / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"FastFlow checkpoint not found: {checkpoint_path}")
    return best_row, checkpoint_path, {"variant": selected_variant}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--input-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant-name", default="")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    repo_root = resolve_repo_root()
    os.chdir(repo_root)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("WM811K_REPO_ROOT", str(repo_root))

    scope = exec_notebook_code_cells((repo_root / NOTEBOOK_PATH), [3, 11, 13])
    build_model_from_checkpoint = scope["build_model_from_checkpoint"]
    collect_maps = scope["collect_maps"]
    reduce_scores = scope["reduce_scores"]
    resolve_device = scope["resolve_device"]

    config = load_toml(repo_root / CONFIG_PATH)
    config["data"]["metadata_csv"] = args.metadata_path

    input_artifact_dir = (repo_root / args.input_artifact_dir).resolve()
    output_dir = (repo_root / args.output_dir)
    evaluation_dir = output_dir / "evaluation"
    plots_dir = output_dir / "plots"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    best_row, checkpoint_path, variant_info = load_selected_variant(input_artifact_dir, args.variant_name)
    selected_variant = str(variant_info["variant"])
    variant = next(variant for variant in config["variants"] if variant["name"] == selected_variant)
    model = build_model_from_checkpoint(checkpoint_path, variant, config).to(resolve_device(args.device))
    device = resolve_device(args.device)

    metadata_path = (repo_root / args.metadata_path).resolve()
    image_size = int(config["data"]["image_size"])
    batch_size = int(config["data"]["batch_size"])
    num_workers = int(config["data"]["num_workers"])

    val_dataset = WaferMapDataset(metadata_path, split="val", image_size=image_size)
    test_dataset = WaferMapDataset(metadata_path, split="test", image_size=image_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    val_maps, _ = collect_maps(model, val_loader, device, image_size, desc="maps_val_holdout")
    test_maps, test_labels = collect_maps(model, test_loader, device, image_size, desc="maps_test_holdout")

    reduction = str(best_row["reduction"])
    topk_ratio = float(best_row["topk_ratio"])
    val_scores = to_numpy_scores(reduce_scores(val_maps, reduction, topk_ratio))
    test_scores = to_numpy_scores(reduce_scores(test_maps, reduction, topk_ratio))

    val_scores_df = pd.DataFrame(
        {"score": val_scores, "is_anomaly": val_dataset.metadata["is_anomaly"].astype(int).to_numpy()}
    )
    test_scores_df = pd.DataFrame({"score": test_scores, "is_anomaly": test_labels.astype(int)})

    threshold_quantile = float(config["scoring"]["threshold_quantile"])
    threshold = float(val_scores_df.loc[val_scores_df["is_anomaly"] == 0, "score"].quantile(threshold_quantile))
    metrics = summarize_threshold_metrics(test_labels.astype(int), test_scores, threshold)
    threshold_sweep_df, best_sweep = sweep_threshold_metrics(test_labels.astype(int), test_scores)

    analysis_df = test_dataset.metadata.reset_index(drop=True).copy()
    analysis_df["score"] = test_scores_df["score"]
    defect_breakdown_df = build_defect_breakdown(analysis_df, threshold)

    val_scores_df.to_csv(evaluation_dir / "val_scores.csv", index=False)
    test_scores_df.to_csv(evaluation_dir / "test_scores.csv", index=False)
    threshold_sweep_df.to_csv(evaluation_dir / "threshold_sweep.csv", index=False)
    defect_breakdown_df.to_csv(evaluation_dir / "defect_breakdown.csv", index=False)
    write_confusion_csv(evaluation_dir / "confusion_matrix.csv", metrics["confusion_matrix"])
    save_threshold_sweep_plot(
        threshold_sweep_df,
        plots_dir / "threshold_sweep.png",
        title=f"FastFlow Holdout ({selected_variant}, {best_row['score_name']})",
    )
    save_defect_breakdown_plot(
        defect_breakdown_df,
        plots_dir / "defect_breakdown.png",
        title=f"FastFlow Holdout Defect Recall ({selected_variant})",
    )

    summary = {
        "experiment": "fastflow_x64_main",
        "protocol": "holdout70k_3p5k",
        "selected_variant": selected_variant,
        "selected_score_name": str(best_row["score_name"]),
        "checkpoint": to_repo_relative(checkpoint_path, repo_root),
        "metadata_csv": to_repo_relative(metadata_path, repo_root),
        "threshold_quantile": threshold_quantile,
        "threshold": threshold,
        "reduction": reduction,
        "topk_ratio": topk_ratio,
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
                "selected_score_name": str(best_row["score_name"]),
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
        "selected_score_name": str(best_row["score_name"]),
        "summary_path": to_repo_relative(output_dir / "summary.json", repo_root),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
