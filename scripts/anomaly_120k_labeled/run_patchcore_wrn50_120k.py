"""Run the baseline 120k labeled WRN50 PatchCore sweep from the command line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from _common import PROJECT_ROOT, load_helper_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-pickle", default="")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--artifact-dir", default="")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--teacher-layers", nargs="+", default=["layer2", "layer3"])
    parser.add_argument("--backbone-input-size", type=int, default=224)
    parser.add_argument("--query-chunk-size", type=int, default=1024)
    parser.add_argument("--memory-chunk-size", type=int, default=4096)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--threshold-strategy", default="validation_f1")
    parser.add_argument("--max-validation-false-positive-rate", type=float, default=-1.0)
    parser.add_argument(
        "--variant-set",
        choices=["auto", "default", "normal_only_improvement"],
        default="auto",
    )
    parser.add_argument("--normal-only-split", action="store_true")
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--overwrite-dataset", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patchcore = load_helper_module("patchcore_wrn50_modal.py", "patchcore_wrn50_modal_cli")

    split_config = (
        patchcore.NORMAL_ONLY_TRAIN_SPLIT_CONFIG.copy()
        if args.normal_only_split
        else patchcore.DEFAULT_SPLIT_CONFIG.copy()
    )
    if args.variant_set == "normal_only_improvement":
        variants = [dict(variant) for variant in patchcore.NORMAL_ONLY_IMPROVEMENT_VARIANTS]
    elif args.variant_set == "default":
        variants = [dict(variant) for variant in patchcore.DEFAULT_VARIANTS]
    else:
        variants = (
            [dict(variant) for variant in patchcore.NORMAL_ONLY_IMPROVEMENT_VARIANTS]
            if args.normal_only_split
            else [dict(variant) for variant in patchcore.DEFAULT_VARIANTS]
        )
    if args.variant:
        requested = set(args.variant)
        variants = [variant for variant in variants if str(variant["name"]) in requested]
        missing = sorted(requested - {str(variant["name"]) for variant in variants})
        if missing:
            raise ValueError(f"Unknown variant name(s): {', '.join(missing)}")

    if not variants:
        raise ValueError("No variants selected.")

    patchcore.set_seed(args.seed)
    device = patchcore.resolve_device(args.device)
    raw_pickle = patchcore.auto_find_raw_pickle(args.raw_pickle or None)
    threshold_strategy = args.threshold_strategy
    if args.normal_only_split and threshold_strategy == "validation_f1":
        threshold_strategy = "validation_normal_quantile"
    data_root = Path(args.data_root).resolve() if args.data_root else patchcore.resolve_data_root(PROJECT_ROOT)
    output_root = Path(args.output_root).resolve() if args.output_root else patchcore.resolve_output_root(PROJECT_ROOT)
    artifact_dir = (
        Path(args.artifact_dir).resolve()
        if args.artifact_dir
        else (
            output_root / "patchcore_wrn50_multilayer_120k_report_normal_only_x224"
            if args.normal_only_split
            else output_root / "patchcore_wrn50_multilayer_120k_5pct"
        )
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = patchcore.prepare_dataset(
        raw_pickle,
        data_root,
        int(args.image_size),
        split_config,
        seed=int(args.seed),
        overwrite=bool(args.overwrite_dataset),
    )
    metadata = pd.read_csv(metadata_path)

    train_dataset = patchcore.WaferArrayDataset(metadata_path, split="train", data_root=data_root)
    val_dataset = patchcore.WaferArrayDataset(metadata_path, split="val", data_root=data_root)
    test_dataset = patchcore.WaferArrayDataset(metadata_path, split="test", data_root=data_root)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    max_validation_false_positive_rate = (
        None if args.max_validation_false_positive_rate < 0 else float(args.max_validation_false_positive_rate)
    )
    variant_results: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []

    print(patchcore.split_summary_wide(metadata).to_string(index=False), flush=True)
    print("", flush=True)
    print(patchcore.defect_type_summary(metadata).head(18).to_string(index=False), flush=True)
    print("", flush=True)
    print(f"Artifact dir: {artifact_dir}", flush=True)
    print(f"Using device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)

    for variant in variants:
        print(f"\n=== Running variant: {variant['name']} ===", flush=True)
        result = patchcore.run_patchcore_variant(
            variant,
            train_dataset=train_dataset,
            val_loader=val_loader,
            test_loader=test_loader,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            output_dir=artifact_dir,
            seed=args.seed,
            teacher_layers=list(args.teacher_layers),
            pretrained=True,
            freeze_backbone=True,
            backbone_input_size=args.backbone_input_size,
            normalize_imagenet=True,
            threshold_quantile=args.threshold_quantile,
            threshold_strategy=threshold_strategy,
            max_validation_false_positive_rate=max_validation_false_positive_rate,
            query_chunk_size=args.query_chunk_size,
            memory_chunk_size=args.memory_chunk_size,
        )
        variant_results[str(variant["name"])] = result
        rows.append(result["row"])
        print(
            "Finished "
            f"{variant['name']} | precision={result['metrics']['precision']:.4f} "
            f"| recall={result['metrics']['recall']:.4f} | f1={result['metrics']['f1']:.4f}",
            flush=True,
        )

    sweep_results_df = pd.DataFrame(rows).sort_values(["f1", "auroc", "auprc"], ascending=False).reset_index(drop=True)
    sweep_results_df.to_csv(artifact_dir / "patchcore_sweep_results.csv", index=False)

    selected_variant_name = str(sweep_results_df.iloc[0]["name"])
    selected_result = variant_results[selected_variant_name]
    test_metadata = metadata.loc[metadata["split"] == "test"].reset_index(drop=True)
    selected_predictions_df = patchcore.attach_scores_to_metadata(
        test_metadata,
        selected_result["test_scores_df"],
        float(selected_result["threshold"]),
    )
    selected_predictions_df.to_csv(artifact_dir / "selected_variant_test_predictions.csv", index=False)

    bundle_summary = {
        "selected_variant": selected_variant_name,
        "split_config": split_config,
        "raw_pickle": str(raw_pickle),
        "metadata_path": str(metadata_path),
        "artifact_dir": str(artifact_dir),
        "threshold_strategy": threshold_strategy,
        "threshold_quantile": float(args.threshold_quantile),
        "max_validation_false_positive_rate": max_validation_false_positive_rate,
        "variant_set": args.variant_set,
        "normal_only_split": bool(args.normal_only_split),
        "teacher_layers": list(args.teacher_layers),
        "variants": sweep_results_df.to_dict(orient="records"),
    }
    (artifact_dir / "bundle_summary.json").write_text(json.dumps(bundle_summary, indent=2), encoding="utf-8")

    print("\n=== Summary ===", flush=True)
    print(sweep_results_df.to_string(index=False), flush=True)
    print("", flush=True)
    print(f"Selected variant: {selected_variant_name}", flush=True)
    print(f"Selected threshold: {float(selected_result['threshold']):.6f}", flush=True)


if __name__ == "__main__":
    main()
