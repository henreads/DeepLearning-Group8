"""Run the 120k labeled WRN50 PatchCore memory-bank feature sweep from the command line."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from _common import PROJECT_ROOT, load_helper_module

DOWNSAMPLE_MAP = {"layer1": 4, "layer2": 8, "layer3": 16, "layer4": 32}
ARTIFACT_BASENAME = "patchcore_wrn50_multilayer_120k_notebook4_normal_only_x224"


def patches_per_image_for_layers(teacher_layers: list[str], backbone_input_size: int = 224) -> int:
    output_spatial = max(1, backbone_input_size // min(DOWNSAMPLE_MAP[layer] for layer in teacher_layers))
    return int(output_spatial * output_spatial)


def default_source_images(memory_bank_size: int, teacher_layers: list[str], backbone_input_size: int = 224) -> int:
    return int(math.ceil(memory_bank_size / patches_per_image_for_layers(teacher_layers, backbone_input_size)))


def make_experiment(
    *,
    name: str,
    group: str,
    image_size: int,
    teacher_layers: list[str],
    memory_bank_size: int = 240_000,
    topk_ratio: float = 0.05,
    reduction: str = "topk_mean",
    patch_nn_k: int = 3,
    memory_source_images: int | None = None,
    normal_only_memory_sampling: bool = True,
    note: str = "",
) -> dict[str, object]:
    teacher_layers = [str(layer).lower() for layer in teacher_layers]
    return {
        "name": str(name),
        "group": str(group),
        "image_size": int(image_size),
        "teacher_layers": teacher_layers,
        "memory_bank_size": int(memory_bank_size),
        "topk_ratio": float(topk_ratio),
        "reduction": str(reduction),
        "patch_nn_k": int(patch_nn_k),
        "memory_source_images": None if memory_source_images is None else int(memory_source_images),
        "normal_only_memory_sampling": bool(normal_only_memory_sampling),
        "note": str(note),
        "expected_patches_per_image": patches_per_image_for_layers(teacher_layers),
        "default_source_images": default_source_images(memory_bank_size, teacher_layers),
    }


EXPERIMENT_SPECS = [
    make_experiment(
        name="coverage__control_224_l23_mb50k_mean_knn1_auto_mixed",
        group="coverage_sampling",
        image_size=224,
        teacher_layers=["layer2", "layer3"],
        memory_bank_size=50_000,
        reduction="mean",
        topk_ratio=0.10,
        patch_nn_k=1,
        memory_source_images=None,
        normal_only_memory_sampling=False,
        note="Current normal-only x224 control before the CT-style scorer and wider source coverage.",
    ),
    make_experiment(
        name="coverage__224_l23_mb240k_topk005_knn3_normals_1024src",
        group="coverage_sampling",
        image_size=224,
        teacher_layers=["layer2", "layer3"],
        memory_source_images=1_024,
        normal_only_memory_sampling=True,
        note="CT-style control recipe with larger bank, k=3 scoring, and moderate normal-only source coverage.",
    ),
    make_experiment(
        name="coverage__224_l23_mb240k_topk005_knn3_normals_2048src",
        group="coverage_sampling",
        image_size=224,
        teacher_layers=["layer2", "layer3"],
        memory_source_images=2048,
        normal_only_memory_sampling=True,
        note="Main coverage candidate for the next pass.",
    ),
    make_experiment(
        name="coverage__224_l23_mb240k_topk005_knn3_normals_8192src",
        group="coverage_sampling",
        image_size=224,
        teacher_layers=["layer2", "layer3"],
        memory_source_images=8192,
        normal_only_memory_sampling=True,
        note="Aggressive coverage test to estimate the upper bound from more normal source wafers.",
    ),
    make_experiment(
        name="image__128_l23_mb240k_topk005_knn3_normals_2048src",
        group="image_size",
        image_size=128,
        teacher_layers=["layer2", "layer3"],
        memory_source_images=2048,
        normal_only_memory_sampling=True,
        note="CT-branch-style pragmatic image size for WRN50 under the stronger normal-only recipe.",
    ),
    make_experiment(
        name="image__224_l23_mb240k_topk005_knn3_normals_2048src",
        group="image_size",
        image_size=224,
        teacher_layers=["layer2", "layer3"],
        memory_source_images=2048,
        normal_only_memory_sampling=True,
        note="Direct x224 comparison point under the stronger normal-only recipe.",
    ),
    make_experiment(
        name="layer__224_l12_mb240k_topk005_knn3_normals_2048src",
        group="layer_sweep",
        image_size=224,
        teacher_layers=["layer1", "layer2"],
        memory_source_images=2048,
        normal_only_memory_sampling=True,
        note="Shallower feature mix to test compact defect sensitivity under the stronger recipe.",
    ),
    make_experiment(
        name="layer__224_l123_mb240k_topk005_knn3_normals_2048src",
        group="layer_sweep",
        image_size=224,
        teacher_layers=["layer1", "layer2", "layer3"],
        memory_source_images=2048,
        normal_only_memory_sampling=True,
        note="Layer1 inclusion while keeping the current deeper features under the stronger recipe.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-pickle", default="")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--artifact-dir", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backbone-input-size", type=int, default=224)
    parser.add_argument("--query-chunk-size", type=int, default=512)
    parser.add_argument("--memory-chunk-size", type=int, default=4096)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--threshold-strategy", default="validation_normal_quantile")
    parser.add_argument("--max-validation-false-positive-rate", type=float, default=-1.0)
    parser.add_argument("--group", action="append", default=[])
    parser.add_argument("--experiment", action="append", default=[])
    parser.add_argument("--overwrite-dataset", action="store_true")
    return parser.parse_args()


def resolve_selected_specs(args: argparse.Namespace) -> list[dict[str, object]]:
    selected_specs = [dict(spec) for spec in EXPERIMENT_SPECS]

    if args.group:
        groups = set(args.group)
        selected_specs = [spec for spec in selected_specs if str(spec["group"]) in groups]
        missing_groups = sorted(groups - {str(spec["group"]) for spec in EXPERIMENT_SPECS})
        if missing_groups:
            raise ValueError(f"Unknown group name(s): {', '.join(missing_groups)}")

    if args.experiment:
        experiments = set(args.experiment)
        selected_specs = [spec for spec in selected_specs if str(spec["name"]) in experiments]
        missing_experiments = sorted(experiments - {str(spec["name"]) for spec in EXPERIMENT_SPECS})
        if missing_experiments:
            raise ValueError(f"Unknown experiment name(s): {', '.join(missing_experiments)}")

    if not selected_specs:
        raise ValueError("No experiments selected.")

    return selected_specs


def main() -> None:
    args = parse_args()
    patchcore = load_helper_module("patchcore_wrn50_modal.py", "patchcore_wrn50_modal_memorybank_cli")

    selected_specs = resolve_selected_specs(args)
    split_config = patchcore.NORMAL_ONLY_TRAIN_SPLIT_CONFIG.copy()
    patchcore.set_seed(args.seed)

    device = patchcore.resolve_device(args.device)
    raw_pickle = patchcore.auto_find_raw_pickle(args.raw_pickle or None)
    data_root = Path(args.data_root).resolve() if args.data_root else patchcore.resolve_data_root(PROJECT_ROOT)
    output_root = Path(args.output_root).resolve() if args.output_root else patchcore.resolve_output_root(PROJECT_ROOT)
    artifact_dir = (
        Path(args.artifact_dir).resolve()
        if args.artifact_dir
        else output_root / ARTIFACT_BASENAME
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)

    max_validation_false_positive_rate = (
        None if args.max_validation_false_positive_rate < 0 else float(args.max_validation_false_positive_rate)
    )
    dataset_cache: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, object]] = []

    def load_split_bundle(image_size: int) -> dict[str, Any]:
        cached = dataset_cache.get(int(image_size))
        if cached is not None:
            return cached

        metadata_path = patchcore.prepare_dataset(
            raw_pickle,
            data_root,
            int(image_size),
            split_config,
            seed=int(args.seed),
            overwrite=bool(args.overwrite_dataset),
        )
        metadata = pd.read_csv(metadata_path)
        train_dataset = patchcore.WaferArrayDataset(metadata_path, split="train", data_root=data_root)
        val_dataset = patchcore.WaferArrayDataset(metadata_path, split="val", data_root=data_root)
        test_dataset = patchcore.WaferArrayDataset(metadata_path, split="test", data_root=data_root)
        bundle = {
            "image_size": int(image_size),
            "metadata_path": metadata_path,
            "metadata": metadata,
            "train_dataset": train_dataset,
            "val_loader": DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
            "test_loader": DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            ),
        }
        dataset_cache[int(image_size)] = bundle
        return bundle

    plan_df = pd.DataFrame(selected_specs)
    plan_df["teacher_layers_label"] = plan_df["teacher_layers"].apply(lambda layers: " + ".join(layers))
    plan_df.to_csv(artifact_dir / "experiment_plan.csv", index=False)
    print("Experiment plan:", flush=True)
    print(plan_df.to_string(index=False), flush=True)
    print("", flush=True)
    print(f"Artifact dir: {artifact_dir}", flush=True)
    print(f"Using device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)

    for spec in selected_specs:
        split_bundle = load_split_bundle(int(spec["image_size"]))
        group_dir = artifact_dir / str(spec["group"])
        group_dir.mkdir(parents=True, exist_ok=True)

        variant = {
            "name": str(spec["name"]),
            "memory_bank_size": int(spec["memory_bank_size"]),
            "reduction": str(spec["reduction"]),
            "topk_ratio": float(spec["topk_ratio"]),
            "patch_nn_k": int(spec["patch_nn_k"]),
            "memory_source_images": spec["memory_source_images"],
            "normal_only_memory_sampling": bool(spec["normal_only_memory_sampling"]),
        }
        print(
            "\n=== Running experiment: "
            f"{spec['name']} | group={spec['group']} | image={spec['image_size']} "
            f"| layers={' + '.join(spec['teacher_layers'])} ===",
            flush=True,
        )
        result = patchcore.run_patchcore_variant(
            variant,
            train_dataset=split_bundle["train_dataset"],
            val_loader=split_bundle["val_loader"],
            test_loader=split_bundle["test_loader"],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            output_dir=group_dir,
            seed=args.seed,
            teacher_layers=list(spec["teacher_layers"]),
            pretrained=True,
            freeze_backbone=True,
            backbone_input_size=args.backbone_input_size,
            normalize_imagenet=True,
            threshold_quantile=args.threshold_quantile,
            threshold_strategy=args.threshold_strategy,
            max_validation_false_positive_rate=max_validation_false_positive_rate,
            query_chunk_size=args.query_chunk_size,
            memory_chunk_size=args.memory_chunk_size,
        )
        row = {
            **result["row"],
            "group": str(spec["group"]),
            "image_size": int(spec["image_size"]),
            "teacher_layers": list(spec["teacher_layers"]),
            "teacher_layers_label": " + ".join(spec["teacher_layers"]),
            "note": str(spec["note"]),
            "default_source_images": int(spec["default_source_images"]),
        }
        rows.append(row)
        print(
            "Finished "
            f"{spec['name']} | precision={result['metrics']['precision']:.4f} "
            f"| recall={result['metrics']['recall']:.4f} | f1={result['metrics']['f1']:.4f}",
            flush=True,
        )

    results_df = pd.DataFrame(rows).sort_values(["f1", "auroc", "auprc"], ascending=False).reset_index(drop=True)
    results_df.to_csv(artifact_dir / "notebook4_results.csv", index=False)

    best_row = results_df.iloc[0].to_dict()
    summary = {
        "best_experiment": str(best_row["name"]),
        "split_config": split_config,
        "raw_pickle": str(raw_pickle),
        "artifact_dir": str(artifact_dir),
        "threshold_strategy": args.threshold_strategy,
        "threshold_quantile": float(args.threshold_quantile),
        "max_validation_false_positive_rate": max_validation_false_positive_rate,
        "groups": sorted({str(spec["group"]) for spec in selected_specs}),
        "experiments": results_df.to_dict(orient="records"),
    }
    (artifact_dir / "notebook4_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Summary ===", flush=True)
    print(results_df.to_string(index=False), flush=True)
    print("", flush=True)
    print(f"Best experiment: {best_row['name']}", flush=True)
    print(f"Best threshold: {float(best_row['threshold']):.6f}", flush=True)


if __name__ == "__main__":
    main()
