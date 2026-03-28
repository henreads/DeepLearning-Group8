"""Batch-evaluate saved anomaly checkpoints on a chosen metadata split."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_patchcore_variant(variant_name: str) -> dict[str, object]:
    if variant_name.startswith("topk_"):
        ratio_token = variant_name.rsplit("_r", maxsplit=1)[-1]
        return {"reduction": "topk_mean", "topk_ratio": int(ratio_token) / 100.0}
    if variant_name.startswith("mean_"):
        return {"reduction": "mean"}
    return {"reduction": "max"}


def maybe_add_entry(entries: list[dict[str, object]], entry: dict[str, object], repo_root: Path) -> None:
    checkpoint = repo_root / str(entry["checkpoint"])
    config = entry.get("config")
    if not checkpoint.exists():
        return
    if config and not (repo_root / str(config)).exists():
        return
    entries.append(entry)


def discover_default_entries(repo_root: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []

    single_runs = [
        {
            "name": "autoencoder_baseline",
            "checkpoint": "artifacts/x64/autoencoder_baseline/best_model.pt",
            "config": "configs/training/train_autoencoder.toml",
            "kind": "autoencoder_scores",
        },
        {
            "name": "autoencoder_batchnorm",
            "checkpoint": "artifacts/x64/autoencoder_batchnorm/best_model.pt",
            "config": "configs/training/train_autoencoder_batchnorm.toml",
            "kind": "autoencoder_scores",
        },
        {
            "name": "autoencoder_batchnorm_dropout_d0p00",
            "checkpoint": "artifacts/x64/autoencoder_batchnorm_dropout/dropout_0p00/best_model.pt",
            "config": "configs/training/train_autoencoder_batchnorm_dropout.toml",
            "kind": "autoencoder_scores",
        },
        {
            "name": "autoencoder_residual",
            "checkpoint": "artifacts/x64/autoencoder_residual/best_model.pt",
            "config": "configs/training/train_autoencoder_residual.toml",
            "kind": "autoencoder_scores",
        },
        {
            "name": "vae_baseline",
            "checkpoint": "artifacts/x64/vae_baseline/best_model.pt",
            "config": "configs/training/train_vae.toml",
        },
        {
            "name": "svdd_baseline",
            "checkpoint": "artifacts/x64/svdd_baseline/best_model.pt",
            "config": "configs/training/train_svdd.toml",
        },
        {
            "name": "ts_resnet18",
            "checkpoint": "artifacts/x64/ts_resnet18/best_model.pt",
            "config": "configs/training/train_ts_resnet18.toml",
            "model_type": "ts_distillation",
        },
        {
            "name": "ts_resnet50",
            "checkpoint": "artifacts/x64/ts_resnet50/best_model_local_format.pt",
            "config": "configs/training/train_ts_resnet50_kaggle.toml",
            "model_type": "ts_distillation",
        },
    ]
    for entry in single_runs:
        maybe_add_entry(entries, entry, repo_root)

    patchcore_families = [
        ("patchcore_ae_bn", "configs/training/train_patchcore.toml"),
        ("patchcore_resnet18", "configs/training/train_patchcore_resnet18.toml"),
        ("patchcore_resnet50", "configs/training/train_patchcore_resnet50.toml"),
    ]
    for family_name, config_path in patchcore_families:
        family_dir = repo_root / "artifacts" / "x64" / family_name
        if not family_dir.exists():
            continue
        for variant_dir in sorted(path for path in family_dir.iterdir() if path.is_dir()):
            if variant_dir.name == "evaluation":
                continue
            checkpoint_path = variant_dir / "best_model.pt"
            if not checkpoint_path.exists():
                continue
            entry = {
                "name": f"{family_name}__{variant_dir.name}",
                "checkpoint": str(checkpoint_path.relative_to(repo_root).as_posix()),
                "config": config_path,
                "model_type": "patchcore",
            }
            entry.update(parse_patchcore_variant(variant_dir.name))
            entries.append(entry)

    generated_cfg_dir = repo_root / "artifacts" / "generated_configs"
    for variant_dir in sorted((repo_root / "artifacts" / "x64").glob("ts_resnet18_layer*")):
        checkpoint_path = variant_dir / "best_model.pt"
        config_path = generated_cfg_dir / f"{variant_dir.name}.toml"
        entry = {
            "name": variant_dir.name,
            "checkpoint": str(checkpoint_path.relative_to(repo_root).as_posix()),
            "config": str(config_path.relative_to(repo_root).as_posix()),
            "model_type": "ts_distillation",
        }
        maybe_add_entry(entries, entry, repo_root)

    return entries


def load_entries(repo_root: Path, manifest_path: Path | None) -> list[dict[str, object]]:
    if manifest_path is None:
        return discover_default_entries(repo_root)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        raise ValueError("Manifest JSON must be a list of entry objects.")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--output-root", default="artifacts/x64/holdout70k_3p5k_evaluations")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    python_executable = Path(sys.executable).resolve()
    evaluation_script = repo_root / "scripts" / "evaluate_reconstruction_model.py"
    autoencoder_score_script = repo_root / "scripts" / "evaluate_autoencoder_scores.py"
    metadata_csv = (repo_root / args.metadata_csv).resolve()
    output_root = (repo_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = (repo_root / args.manifest).resolve() if args.manifest else None
    entries = load_entries(repo_root, manifest_path)
    if not entries:
        raise ValueError("No evaluation entries found.")

    print(f"Loaded {len(entries)} evaluation entries.")
    for entry in entries:
        print(f"- {entry['name']}")
    if args.dry_run:
        return

    rows: list[dict[str, object]] = []
    for entry in entries:
        name = str(entry["name"])
        checkpoint = (repo_root / str(entry["checkpoint"])).resolve()
        output_dir = output_root / name
        kind = str(entry.get("kind", "reconstruction"))
        summary_path = output_dir / ("score_summary.json" if kind == "autoencoder_scores" else "summary.json")

        if args.skip_existing and summary_path.exists():
            print(f"Skipping existing run: {name}")
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            script_path = evaluation_script if kind == "reconstruction" else autoencoder_score_script
            command = [
                str(python_executable),
                str(script_path),
                "--checkpoint",
                str(checkpoint),
                "--metadata-csv",
                str(metadata_csv),
                "--output-dir",
                str(output_dir),
                "--threshold-quantile",
                str(args.threshold_quantile),
                "--device",
                args.device,
            ]
            if args.batch_size > 0:
                command.extend(["--batch-size", str(args.batch_size)])
            if entry.get("config"):
                command.extend(["--config", str((repo_root / str(entry["config"])).resolve())])
            if kind == "reconstruction" and entry.get("model_type"):
                command.extend(["--model-type", str(entry["model_type"])])
            if kind == "reconstruction" and entry.get("reduction"):
                command.extend(["--reduction", str(entry["reduction"])])
            if kind == "reconstruction" and entry.get("topk_ratio") is not None:
                command.extend(["--topk-ratio", str(entry["topk_ratio"])])
            if kind == "reconstruction" and entry.get("score_student_weight") is not None:
                command.extend(["--score-student-weight", str(entry["score_student_weight"])])
            if kind == "reconstruction" and entry.get("score_autoencoder_weight") is not None:
                command.extend(["--score-autoencoder-weight", str(entry["score_autoencoder_weight"])])

            print(f"Running {name} ...")
            subprocess.run(command, check=True, cwd=repo_root)

        if kind == "autoencoder_scores":
            score_summary_path = output_dir / "score_summary.csv"
            score_df = pd.read_csv(score_summary_path)
            best_row = score_df.sort_values(
                ["val_threshold_f1", "auprc", "best_sweep_f1"],
                ascending=False,
            ).iloc[0]
            rows.append(
                {
                    "name": name,
                    "model_type": "autoencoder",
                    "selected_score": best_row["score_name"],
                    "checkpoint": str(checkpoint),
                    "precision": best_row["val_threshold_precision"],
                    "recall": best_row["val_threshold_recall"],
                    "f1": best_row["val_threshold_f1"],
                    "auroc": best_row["auroc"],
                    "auprc": best_row["auprc"],
                    "threshold": best_row["threshold"],
                    "best_sweep_f1": best_row["best_sweep_f1"],
                    "best_sweep_threshold": best_row["best_sweep_threshold"],
                    "test_normal": "",
                    "test_anomaly": "",
                    "output_dir": str(output_dir),
                }
            )
        else:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            metrics = summary["metrics_at_validation_threshold"]
            best_sweep = summary["best_threshold_sweep"]
            rows.append(
                {
                    "name": name,
                    "model_type": summary.get("model_type", entry.get("model_type", "")),
                    "selected_score": "",
                    "checkpoint": summary.get("checkpoint", str(checkpoint)),
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "auroc": metrics["auroc"],
                    "auprc": metrics["auprc"],
                    "threshold": summary["threshold"],
                    "best_sweep_f1": best_sweep["f1"],
                    "best_sweep_threshold": best_sweep["threshold"],
                    "test_normal": summary["counts"]["test_normal"],
                    "test_anomaly": summary["counts"]["test_anomaly"],
                    "output_dir": str(output_dir),
                }
            )

    leaderboard = pd.DataFrame(rows).sort_values(["f1", "auprc", "auroc"], ascending=False)
    leaderboard_path = output_root / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    print(f"Saved leaderboard to {leaderboard_path}")
    print(leaderboard.to_string(index=False))


if __name__ == "__main__":
    main()
