"""Run a simple VAE beta sweep and aggregate evaluation summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from wafer_defect.config import load_toml


def format_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def dump_toml(config: dict[str, Any]) -> str:
    lines: list[str] = []
    for section, values in config.items():
        lines.append(f"[{section}]")
        for key, value in values.items():
            lines.append(f"{key} = {format_toml_value(value)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def beta_tag(beta: float) -> str:
    return str(beta).replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training/train_vae.toml")
    parser.add_argument("--betas", nargs="+", type=float, default=[0.001, 0.005, 0.01, 0.05])
    parser.add_argument("--sweep-root", default="artifacts/x64/vae_beta_sweep")
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    base_config = load_toml(repo_root / args.config)

    sweep_root = repo_root / args.sweep_root
    configs_dir = sweep_root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []

    for beta in args.betas:
        run_config = {
            section: values.copy()
            for section, values in base_config.items()
        }
        tag = beta_tag(beta)
        run_output_dir = sweep_root / f"beta_{tag}"

        run_config["run"]["output_dir"] = str(run_output_dir.relative_to(repo_root)).replace("\\", "/")
        run_config["model"]["beta"] = beta

        config_path = configs_dir / f"train_vae_beta_{tag}.toml"
        config_path.write_text(dump_toml(run_config), encoding="utf-8")

        train_cmd = [
            sys.executable,
            "scripts/train_vae.py",
            "--config",
            str(config_path.relative_to(repo_root)),
        ]
        eval_cmd = [
            sys.executable,
            "scripts/evaluate_reconstruction_model.py",
            "--checkpoint",
            str((run_output_dir / "best_model.pt").relative_to(repo_root)),
            "--config",
            str(config_path.relative_to(repo_root)),
            "--output-dir",
            str((run_output_dir / "evaluation").relative_to(repo_root)),
            "--threshold-quantile",
            str(args.threshold_quantile),
        ]

        print(f"Running beta={beta} with config {config_path.relative_to(repo_root)}")
        subprocess.run(train_cmd, cwd=repo_root, check=True)
        subprocess.run(eval_cmd, cwd=repo_root, check=True)

        summary_path = run_output_dir / "evaluation" / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = summary["metrics_at_validation_threshold"]
        best_sweep = summary["best_threshold_sweep"]

        results.append(
            {
                "beta": beta,
                "output_dir": str(run_output_dir.relative_to(repo_root)).replace("\\", "/"),
                "threshold": metrics["threshold"],
                "val_threshold_precision": metrics["precision"],
                "val_threshold_recall": metrics["recall"],
                "val_threshold_f1": metrics["f1"],
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "best_sweep_threshold": best_sweep["threshold"],
                "best_sweep_precision": best_sweep["precision"],
                "best_sweep_recall": best_sweep["recall"],
                "best_sweep_f1": best_sweep["f1"],
            }
        )

    results.sort(key=lambda row: row["val_threshold_f1"], reverse=True)
    summary_out = {
        "base_config": args.config,
        "betas": args.betas,
        "results": results,
    }
    (sweep_root / "beta_sweep_summary.json").write_text(
        json.dumps(summary_out, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary_out, indent=2))


if __name__ == "__main__":
    main()
