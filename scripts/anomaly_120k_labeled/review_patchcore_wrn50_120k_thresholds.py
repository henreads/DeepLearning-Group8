"""Review threshold policies for a saved 120k WRN50 PatchCore bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from _common import PROJECT_ROOT, load_helper_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", default="")
    parser.add_argument("--variant-name", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--min-recall", type=float, default=0.70)
    parser.add_argument("--max-false-positive-rate", type=float, default=0.03)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tools = load_helper_module("patchcore_threshold_tools.py", "patchcore_threshold_tools_cli")

    default_bundle_dir = PROJECT_ROOT / "artifacts" / "patchcore_wrn50_multilayer_120k_5pct"
    bundle_dir = Path(args.bundle_dir).resolve() if args.bundle_dir else default_bundle_dir
    bundle_summary_path = bundle_dir / "bundle_summary.json"
    if not bundle_summary_path.exists():
        raise FileNotFoundError(f"Bundle summary not found: {bundle_summary_path}")

    bundle_summary = json.loads(bundle_summary_path.read_text(encoding="utf-8"))
    variant_name = args.variant_name or str(bundle_summary["selected_variant"])
    summary, val_scores_df, test_scores_df = tools.load_variant_artifacts(bundle_dir, variant_name)

    current_threshold = float(summary["threshold"])
    policy_df = tools.build_single_threshold_policy_table(
        val_scores_df,
        test_scores_df,
        current_threshold=current_threshold,
        min_recall=float(args.min_recall),
        max_false_positive_rate=float(args.max_false_positive_rate),
    )
    val_threshold_sweep_df = tools.build_threshold_sweep(val_scores_df)
    val_auto_normal_df = tools.build_auto_normal_sweep(val_scores_df)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else bundle_dir / variant_name / "threshold_review"
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_df.to_csv(output_dir / "threshold_policy_table.csv", index=False)
    val_threshold_sweep_df.to_csv(output_dir / "validation_threshold_sweep.csv", index=False)
    val_auto_normal_df.to_csv(output_dir / "validation_auto_normal_sweep.csv", index=False)

    review_summary = {
        "bundle_dir": str(bundle_dir),
        "variant_name": variant_name,
        "current_threshold": current_threshold,
        "min_recall": float(args.min_recall),
        "max_false_positive_rate": float(args.max_false_positive_rate),
        "policies": policy_df.to_dict(orient="records"),
    }
    (output_dir / "threshold_review_summary.json").write_text(json.dumps(review_summary, indent=2), encoding="utf-8")

    print("Threshold policies:", flush=True)
    print(policy_df.to_string(index=False), flush=True)
    print("", flush=True)
    print("Saved review outputs to:", flush=True)
    print(output_dir, flush=True)


if __name__ == "__main__":
    main()
