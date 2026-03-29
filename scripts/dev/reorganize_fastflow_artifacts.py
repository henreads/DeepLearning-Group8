from __future__ import annotations

import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "experiments" / "anomaly_detection" / "fastflow" / "x64" / "main" / "artifacts" / "fastflow_variant_sweep"
RESULTS_DIR = OUTPUT_DIR / "results"
PLOTS_DIR = OUTPUT_DIR / "plots"

VARIANTS = [
    "wrn50_l23_s6",
    "wrn50_l2_s6",
    "wrn50_l23_s4",
]


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        (OUTPUT_DIR / variant / "checkpoints").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / variant / "results").mkdir(parents=True, exist_ok=True)


def move_if_needed(src: Path, dst: Path) -> None:
    if not src.exists() or dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def main() -> None:
    ensure_dirs()

    move_if_needed(OUTPUT_DIR / "fastflow_variant_summary.csv", RESULTS_DIR / "fastflow_variant_summary.csv")
    move_if_needed(OUTPUT_DIR / "run_manifest.json", RESULTS_DIR / "run_manifest.json")

    for variant in VARIANTS:
        variant_root = OUTPUT_DIR / variant
        checkpoints_dir = variant_root / "checkpoints"
        results_dir = variant_root / "results"

        move_if_needed(OUTPUT_DIR / f"{variant}_history.csv", results_dir / "history.csv")
        move_if_needed(OUTPUT_DIR / f"{variant}_scores.csv", results_dir / "scores.csv")
        move_if_needed(OUTPUT_DIR / f"{variant}_defect_breakdown.csv", results_dir / "defect_breakdown.csv")
        move_if_needed(OUTPUT_DIR / f"{variant}_best_row.csv", results_dir / "best_row.csv")
        move_if_needed(OUTPUT_DIR / f"{variant}_summary.json", results_dir / "summary.json")

        move_if_needed(OUTPUT_DIR / f"{variant}_best_model.pt", checkpoints_dir / "best_model.pt")
        move_if_needed(OUTPUT_DIR / f"{variant}_latest_checkpoint.pt", checkpoints_dir / "latest_checkpoint.pt")

        for epoch_checkpoint in OUTPUT_DIR.glob(f"{variant}_checkpoint_epoch_*.pt"):
            target_name = epoch_checkpoint.name.replace(f"{variant}_", "", 1)
            move_if_needed(epoch_checkpoint, checkpoints_dir / target_name)

    print(f"Reorganized FastFlow artifacts under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
