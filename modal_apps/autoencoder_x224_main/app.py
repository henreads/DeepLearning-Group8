from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import modal


APP_NAME = "wafer-defect-autoencoder-x224-main"
RAW_VOLUME_NAME = "wafer-defect-lswmd-raw"
PROCESSED_VOLUME_NAME = "wafer-defect-wm811k-x224-processed"
ARTIFACT_VOLUME_NAME = "wafer-defect-autoencoder-x224-main-artifacts"


def _resolve_local_repo_root() -> Path:
    here = Path(__file__).resolve()
    cwd = Path.cwd().resolve()
    candidates = [here.parent, *here.parents, cwd, *cwd.parents]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "modal_apps").exists() and (candidate / "experiments").exists():
            return candidate
    return cwd


LOCAL_REPO_ROOT = _resolve_local_repo_root()
LOCAL_RAW_PICKLE = LOCAL_REPO_ROOT / "data" / "raw" / "LSWMD.pkl"
LOCAL_ARTIFACT_DIR = (
    LOCAL_REPO_ROOT / "experiments" / "anomaly_detection" / "autoencoder" / "x224" / "main" / "artifacts"
)

REMOTE_PROJECT_ROOT = "/root/project"
REMOTE_RAW_DIR = f"{REMOTE_PROJECT_ROOT}/data/raw"
REMOTE_PROCESSED_DIR = f"{REMOTE_PROJECT_ROOT}/data/processed/x224/wm811k"
REMOTE_ARTIFACT_DIR = (
    f"{REMOTE_PROJECT_ROOT}/experiments/anomaly_detection/autoencoder/x224/main/artifacts"
)
REMOTE_RUNNER = f"{REMOTE_PROJECT_ROOT}/scripts/run_autoencoder_x224_main_notebook.py"
REMOTE_PREPARE_SCRIPT = f"{REMOTE_PROJECT_ROOT}/scripts/prepare_wm811k.py"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.0",
        "torchvision>=0.15",
        "numpy>=1.24",
        "pandas>=1.5",
        "scikit-learn>=1.3",
        "toml>=0.10",
        "matplotlib>=3.6",
        "jupyter>=1.0",
        "nbconvert>=7.0",
        "tqdm>=4.66",
    )
)

app = modal.App(name=APP_NAME)


@app.function(
    image=image,
    gpu="A10G",
    timeout=8 * 3600,
    volumes={
        RAW_VOLUME_NAME: modal.Volume.from_name(RAW_VOLUME_NAME, create_if_missing=True),
        PROCESSED_VOLUME_NAME: modal.Volume.from_name(PROCESSED_VOLUME_NAME, create_if_missing=True),
        ARTIFACT_VOLUME_NAME: modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True),
    },
)
def main(
    run_train: bool = True,
    run_eval: bool = True,
    run_sweep: bool = True,
) -> dict[str, Any]:
    """
    Main training pipeline for Autoencoder x224 Main.

    Phases:
    1. Prepare: Ensure x224 dataset is ready
    2. Train: Train autoencoder checkpoint
    3. Eval: Evaluate on test split
    4. Sweep: Generate threshold sweep and metrics
    """
    from pathlib import Path

    REMOTE_ROOT = Path(REMOTE_PROJECT_ROOT)
    RAW_VOL_MOUNT = Path("/mnt/raw")
    PROCESSED_VOL_MOUNT = Path("/mnt/processed")
    ARTIFACT_VOL_MOUNT = Path("/mnt/artifacts")

    # Mount volumes
    raw_vol = modal.Volume.from_name(RAW_VOLUME_NAME)
    processed_vol = modal.Volume.from_name(PROCESSED_VOLUME_NAME)
    artifact_vol = modal.Volume.from_name(ARTIFACT_VOLUME_NAME)

    # Ensure directories exist
    RAW_VOL_MOUNT.mkdir(parents=True, exist_ok=True)
    PROCESSED_VOL_MOUNT.mkdir(parents=True, exist_ok=True)
    ARTIFACT_VOL_MOUNT.mkdir(parents=True, exist_ok=True)

    # Check if x224 dataset needs preparation
    metadata_path = PROCESSED_VOL_MOUNT / "metadata_50k_5pct.csv"
    if not metadata_path.exists():
        print("Preparing x224 dataset...")
        env = {"PYTHONPATH": str(REMOTE_ROOT / "src")}
        subprocess.run(
            [
                "python",
                str(REMOTE_PREPARE_SCRIPT),
                "--pkl-path", str(RAW_VOL_MOUNT / "LSWMD.pkl"),
                "--output-dir", str(PROCESSED_VOL_MOUNT),
                "--target-size", "224",
            ],
            cwd=REMOTE_ROOT,
            env={**subprocess.os.environ, **env},
            check=True,
        )
        processed_vol.commit()
        print("x224 dataset preparation complete.")
    else:
        print(f"Found existing x224 dataset at {metadata_path}")

    # Train or load artifact
    train_config_path = REMOTE_ROOT / "experiments/anomaly_detection/autoencoder/x224/main/train_config.toml"
    runner_result = {}

    if run_train:
        print("Phase 1: Training...")
        runner_result["train"] = subprocess.run(
            ["python", str(REMOTE_RUNNER), "train"],
            cwd=REMOTE_ROOT,
            env={**subprocess.os.environ, "PYTHONPATH": str(REMOTE_ROOT / "src")},
            capture_output=True,
            text=True,
            check=True,
        ).returncode
        artifact_vol.commit()
        print("Training phase complete.")

    if run_eval:
        print("Phase 2: Evaluation...")
        runner_result["eval"] = subprocess.run(
            ["python", str(REMOTE_RUNNER), "eval"],
            cwd=REMOTE_ROOT,
            env={**subprocess.os.environ, "PYTHONPATH": str(REMOTE_ROOT / "src")},
            capture_output=True,
            text=True,
            check=True,
        ).returncode
        artifact_vol.commit()
        print("Evaluation phase complete.")

    if run_sweep:
        print("Phase 3: Threshold sweep and metrics...")
        runner_result["sweep"] = subprocess.run(
            ["python", str(REMOTE_RUNNER), "sweep"],
            cwd=REMOTE_ROOT,
            env={**subprocess.os.environ, "PYTHONPATH": str(REMOTE_ROOT / "src")},
            capture_output=True,
            text=True,
            check=True,
        ).returncode
        artifact_vol.commit()
        print("Threshold sweep phase complete.")

    return {
        "status": "success",
        "phases": runner_result,
        "artifact_dir": REMOTE_ARTIFACT_DIR,
    }


@app.function(
    image=image,
    volumes={
        ARTIFACT_VOLUME_NAME: modal.Volume.from_name(ARTIFACT_VOLUME_NAME),
    },
)
def download_artifacts() -> str:
    """Download artifacts from Modal volume back to local repo."""
    import shutil
    from pathlib import Path

    ARTIFACT_VOL_MOUNT = Path("/mnt/artifacts")
    LOCAL_ARTIFACT_ROOT = LOCAL_ARTIFACT_DIR

    if not ARTIFACT_VOL_MOUNT.exists():
        return f"No artifacts found in volume at {ARTIFACT_VOL_MOUNT}"

    LOCAL_ARTIFACT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    if LOCAL_ARTIFACT_ROOT.exists():
        shutil.rmtree(LOCAL_ARTIFACT_ROOT)

    shutil.copytree(ARTIFACT_VOL_MOUNT / "autoencoder_x224", LOCAL_ARTIFACT_ROOT)
    return f"Artifacts downloaded to {LOCAL_ARTIFACT_ROOT}"


@app.local_entrypoint()
def cli():
    """CLI entry point for training."""
    print(f"Starting {APP_NAME}...")
    result = main.remote(run_train=True, run_eval=True, run_sweep=True)
    print(f"Training result: {result}")
    print("\nDownloading artifacts...")
    download_msg = download_artifacts.remote()
    print(download_msg)
