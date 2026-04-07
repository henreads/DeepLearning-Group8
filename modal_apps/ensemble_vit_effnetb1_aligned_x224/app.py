"""Modal app: aligned ViT-B/16 + EfficientNet-B1 ensemble re-scoring.

Both models are re-scored on the same shared test set (seed=42, LSWMD.pkl pipeline)
so that per-class defect recall can be computed correctly.

Setup (one-time, if not already done):
    modal volume create wafer-defect-lswmd-raw
    modal run modal_apps/patchcore_vit_b16_x224_main/app.py::upload_raw_data

Run:
    modal run modal_apps/ensemble_vit_effnetb1_aligned_x224/app.py

Download results:
    modal run modal_apps/ensemble_vit_effnetb1_aligned_x224/app.py::download_artifacts
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import modal


APP_NAME             = "wafer-defect-ensemble-vit-effnetb1-aligned-x224"
RAW_VOLUME_NAME      = "wafer-defect-lswmd-raw"
VIT_ARTIFACT_VOLUME  = "wafer-defect-patchcore-vit-b16-x224-main-artifacts"
EFF_ARTIFACT_VOLUME  = "wafer-defect-patchcore-effb1-x240-artifacts"
OUT_ARTIFACT_VOLUME  = "wafer-defect-ensemble-vit-effnetb1-aligned-artifacts"


def _resolve_local_repo_root() -> Path:
    here = Path(__file__).resolve()
    cwd  = Path.cwd().resolve()
    for candidate in [*here.parents, cwd, *cwd.parents]:
        if (candidate / "modal_apps").exists() and (candidate / "experiments").exists():
            return candidate
    return cwd


LOCAL_REPO_ROOT   = _resolve_local_repo_root()
LOCAL_RAW_PICKLE  = LOCAL_REPO_ROOT / "data" / "raw" / "LSWMD.pkl"
LOCAL_ARTIFACT_DIR = (
    LOCAL_REPO_ROOT
    / "experiments/anomaly_detection/ensemble/x224/vit_effnetb1_ensemble/artifacts"
)

REMOTE_PROJECT_ROOT = "/root/project"
REMOTE_RAW_DIR      = f"{REMOTE_PROJECT_ROOT}/data/raw"
REMOTE_VIT_ARTIFACT = (
    f"{REMOTE_PROJECT_ROOT}/experiments/anomaly_detection"
    "/patchcore/vit_b16/x224/main/artifacts"
)
REMOTE_EFF_ARTIFACT = (
    f"{REMOTE_PROJECT_ROOT}/experiments/anomaly_detection"
    "/patchcore/efficientnet_b1/x240/main_one_layer/artifacts"
)
REMOTE_OUT_ARTIFACT = (
    f"{REMOTE_PROJECT_ROOT}/experiments/anomaly_detection"
    "/ensemble/x224/vit_effnetb1_ensemble/artifacts"
)
REMOTE_RUNNER = f"{REMOTE_PROJECT_ROOT}/scripts/run_ensemble_vit_effnetb1_aligned_x224.py"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.26",
        "pandas>=2.2",
        "scikit-learn>=1.5",
        "matplotlib>=3.9",
        "torch>=2.2",
        "torchvision>=0.17",
        "tqdm>=4.66",
        "timm>=1.0",
        "ipython>=8.0",   # for IPython.display.display used in bootstrap globals
    )
    .add_local_python_source("wafer_defect", remote_path=f"{REMOTE_PROJECT_ROOT}/src/wafer_defect", copy=True)
    .add_local_dir(
        "scripts",
        remote_path=f"{REMOTE_PROJECT_ROOT}/scripts",
        copy=True,
    )
    .add_local_dir(
        "experiments/anomaly_detection/ensemble/x224/vit_effnetb1_ensemble",
        remote_path=(
            f"{REMOTE_PROJECT_ROOT}/experiments/anomaly_detection"
            "/ensemble/x224/vit_effnetb1_ensemble"
        ),
        copy=True,
        ignore=["artifacts", "artifacts/**"],
    )
    # Bake in a minimal repo skeleton so REPO_ROOT detection works
    .add_local_dir("configs", remote_path=f"{REMOTE_PROJECT_ROOT}/configs", copy=True)
)

raw_volume     = modal.Volume.from_name(RAW_VOLUME_NAME,     create_if_missing=True)
vit_volume     = modal.Volume.from_name(VIT_ARTIFACT_VOLUME, create_if_missing=True)
eff_volume     = modal.Volume.from_name(EFF_ARTIFACT_VOLUME, create_if_missing=True)
out_volume     = modal.Volume.from_name(OUT_ARTIFACT_VOLUME, create_if_missing=True)
app = modal.App(APP_NAME, image=image)


def _run_modal_cli(args: list[str], *, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "modal", *args],
        check=True,
        cwd=LOCAL_REPO_ROOT,
        text=True,
        capture_output=capture_output,
    )


def _download_artifacts(local_dir: Path, volume_name: str, remote_subdir: str) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    listing = _run_modal_cli(
        ["volume", "ls", volume_name, remote_subdir, "--json"],
        capture_output=True,
    )
    for entry in json.loads(listing.stdout):
        remote_name = str(entry["Filename"])
        local_target = local_dir / Path(remote_name)
        if str(entry.get("Type", "")).lower() == "dir":
            local_target.mkdir(parents=True, exist_ok=True)
        else:
            local_target.parent.mkdir(parents=True, exist_ok=True)
            _run_modal_cli(
                ["volume", "get", volume_name, f"/{remote_name}", str(local_target), "--force"]
            )


@app.function(
    gpu="A10G",
    timeout=60 * 60,   # 1 h — generous for CPU pickle loading + GPU inference
    volumes={
        REMOTE_RAW_DIR:      raw_volume,
        REMOTE_VIT_ARTIFACT: vit_volume,
        REMOTE_EFF_ARTIFACT: eff_volume,
        REMOTE_OUT_ARTIFACT: out_volume,
    },
)
def run_aligned_ensemble(batch_size: int = 64) -> dict[str, Any]:
    command = [
        "python", "-u", REMOTE_RUNNER,
        "--raw-pickle", f"{REMOTE_RAW_DIR}/LSWMD.pkl",
        "--output-dir", REMOTE_OUT_ARTIFACT,
        "--batch-size", str(batch_size),
    ]
    print(f"[aligned-ensemble] launching: {' '.join(command)}", flush=True)
    subprocess.run(command, check=True, cwd=REMOTE_PROJECT_ROOT)
    out_volume.commit()

    manifest_path = Path(REMOTE_OUT_ARTIFACT) / "aligned_run_manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {"output_dir": REMOTE_OUT_ARTIFACT}


@app.local_entrypoint()
def main(batch_size: int = 64, sync_back: bool = True) -> None:
    result = run_aligned_ensemble.remote(batch_size=batch_size)
    print(json.dumps(result, indent=2))
    if sync_back:
        _download_artifacts(LOCAL_ARTIFACT_DIR, OUT_ARTIFACT_VOLUME, "/")


@app.local_entrypoint()
def download_artifacts(local_artifact_dir: str = str(LOCAL_ARTIFACT_DIR)) -> None:
    _download_artifacts(Path(local_artifact_dir), OUT_ARTIFACT_VOLUME, "/")


@app.local_entrypoint()
def upload_raw_data(local_raw_pickle: str = str(LOCAL_RAW_PICKLE)) -> None:
    local_path = Path(local_raw_pickle).resolve()
    if not local_path.exists():
        raise FileNotFoundError(f"Raw pickle not found: {local_path}")
    _run_modal_cli(["volume", "put", RAW_VOLUME_NAME, str(local_path), "/LSWMD.pkl"])
