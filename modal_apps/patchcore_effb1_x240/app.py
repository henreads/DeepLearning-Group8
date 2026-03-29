from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

import modal


APP_NAME = "wafer-defect-patchcore-effb1-x240"
RAW_VOLUME_NAME = "wafer-defect-lswmd-raw"
ARTIFACT_VOLUME_NAME = "wafer-defect-patchcore-effb1-x240-artifacts"


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
    LOCAL_REPO_ROOT
    / "experiments"
    / "anomaly_detection"
    / "patchcore"
    / "efficientnet_b1"
    / "x240"
    / "main"
    / "artifacts"
)

REMOTE_PROJECT_ROOT = "/root/project"
REMOTE_RAW_DIR = f"{REMOTE_PROJECT_ROOT}/data/raw"
REMOTE_ARTIFACT_DIR = (
    f"{REMOTE_PROJECT_ROOT}/experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main/artifacts"
)
REMOTE_RUNNER = f"{REMOTE_PROJECT_ROOT}/scripts/run_patchcore_effb1_x240_notebook.py"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.26",
        "pandas>=2.2",
        "scikit-learn>=1.5",
        "matplotlib>=3.9",
        "umap-learn>=0.5.6",
        "torch>=2.2",
        "torchvision>=0.17",
        "tqdm>=4.66",
    )
    .add_local_python_source("wafer_defect", copy=True)
    .add_local_dir("scripts", remote_path=f"{REMOTE_PROJECT_ROOT}/scripts", copy=True)
    .add_local_dir(
        "experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main",
        remote_path=f"{REMOTE_PROJECT_ROOT}/experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main",
        copy=True,
        ignore=["artifacts", "artifacts/**"],
    )
)

raw_volume = modal.Volume.from_name(RAW_VOLUME_NAME, create_if_missing=True)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=image)


def _run_modal_cli(args: list[str]) -> None:
    subprocess.run(["modal", *args], check=True, cwd=LOCAL_REPO_ROOT)


def _download_artifacts(local_artifact_dir: str) -> None:
    local_dir = Path(local_artifact_dir).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    _run_modal_cli(["volume", "get", ARTIFACT_VOLUME_NAME, "/", str(local_dir), "--force"])


def _build_runner_command(*, num_workers: int, phase: str) -> list[str]:
    return [
        "python",
        "-u",
        REMOTE_RUNNER,
        "--raw-pickle",
        f"{REMOTE_RAW_DIR}/LSWMD.pkl",
        "--output-dir",
        "experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main/artifacts/patchcore_efficientnet_b1_one_layer",
        "--num-workers",
        str(num_workers),
        "--phase",
        phase,
    ]


@app.function(
    gpu="A10G",
    timeout=60 * 60 * 8,
    volumes={
        REMOTE_RAW_DIR: raw_volume,
        REMOTE_ARTIFACT_DIR: artifact_volume,
    },
)
def run_patchcore_remote(num_workers: int = 4, run_extras: bool = False) -> dict[str, Any]:
    main_command = _build_runner_command(num_workers=num_workers, phase="main")
    print(f"[patchcore-effb1-x240] launching main phase: {' '.join(main_command)}", flush=True)
    subprocess.run(main_command, check=True, cwd=REMOTE_PROJECT_ROOT)
    artifact_volume.commit()
    print(
        "[patchcore-effb1-x240] main benchmark artifacts committed; "
        "you can download them now from another terminal if you want.",
        flush=True,
    )

    if run_extras:
        extras_command = _build_runner_command(num_workers=num_workers, phase="extras")
        print(f"[patchcore-effb1-x240] launching extras phase: {' '.join(extras_command)}", flush=True)
        subprocess.run(extras_command, check=True, cwd=REMOTE_PROJECT_ROOT)

    artifact_volume.commit()
    print("[patchcore-effb1-x240] artifact volume committed", flush=True)

    manifest_path = Path(REMOTE_ARTIFACT_DIR) / "patchcore_efficientnet_b1_one_layer" / "run_manifest.json"
    if not manifest_path.exists():
        manifest_path = Path(REMOTE_ARTIFACT_DIR) / "patchcore_efficientnet_b1_one_layer" / "main_phase_manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {"output_dir": REMOTE_ARTIFACT_DIR}


@app.local_entrypoint()
def main(num_workers: int = 4, run_extras: bool = False, sync_back: bool = True) -> None:
    result = run_patchcore_remote.remote(num_workers=num_workers, run_extras=run_extras)
    print(json.dumps(result, indent=2))
    if sync_back:
        _download_artifacts(str(LOCAL_ARTIFACT_DIR))


@app.local_entrypoint()
def upload_raw_data(local_raw_pickle: str = str(LOCAL_RAW_PICKLE)) -> None:
    local_path = Path(local_raw_pickle).resolve()
    if not local_path.exists():
        raise FileNotFoundError(f"Raw pickle not found: {local_path}")
    _run_modal_cli(["volume", "put", RAW_VOLUME_NAME, str(local_path), "/LSWMD.pkl"])


@app.local_entrypoint()
def download_artifacts(local_artifact_dir: str = str(LOCAL_ARTIFACT_DIR)) -> None:
    _download_artifacts(local_artifact_dir)
