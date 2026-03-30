from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import modal


APP_NAME = "wafer-defect-ts-wrn50-x64-multilayer"
RAW_VOLUME_NAME = "wafer-defect-lswmd-raw"
ARTIFACT_VOLUME_NAME = "wafer-defect-ts-wrn50-x64-multilayer-artifacts"


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
    / "teacher_student"
    / "wideresnet50_2"
    / "x64"
    / "multilayer_self_contained"
    / "artifacts"
)

REMOTE_PROJECT_ROOT = "/root/project"
REMOTE_RAW_DIR = f"{REMOTE_PROJECT_ROOT}/data/raw"
REMOTE_ARTIFACT_DIR = (
    f"{REMOTE_PROJECT_ROOT}/experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/multilayer_self_contained/artifacts"
)
REMOTE_RUNNER = f"{REMOTE_PROJECT_ROOT}/scripts/run_ts_wrn50_x64_multilayer_notebook.py"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "ipython>=8.18",
        "numpy>=1.26",
        "pandas>=2.2",
        "scikit-learn>=1.5",
        "matplotlib>=3.9",
        "torch>=2.2",
        "torchvision>=0.17",
        "tqdm>=4.66",
    )
    .add_local_python_source("wafer_defect", copy=True)
    .add_local_dir("scripts", remote_path=f"{REMOTE_PROJECT_ROOT}/scripts", copy=True)
    .add_local_dir(
        "experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/multilayer_self_contained",
        remote_path=f"{REMOTE_PROJECT_ROOT}/experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/multilayer_self_contained",
        copy=True,
        ignore=["artifacts", "artifacts/**"],
    )
)

raw_volume = modal.Volume.from_name(RAW_VOLUME_NAME, create_if_missing=True)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=image)


def _run_modal_cli(args: list[str], *, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "modal", *args],
        check=True,
        cwd=LOCAL_REPO_ROOT,
        text=True,
        capture_output=capture_output,
    )


def _download_artifacts(local_artifact_dir: str, remote_subdir: str) -> None:
    local_dir = Path(local_artifact_dir).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    listing = _run_modal_cli(["volume", "ls", ARTIFACT_VOLUME_NAME, remote_subdir, "--json"], capture_output=True)
    entries = json.loads(listing.stdout)
    for entry in entries:
        remote_name = str(entry["Filename"])
        if Path(remote_name).name == "processed":
            continue
        local_target = local_dir / Path(remote_name)
        if str(entry.get("Type", "")).lower() == "dir":
            local_target.mkdir(parents=True, exist_ok=True)
        else:
            local_target.parent.mkdir(parents=True, exist_ok=True)
        _run_modal_cli(["volume", "get", ARTIFACT_VOLUME_NAME, f"/{remote_name}", str(local_target), "--force"])


@app.function(
    gpu="A10G",
    timeout=60 * 60 * 10,
    volumes={
        REMOTE_RAW_DIR: raw_volume,
        REMOTE_ARTIFACT_DIR: artifact_volume,
    },
)
def run_ts_remote(num_workers: int = 8) -> dict[str, Any]:
    manifests: dict[str, Any] = {}
    for phase in ["train", "eval", "sweep"]:
        command = [
            "python",
            "-u",
            REMOTE_RUNNER,
            "--output-dir",
            "experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/multilayer_self_contained/artifacts/ts_wideresnet50_multilayer",
            "--phase",
            phase,
            "--num-workers",
            str(num_workers),
        ]
        print(f"[ts-wrn50-x64-multilayer] launching phase={phase}: {' '.join(command)}", flush=True)
        subprocess.run(command, check=True, cwd=REMOTE_PROJECT_ROOT)
        artifact_volume.commit()
        print(f"[ts-wrn50-x64-multilayer] artifact volume committed after phase={phase}", flush=True)
        manifest_path = Path(REMOTE_ARTIFACT_DIR) / "ts_wideresnet50_multilayer" / f"{phase}_phase_manifest.json"
        if manifest_path.exists():
            manifests[phase] = json.loads(manifest_path.read_text(encoding="utf-8"))
    final_manifest = Path(REMOTE_ARTIFACT_DIR) / "ts_wideresnet50_multilayer" / "run_manifest.json"
    if final_manifest.exists():
        manifests["final"] = json.loads(final_manifest.read_text(encoding="utf-8"))
    return manifests


@app.local_entrypoint()
def main(num_workers: int = 8, sync_back: bool = True) -> None:
    result = run_ts_remote.remote(num_workers=num_workers)
    print(json.dumps(result, indent=2))
    if sync_back:
        _download_artifacts(str(LOCAL_ARTIFACT_DIR), "/ts_wideresnet50_multilayer")


@app.local_entrypoint()
def upload_raw_data(local_raw_pickle: str = str(LOCAL_RAW_PICKLE)) -> None:
    local_path = Path(local_raw_pickle).resolve()
    if not local_path.exists():
        raise FileNotFoundError(f"Raw pickle not found: {local_path}")
    _run_modal_cli(["volume", "put", RAW_VOLUME_NAME, str(local_path), "/LSWMD.pkl"])


@app.local_entrypoint()
def download_artifacts(local_artifact_dir: str = str(LOCAL_ARTIFACT_DIR)) -> None:
    _download_artifacts(local_artifact_dir, "/ts_wideresnet50_multilayer")
