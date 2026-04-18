from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import modal


APP_NAME = "wafer-defect-ensemble-vit-effnetb1-x224"
ARTIFACT_VOLUME_NAME = "wafer-defect-ensemble-vit-effnetb1-x224-artifacts"


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
LOCAL_ARTIFACT_DIR = (
    LOCAL_REPO_ROOT / "experiments" / "anomaly_detection" / "ensemble" / "x224" / "vit_effnetb1_ensemble" / "artifacts"
)
REMOTE_PROJECT_ROOT = "/root/project"
REMOTE_ARTIFACT_DIR = (
    f"{REMOTE_PROJECT_ROOT}/experiments/anomaly_detection/ensemble/x224/vit_effnetb1_ensemble/artifacts"
)
REMOTE_RUNNER = f"{REMOTE_PROJECT_ROOT}/scripts/run_ensemble_vit_effnetb1_x224_notebook.py"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.26",
        "pandas>=2.2",
        "scikit-learn>=1.5",
        "matplotlib>=3.9",
    )
    .add_local_dir("scripts", remote_path=f"{REMOTE_PROJECT_ROOT}/scripts", copy=True)
    .add_local_dir(
        "experiments/anomaly_detection/ensemble/x224/vit_effnetb1_ensemble",
        remote_path=f"{REMOTE_PROJECT_ROOT}/experiments/anomaly_detection/ensemble/x224/vit_effnetb1_ensemble",
        copy=True,
        ignore=["artifacts", "artifacts/**"],
    )
    .add_local_dir(
        "experiments/anomaly_detection/patchcore/vit_b16/x224/main/artifacts/patchcore_vit_b16_5pct/main_5pct/results/evaluation",
        remote_path=(
            f"{REMOTE_PROJECT_ROOT}/experiments/anomaly_detection/patchcore/vit_b16/x224/main/"
            "artifacts/patchcore_vit_b16_5pct/main_5pct/results/evaluation"
        ),
        copy=True,
    )
    .add_local_dir(
        "experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main_one_layer/"
        "artifacts/patchcore_efficientnet_b1_one_layer/results/evaluation",
        remote_path=(
            f"{REMOTE_PROJECT_ROOT}/experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main_one_layer/"
            "artifacts/patchcore_efficientnet_b1_one_layer/results/evaluation"
        ),
        copy=True,
    )
)

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
        local_target = local_dir / Path(remote_name)
        if str(entry.get("Type", "")).lower() == "dir":
            local_target.mkdir(parents=True, exist_ok=True)
        else:
            local_target.parent.mkdir(parents=True, exist_ok=True)
        _run_modal_cli(["volume", "get", ARTIFACT_VOLUME_NAME, f"/{remote_name}", str(local_target), "--force"])


@app.function(
    timeout=60 * 30,
    volumes={REMOTE_ARTIFACT_DIR: artifact_volume},
)
def run_ensemble_remote() -> dict[str, Any]:
    command = [
        "python",
        "-u",
        REMOTE_RUNNER,
        "--output-dir",
        "experiments/anomaly_detection/ensemble/x224/vit_effnetb1_ensemble/artifacts/vit_effnetb1_ensemble",
    ]
    print(f"[ensemble-vit-effnetb1-x224] launching runner: {' '.join(command)}", flush=True)
    subprocess.run(command, check=True, cwd=REMOTE_PROJECT_ROOT)
    artifact_volume.commit()
    manifest_path = Path(REMOTE_ARTIFACT_DIR) / "vit_effnetb1_ensemble" / "run_manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {"output_dir": REMOTE_ARTIFACT_DIR}


@app.local_entrypoint()
def main(sync_back: bool = True) -> None:
    result = run_ensemble_remote.remote()
    print(json.dumps(result, indent=2))
    if sync_back:
        _download_artifacts(str(LOCAL_ARTIFACT_DIR), "/vit_effnetb1_ensemble")


@app.local_entrypoint()
def download_artifacts(local_artifact_dir: str = str(LOCAL_ARTIFACT_DIR)) -> None:
    _download_artifacts(local_artifact_dir, "/vit_effnetb1_ensemble")
