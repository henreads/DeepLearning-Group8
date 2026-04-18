from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

import modal


APP_NAME = "wafer-defect-fastflow-x64"
DATA_VOLUME_NAME = "wafer-defect-fastflow-x64-data"
ARTIFACT_VOLUME_NAME = "wafer-defect-fastflow-x64-artifacts"

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
LOCAL_PROCESSED_DIR = LOCAL_REPO_ROOT / "data" / "processed" / "x64" / "wm811k"
LOCAL_FASTFLOW_METADATA = LOCAL_PROCESSED_DIR / "metadata_50k_5pct.csv"
LOCAL_FASTFLOW_ARRAYS = LOCAL_PROCESSED_DIR / "arrays_50k_5pct"
LOCAL_ARTIFACT_DIR = (
    LOCAL_REPO_ROOT
    / "experiments"
    / "anomaly_detection"
    / "fastflow"
    / "x64"
    / "main"
    / "artifacts"
    / "fastflow_variant_sweep"
)

REMOTE_PROJECT_ROOT = "/root/project"
REMOTE_DATA_ROOT = f"{REMOTE_PROJECT_ROOT}/data"
REMOTE_ARTIFACT_DIR = (
    f"{REMOTE_PROJECT_ROOT}/experiments/anomaly_detection/fastflow/x64/main/artifacts/fastflow_variant_sweep"
)
REMOTE_RUNNER = f"{REMOTE_PROJECT_ROOT}/scripts/run_fastflow_x64_notebook.py"

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
    )
    .add_local_python_source("wafer_defect", copy=True)
    .add_local_dir("scripts", remote_path=f"{REMOTE_PROJECT_ROOT}/scripts", copy=True)
    .add_local_dir(
        "experiments/anomaly_detection/fastflow/x64/main",
        remote_path=f"{REMOTE_PROJECT_ROOT}/experiments/anomaly_detection/fastflow/x64/main",
        copy=True,
        ignore=["artifacts", "artifacts/**"],
    )
)

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=image)


def _run_modal_cli(args: list[str]) -> None:
    subprocess.run(["modal", *args], check=True, cwd=LOCAL_REPO_ROOT)


def _download_artifacts(local_artifact_dir: str) -> None:
    local_dir = Path(local_artifact_dir).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    _run_modal_cli(["volume", "get", ARTIFACT_VOLUME_NAME, "/", str(local_dir), "--force"])


def _upload_fastflow_data(local_processed_dir: str, *, upload_all: bool) -> None:
    local_dir = Path(local_processed_dir).resolve()
    if not local_dir.exists():
        raise FileNotFoundError(f"Processed data directory not found: {local_dir}")

    if upload_all:
        _run_modal_cli(["volume", "put", DATA_VOLUME_NAME, str(local_dir), "/processed/x64/wm811k"])
        return

    metadata_path = local_dir / LOCAL_FASTFLOW_METADATA.name
    arrays_dir = local_dir / LOCAL_FASTFLOW_ARRAYS.name
    if not metadata_path.exists():
        raise FileNotFoundError(f"FastFlow metadata CSV not found: {metadata_path}")
    if not arrays_dir.exists():
        raise FileNotFoundError(f"FastFlow arrays directory not found: {arrays_dir}")

    _run_modal_cli(
        ["volume", "put", DATA_VOLUME_NAME, str(metadata_path), "/processed/x64/wm811k/metadata_50k_5pct.csv"]
    )
    _run_modal_cli(
        ["volume", "put", DATA_VOLUME_NAME, str(arrays_dir), "/processed/x64/wm811k/arrays_50k_5pct"]
    )


@app.function(
    gpu="A10G",
    timeout=60 * 60 * 8,
    volumes={
        REMOTE_DATA_ROOT: data_volume,
        REMOTE_ARTIFACT_DIR: artifact_volume,
    },
)
def run_fastflow_remote(
    *,
    run_missing_variants: bool = True,
    force_retrain_variants: bool = False,
    qualitative_variant: str = "wrn50_l23_s4",
    qualitative_max_examples: int = 6,
) -> dict[str, Any]:
    command = ["python", "-u", str(REMOTE_RUNNER)]
    print(f"[fastflow] launching notebook runner: {' '.join(command)}", flush=True)
    if run_missing_variants:
        command.append("--run-missing-variants")
    if force_retrain_variants:
        command.append("--force-retrain-variants")
    command.extend(
        [
            "--qualitative-variant",
            qualitative_variant,
            "--qualitative-max-examples",
            str(qualitative_max_examples),
        ]
    )

    subprocess.run(command, check=True, cwd=REMOTE_PROJECT_ROOT)
    print("[fastflow] notebook runner completed", flush=True)
    artifact_volume.commit()
    print("[fastflow] artifact volume committed", flush=True)

    manifest_path = Path(REMOTE_ARTIFACT_DIR) / "run_manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "output_dir": REMOTE_ARTIFACT_DIR,
        "run_missing_variants": run_missing_variants,
        "force_retrain_variants": force_retrain_variants,
    }


@app.local_entrypoint()
def main(
    run_missing_variants: bool = True,
    force_retrain_variants: bool = False,
    sync_back: bool = True,
    qualitative_variant: str = "wrn50_l23_s4",
    qualitative_max_examples: int = 6,
) -> None:
    result = run_fastflow_remote.remote(
        run_missing_variants=run_missing_variants,
        force_retrain_variants=force_retrain_variants,
        qualitative_variant=qualitative_variant,
        qualitative_max_examples=qualitative_max_examples,
    )
    print(json.dumps(result, indent=2))

    if sync_back:
        _download_artifacts(str(LOCAL_ARTIFACT_DIR))


@app.local_entrypoint()
def upload_data(local_processed_dir: str = str(LOCAL_PROCESSED_DIR), upload_all: bool = False) -> None:
    _upload_fastflow_data(local_processed_dir, upload_all=upload_all)


@app.local_entrypoint()
def download_artifacts(local_artifact_dir: str = str(LOCAL_ARTIFACT_DIR)) -> None:
    _download_artifacts(local_artifact_dir)
