from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import modal


APP_NAME = "wafer-defect-patchcore-effb1-x240"
RAW_VOLUME_NAME = "wafer-defect-lswmd-raw"
PROCESSED_VOLUME_NAME = "wafer-defect-wm811k-x240-processed"
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
REMOTE_PROCESSED_DIR = f"{REMOTE_PROJECT_ROOT}/data/processed/x240/wm811k"
REMOTE_ARTIFACT_DIR = (
    f"{REMOTE_PROJECT_ROOT}/experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main/artifacts"
)
REMOTE_RUNNER = f"{REMOTE_PROJECT_ROOT}/scripts/run_patchcore_effb1_x240_notebook.py"
REMOTE_PREPARE_SCRIPT = f"{REMOTE_PROJECT_ROOT}/scripts/prepare_wm811k.py"

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
    .add_local_dir("configs", remote_path=f"{REMOTE_PROJECT_ROOT}/configs", copy=True)
    .add_local_dir("scripts", remote_path=f"{REMOTE_PROJECT_ROOT}/scripts", copy=True)
    .add_local_dir(
        "data/dataset/x240/benchmark_50k_5pct",
        remote_path=f"{REMOTE_PROJECT_ROOT}/data/dataset/x240/benchmark_50k_5pct",
        copy=True,
    )
    .add_local_dir(
        "experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main_one_layer",
        remote_path=f"{REMOTE_PROJECT_ROOT}/experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main_one_layer",
        copy=True,
        ignore=["artifacts", "artifacts/**"],
    )
)

raw_volume = modal.Volume.from_name(RAW_VOLUME_NAME, create_if_missing=True)
processed_volume = modal.Volume.from_name(PROCESSED_VOLUME_NAME, create_if_missing=True)
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


def _download_artifacts(local_artifact_dir: str) -> None:
    local_dir = Path(local_artifact_dir).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    remote_base = "/patchcore_efficientnet_b1_one_layer"
    local_base = local_dir / "patchcore_efficientnet_b1_one_layer"
    local_base.mkdir(parents=True, exist_ok=True)
    listing = _run_modal_cli(
        ["volume", "ls", ARTIFACT_VOLUME_NAME, remote_base, "--json"],
        capture_output=True,
    )
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


def _cached_dataset_is_valid(metadata_path: Path, arrays_dir: Path) -> bool:
    if not metadata_path.exists() or not arrays_dir.exists():
        return False

    expected_prefix = f"data/processed/x240/wm811k/{arrays_dir.name}/"
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        header = handle.readline()
        if not header:
            return False
        first_row = handle.readline().strip()

    if not first_row:
        return False

    columns = [c.strip() for c in header.strip().split(",")]
    values = [v.strip() for v in first_row.split(",")]
    row = dict(zip(columns, values))
    sample_path = str(row.get("array_path", "")).strip()
    if not sample_path:
        return False
    if ":\\" in sample_path or sample_path.startswith("C:/") or "\\" in sample_path:
        return False
    if not sample_path.startswith(expected_prefix):
        return False
    return (Path(REMOTE_PROJECT_ROOT) / sample_path).exists()


def _clear_processed_cache(metadata_path: Path, arrays_dir: Path) -> None:
    for candidate in metadata_path.parent.glob("metadata*.csv"):
        if candidate.exists():
            candidate.unlink()
    if arrays_dir.exists():
        shutil.rmtree(arrays_dir)


def _prepare_processed_dataset() -> Path:
    metadata_path = Path(REMOTE_PROCESSED_DIR) / "metadata_50k_5pct.csv"
    arrays_dir = Path(REMOTE_PROCESSED_DIR) / "arrays_50k_5pct"
    if _cached_dataset_is_valid(metadata_path, arrays_dir):
        print(f"[patchcore-effb1-x240] reusing cached processed dataset: {metadata_path}", flush=True)
        return metadata_path
    if metadata_path.exists() or arrays_dir.exists():
        print("[patchcore-effb1-x240] cached processed dataset is stale; rebuilding", flush=True)
        _clear_processed_cache(metadata_path, arrays_dir)

    config_path = Path("/tmp/patchcore_effb1_x240_prepare_wm811k.toml")
    config_path.write_text(
        "\n".join(
            [
                "[dataset]",
                'name = "wm811k"',
                f'raw_pickle = "{REMOTE_RAW_DIR}/LSWMD.pkl"',
                f'processed_root = "{REMOTE_PROCESSED_DIR}"',
                f'metadata_csv = "{REMOTE_PROCESSED_DIR}/metadata.csv"',
                f'metadata_50k_csv = "{REMOTE_PROCESSED_DIR}/metadata_50k.csv"',
                f'metadata_50k_5pct_csv = "{REMOTE_PROCESSED_DIR}/metadata_50k_5pct.csv"',
                f'dev_metadata_csv = "{REMOTE_PROCESSED_DIR}/metadata_dev.csv"',
                "image_size = 240",
                'normal_label = "none"',
                'defect_label = "pattern"',
                "",
                "[split_generation]",
                'mode = "normal_only_test_defects"',
                "",
                "[splits]",
                "train_normal_fraction = 0.8",
                "val_normal_fraction = 0.1",
                "test_normal_fraction = 0.1",
                "random_seed = 42",
                "",
                "[dev_subset]",
                "enabled = true",
                "normal_count = 2000",
                "defect_count = 400",
                "",
                "[train_subset]",
                "normal_count = 50000",
                "use_all_defects_for_test = false",
                "test_defect_fraction_of_test_normals = 0.05",
                "",
            ]
        ),
        encoding="utf-8",
    )
    command = ["python", "-u", REMOTE_PREPARE_SCRIPT, "--config", str(config_path)]
    print(f"[patchcore-effb1-x240] preparing processed dataset: {' '.join(command)}", flush=True)
    subprocess.run(command, check=True, cwd=REMOTE_PROJECT_ROOT)
    processed_volume.commit()
    print("[patchcore-effb1-x240] processed dataset volume committed", flush=True)
    return metadata_path


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
        REMOTE_PROCESSED_DIR: processed_volume,
        REMOTE_ARTIFACT_DIR: artifact_volume,
    },
)
def run_patchcore_remote(num_workers: int = 4, run_extras: bool = False) -> dict[str, Any]:
    _prepare_processed_dataset()
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


@app.function(
    timeout=60 * 60,
    volumes={
        REMOTE_RAW_DIR: raw_volume,
        REMOTE_PROCESSED_DIR: processed_volume,
    },
)
def prepare_processed_remote() -> dict[str, str]:
    metadata_path = _prepare_processed_dataset()
    return {
        "processed_volume": PROCESSED_VOLUME_NAME,
        "metadata_path": metadata_path.as_posix(),
        "arrays_dir": f"{REMOTE_PROCESSED_DIR}/arrays_50k_5pct",
    }


@app.local_entrypoint()
def prepare_processed_data() -> None:
    result = prepare_processed_remote.remote()
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def download_artifacts(local_artifact_dir: str = str(LOCAL_ARTIFACT_DIR)) -> None:
    _download_artifacts(local_artifact_dir)
