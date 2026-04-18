from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import modal


APP_NAME = "wafer-defect-rd4ad-wrn50-x224-main"
RAW_VOLUME_NAME = "wafer-defect-lswmd-raw"
PROCESSED_VOLUME_NAME = "wafer-defect-wm811k-x224-processed"
ARTIFACT_VOLUME_NAME = "wafer-defect-rd4ad-wrn50-x224-main-artifacts"

ARTIFACT_SUBDIR = "rd4ad_wrn50_x224"


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
    / "rd4ad"
    / "wideresnet50"
    / "x224"
    / "main"
    / "artifacts"
)

REMOTE_PROJECT_ROOT = "/root/project"
REMOTE_RAW_DIR = f"{REMOTE_PROJECT_ROOT}/data/raw"
REMOTE_PROCESSED_DIR = f"{REMOTE_PROJECT_ROOT}/data/processed/x224/wm811k"
REMOTE_ARTIFACT_DIR = (
    f"{REMOTE_PROJECT_ROOT}/experiments/anomaly_detection/rd4ad/wideresnet50/x224/main/artifacts"
)
REMOTE_TRAIN_CONFIG = (
    f"{REMOTE_PROJECT_ROOT}/experiments/anomaly_detection/rd4ad/wideresnet50/x224/main/train_config.toml"
)
REMOTE_RUNNER = f"{REMOTE_PROJECT_ROOT}/scripts/train_rd4ad.py"
REMOTE_PREPARE_SCRIPT = f"{REMOTE_PROJECT_ROOT}/scripts/prepare_wm811k.py"
ARTIFACT_COMMIT_INTERVAL_SECONDS = 300

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
        "umap-learn>=0.5.6",
    )
    .add_local_python_source("wafer_defect", copy=True)
    .add_local_dir("configs", remote_path=f"{REMOTE_PROJECT_ROOT}/configs", copy=True)
    .add_local_dir("scripts", remote_path=f"{REMOTE_PROJECT_ROOT}/scripts", copy=True)
    .add_local_dir(
        "experiments/anomaly_detection/rd4ad/wideresnet50/x224/main",
        remote_path=f"{REMOTE_PROJECT_ROOT}/experiments/anomaly_detection/rd4ad/wideresnet50/x224/main",
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


def _download_artifacts(local_artifact_dir: str, remote_subdir: str) -> None:
    local_dir = Path(local_artifact_dir).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    keep_checkpoint_names = {"best_model.pt", "latest_checkpoint.pt"}

    def _download_tree(remote_dir: str) -> None:
        listing = _run_modal_cli(["volume", "ls", ARTIFACT_VOLUME_NAME, remote_dir, "--json"], capture_output=True)
        entries = json.loads(listing.stdout)
        for entry in entries:
            remote_name = str(entry["Filename"])
            local_target = local_dir / Path(remote_name)
            entry_type = str(entry.get("Type", "")).lower()
            if entry_type == "dir":
                local_target.mkdir(parents=True, exist_ok=True)
                _download_tree(f"/{remote_name}")
                continue
            if (
                local_target.suffix == ".pt"
                and (
                    local_target.name.startswith("checkpoint_epoch_")
                    or (
                        "checkpoints" in local_target.parts
                        and local_target.name not in keep_checkpoint_names
                    )
                )
            ):
                continue
            local_target.parent.mkdir(parents=True, exist_ok=True)
            _run_modal_cli(["volume", "get", ARTIFACT_VOLUME_NAME, f"/{remote_name}", str(local_target), "--force"])

    _download_tree(remote_subdir)


def _cached_dataset_is_valid(metadata_path: Path, arrays_dir: Path) -> bool:
    if not metadata_path.exists() or not arrays_dir.exists():
        return False

    expected_prefix = f"data/processed/x224/wm811k/{arrays_dir.name}/"
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        first_row = next(reader, None)

    if not first_row:
        return False

    sample_path = str(first_row.get("array_path", "")).strip()
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
        print(f"[{APP_NAME}] reusing cached processed dataset: {metadata_path}", flush=True)
        return metadata_path
    if metadata_path.exists() or arrays_dir.exists():
        print(f"[{APP_NAME}] cached processed dataset is stale; rebuilding", flush=True)
        _clear_processed_cache(metadata_path, arrays_dir)

    config_path = Path(f"/tmp/{APP_NAME}_prepare_wm811k.toml")
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
                "image_size = 224",
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
    print(f"[{APP_NAME}] preparing processed dataset: {' '.join(command)}", flush=True)
    subprocess.run(command, check=True, cwd=REMOTE_PROJECT_ROOT)
    processed_volume.commit()
    print(f"[{APP_NAME}] processed dataset volume committed", flush=True)
    return metadata_path


@app.function(
    gpu="A10G",
    timeout=60 * 60 * 8,
    volumes={
        REMOTE_RAW_DIR: raw_volume,
        REMOTE_PROCESSED_DIR: processed_volume,
        REMOTE_ARTIFACT_DIR: artifact_volume,
    },
)
def prepare_processed_remote() -> dict[str, str]:
    metadata_path = _prepare_processed_dataset()
    return {"metadata_csv": metadata_path.as_posix()}


@app.function(
    gpu="A10G",
    timeout=60 * 60 * 8,
    volumes={
        REMOTE_RAW_DIR: raw_volume,
        REMOTE_PROCESSED_DIR: processed_volume,
        REMOTE_ARTIFACT_DIR: artifact_volume,
    },
)
def run_rd4ad_remote() -> dict[str, Any]:
    _prepare_processed_dataset()
    command = [
        "python",
        "-u",
        REMOTE_RUNNER,
        "--config",
        REMOTE_TRAIN_CONFIG,
    ]
    print(f"[{APP_NAME}] launching runner: {' '.join(command)}", flush=True)
    process = subprocess.Popen(command, cwd=REMOTE_PROJECT_ROOT)
    last_commit_time = time.monotonic()
    while True:
        return_code = process.poll()
        now = time.monotonic()
        if now - last_commit_time >= ARTIFACT_COMMIT_INTERVAL_SECONDS:
            artifact_volume.commit()
            print(f"[{APP_NAME}] periodic artifact volume commit", flush=True)
            last_commit_time = now
        if return_code is not None:
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)
            break
        time.sleep(15)
    artifact_volume.commit()
    print(f"[{APP_NAME}] artifact volume committed", flush=True)

    summary_path = Path(REMOTE_ARTIFACT_DIR) / ARTIFACT_SUBDIR / "results" / "summary.json"
    history_path = Path(REMOTE_ARTIFACT_DIR) / ARTIFACT_SUBDIR / "results" / "history.json"
    result: dict[str, Any] = {
        "output_dir": f"{REMOTE_ARTIFACT_DIR}/{ARTIFACT_SUBDIR}",
        "train_config": REMOTE_TRAIN_CONFIG,
    }
    if summary_path.exists():
        result["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        result["history_len"] = len(history)
    return result


@app.local_entrypoint()
def main(sync_back: bool = True) -> None:
    result = run_rd4ad_remote.remote()
    print(json.dumps(result, indent=2))
    if sync_back:
        _download_artifacts(str(LOCAL_ARTIFACT_DIR), f"/{ARTIFACT_SUBDIR}")


@app.local_entrypoint()
def upload_raw_data(local_raw_pickle: str = str(LOCAL_RAW_PICKLE)) -> None:
    local_path = Path(local_raw_pickle).resolve()
    if not local_path.exists():
        raise FileNotFoundError(f"Raw pickle not found: {local_path}")
    _run_modal_cli(["volume", "put", RAW_VOLUME_NAME, str(local_path), "/LSWMD.pkl"])


@app.local_entrypoint()
def prepare_processed_data() -> None:
    result = prepare_processed_remote.remote()
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def download_artifacts(local_artifact_dir: str = str(LOCAL_ARTIFACT_DIR)) -> None:
    _download_artifacts(local_artifact_dir, f"/{ARTIFACT_SUBDIR}")
