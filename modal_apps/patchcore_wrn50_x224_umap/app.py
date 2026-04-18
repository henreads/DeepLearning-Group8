from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import modal


APP_NAME = "wafer-defect-patchcore-wrn50-x224-umap"
RAW_VOLUME_NAME = "wafer-defect-lswmd-raw"
PROCESSED_VOLUME_NAME = "wafer-defect-wm811k-x224-processed"
ARTIFACT_VOLUME_NAME = "wafer-defect-patchcore-wrn50-x224-umap-artifacts"


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
    / "wideresnet50"
    / "x224"
    / "multilayer_umap"
    / "artifacts"
)

REMOTE_PROJECT_ROOT = "/root/project"
REMOTE_RAW_DIR = f"{REMOTE_PROJECT_ROOT}/data/raw"
REMOTE_PROCESSED_DIR = f"{REMOTE_PROJECT_ROOT}/data/processed/x224/wm811k"
REMOTE_ARTIFACT_DIR = (
    f"{REMOTE_PROJECT_ROOT}/experiments/anomaly_detection/patchcore"
    f"/wideresnet50/x224/multilayer_umap/artifacts"
)
REMOTE_RUNNER = f"{REMOTE_PROJECT_ROOT}/scripts/run_patchcore_wrn50_x224_umap.py"
REMOTE_PREPARE_SCRIPT = f"{REMOTE_PROJECT_ROOT}/scripts/prepare_wm811k.py"
REMOTE_RUN_SUBDIR = "patchcore-wideresnet50-multilayer-umap"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "ipython>=8.18",
        "numpy>=1.26",
        "pandas>=2.2",
        "scikit-learn>=1.5",
        "matplotlib>=3.9",
        "seaborn>=0.13",
        "torch>=2.2",
        "torchvision>=0.17",
        "tqdm>=4.66",
        "umap-learn>=0.5.6",
    )
    .add_local_python_source("wafer_defect", copy=True)
    .add_local_dir("configs", remote_path=f"{REMOTE_PROJECT_ROOT}/configs", copy=True)
    .add_local_dir("scripts", remote_path=f"{REMOTE_PROJECT_ROOT}/scripts", copy=True)
    .add_local_dir(
        "experiments/anomaly_detection/patchcore/wideresnet50/x224/multilayer_umap",
        remote_path=(
            f"{REMOTE_PROJECT_ROOT}/experiments/anomaly_detection/patchcore"
            f"/wideresnet50/x224/multilayer_umap"
        ),
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

    def _download_tree(remote_dir: str) -> None:
        listing = _run_modal_cli(["volume", "ls", ARTIFACT_VOLUME_NAME, remote_dir, "--json"], capture_output=True)
        entries = json.loads(listing.stdout)
        for entry in entries:
            remote_name = str(entry["Filename"])
            if Path(remote_name).name == "processed":
                continue
            local_target = local_dir / Path(remote_name)
            entry_type = str(entry.get("Type", "")).lower()
            if entry_type == "dir":
                local_target.mkdir(parents=True, exist_ok=True)
                _download_tree(f"/{remote_name}")
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
        print(f"[patchcore-wrn50-x224-umap] reusing cached processed dataset: {metadata_path}", flush=True)
        return metadata_path
    if metadata_path.exists() or arrays_dir.exists():
        print("[patchcore-wrn50-x224-umap] cached processed dataset is stale; rebuilding", flush=True)
        _clear_processed_cache(metadata_path, arrays_dir)

    config_path = Path("/tmp/patchcore_wrn50_x224_umap_prepare_wm811k.toml")
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
    print(f"[patchcore-wrn50-x224-umap] preparing processed dataset: {' '.join(command)}", flush=True)
    subprocess.run(command, check=True, cwd=REMOTE_PROJECT_ROOT)
    processed_volume.commit()
    print("[patchcore-wrn50-x224-umap] processed dataset volume committed", flush=True)
    return metadata_path


def _write_remote_data_config(metadata_path: Path) -> Path:
    config_path = Path("/tmp/patchcore_wrn50_x224_umap_data_config.toml")
    config_path.write_text(
        "\n".join(
            [
                "[dataset]",
                'name = "wm811k"',
                f'raw_pickle = "{REMOTE_RAW_DIR}/LSWMD.pkl"',
                f'processed_root = "{REMOTE_PROCESSED_DIR}"',
                f'metadata_csv = "{REMOTE_PROCESSED_DIR}/metadata.csv"',
                f'metadata_50k_csv = "{REMOTE_PROCESSED_DIR}/metadata_50k.csv"',
                f'metadata_50k_5pct_csv = "{metadata_path.as_posix()}"',
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
    return config_path


def _core_artifacts_ready() -> bool:
    run_dir = Path(REMOTE_ARTIFACT_DIR) / REMOTE_RUN_SUBDIR
    results_dir = run_dir / "results"
    if not (run_dir / "run_manifest.json").exists():
        return False
    if not (results_dir / "patchcore_sweep_results.csv").exists():
        return False
    if not (results_dir / "selected_checkpoint.json").exists():
        return False
    for variant in ["topk_mb50k_r010_x224", "topk_mb50k_r005_x224"]:
        if not (run_dir / variant / "checkpoints" / "best_model.pt").exists():
            return False
        if not (run_dir / variant / "results" / "evaluation" / "val_scores.csv").exists():
            return False
        if not (run_dir / variant / "results" / "evaluation" / "test_scores.csv").exists():
            return False
    return True


def _umap_artifacts_ready() -> bool:
    run_dir = Path(REMOTE_ARTIFACT_DIR) / REMOTE_RUN_SUBDIR
    for variant in ["topk_mb50k_r010_x224", "topk_mb50k_r005_x224"]:
        umap_dir = run_dir / variant / "results" / "umap"
        required = [
            umap_dir / "train_embeddings.npy",
            umap_dir / "val_embeddings.npy",
            umap_dir / "test_embeddings.npy",
            umap_dir / "val_labels.npy",
            umap_dir / "test_labels.npy",
            umap_dir / "umap_points.csv",
            umap_dir / "umap_summary.json",
            umap_dir / "umap_by_split.png",
            umap_dir / "umap_by_score.png",
        ]
        if not all(path.exists() for path in required):
            return False
    return True


@app.function(
    gpu="A10G",
    timeout=60 * 60 * 10,
    volumes={
        REMOTE_RAW_DIR: raw_volume,
        REMOTE_PROCESSED_DIR: processed_volume,
        REMOTE_ARTIFACT_DIR: artifact_volume,
    },
)
def run_patchcore_remote(num_workers: int = 4, skip_umap: bool = False) -> dict[str, Any]:
    metadata_path = _prepare_processed_dataset()
    data_config_path = _write_remote_data_config(metadata_path)
    base_command = [
        "python",
        "-u",
        REMOTE_RUNNER,
        "--raw-pickle",
        f"{REMOTE_RAW_DIR}/LSWMD.pkl",
        "--data-config",
        str(data_config_path),
        "--train-config",
        (
            f"{REMOTE_PROJECT_ROOT}/experiments/anomaly_detection/patchcore"
            f"/wideresnet50/x224/multilayer_umap/train_config.toml"
        ),
        "--output-dir",
        "experiments/anomaly_detection/patchcore/wideresnet50/x224/multilayer_umap/artifacts/patchcore-wideresnet50-multilayer-umap",
        "--num-workers",
        str(num_workers),
    ]

    if _core_artifacts_ready():
        print("[patchcore-wrn50-x224-umap] reusing existing core artifacts and checkpoints", flush=True)
    else:
        train_command = [*base_command, "--skip-umap"]
        print(f"[patchcore-wrn50-x224-umap] launching core phase: {' '.join(train_command)}", flush=True)
        subprocess.run(train_command, check=True, cwd=REMOTE_PROJECT_ROOT)
        artifact_volume.commit()
        print("[patchcore-wrn50-x224-umap] artifact volume committed after core phase", flush=True)

    if not skip_umap:
        if _umap_artifacts_ready():
            print("[patchcore-wrn50-x224-umap] reusing existing UMAP artifacts", flush=True)
        else:
            umap_command = [*base_command, "--umap-only"]
            print(f"[patchcore-wrn50-x224-umap] launching UMAP phase: {' '.join(umap_command)}", flush=True)
            subprocess.run(umap_command, check=True, cwd=REMOTE_PROJECT_ROOT)
            artifact_volume.commit()
            print("[patchcore-wrn50-x224-umap] artifact volume committed after UMAP phase", flush=True)

    manifest_path = (
        Path(REMOTE_ARTIFACT_DIR) / REMOTE_RUN_SUBDIR / "run_manifest.json"
    )
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {"output_dir": REMOTE_ARTIFACT_DIR}


@app.local_entrypoint()
def main(num_workers: int = 4, sync_back: bool = True, skip_umap: bool = False) -> None:
    result = run_patchcore_remote.remote(num_workers=num_workers, skip_umap=skip_umap)
    print(json.dumps(result, indent=2))
    if sync_back:
        _download_artifacts(str(LOCAL_ARTIFACT_DIR), f"/{REMOTE_RUN_SUBDIR}")


@app.local_entrypoint()
def upload_raw_data(local_raw_pickle: str = str(LOCAL_RAW_PICKLE)) -> None:
    local_path = Path(local_raw_pickle).resolve()
    if not local_path.exists():
        raise FileNotFoundError(f"Raw pickle not found: {local_path}")
    _run_modal_cli(["volume", "put", RAW_VOLUME_NAME, str(local_path), "/LSWMD.pkl"])


@app.local_entrypoint()
def download_artifacts(local_artifact_dir: str = str(LOCAL_ARTIFACT_DIR)) -> None:
    _download_artifacts(local_artifact_dir, f"/{REMOTE_RUN_SUBDIR}")
