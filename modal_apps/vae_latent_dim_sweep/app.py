from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import modal


APP_NAME = "wafer-defect-vae-latent-dim-sweep"
RAW_VOLUME_NAME = "wafer-defect-lswmd-raw"
PROCESSED_VOLUME_NAME = "wafer-defect-wm811k-x64-processed"
ARTIFACT_VOLUME_NAME = "wafer-defect-vae-latent-dim-sweep-artifacts"


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
    LOCAL_REPO_ROOT / "experiments" / "anomaly_detection" / "vae" / "x64" / "latent_dim_sweep" / "artifacts"
)

REMOTE_PROJECT_ROOT = "/root/project"
REMOTE_RAW_DIR = f"{REMOTE_PROJECT_ROOT}/data/raw"
REMOTE_PROCESSED_DIR = f"{REMOTE_PROJECT_ROOT}/data/processed/x64/wm811k"
REMOTE_ARTIFACT_DIR = (
    f"{REMOTE_PROJECT_ROOT}/experiments/anomaly_detection/vae/x64/latent_dim_sweep/artifacts"
)
REMOTE_RUNNER = f"{REMOTE_PROJECT_ROOT}/scripts/run_vae_latent_dim_sweep_notebook.py"
REMOTE_PREPARE_SCRIPT = f"{REMOTE_PROJECT_ROOT}/scripts/prepare_wm811k.py"

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
    )
    .add_local_python_source("wafer_defect", copy=True)
    .add_local_dir("configs", remote_path=f"{REMOTE_PROJECT_ROOT}/configs", copy=True)
    .add_local_dir("scripts", remote_path=f"{REMOTE_PROJECT_ROOT}/scripts", copy=True)
    .add_local_dir(
        "experiments/anomaly_detection/vae/x64/latent_dim_sweep",
        remote_path=f"{REMOTE_PROJECT_ROOT}/experiments/anomaly_detection/vae/x64/latent_dim_sweep",
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


def _cached_dataset_is_valid(metadata_path: Path, arrays_dir: Path) -> bool:
    if not metadata_path.exists() or not arrays_dir.exists():
        return False

    expected_prefix = f"data/processed/x64/wm811k/{arrays_dir.name}/"
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


def _prepare_processed_dataset() -> None:
    metadata_path = Path(REMOTE_PROCESSED_DIR) / "metadata_50k_5pct.csv"
    arrays_dir = Path(REMOTE_PROCESSED_DIR) / "arrays_50k_5pct"
    if _cached_dataset_is_valid(metadata_path, arrays_dir):
        print(
            f"[vae-latent-dim-sweep] reusing cached processed dataset: {metadata_path}",
            flush=True,
        )
        return
    if metadata_path.exists() or arrays_dir.exists():
        print("[vae-latent-dim-sweep] cached processed dataset is stale; rebuilding", flush=True)
        _clear_processed_cache(metadata_path, arrays_dir)

    config_path = Path("/tmp/vae_latent_dim_sweep_prepare_wm811k.toml")
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
                "image_size = 64",
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
    command = [
        "python",
        "-u",
        REMOTE_PREPARE_SCRIPT,
        "--config",
        str(config_path),
    ]
    print(f"[vae-latent-dim-sweep] preparing processed dataset: {' '.join(command)}", flush=True)
    subprocess.run(command, check=True, cwd=REMOTE_PROJECT_ROOT)
    processed_volume.commit()
    print("[vae-latent-dim-sweep] processed dataset volume committed", flush=True)


@app.function(
    gpu="A10G",
    timeout=60 * 60 * 12,
    volumes={
        REMOTE_RAW_DIR: raw_volume,
        REMOTE_PROCESSED_DIR: processed_volume,
        REMOTE_ARTIFACT_DIR: artifact_volume,
    },
)
def run_vae_remote(retrain: bool = False, full_retrain: bool = False) -> dict[str, Any]:
    _prepare_processed_dataset()
    manifests: dict[str, Any] = {}
    for phase in ["train", "eval", "sweep"]:
        command = [
            "python",
            "-u",
            REMOTE_RUNNER,
            "--output-dir",
            "experiments/anomaly_detection/vae/x64/latent_dim_sweep/artifacts/vae_latent_dim_sweep",
            "--phase",
            phase,
        ]
        if retrain:
            command.append("--retrain")
        if full_retrain:
            command.append("--full-retrain")
        print(f"[vae-latent-dim-sweep] launching phase={phase}: {' '.join(command)}", flush=True)
        subprocess.run(command, check=True, cwd=REMOTE_PROJECT_ROOT)
        artifact_volume.commit()
        print(f"[vae-latent-dim-sweep] artifact volume committed after phase={phase}", flush=True)
        manifest_path = Path(REMOTE_ARTIFACT_DIR) / "vae_latent_dim_sweep" / f"{phase}_phase_manifest.json"
        if manifest_path.exists():
            manifests[phase] = json.loads(manifest_path.read_text(encoding="utf-8"))
    final_manifest = Path(REMOTE_ARTIFACT_DIR) / "vae_latent_dim_sweep" / "run_manifest.json"
    if final_manifest.exists():
        manifests["final"] = json.loads(final_manifest.read_text(encoding="utf-8"))
    return manifests


@app.local_entrypoint()
def main(sync_back: bool = True, retrain: bool = False, full_retrain: bool = False) -> None:
    result = run_vae_remote.remote(retrain=retrain, full_retrain=full_retrain)
    print(json.dumps(result, indent=2))
    if sync_back:
        _download_artifacts(str(LOCAL_ARTIFACT_DIR), "/vae_latent_dim_sweep")


@app.local_entrypoint()
def upload_raw_data(local_raw_pickle: str = str(LOCAL_RAW_PICKLE)) -> None:
    local_path = Path(local_raw_pickle).resolve()
    if not local_path.exists():
        raise FileNotFoundError(f"Raw pickle not found: {local_path}")
    _run_modal_cli(["volume", "put", RAW_VOLUME_NAME, str(local_path), "/LSWMD.pkl"])


@app.local_entrypoint()
def download_artifacts(local_artifact_dir: str = str(LOCAL_ARTIFACT_DIR)) -> None:
    _download_artifacts(local_artifact_dir, "/vae_latent_dim_sweep")
