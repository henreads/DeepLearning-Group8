from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import modal


APP_NAME = "wafer-defect-supervised-sweep-vit-b16-x224"
RAW_VOLUME_NAME = "wafer-defect-lswmd-raw"
PROCESSED_VOLUME_NAME = "wafer-defect-wm811k-x224-processed"
ARTIFACT_VOLUME_NAME = "wafer-defect-supervised-sweep-vit-b16-x224-artifacts"


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
    / "anomaly_detection_defect"
    / "supervised_sweep"
    / "vit_b16"
    / "x224"
    / "main"
    / "artifacts"
)

REMOTE_PROJECT_ROOT = "/root/project"
REMOTE_RAW_DIR = f"{REMOTE_PROJECT_ROOT}/data/raw"
REMOTE_PROCESSED_DIR = f"{REMOTE_PROJECT_ROOT}/data/processed/x224/wm811k"
REMOTE_ARTIFACT_DIR = (
    f"{REMOTE_PROJECT_ROOT}/experiments/anomaly_detection_defect"
    f"/supervised_sweep/vit_b16/x224/main/artifacts"
)
REMOTE_RUNNER = f"{REMOTE_PROJECT_ROOT}/scripts/run_supervised_sweep_vit_b16_x224.py"
REMOTE_PREPARE_SCRIPT = f"{REMOTE_PROJECT_ROOT}/scripts/prepare_wm811k.py"
REMOTE_RUN_SUBDIR = "supervised_vit_sweep"


# Pre-download ViT weights during image build to avoid cold-start downloads
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.26",
        "pandas>=2.2",
        "scikit-learn>=1.5",
        "torch>=2.2",
        "torchvision>=0.17",
        "timm>=0.9",
        "tqdm>=4.66",
    )
    .run_commands(
        "python -c \""
        "import timm; "
        "timm.create_model('vit_base_patch16_224.augreg_in21k_ft_in1k', pretrained=True)"
        "\""
    )
    .add_local_python_source("wafer_defect", copy=True)
    .add_local_dir("configs", remote_path=f"{REMOTE_PROJECT_ROOT}/configs", copy=True)
    .add_local_dir("scripts", remote_path=f"{REMOTE_PROJECT_ROOT}/scripts", copy=True)
    .add_local_dir(
        "experiments/anomaly_detection_defect/supervised_sweep/vit_b16/x224/main",
        remote_path=(
            f"{REMOTE_PROJECT_ROOT}/experiments/anomaly_detection_defect"
            f"/supervised_sweep/vit_b16/x224/main"
        ),
        copy=True,
        ignore=["artifacts", "artifacts/**"],
    )
)

raw_volume = modal.Volume.from_name(RAW_VOLUME_NAME, create_if_missing=True)
processed_volume = modal.Volume.from_name(PROCESSED_VOLUME_NAME, create_if_missing=True)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=image)


def _run_modal_cli(
    args: list[str], *, capture_output: bool = False
) -> subprocess.CompletedProcess[str]:
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
        listing = _run_modal_cli(
            ["volume", "ls", ARTIFACT_VOLUME_NAME, remote_dir, "--json"],
            capture_output=True,
        )
        entries = json.loads(listing.stdout)
        for entry in entries:
            remote_name = str(entry["Filename"])
            local_target = local_dir / Path(remote_name)
            entry_type = str(entry.get("Type", "")).lower()
            if entry_type == "dir":
                local_target.mkdir(parents=True, exist_ok=True)
                _download_tree(f"/{remote_name}")
                continue
            local_target.parent.mkdir(parents=True, exist_ok=True)
            _run_modal_cli(
                ["volume", "get", ARTIFACT_VOLUME_NAME, f"/{remote_name}", str(local_target), "--force"]
            )

    _download_tree(remote_subdir)


def _prepare_processed_dataset() -> Path:
    """Ensure the x224 processed normal arrays exist with remote-compatible paths."""
    import csv

    metadata_path = Path(REMOTE_PROCESSED_DIR) / "metadata_50k_5pct.csv"
    arrays_dir = Path(REMOTE_PROCESSED_DIR) / "arrays_50k_5pct"

    # Quick validity check: first row array_path must be repo-relative and remote-loadable.
    if metadata_path.exists() and arrays_dir.exists():
        with metadata_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            first = next(reader, None)
        if first:
            sample = str(first.get("array_path", "")).strip()
            expected_prefix = "data/processed/x224/wm811k/"
            is_windows_path = ":\\" in sample or sample.startswith("C:/") or "\\" in sample
            if (
                not is_windows_path
                and sample.startswith(expected_prefix)
                and (Path(REMOTE_PROJECT_ROOT) / sample).exists()
            ):
                print(
                    f"[supervised-sweep] reusing cached processed dataset: {metadata_path}",
                    flush=True,
                )
                return metadata_path

    if metadata_path.exists() or arrays_dir.exists():
        print("[supervised-sweep] cached processed dataset is stale; rebuilding", flush=True)
        for candidate in metadata_path.parent.glob("metadata*.csv"):
            if candidate.exists():
                candidate.unlink()
        if arrays_dir.exists():
            shutil.rmtree(arrays_dir)

    print("[supervised-sweep] rebuilding processed dataset ...", flush=True)
    config_path = Path("/tmp/supervised_sweep_prepare_wm811k.toml")
    config_path.write_text(
        "\n".join([
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
        ]),
        encoding="utf-8",
    )
    subprocess.run(
        ["python", "-u", REMOTE_PREPARE_SCRIPT, "--config", str(config_path)],
        check=True,
        cwd=REMOTE_PROJECT_ROOT,
    )
    processed_volume.commit()
    return metadata_path


def _artifacts_ready() -> bool:
    run_dir = Path(REMOTE_ARTIFACT_DIR) / REMOTE_RUN_SUBDIR
    summary = run_dir / "results" / "sweep_summary.json"
    return summary.exists()


@app.function(
    gpu="A10G",
    timeout=60 * 60 * 6,
    volumes={
        REMOTE_RAW_DIR: raw_volume,
        REMOTE_PROCESSED_DIR: processed_volume,
        REMOTE_ARTIFACT_DIR: artifact_volume,
    },
)
def run_sweep_remote(num_workers: int = 4) -> dict[str, Any]:
    if _artifacts_ready():
        print("[supervised-sweep] reusing existing artifacts", flush=True)
    else:
        metadata_path = _prepare_processed_dataset()

        command = [
            "python", "-u", REMOTE_RUNNER,
            "--config",
            (
                f"{REMOTE_PROJECT_ROOT}/experiments/anomaly_detection_defect"
                f"/supervised_sweep/vit_b16/x224/main/train_config.toml"
            ),
            "--raw-pickle", f"{REMOTE_RAW_DIR}/LSWMD.pkl",
            "--normal-metadata-csv", str(metadata_path),
            "--output-dir",
            f"{REMOTE_ARTIFACT_DIR}/{REMOTE_RUN_SUBDIR}",
        ]
        print(f"[supervised-sweep] running: {' '.join(command)}", flush=True)
        proc = subprocess.Popen(command, cwd=REMOTE_PROJECT_ROOT)
        try:
            while True:
                try:
                    proc.wait(timeout=300)   # poll every 5 min
                    break
                except subprocess.TimeoutExpired:
                    artifact_volume.commit()
                    print("[supervised-sweep] intermediate volume commit", flush=True)
        finally:
            if proc.poll() is None:
                proc.wait()
            artifact_volume.commit()
            print("[supervised-sweep] artifact volume committed", flush=True)
            if proc.returncode not in (0, None):
                raise subprocess.CalledProcessError(proc.returncode, command)

    summary_path = Path(REMOTE_ARTIFACT_DIR) / REMOTE_RUN_SUBDIR / "results" / "sweep_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return {"output_dir": REMOTE_ARTIFACT_DIR}


@app.local_entrypoint()
def main(num_workers: int = 4, sync_back: bool = True) -> None:
    result = run_sweep_remote.remote(num_workers=num_workers)
    print(json.dumps(result, indent=2))
    if sync_back:
        _download_artifacts(str(LOCAL_ARTIFACT_DIR), f"/{REMOTE_RUN_SUBDIR}")


@app.local_entrypoint()
def download_artifacts(local_artifact_dir: str = str(LOCAL_ARTIFACT_DIR)) -> None:
    _download_artifacts(local_artifact_dir, f"/{REMOTE_RUN_SUBDIR}")
