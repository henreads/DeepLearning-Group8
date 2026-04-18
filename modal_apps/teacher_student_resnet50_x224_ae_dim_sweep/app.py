from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import modal


APP_NAME = "wafer-defect-ts-resnet50-x224-ae-dim-sweep"
RAW_VOLUME_NAME = "wafer-defect-lswmd-raw"
PROCESSED_VOLUME_NAME = "wafer-defect-wm811k-x224-processed"
ARTIFACT_VOLUME_NAME = "wafer-defect-ts-resnet50-x224-ae-dim-sweep-artifacts"
FULL_AE_HIDDEN_DIMS = [64, 128, 256, 512, 768]
QUICK_AE_HIDDEN_DIMS = [64, 128, 256]
QUICK_SWEEP_EPOCHS = 10
FULL_SWEEP_EPOCHS = 30


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
    / "resnet50"
    / "x224"
    / "feature_autoencoder_dim_sweep"
    / "artifacts"
)

REMOTE_PROJECT_ROOT = "/root/project"
REMOTE_RAW_DIR = f"{REMOTE_PROJECT_ROOT}/data/raw"
REMOTE_PROCESSED_DIR = f"{REMOTE_PROJECT_ROOT}/data/processed/x224/wm811k"
REMOTE_EXPERIMENT_DIR = (
    f"{REMOTE_PROJECT_ROOT}/experiments/anomaly_detection/teacher_student/resnet50/x224/feature_autoencoder_dim_sweep"
)
REMOTE_ARTIFACT_DIR = f"{REMOTE_EXPERIMENT_DIR}/artifacts"
REMOTE_SWEEP_ARTIFACT_DIR = f"{REMOTE_ARTIFACT_DIR}/ts_resnet50_x224_ae_dim_sweep"
REMOTE_CONFIG_PATH = f"{REMOTE_EXPERIMENT_DIR}/train_config.toml"
REMOTE_PREPARE_SCRIPT = f"{REMOTE_PROJECT_ROOT}/scripts/prepare_wm811k.py"
REMOTE_TRAIN_SCRIPT = f"{REMOTE_PROJECT_ROOT}/scripts/train_ts_distillation.py"
REMOTE_EVAL_SCRIPT = f"{REMOTE_PROJECT_ROOT}/scripts/evaluate_reconstruction_model.py"

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
        "timm>=1.0",
    )
    .add_local_python_source("wafer_defect", copy=True)
    .add_local_dir("configs", remote_path=f"{REMOTE_PROJECT_ROOT}/configs", copy=True)
    .add_local_dir("scripts", remote_path=f"{REMOTE_PROJECT_ROOT}/scripts", copy=True)
    .add_local_dir(
        "experiments/anomaly_detection/teacher_student/resnet50/x224/feature_autoencoder_dim_sweep",
        remote_path=REMOTE_EXPERIMENT_DIR,
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


def _prepare_processed_dataset() -> None:
    metadata_path = Path(REMOTE_PROCESSED_DIR) / "metadata_50k_5pct.csv"
    arrays_dir = Path(REMOTE_PROCESSED_DIR) / "arrays_50k_5pct"
    if _cached_dataset_is_valid(metadata_path, arrays_dir):
        print(f"[ts-resnet50-x224-ae-dim-sweep] reusing cached processed dataset: {metadata_path}", flush=True)
        return
    if metadata_path.exists() or arrays_dir.exists():
        print("[ts-resnet50-x224-ae-dim-sweep] cached processed dataset is stale; rebuilding", flush=True)
        _clear_processed_cache(metadata_path, arrays_dir)

    config_path = Path("/tmp/ts_resnet50_x224_ae_dim_sweep_prepare_wm811k.toml")
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
    subprocess.run(
        ["python", "-u", REMOTE_PREPARE_SCRIPT, "--config", str(config_path)],
        check=True,
        cwd=REMOTE_PROJECT_ROOT,
    )
    processed_volume.commit()
    print("[ts-resnet50-x224-ae-dim-sweep] processed dataset volume committed", flush=True)


def _format_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_format_toml_value(item) for item in value) + "]"
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _dump_toml(config_dict: dict[str, Any]) -> str:
    lines: list[str] = []
    for section, values in config_dict.items():
        lines.append(f"[{section}]")
        for key, value in values.items():
            lines.append(f"{key} = {_format_toml_value(value)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


@app.function(
    gpu="A10G",
    timeout=60 * 60 * 12,
    volumes={
        REMOTE_RAW_DIR: raw_volume,
        REMOTE_PROCESSED_DIR: processed_volume,
        REMOTE_ARTIFACT_DIR: artifact_volume,
    },
)
def run_dimension_sweep(fresh_train: bool = False, quick: bool = True, epochs: int | None = None) -> dict[str, Any]:
    import pandas as pd
    from wafer_defect.config import load_toml

    def normalize_training_artifacts(artifact_dir: Path) -> None:
        checkpoint_dir = artifact_dir / "checkpoints"
        results_dir = artifact_dir / "results"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        for name in ["best_model.pt", "last_model.pt", "latest_checkpoint.pt", "final_model.pt"]:
            src = artifact_dir / name
            if src.exists():
                shutil.move(str(src), str(checkpoint_dir / name))
        for checkpoint_path in artifact_dir.glob("checkpoint_epoch_*.pt"):
            shutil.move(str(checkpoint_path), str(checkpoint_dir / checkpoint_path.name))
        for name in ["history.json", "summary.json"]:
            src = artifact_dir / name
            if src.exists():
                shutil.move(str(src), str(results_dir / name))

    def copy_configs(artifact_dir: Path, effective_config_path: Path) -> None:
        config_dir = artifact_dir / "results" / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        source_dir = Path(REMOTE_EXPERIMENT_DIR)
        for config_name in ["train_config.toml"]:
            source_path = source_dir / config_name
            if source_path.exists():
                (config_dir / config_name).write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
        (config_dir / "train_config_effective.toml").write_text(
            effective_config_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    _prepare_processed_dataset()
    sweep_root = Path(REMOTE_SWEEP_ARTIFACT_DIR)
    sweep_root.mkdir(parents=True, exist_ok=True)

    ae_hidden_dims = QUICK_AE_HIDDEN_DIMS if quick else FULL_AE_HIDDEN_DIMS
    rows: list[dict[str, Any]] = []
    for ae_hidden_dim in ae_hidden_dims:
        dim_name = f"ae_dim_{ae_hidden_dim}"
        dim_artifact_dir = sweep_root / dim_name
        dim_checkpoint_dir = dim_artifact_dir / "checkpoints"
        dim_results_dir = dim_artifact_dir / "results"
        dim_evaluation_dir = dim_results_dir / "evaluation"
        dim_summary_path = dim_evaluation_dir / "summary.json"
        dim_checkpoint_path = dim_checkpoint_dir / "best_model.pt"
        latest_checkpoint_path = dim_checkpoint_dir / "latest_checkpoint.pt"

        if fresh_train and dim_artifact_dir.exists():
            shutil.rmtree(dim_artifact_dir)

        dim_evaluation_dir.mkdir(parents=True, exist_ok=True)
        (dim_artifact_dir / "plots").mkdir(parents=True, exist_ok=True)

        base_config = load_toml(REMOTE_CONFIG_PATH)
        base_config["model"]["feature_autoencoder_hidden_dim"] = ae_hidden_dim
        base_config["run"]["output_dir"] = dim_artifact_dir.as_posix()
        base_config["data"]["batch_size"] = 512
        base_config["data"]["num_workers"] = 4
        base_config["training"]["epochs"] = int(
            epochs if epochs is not None else (QUICK_SWEEP_EPOCHS if quick else FULL_SWEEP_EPOCHS)
        )
        if latest_checkpoint_path.exists() and not fresh_train:
            base_config["training"]["resume_from"] = latest_checkpoint_path.as_posix()
        else:
            base_config["training"].pop("resume_from", None)

        effective_config_path = Path(f"/tmp/ts_resnet50_x224_ae_dim_{ae_hidden_dim}.toml")
        effective_config_path.write_text(_dump_toml(base_config), encoding="utf-8")
        copy_configs(dim_artifact_dir, effective_config_path)

        print(f"[ts-resnet50-x224-ae-dim-sweep] processing {dim_name}", flush=True)

        if not dim_checkpoint_path.exists():
            subprocess.run(
                ["python", "-u", REMOTE_TRAIN_SCRIPT, "--config", str(effective_config_path)],
                check=True,
                cwd=REMOTE_PROJECT_ROOT,
            )
            normalize_training_artifacts(dim_artifact_dir)
            copy_configs(dim_artifact_dir, effective_config_path)
            artifact_volume.commit()
            print(f"[ts-resnet50-x224-ae-dim-sweep] artifact volume committed after train for {dim_name}", flush=True)

        if not dim_checkpoint_path.exists():
            print(f"[ts-resnet50-x224-ae-dim-sweep] skipping {dim_name}; checkpoint missing after training", flush=True)
            continue

        if not dim_summary_path.exists():
            subprocess.run(
                [
                    "python",
                    "-u",
                    REMOTE_EVAL_SCRIPT,
                    "--checkpoint",
                    str(dim_checkpoint_path),
                    "--config",
                    str(effective_config_path),
                    "--model-type",
                    "ts_distillation",
                    "--output-dir",
                    str(dim_evaluation_dir),
                ],
                check=True,
                cwd=REMOTE_PROJECT_ROOT,
            )
            artifact_volume.commit()
            print(f"[ts-resnet50-x224-ae-dim-sweep] artifact volume committed after eval for {dim_name}", flush=True)

        if dim_summary_path.exists():
            summary = json.loads(dim_summary_path.read_text(encoding="utf-8"))
            metrics = summary.get("metrics_at_validation_threshold", {})
            rows.append(
                {
                    "ae_hidden_dim": ae_hidden_dim,
                    "f1": metrics.get("f1"),
                    "auroc": metrics.get("auroc"),
                    "auprc": metrics.get("auprc"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                }
            )

    summary_path = sweep_root / "ae_dimension_sweep_summary.csv"
    best_result: dict[str, Any] | None = None
    if rows:
        sweep_df = pd.DataFrame(rows).sort_values(["f1", "auprc", "auroc"], ascending=False).reset_index(drop=True)
        sweep_df.to_csv(summary_path, index=False)
        artifact_volume.commit()
        best_result = sweep_df.iloc[0].to_dict()

    return {
        "artifact_dir": REMOTE_SWEEP_ARTIFACT_DIR,
        "quick_mode": quick,
        "epochs": int(epochs if epochs is not None else QUICK_SWEEP_EPOCHS if quick else FULL_SWEEP_EPOCHS),
        "dimensions_requested": ae_hidden_dims,
        "dimensions_completed": [row["ae_hidden_dim"] for row in rows],
        "summary_csv": summary_path.as_posix() if summary_path.exists() else None,
        "best_result": best_result,
    }


@app.local_entrypoint()
def main(sync_back: bool = True, fresh_train: bool = False, quick: bool = True, epochs: int | None = None) -> None:
    result = run_dimension_sweep.remote(fresh_train=fresh_train, quick=quick, epochs=epochs)
    print(json.dumps(result, indent=2))
    if sync_back:
        _download_artifacts(str(LOCAL_ARTIFACT_DIR), "/ts_resnet50_x224_ae_dim_sweep")


@app.local_entrypoint()
def upload_raw_data(local_raw_pickle: str = str(LOCAL_RAW_PICKLE)) -> None:
    local_path = Path(local_raw_pickle).resolve()
    if not local_path.exists():
        raise FileNotFoundError(f"Raw pickle not found: {local_path}")
    _run_modal_cli(["volume", "put", RAW_VOLUME_NAME, str(local_path), "/LSWMD.pkl"])


@app.local_entrypoint()
def download_artifacts(local_artifact_dir: str = str(LOCAL_ARTIFACT_DIR)) -> None:
    _download_artifacts(local_artifact_dir, "/ts_resnet50_x224_ae_dim_sweep")
