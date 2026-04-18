from __future__ import annotations

import json
import csv
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import modal


APP_NAME = "wafer-defect-ts-resnet50-x64-layer-ablation"
RAW_VOLUME_NAME = "wafer-defect-lswmd-raw"
PROCESSED_VOLUME_NAME = "wafer-defect-wm811k-x64-processed"
ARTIFACT_VOLUME_NAME = "wafer-defect-ts-resnet50-x64-layer-ablation-artifacts"

FULL_VARIANT_SPECS: list[dict[str, Any]] = [
    {"name": "ts_resnet50_layer2_topk20_sw2p0_aw1p0", "teacher_layer": "layer2", "topk_ratio": 0.2, "score_student_weight": 2.0, "score_autoencoder_weight": 1.0},
    {"name": "ts_resnet50_layer1_topk10_sw1p0_aw0p5", "teacher_layer": "layer1", "topk_ratio": 0.1, "score_student_weight": 1.0, "score_autoencoder_weight": 0.5},
    {"name": "ts_resnet50_layer1_topk10_sw2p0_aw1p0", "teacher_layer": "layer1", "topk_ratio": 0.1, "score_student_weight": 2.0, "score_autoencoder_weight": 1.0},
    {"name": "ts_resnet50_layer1_topk10_sw2p0_aw0p5", "teacher_layer": "layer1", "topk_ratio": 0.1, "score_student_weight": 2.0, "score_autoencoder_weight": 0.5},
    {"name": "ts_resnet50_layer1_topk20_sw2p0_aw1p0", "teacher_layer": "layer1", "topk_ratio": 0.2, "score_student_weight": 2.0, "score_autoencoder_weight": 1.0},
    {"name": "ts_resnet50_layer1_topk20_sw1p0_aw0p5", "teacher_layer": "layer1", "topk_ratio": 0.2, "score_student_weight": 1.0, "score_autoencoder_weight": 0.5},
]
QUICK_VARIANT_SPECS: list[dict[str, Any]] = [
    FULL_VARIANT_SPECS[0],
    FULL_VARIANT_SPECS[2],
    FULL_VARIANT_SPECS[4],
]

THRESHOLD_QUANTILE = 0.95
MODAL_BATCH_SIZE = 512
MODAL_NUM_WORKERS = 4
QUICK_SWEEP_EPOCHS = 10
FULL_SWEEP_EPOCHS = 30
LAYER1_MODAL_BATCH_SIZE = 128


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
    / "x64"
    / "layer_ablation"
    / "artifacts"
)

REMOTE_PROJECT_ROOT = "/root/project"
REMOTE_RAW_DIR = f"{REMOTE_PROJECT_ROOT}/data/raw"
REMOTE_PROCESSED_DIR = f"{REMOTE_PROJECT_ROOT}/data/processed/x64/wm811k"
REMOTE_EXPERIMENT_DIR = (
    f"{REMOTE_PROJECT_ROOT}/experiments/anomaly_detection/teacher_student/resnet50/x64/layer_ablation"
)
REMOTE_ARTIFACT_DIR = f"{REMOTE_EXPERIMENT_DIR}/artifacts"
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
        "experiments/anomaly_detection/teacher_student/resnet50/x64/layer_ablation",
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


def _download_artifacts(local_artifact_dir: str) -> None:
    local_dir = Path(local_artifact_dir).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    listing = _run_modal_cli(["volume", "ls", ARTIFACT_VOLUME_NAME, "/", "--json"], capture_output=True)
    entries = json.loads(listing.stdout)
    for entry in entries:
        remote_name = str(entry["Filename"])
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
        print(f"[ts-resnet50-x64-layer-ablation] reusing cached processed dataset: {metadata_path}", flush=True)
        return
    if metadata_path.exists() or arrays_dir.exists():
        print("[ts-resnet50-x64-layer-ablation] cached processed dataset is stale; rebuilding", flush=True)
        _clear_processed_cache(metadata_path, arrays_dir)

    config_path = Path("/tmp/ts_resnet50_x64_layer_ablation_prepare_wm811k.toml")
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
        ]),
        encoding="utf-8",
    )
    subprocess.run(
        ["python", "-u", REMOTE_PREPARE_SCRIPT, "--config", str(config_path)],
        check=True,
        cwd=REMOTE_PROJECT_ROOT,
    )
    processed_volume.commit()
    print("[ts-resnet50-x64-layer-ablation] processed dataset volume committed", flush=True)


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
def run_layer_ablation(fresh_train: bool = False, quick: bool = True, epochs: int | None = None) -> dict[str, Any]:
    import pandas as pd
    from wafer_defect.config import load_toml

    _prepare_processed_dataset()

    artifact_root = Path(REMOTE_ARTIFACT_DIR)
    results_dir = artifact_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    generated_config_dir = artifact_root / "generated_configs"
    generated_config_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_toml(REMOTE_CONFIG_PATH)
    base_config["data"]["batch_size"] = MODAL_BATCH_SIZE
    base_config["data"]["num_workers"] = MODAL_NUM_WORKERS

    variant_specs = QUICK_VARIANT_SPECS if quick else FULL_VARIANT_SPECS
    effective_epochs = int(epochs if epochs is not None else (QUICK_SWEEP_EPOCHS if quick else FULL_SWEEP_EPOCHS))
    rows: list[dict[str, Any]] = []

    for spec in variant_specs:
        variant_name = spec["name"]
        variant_dir = artifact_root / variant_name
        checkpoint_path = variant_dir / "best_model.pt"
        evaluation_dir = variant_dir / "evaluation"
        summary_path = evaluation_dir / "summary.json"

        if fresh_train and variant_dir.exists():
            shutil.rmtree(variant_dir)

        variant_dir.mkdir(parents=True, exist_ok=True)
        evaluation_dir.mkdir(parents=True, exist_ok=True)

        variant_config = {**{k: dict(v) for k, v in base_config.items()}}
        variant_batch_size = LAYER1_MODAL_BATCH_SIZE if spec["teacher_layer"] == "layer1" else MODAL_BATCH_SIZE
        variant_config["run"]["output_dir"] = variant_dir.as_posix()
        variant_config["data"]["batch_size"] = variant_batch_size
        variant_config["model"]["teacher_layer"] = spec["teacher_layer"]
        variant_config["model"]["topk_ratio"] = float(spec["topk_ratio"])
        variant_config["model"]["score_student_weight"] = float(spec["score_student_weight"])
        variant_config["model"]["score_autoencoder_weight"] = float(spec["score_autoencoder_weight"])
        variant_config["training"]["epochs"] = effective_epochs
        variant_config["training"]["resume_from"] = ""

        effective_config_path = generated_config_dir / f"{variant_name}.toml"
        effective_config_path.write_text(_dump_toml(variant_config), encoding="utf-8")

        print(f"[ts-resnet50-x64-layer-ablation] processing variant: {variant_name}", flush=True)

        if not checkpoint_path.exists():
            subprocess.run(
                ["python", "-u", REMOTE_TRAIN_SCRIPT, "--config", str(effective_config_path)],
                check=True,
                cwd=REMOTE_PROJECT_ROOT,
            )
            artifact_volume.commit()
            print(f"[ts-resnet50-x64-layer-ablation] committed after train: {variant_name}", flush=True)

        if not checkpoint_path.exists():
            print(f"[ts-resnet50-x64-layer-ablation] skipping {variant_name}; checkpoint missing after training", flush=True)
            continue

        if not summary_path.exists():
            subprocess.run(
                [
                    "python", "-u", REMOTE_EVAL_SCRIPT,
                    "--checkpoint", str(checkpoint_path),
                    "--config", str(effective_config_path),
                    "--model-type", "ts_distillation",
                    "--threshold-quantile", str(THRESHOLD_QUANTILE),
                    "--output-dir", str(evaluation_dir),
                ],
                check=True,
                cwd=REMOTE_PROJECT_ROOT,
            )
            artifact_volume.commit()
            print(f"[ts-resnet50-x64-layer-ablation] committed after eval: {variant_name}", flush=True)

        if summary_path.exists():
            evaluation_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            metrics = evaluation_summary["metrics_at_validation_threshold"]
            best_sweep = evaluation_summary["best_threshold_sweep"]
            rows.append({
                "variant": variant_name,
                "source": "modal_rerun",
                "teacher_layer": spec["teacher_layer"],
                "batch_size": variant_batch_size,
                "topk_ratio": float(spec["topk_ratio"]),
                "score_student_weight": float(spec["score_student_weight"]),
                "score_autoencoder_weight": float(spec["score_autoencoder_weight"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "auroc": float(metrics["auroc"]),
                "auprc": float(metrics["auprc"]),
                "best_sweep_f1": float(best_sweep["f1"]),
                "threshold": float(metrics["threshold"]),
            })

    local_summary_path = results_dir / "local_variant_summary.csv"
    best_result: dict[str, Any] | None = None
    if rows:
        summary_df = pd.DataFrame(rows).sort_values(["f1", "auroc"], ascending=False).reset_index(drop=True)
        summary_df.to_csv(local_summary_path, index=False)
        artifact_volume.commit()
        best_result = summary_df.iloc[0].to_dict()

    return {
        "artifact_dir": REMOTE_ARTIFACT_DIR,
        "quick_mode": quick,
        "epochs": effective_epochs,
        "variants_requested": [s["name"] for s in variant_specs],
        "variants_completed": [row["variant"] for row in rows],
        "local_summary_csv": local_summary_path.as_posix() if local_summary_path.exists() else None,
        "best_result": best_result,
    }


@app.local_entrypoint()
def main(sync_back: bool = True, fresh_train: bool = False, quick: bool = True, epochs: int | None = None) -> None:
    result = run_layer_ablation.remote(fresh_train=fresh_train, quick=quick, epochs=epochs)
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
