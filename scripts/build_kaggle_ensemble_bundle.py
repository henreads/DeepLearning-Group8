"""Build a Kaggle-ready deployment bundle for the multiclass classifier ensemble."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


README_TEMPLATE = """Ensemble ABC Kaggle Deployment Bundle

This bundle is prepared for Kaggle release and inference deployment of the WM-811K multiclass classifier ensemble.

What is included

- `scripts/predict_unlabeled_multiclass_ensemble.py` for direct ensemble inference
- `scripts/predict_unlabeled_multiclass.py` for single-checkpoint inference
- `scripts/ensemble_multiclass_classifier.py` for ensemble evaluation plus optional unlabeled inference
- optional `ensemble_combiner.json` for applying a saved stacking combiner at inference time
- output exports for `unlabeled_predictions.csv`, `unlabeled_predictions.defect_candidates.csv`, and `unlabeled_predictions.accepted_pseudo_labels.csv`
- `src/wafer_defect/...` runtime code required by the scripts
- `configs/data/data_multiclass_50k.toml`
- `Outputs/model_a`, `Outputs/model_b`, and `Outputs/model_c` with the released checkpoints
- `Outputs/ensemble_abc/metrics.json` with the test-set ensemble result
- `ensemble_manifest.json` listing the released checkpoint files
- `dataset-metadata.template.json` for Kaggle CLI upload setup

Released ensemble result

- test accuracy: `{test_accuracy:.4f}`
- test balanced accuracy: `{test_balanced_accuracy:.4f}`

Recommended Kaggle setup

1. Create one Kaggle Dataset for this release bundle.
2. Create or mount a separate Kaggle Dataset that contains `LSWMD.pkl`.
3. Copy this bundle from `/kaggle/input/...` to `/kaggle/working/...` before running inference.
4. Set `PYTHONPATH` to the copied `src` directory.
5. Pass the mounted `LSWMD.pkl` path through `--raw-pickle` so you do not need to edit the config file.

Recommended first notebook cell

```python
from pathlib import Path
import os
import shutil
import sys

def resolve_unique_input_path(pattern: str, description: str) -> Path:
    matches = sorted(Path("/kaggle/input").glob(pattern))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Could not find {{description}} with pattern '/kaggle/input/{{pattern}}'. "
            "Attach the Kaggle dataset first or set SOURCE manually."
        )
    raise RuntimeError(
        f"Found multiple {{description}} matches: {{matches}}. "
        "Set SOURCE manually to the correct mounted path."
    )

SOURCE = None  # Optional manual override: Path("/kaggle/input/<dataset>/{bundle_dir_name}")
if SOURCE is None:
    SOURCE = resolve_unique_input_path("*/{bundle_dir_name}", "release bundle")
else:
    SOURCE = Path(SOURCE)
TARGET = Path("/kaggle/working/{bundle_dir_name}")

if TARGET.exists():
    shutil.rmtree(TARGET)
shutil.copytree(SOURCE, TARGET)
os.chdir(TARGET)

SRC_ROOT = TARGET / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

print("Bundle source:", SOURCE)
print("Running from:", TARGET)
```

Recommended configuration cell

```python
from pathlib import Path

def resolve_unique_input_path(pattern: str, description: str) -> Path:
    matches = sorted(Path("/kaggle/input").glob(pattern))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Could not find {{description}} with pattern '/kaggle/input/{{pattern}}'. "
            "Attach the Kaggle dataset first or set RAW_PICKLE manually."
        )
    raise RuntimeError(
        f"Found multiple {{description}} matches: {{matches}}. "
        "Set RAW_PICKLE manually to the correct mounted path."
    )

REPO_ROOT = Path("/kaggle/working/{bundle_dir_name}")
RAW_PICKLE = None  # Optional manual override: Path("/kaggle/input/<dataset>/LSWMD.pkl")
if RAW_PICKLE is None:
    RAW_PICKLE = resolve_unique_input_path("*/LSWMD.pkl", "raw WM-811K pickle")
else:
    RAW_PICKLE = Path(RAW_PICKLE)

OUTPUT_DIR = REPO_ROOT / "artifacts/ensemble_abc"
OUTPUT_CSV = OUTPUT_DIR / "unlabeled_predictions.csv"
DEFECT_CSV = OUTPUT_DIR / "unlabeled_predictions.defect_candidates.csv"
ACCEPTED_CSV = OUTPUT_DIR / "unlabeled_predictions.accepted_pseudo_labels.csv"
SUMMARY_JSON = OUTPUT_DIR / "unlabeled_predictions.summary.json"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COMBINER_JSON = REPO_ROOT / "ensemble_combiner.json"
if not COMBINER_JSON.exists():
    COMBINER_JSON = None

print("Raw pickle:", RAW_PICKLE)
```

Recommended inference command

```python
import os
import subprocess
import sys

command = [
    sys.executable,
    str(REPO_ROOT / "scripts/predict_unlabeled_multiclass_ensemble.py"),
    "--manifest",
    str(REPO_ROOT / "ensemble_manifest.json"),
    "--config",
    str(REPO_ROOT / "configs/data/data_multiclass_50k.toml"),
    "--raw-pickle",
    str(RAW_PICKLE),
    "--output-csv",
    str(OUTPUT_CSV),
    "--defect-csv",
    str(DEFECT_CSV),
    "--accepted-csv",
    str(ACCEPTED_CSV),
    "--summary-json",
    str(SUMMARY_JSON),
    "--batch-size",
    "256",
    "--min-confidence",
    "0.98",
]
if COMBINER_JSON is not None:
    command.extend(["--combiner-json", str(COMBINER_JSON)])

env = os.environ.copy()
env["PYTHONPATH"] = str(REPO_ROOT / "src")
subprocess.run(command, check=True, cwd=REPO_ROOT, env=env)
```

Notes

- Run from `/kaggle/working`, not `/kaggle/input`, because the scripts write outputs.
- The raw data path is intentionally supplied via `--raw-pickle` for Kaggle portability.
- The default `num_workers` is `0` for compatibility. Increase it only if your Kaggle runtime supports it cleanly.
- If `ensemble_combiner.json` is present, the recommended command will automatically use it for stacking inference.
- `unlabeled_predictions.csv` keeps every row, while the defect-candidate and accepted-pseudo-label CSVs make review and downstream tracking easier.
- This bundle is prepared locally. It is not uploaded automatically by this script.
"""


DATASET_METADATA_TEMPLATE = {
    "title": "WM811K Multiclass Ensemble ABC Release",
    "id": "YOUR_KAGGLE_USERNAME/wm811k-multiclass-ensemble-abc-release",
    "licenses": [{"name": "CC0-1.0"}],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", default="kaggle_upload/ensemble_abc_release_bundle")
    parser.add_argument("--zip-path", default="kaggle_upload/ensemble_abc_release_bundle.zip")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=[
            "kaggle_upload/Outputs/model_a/best_model.pt",
            "kaggle_upload/Outputs/model_b/best_model.pt",
            "kaggle_upload/Outputs/model_c/best_model.pt",
        ],
    )
    parser.add_argument("--ensemble-metrics", default="kaggle_upload/Outputs/ensemble_abc/metrics.json")
    parser.add_argument("--combiner-json", default=None)
    return parser.parse_args()


def copy_file(repo_root: Path, source: str | Path, destination: Path) -> None:
    source_path = (repo_root / Path(source)).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)


def copy_tree(repo_root: Path, source: str | Path, destination: Path) -> None:
    source_path = (repo_root / Path(source)).resolve()
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source_path, destination)


def write_zip(bundle_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()

    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        for path in bundle_dir.rglob("*"):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(bundle_dir.parent))


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    bundle_dir = (repo_root / args.bundle_dir).resolve()
    zip_path = (repo_root / args.zip_path).resolve()

    checkpoint_paths = [Path(path) for path in args.checkpoints]
    ensemble_metrics_path = (repo_root / args.ensemble_metrics).resolve()
    ensemble_metrics = json.loads(ensemble_metrics_path.read_text(encoding="utf-8"))

    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    copy_file(repo_root, "requirements.txt", bundle_dir / "requirements.txt")
    copy_file(repo_root, "configs/data/data_multiclass_50k.toml", bundle_dir / "configs/data/data_multiclass_50k.toml")

    for script_name in [
        "predict_unlabeled_multiclass.py",
        "predict_unlabeled_multiclass_ensemble.py",
        "ensemble_multiclass_classifier.py",
    ]:
        copy_file(repo_root, f"scripts/{script_name}", bundle_dir / "scripts" / script_name)

    copy_tree(repo_root, "src/wafer_defect", bundle_dir / "src/wafer_defect")

    released_checkpoints: list[str] = []
    for checkpoint_path in checkpoint_paths:
        checkpoint_path = (repo_root / checkpoint_path).resolve()
        model_dir = checkpoint_path.parent
        relative_model_dir = Path("Outputs") / model_dir.name
        copy_tree(repo_root, model_dir.relative_to(repo_root), bundle_dir / relative_model_dir)
        released_checkpoints.append((relative_model_dir / checkpoint_path.name).as_posix())

    copy_tree(repo_root, "kaggle_upload/Outputs/ensemble_abc", bundle_dir / "Outputs/ensemble_abc")

    if args.combiner_json:
        copy_file(repo_root, args.combiner_json, bundle_dir / "ensemble_combiner.json")

    manifest_description = "Three-checkpoint WM811K multiclass ensemble of model_a, model_b, and model_c."
    if args.combiner_json:
        manifest_description = (
            "Three-checkpoint WM811K multiclass ensemble of model_a, model_b, and model_c "
            "with a saved stacking combiner for deployment inference."
        )

    manifest = {
        "name": "ensemble_abc",
        "description": manifest_description,
        "ensemble_size": len(released_checkpoints),
        "checkpoints": released_checkpoints,
        "combiner_json": "ensemble_combiner.json" if args.combiner_json else None,
        "recommended_min_confidence": ensemble_metrics["min_confidence"],
        "test_accuracy": ensemble_metrics["test"]["accuracy"],
        "test_balanced_accuracy": ensemble_metrics["test"]["balanced_accuracy"],
    }
    (bundle_dir / "ensemble_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (bundle_dir / "dataset-metadata.template.json").write_text(
        json.dumps(DATASET_METADATA_TEMPLATE, indent=2),
        encoding="utf-8",
    )
    (bundle_dir / "README_KAGGLE.md").write_text(
        README_TEMPLATE.format(
            test_accuracy=ensemble_metrics["test"]["accuracy"],
            test_balanced_accuracy=ensemble_metrics["test"]["balanced_accuracy"],
            bundle_dir_name=bundle_dir.name,
        ),
        encoding="utf-8",
    )

    write_zip(bundle_dir, zip_path)

    print(f"Built bundle at {bundle_dir}")
    print(f"Built zip at {zip_path}")


if __name__ == "__main__":
    main()
