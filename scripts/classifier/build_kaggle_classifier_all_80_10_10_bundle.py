"""Build a Kaggle-ready training bundle for the all-labeled 80/10/10 classifier ensemble workflow."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


README_TEMPLATE = """WM-811K Classifier All-Labeled 80/10/10 Kaggle Bundle

This bundle is prepared for retraining the multiclass wafer-defect ensemble on all currently labeled WM-811K rows with a stratified `80 / 10 / 10` split.

Why this bundle exists

- the earlier `50k` classifier subset already used every labeled defect row, so it could not provide a true unseen-defect holdout
- this bundle rebuilds the classifier on the full labeled pool so validation and test both contain held-out examples from every labeled class
- evaluation reports multiclass metrics such as accuracy, balanced accuracy, precision, recall, and F1
- evaluation also exports a binary `defect vs none` view using `1 - P(none)` so the classifier can still be read in a report-style anomaly framing

What is included

- `notebooks/classifier/5_multiclass_classifier_all_labeled_kaggle.ipynb`
- `scripts/classifier/prepare_wm811k_multiclass.py`
- `scripts/classifier/train_multiclass_classifier.py`
- `scripts/classifier/evaluate_multiclass_classifier_metrics.py`
- `src/wafer_defect/...` runtime code needed by the workflow
- `configs/data/classifier/data_multiclass_all_80_10_10.toml`
- seed configs for `7`, `13`, and `21`
- `requirements.txt`
- `dataset-metadata.template.json` for Kaggle CLI upload setup

Recommended Kaggle setup

1. Create one Kaggle Dataset for this release bundle.
2. Create or mount a separate Kaggle Dataset that contains `LSWMD.pkl`.
3. Attach both datasets to a Kaggle Notebook.
4. Open `notebooks/classifier/5_multiclass_classifier_all_labeled_kaggle.ipynb` from this bundle or copy its cells into your notebook.
5. Run from `/kaggle/working`, not `/kaggle/input`, because the workflow writes processed arrays, checkpoints, and metrics.

Important comparison note

- this split is intentionally different from the older anomaly leaderboard split in `REPORT.md`
- use the same metric family for consistency, but do not merge these numbers directly into the older `40k / 5k / 5k + 250` anomaly table

Notes

- raw `LSWMD.pkl` is intentionally not bundled here; attach it separately on Kaggle
- the notebook rewrites only the raw-pickle path at runtime, so the included configs stay version-controlled and reusable
- building the full processed array cache can take time and several GB of working storage
- this bundle is prepared locally and is not uploaded automatically by this script
"""


DATASET_METADATA_TEMPLATE = {
    "title": "WM811K Classifier All Labeled 80 10 10 Bundle",
    "id": "YOUR_KAGGLE_USERNAME/wm811k-classifier-all-labeled-80-10-10-bundle",
    "licenses": [{"name": "CC0-1.0"}],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", default="kaggle_upload/classifier_all_80_10_10_bundle")
    parser.add_argument("--zip-path", default="kaggle_upload/classifier_all_80_10_10_bundle.zip")
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
    repo_root = Path(__file__).resolve().parents[2]
    bundle_dir = (repo_root / args.bundle_dir).resolve()
    zip_path = (repo_root / args.zip_path).resolve()

    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    copy_file(repo_root, "requirements.txt", bundle_dir / "requirements.txt")
    copy_file(repo_root, "scripts/classifier/README.md", bundle_dir / "scripts/classifier/README.md")
    copy_file(repo_root, "notebooks/classifier/README.md", bundle_dir / "notebooks/classifier/README.md")
    copy_file(
        repo_root,
        "notebooks/classifier/5_multiclass_classifier_all_labeled_kaggle.ipynb",
        bundle_dir / "notebooks/classifier/5_multiclass_classifier_all_labeled_kaggle.ipynb",
    )

    for script_name in [
        "prepare_wm811k_multiclass.py",
        "train_multiclass_classifier.py",
        "evaluate_multiclass_classifier_metrics.py",
    ]:
        copy_file(repo_root, f"scripts/classifier/{script_name}", bundle_dir / "scripts/classifier" / script_name)

    for config_name in [
        "data_multiclass_all_80_10_10.toml",
    ]:
        copy_file(
            repo_root,
            f"configs/data/classifier/{config_name}",
            bundle_dir / "configs/data/classifier" / config_name,
        )

    for config_name in [
        "train_multiclass_classifier_all_80_10_10_seed07.toml",
        "train_multiclass_classifier_all_80_10_10_seed13.toml",
        "train_multiclass_classifier_all_80_10_10_seed21.toml",
    ]:
        copy_file(
            repo_root,
            f"configs/training/classifier/{config_name}",
            bundle_dir / "configs/training/classifier" / config_name,
        )

    copy_tree(repo_root, "src/wafer_defect", bundle_dir / "src/wafer_defect")

    manifest = {
        "bundle_name": bundle_dir.name,
        "dataset_config": "configs/data/classifier/data_multiclass_all_80_10_10.toml",
        "training_configs": [
            "configs/training/classifier/train_multiclass_classifier_all_80_10_10_seed07.toml",
            "configs/training/classifier/train_multiclass_classifier_all_80_10_10_seed13.toml",
            "configs/training/classifier/train_multiclass_classifier_all_80_10_10_seed21.toml",
        ],
        "evaluation_script": "scripts/classifier/evaluate_multiclass_classifier_metrics.py",
        "kaggle_notebook": "notebooks/classifier/5_multiclass_classifier_all_labeled_kaggle.ipynb",
        "notes": {
            "split": "all labeled rows, stratified 80/10/10",
            "recommended_ensemble": "average probabilities across the three baseline seed checkpoints",
            "comparison_caution": "Use the same metric family as REPORT.md, but do not merge results directly into the older anomaly leaderboard because the split is different.",
        },
    }
    (bundle_dir / "classifier_bundle_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (bundle_dir / "dataset-metadata.template.json").write_text(
        json.dumps(DATASET_METADATA_TEMPLATE, indent=2),
        encoding="utf-8",
    )
    (bundle_dir / "README_KAGGLE.md").write_text(README_TEMPLATE, encoding="utf-8")

    write_zip(bundle_dir, zip_path)

    print(f"Built bundle at {bundle_dir}")
    print(f"Built zip at {zip_path}")


if __name__ == "__main__":
    main()
