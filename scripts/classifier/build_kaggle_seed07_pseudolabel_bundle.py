"""Build a Kaggle-ready bundle for seed07 unlabeled pseudo-labeling only."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


README_TEMPLATE = """WM-811K Seed07 Pseudo-Labeling Kaggle Bundle

This bundle is prepared specifically for applying the saved `seed07` multiclass classifier
checkpoint to the unlabeled WM-811K rows and exporting pseudo labels plus confidence scores.

Why this bundle exists

- it keeps the pseudo-labeling workflow separate from the much heavier all-labeled retraining workflow
- it reuses the finished `seed07` checkpoint directly instead of asking Kaggle to retrain the classifier
- it produces confidence-scored pseudo labels that can later support anomaly-threshold validation and review

What is included

- `notebooks/classifier/6_seed07_unlabeled_pseudolabeling.ipynb`
- `scripts/classifier/predict_unlabeled_multiclass.py`
- `configs/data/classifier/data_multiclass_all_80_10_10.toml`
- exported `seed07` checkpoint at `artifacts/multiclass_classifier_all_80_10_10_seed07/best_model.pt`
- exported `seed07` metrics at `artifacts/multiclass_classifier_all_80_10_10_seed07/metrics.json`
- `src/wafer_defect/...` runtime code needed by the workflow
- `requirements.txt`
- `dataset-metadata.template.json` for Kaggle CLI upload setup

Recommended Kaggle setup

1. Create one Kaggle Dataset for this pseudo-labeling bundle.
2. Create or mount a separate Kaggle Dataset that contains `LSWMD.pkl`.
3. Attach both datasets to a Kaggle Notebook.
4. Open `notebooks/classifier/6_seed07_unlabeled_pseudolabeling.ipynb` from this bundle or copy its cells into your notebook.
5. Run from `/kaggle/working`, not `/kaggle/input`, because the notebook writes pseudo-label CSV outputs.

Notes

- raw `LSWMD.pkl` is intentionally not bundled here; attach it separately on Kaggle
- the notebook rewrites only the raw-pickle path at runtime, so the included config stays reusable
- this bundle is prepared locally and is not uploaded automatically by this script
"""


DATASET_METADATA_TEMPLATE = {
    "title": "WM811K Seed07 Pseudolabel Bundle",
    "id": "YOUR_KAGGLE_USERNAME/wm811k-seed07-pseudolabel-bundle",
    "licenses": [{"name": "CC0-1.0"}],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", default="kaggle_upload/seed07_pseudolabel_bundle")
    parser.add_argument("--zip-path", default="kaggle_upload/seed07_pseudolabel_bundle.zip")
    parser.add_argument("--seed07-checkpoint", default="")
    parser.add_argument("--seed07-metrics", default="")
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


def resolve_existing_file(repo_root: Path, candidates: list[str | Path], description: str) -> Path:
    for candidate in candidates:
        candidate_path = (repo_root / Path(candidate)).resolve()
        if candidate_path.exists():
            return candidate_path
    candidate_list = "\n".join(str((repo_root / Path(candidate)).resolve()) for candidate in candidates)
    raise FileNotFoundError(f"Could not find {description}. Checked:\n{candidate_list}")


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

    seed07_checkpoint_path = resolve_existing_file(
        repo_root,
        candidates=[
            args.seed07_checkpoint,
            "artifacts/multiclass_classifier_all_80_10_10_seed07/best_model.pt",
            "outputs/_output_inspect/classifier_all_80_10_10_bundle/artifacts/multiclass_classifier_all_80_10_10_seed07/best_model.pt",
        ]
        if args.seed07_checkpoint
        else [
            "artifacts/multiclass_classifier_all_80_10_10_seed07/best_model.pt",
            "outputs/_output_inspect/classifier_all_80_10_10_bundle/artifacts/multiclass_classifier_all_80_10_10_seed07/best_model.pt",
        ],
        description="seed07 checkpoint",
    )
    seed07_metrics_path = resolve_existing_file(
        repo_root,
        candidates=[
            args.seed07_metrics,
            "artifacts/multiclass_classifier_all_80_10_10_seed07/metrics.json",
            "outputs/_output_inspect/classifier_all_80_10_10_bundle/artifacts/multiclass_classifier_all_80_10_10_seed07/metrics.json",
        ]
        if args.seed07_metrics
        else [
            "artifacts/multiclass_classifier_all_80_10_10_seed07/metrics.json",
            "outputs/_output_inspect/classifier_all_80_10_10_bundle/artifacts/multiclass_classifier_all_80_10_10_seed07/metrics.json",
        ],
        description="seed07 metrics",
    )

    copy_file(repo_root, "requirements.txt", bundle_dir / "requirements.txt")
    copy_file(repo_root, "scripts/classifier/README.md", bundle_dir / "scripts/classifier/README.md")
    copy_file(repo_root, "notebooks/classifier/README.md", bundle_dir / "notebooks/classifier/README.md")
    copy_file(
        repo_root,
        "notebooks/classifier/6_seed07_unlabeled_pseudolabeling.ipynb",
        bundle_dir / "notebooks/classifier/6_seed07_unlabeled_pseudolabeling.ipynb",
    )
    copy_file(
        repo_root,
        "scripts/classifier/predict_unlabeled_multiclass.py",
        bundle_dir / "scripts/classifier/predict_unlabeled_multiclass.py",
    )
    copy_file(
        repo_root,
        "configs/data/classifier/data_multiclass_all_80_10_10.toml",
        bundle_dir / "configs/data/classifier/data_multiclass_all_80_10_10.toml",
    )
    copy_tree(repo_root, "src/wafer_defect", bundle_dir / "src/wafer_defect")
    copy_file(
        repo_root,
        seed07_checkpoint_path.relative_to(repo_root),
        bundle_dir / "artifacts/multiclass_classifier_all_80_10_10_seed07/best_model.pt",
    )
    copy_file(
        repo_root,
        seed07_metrics_path.relative_to(repo_root),
        bundle_dir / "artifacts/multiclass_classifier_all_80_10_10_seed07/metrics.json",
    )

    manifest = {
        "bundle_name": bundle_dir.name,
        "workflow": "seed07_unlabeled_pseudolabeling",
        "kaggle_notebook": "notebooks/classifier/6_seed07_unlabeled_pseudolabeling.ipynb",
        "prediction_script": "scripts/classifier/predict_unlabeled_multiclass.py",
        "dataset_config": "configs/data/classifier/data_multiclass_all_80_10_10.toml",
        "seed07_checkpoint": "artifacts/multiclass_classifier_all_80_10_10_seed07/best_model.pt",
        "seed07_metrics": "artifacts/multiclass_classifier_all_80_10_10_seed07/metrics.json",
        "notes": {
            "intended_use": "Pseudo-label the unlabeled WM-811K rows with the finished seed07 classifier.",
            "confidence_outputs": "The notebook exports pseudo labels, confidence percentages, second-choice confidence, and a high-confidence subset CSV.",
        },
    }
    (bundle_dir / "seed07_pseudolabel_bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
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
