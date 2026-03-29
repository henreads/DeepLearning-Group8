from __future__ import annotations

import json
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

RENAMES = [
    (
        REPO_ROOT / "experiments/anomaly_detection/teacher_student/resnet50/x64/main",
        REPO_ROOT / "experiments/anomaly_detection/teacher_student/resnet50/x64/main",
    ),
    (
        REPO_ROOT / "experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/all_in_one",
        REPO_ROOT / "experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/layer2_self_contained",
    ),
    (
        REPO_ROOT / "experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/multilayer",
        REPO_ROOT / "experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/multilayer_self_contained",
    ),
]


TEXT_FILES = [
    REPO_ROOT / "experiments/anomaly_detection/teacher_student/README.md",
    REPO_ROOT / "experiments/anomaly_detection/teacher_student/resnet50/x64/main/README.md",
    REPO_ROOT / "experiments/anomaly_detection/teacher_student/resnet50/x64/main/train_config.toml",
    REPO_ROOT / "experiments/anomaly_detection/teacher_student/resnet50/x64/layer_ablation/README.md",
    REPO_ROOT / "experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/layer2_self_contained/README.md",
    REPO_ROOT / "experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/layer2_self_contained/train_config.toml",
    REPO_ROOT / "experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/multilayer_self_contained/README.md",
    REPO_ROOT / "experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/multilayer_self_contained/train_config.toml",
    REPO_ROOT / "scripts/dev/reorganize_teacher_student_artifacts.py",
]


NOTEBOOK_FILES = [
    REPO_ROOT / "experiments/anomaly_detection/teacher_student/resnet50/x64/main/notebook.ipynb",
    REPO_ROOT / "experiments/anomaly_detection/teacher_student/resnet50/x64/layer_ablation/notebook.ipynb",
    REPO_ROOT / "experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/layer2_self_contained/notebook.ipynb",
    REPO_ROOT / "experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/multilayer_self_contained/notebook.ipynb",
    REPO_ROOT / "experiments/anomaly_detection/ensemble/x64/score_ensemble/notebook.ipynb",
]


TEXT_REPLACEMENTS = [
    (
        "experiments/anomaly_detection/teacher_student/resnet50/x64/main",
        "experiments/anomaly_detection/teacher_student/resnet50/x64/main",
    ),
    (
        "experiments/anomaly_detection/teacher_student/resnet50/x64/kaggle_import",
        "experiments/anomaly_detection/teacher_student/resnet50/x64/main",
    ),
    (
        "experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/all_in_one",
        "experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/layer2_self_contained",
    ),
    (
        "experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/multilayer",
        "experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/multilayer_self_contained",
    ),
    (
        "artifacts/wideresnet50_2_modal",
        "artifacts/ts_wideresnet50_layer2",
    ),
    (
        "artifacts/ts_wideresnet50",
        "artifacts/ts_wideresnet50_layer2",
    ),
]


def move_directory(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")
    src.rename(dst)


def replace_in_text_file(path: Path) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    updated = text
    for old, new in TEXT_REPLACEMENTS:
        updated = updated.replace(old, new)
    updated = updated.replace("# TS-ResNet50 Kaggle Import", "# TS-ResNet50 Training And Import Analysis")
    updated = updated.replace("kaggle import notebook", "training-and-import-analysis notebook")
    updated = updated.replace(
        "# TS-WideResNet50-2 All-in-One",
        "# TS-WideResNet50-2 Layer2 Self-Contained",
    )
    updated = updated.replace(
        "# TS-WideResNet50-2 Multilayer",
        "# TS-WideResNet50-2 Multilayer Self-Contained",
    )
    updated = updated.replace(
        "rerun-oriented WideResNet50-2 teacher-student notebook and configs.",
        "self-contained single-layer WideResNet50-2 teacher-student notebook and configs.",
    )
    updated = updated.replace(
        "saved WideResNet50-2 run in the family. It uses multilayer teacher features",
        "self-contained multilayer WideResNet50-2 teacher-student notebook. It uses teacher layers `layer2` and `layer3`",
    )
    updated = updated.replace(
        "The notebook defaults to `run_training = false` through the config-backed run settings, so opening and running it should reuse the saved artifacts first.",
        "The notebook is a self-contained multilayer branch. It is configured to avoid retraining by default, but it does not currently ship with a finished saved multilayer run in the repo.",
    )
    updated = updated.replace(
        "- checkpoints: `artifacts/wideresnet50_2_modal/checkpoints/`\n- results: `artifacts/wideresnet50_2_modal/results/`",
        "- checkpoints: `artifacts/ts_wideresnet50_multilayer/checkpoints/`\n- results: `artifacts/ts_wideresnet50_multilayer/results/`",
    )
    path.write_text(updated, encoding="utf-8")


def replace_in_notebook(path: Path) -> None:
    if not path.exists():
        return
    notebook = json.loads(path.read_text(encoding="utf-8-sig"))
    changed = False
    for cell in notebook["cells"]:
        source = cell.get("source")
        if not isinstance(source, list):
            continue
        joined = "".join(source)
        updated = joined
        for old, new in TEXT_REPLACEMENTS:
            updated = updated.replace(old, new)
        updated = updated.replace(
            "# TS-ResNet50 Training and Import Analysis",
            "# TS-ResNet50 Training And Import Analysis",
        )
        updated = updated.replace(
            "# WideResNet50-2 Teacher-Student Distillation (All-in-One)",
            "# WideResNet50-2 Teacher-Student Distillation (Layer2 Self-Contained)"
            if "teacher_layer\": \"layer2\"" in joined or "teacher_layer = \"layer2\"" in joined or "layer2_self_contained" in str(path)
            else "# WideResNet50-2 Teacher-Student Distillation (Multilayer Self-Contained)",
        )
        updated = updated.replace(
            '"output_dir": "root/ts_wideresnet50_experiment"',
            '"output_dir": "experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/multilayer_self_contained/artifacts/ts_wideresnet50_multilayer"',
        )
        if updated != joined:
            cell["source"] = updated.splitlines(keepends=True)
            changed = True
    if changed:
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def patch_multilayer_train_config(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        'output_dir = "experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/multilayer_self_contained/artifacts/ts_wideresnet50_layer2"',
        'output_dir = "experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/multilayer_self_contained/artifacts/ts_wideresnet50_multilayer"',
    )
    text = text.replace('teacher_layer = "layer2"', 'teacher_layers = ["layer2", "layer3"]')
    text = text.replace('score_student_weight = 1.0', 'score_student_weight = 2.0')
    text = text.replace('score_autoencoder_weight = 0.0', 'score_autoencoder_weight = 1.0')
    text = text.replace('topk_ratio = 0.20', 'topk_ratio = 0.25')
    path.write_text(text, encoding="utf-8")


def move_wrn_artifacts() -> None:
    src = (
        REPO_ROOT
        / "experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/multilayer_self_contained/artifacts/ts_wideresnet50_layer2"
    )
    dst = (
        REPO_ROOT
        / "experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/layer2_self_contained/artifacts/ts_wideresnet50_layer2"
    )
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            raise FileExistsError(f"Destination already exists: {dst}")
        shutil.move(str(src), str(dst))


def main() -> None:
    for src, dst in RENAMES:
        move_directory(src, dst)

    move_wrn_artifacts()

    for path in TEXT_FILES:
        replace_in_text_file(path)

    multilayer_train_config = REPO_ROOT / "experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/multilayer_self_contained/train_config.toml"
    if multilayer_train_config.exists():
        patch_multilayer_train_config(multilayer_train_config)

    for path in NOTEBOOK_FILES:
        replace_in_notebook(path)

    print("Renamed teacher-student branches and patched references.")


if __name__ == "__main__":
    main()
