from __future__ import annotations

import json
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


RESNET18_NOTEBOOK = REPO_ROOT / "experiments/anomaly_detection/teacher_student/resnet18/x64/main/notebook.ipynb"
RESNET18_RUN_DIR = REPO_ROOT / "experiments/anomaly_detection/teacher_student/resnet18/x64/main/artifacts/ts_resnet18"

RESNET50_NOTEBOOK = REPO_ROOT / "experiments/anomaly_detection/teacher_student/resnet50/x64/main/notebook.ipynb"
RESNET50_RUN_DIR = REPO_ROOT / "experiments/anomaly_detection/teacher_student/resnet50/x64/main/artifacts/ts_resnet50"

WRN_NOTEBOOK = REPO_ROOT / "experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/layer2_self_contained/notebook.ipynb"
WRN_RUN_DIR = REPO_ROOT / "experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/layer2_self_contained/artifacts/ts_wideresnet50_layer2"


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_notebook(path: Path, notebook: dict) -> None:
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def replace_in_notebook(path: Path, replacements: list[tuple[str, str]]) -> bool:
    notebook = load_notebook(path)
    changed = False
    for cell in notebook["cells"]:
        source = cell.get("source")
        if not isinstance(source, list):
            continue
        joined = "".join(source)
        updated = joined
        for old, new in replacements:
            updated = updated.replace(old, new)
        if updated != joined:
            cell["source"] = updated.splitlines(keepends=True)
            changed = True
    if changed:
        save_notebook(path, notebook)
    return changed


def move_file(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if src.read_bytes() == dst.read_bytes():
            src.unlink()
            return
        dst.unlink()
    shutil.move(str(src), str(dst))


def move_dir(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        for child in src.iterdir():
            target = dst / child.name
            if child.is_dir():
                move_dir(child, target)
            else:
                move_file(child, target)
        src.rmdir()
        return
    shutil.move(str(src), str(dst))


def patch_resnet18_notebook() -> bool:
    replacements = [
        ("RUN_TRAINING = True", "RUN_TRAINING = False"),
        ("RUN_DEFAULT_EVALUATION = True", "RUN_DEFAULT_EVALUATION = False"),
        ("RUN_ABLATION_SWEEP = True", "RUN_ABLATION_SWEEP = False"),
        ('evaluation_dir = output_dir / "evaluation"', 'evaluation_dir = output_dir / "results" / "evaluation"'),
        ('best_model_path = output_dir / "best_model.pt"', 'best_model_path = output_dir / "checkpoints" / "best_model.pt"'),
        ('history_df = pd.read_json(output_dir / "history.json")', 'history_df = pd.read_json(output_dir / "results" / "history.json")'),
        ('training_summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))', 'training_summary = json.loads((output_dir / "results" / "summary.json").read_text(encoding="utf-8"))'),
        ('generated_config_dir = REPO_ROOT / "artifacts" / "generated_configs"', 'generated_config_dir = output_dir.parent / "generated_configs"'),
        ('variant_config["run"]["output_dir"] = f"artifacts/{image_size_name}/{variant_name}"', 'variant_config["run"]["output_dir"] = f"experiments/anomaly_detection/teacher_student/resnet18/x64/main/artifacts/ablation_variants/{variant_name}"'),
        ('checkpoint_path = variant_output_dir / "best_model.pt"', 'checkpoint_path = variant_output_dir / "checkpoints" / "best_model.pt"'),
        ('evaluation_output_dir = variant_output_dir / "evaluation"', 'evaluation_output_dir = variant_output_dir / "results" / "evaluation"'),
        ('train_summary = json.loads((variant_output_dir / "summary.json").read_text(encoding="utf-8"))', 'train_summary = json.loads((variant_output_dir / "results" / "summary.json").read_text(encoding="utf-8"))'),
        ('repo_root / "artifacts" / "x64" / "ts_resnet18" / "evaluation",', 'repo_root / "experiments" / "anomaly_detection" / "teacher_student" / "resnet18" / "x64" / "main" / "artifacts" / "ts_resnet18" / "results" / "evaluation",'),
        ('save_path = evaluation_dir / "selected_score_variant.json"', 'save_path = evaluation_dir / "selected_score_variant.json"'),
    ]
    return replace_in_notebook(RESNET18_NOTEBOOK, replacements)


def patch_resnet50_notebook() -> bool:
    replacements = [
        ('CHECKPOINT_PATH = ARTIFACT_DIR / "best_model.pt"', 'CHECKPOINT_PATH = ARTIFACT_DIR / "checkpoints" / "best_model.pt"'),
        ('CONVERTED_CHECKPOINT_PATH = ARTIFACT_DIR / "best_model_local_format.pt"', 'CONVERTED_CHECKPOINT_PATH = ARTIFACT_DIR / "checkpoints" / "best_model_local_format.pt"'),
        ('EVALUATION_DIR = ARTIFACT_DIR / "evaluation"', 'EVALUATION_DIR = ARTIFACT_DIR / "results" / "evaluation_imported"'),
        ('LOCAL_EVALUATION_DIR = ARTIFACT_DIR / "evaluation_local"', 'LOCAL_EVALUATION_DIR = ARTIFACT_DIR / "results" / "evaluation_local"'),
        ("RUN_IMPORT_REMAP = True", "RUN_IMPORT_REMAP = False"),
        ("RUN_LOCAL_RE_EVALUATION = True", "RUN_LOCAL_RE_EVALUATION = False"),
        ("RUN_SCORE_SWEEP = True", "RUN_SCORE_SWEEP = False"),
        ('imported_summary_path = ARTIFACT_DIR / "summary.json"', 'imported_summary_path = ARTIFACT_DIR / "results" / "summary.json"'),
        ('history_path = ARTIFACT_DIR / "history.json"', 'history_path = ARTIFACT_DIR / "results" / "history.json"'),
    ]
    return replace_in_notebook(RESNET50_NOTEBOOK, replacements)


def patch_wrn_notebook() -> bool:
    replacements = [
        ('evaluation_dir = output_dir / "evaluation"', 'evaluation_dir = output_dir / "results" / "evaluation"'),
        ('with (output_dir / "config.json").open("w", encoding="utf-8") as f:', 'with (output_dir / "results" / "config.json").open("w", encoding="utf-8") as f:'),
        ('output_dir / "final_model.pt"', 'output_dir / "checkpoints" / "final_model.pt"'),
        ('history_path = output_dir / "history.json"', 'history_path = output_dir / "results" / "history.json"'),
        ('history_df.to_csv(output_dir / "training_history.csv", index=False)', 'history_df.to_csv(output_dir / "results" / "training_history.csv", index=False)'),
        ('"training_history_csv": str(output_dir / "training_history.csv"),', '"training_history_csv": str(output_dir / "results" / "training_history.csv"),'),
        ('"final_model_path": str(output_dir / "final_model.pt"),', '"final_model_path": str(output_dir / "checkpoints" / "final_model.pt"),'),
        ('with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:', 'with (output_dir / "results" / "summary.json").open("w", encoding="utf-8") as handle:'),
        ('RUN_TRAINING = bool(CONFIG["run"].get("run_training", True))', 'RUN_TRAINING = bool(CONFIG["run"].get("run_training", False))'),
        ('RUN_SCORE_SWEEP = bool(CONFIG["run"].get("run_score_sweep", False))', 'RUN_SCORE_SWEEP = bool(CONFIG["run"].get("run_score_sweep", False))'),
        ('print(f"Saved final model: {output_dir / \'final_model.pt\'}")', 'print(f"Saved final model: {output_dir / \'checkpoints\' / \'final_model.pt\'}")'),
    ]
    return replace_in_notebook(WRN_NOTEBOOK, replacements)


def reorganize_resnet18() -> None:
    (RESNET18_RUN_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
    (RESNET18_RUN_DIR / "results").mkdir(parents=True, exist_ok=True)
    for name in [
        "best_model.pt",
        "last_model.pt",
        "latest_checkpoint.pt",
        "checkpoint_epoch_5.pt",
        "checkpoint_epoch_10.pt",
        "checkpoint_epoch_15.pt",
        "checkpoint_epoch_20.pt",
        "checkpoint_epoch_25.pt",
        "checkpoint_epoch_30.pt",
    ]:
        move_file(RESNET18_RUN_DIR / name, RESNET18_RUN_DIR / "checkpoints" / name)
    for name in ["history.json", "summary.json"]:
        move_file(RESNET18_RUN_DIR / name, RESNET18_RUN_DIR / "results" / name)
    move_dir(RESNET18_RUN_DIR / "evaluation", RESNET18_RUN_DIR / "results" / "evaluation")


def reorganize_resnet50() -> None:
    (RESNET50_RUN_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
    (RESNET50_RUN_DIR / "results").mkdir(parents=True, exist_ok=True)
    for name in ["best_model.pt", "best_model_local_format.pt"]:
        move_file(RESNET50_RUN_DIR / name, RESNET50_RUN_DIR / "checkpoints" / name)
    for name in ["history.json"]:
        move_file(RESNET50_RUN_DIR / name, RESNET50_RUN_DIR / "results" / name)
    move_dir(RESNET50_RUN_DIR / "evaluation", RESNET50_RUN_DIR / "results" / "evaluation_imported")
    move_dir(RESNET50_RUN_DIR / "evaluation_local", RESNET50_RUN_DIR / "results" / "evaluation_local")
    move_dir(RESNET50_RUN_DIR / "evaluation_holdout70k_3p5k", RESNET50_RUN_DIR / "results" / "evaluation_holdout70k_3p5k")


def reorganize_wrn() -> None:
    (WRN_RUN_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
    (WRN_RUN_DIR / "results").mkdir(parents=True, exist_ok=True)
    for name in ["best_model.pt", "checkpoint_epoch_5.pt", "checkpoint_epoch_10.pt", "final_model.pt"]:
        move_file(WRN_RUN_DIR / name, WRN_RUN_DIR / "checkpoints" / name)
    for name in ["config.json", "history.json", "summary.json", "training_history.csv"]:
        move_file(WRN_RUN_DIR / name, WRN_RUN_DIR / "results" / name)
    move_dir(WRN_RUN_DIR / "eval", WRN_RUN_DIR / "results" / "evaluation")
    old_eval_dir = WRN_RUN_DIR / "evaluation"
    if old_eval_dir.exists() and not any(old_eval_dir.iterdir()):
        old_eval_dir.rmdir()


def main() -> None:
    changed = []
    if patch_resnet18_notebook():
        changed.append(RESNET18_NOTEBOOK)
    if patch_resnet50_notebook():
        changed.append(RESNET50_NOTEBOOK)
    if patch_wrn_notebook():
        changed.append(WRN_NOTEBOOK)

    reorganize_resnet18()
    reorganize_resnet50()
    reorganize_wrn()

    print("Patched notebooks:")
    for path in changed:
        print(f" - {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
