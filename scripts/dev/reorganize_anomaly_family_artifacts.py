from __future__ import annotations

import json
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


AUTOENCODER_NOTEBOOKS = [
    REPO_ROOT / "experiments/anomaly_detection/autoencoder/x64/baseline/notebook.ipynb",
    REPO_ROOT / "experiments/anomaly_detection/autoencoder/x64/batchnorm/notebook.ipynb",
    REPO_ROOT / "experiments/anomaly_detection/autoencoder/x64/residual/notebook.ipynb",
    REPO_ROOT / "experiments/anomaly_detection/autoencoder/x64/batchnorm_dropout/notebook.ipynb",
    REPO_ROOT / "experiments/anomaly_detection/autoencoder/x128/baseline/notebook.ipynb",
]

AUTOENCODER_RUN_DIRS = [
    REPO_ROOT / "experiments/anomaly_detection/autoencoder/x64/baseline/artifacts/autoencoder_baseline",
    REPO_ROOT / "experiments/anomaly_detection/autoencoder/x64/batchnorm/artifacts/autoencoder_batchnorm",
    REPO_ROOT / "experiments/anomaly_detection/autoencoder/x64/residual/artifacts/autoencoder_residual",
    REPO_ROOT / "experiments/anomaly_detection/autoencoder/x64/batchnorm_dropout/artifacts/autoencoder_batchnorm_dropout/dropout_0p00",
    REPO_ROOT / "experiments/anomaly_detection/autoencoder/x128/baseline/artifacts/autoencoder_baseline",
]

AUTOENCODER_SWEEP_ROOT = REPO_ROOT / "experiments/anomaly_detection/autoencoder/x64/batchnorm_dropout/artifacts/autoencoder_batchnorm_dropout"

SVDD_NOTEBOOK = REPO_ROOT / "experiments/anomaly_detection/svdd/x64/baseline/notebook.ipynb"
SVDD_RUN_DIR = REPO_ROOT / "experiments/anomaly_detection/svdd/x64/baseline/artifacts/svdd_baseline"

VAE_BASE_NOTEBOOK = REPO_ROOT / "experiments/anomaly_detection/vae/x64/baseline/notebook.ipynb"
VAE_BASE_RUN_DIR = REPO_ROOT / "experiments/anomaly_detection/vae/x64/baseline/artifacts/vae_baseline"

VAE_SWEEP_NOTEBOOK = REPO_ROOT / "experiments/anomaly_detection/vae/x64/beta_sweep/notebook.ipynb"
VAE_SWEEP_ROOT = REPO_ROOT / "experiments/anomaly_detection/vae/x64/beta_sweep/artifacts/vae_beta_sweep"


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def move_dir_contents(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for child in list(src.iterdir()):
        target = dst / child.name
        if child.is_dir():
            if target.exists() and target.is_dir():
                move_dir_contents(child, target)
                if child.exists():
                    child.rmdir()
            else:
                shutil.move(str(child), str(target))
        else:
            move_file(child, target)
    if src.exists():
        src.rmdir()


def patch_autoencoder_notebooks() -> list[Path]:
    replacements = [
        ('output_dir / "history.json"', 'output_dir / "results" / "history.json"'),
        ('output_dir / "summary.json"', 'output_dir / "results" / "summary.json"'),
        ('output_dir / "best_model.pt"', 'output_dir / "checkpoints" / "best_model.pt"'),
        ('output_dir / "latest_checkpoint.pt"', 'output_dir / "checkpoints" / "latest_checkpoint.pt"'),
        ('output_dir / "last_model.pt"', 'output_dir / "checkpoints" / "last_model.pt"'),
        ('output_dir / "model.pt"', 'output_dir / "checkpoints" / "model.pt"'),
        ('output_dir / f"checkpoint_epoch_{epoch + 1}.pt"', 'output_dir / "checkpoints" / f"checkpoint_epoch_{epoch + 1}.pt"'),
        ('output_dir / "test_scores.csv"', 'output_dir / "results" / "test_scores.csv"'),
        ('output_dir / "metrics.csv"', 'output_dir / "results" / "metrics.csv"'),
        ('output_dir / "threshold_sweep.csv"', 'output_dir / "results" / "threshold_sweep.csv"'),
        ('output_dir / "failure_analysis.csv"', 'output_dir / "results" / "failure_analysis.csv"'),
        ('output_dir / "failure_error_summary.csv"', 'output_dir / "results" / "failure_error_summary.csv"'),
        ('output_dir / "failure_defect_recall.csv"', 'output_dir / "results" / "failure_defect_recall.csv"'),
        ('output_dir / "failure_false_positive_breakdown.csv"', 'output_dir / "results" / "failure_false_positive_breakdown.csv"'),
        ('output_dir / "training_curve.png"', 'output_dir / "plots" / "training_curves.png"'),
        ('output_dir / "confusion_matrix.png"', 'output_dir / "plots" / "confusion_matrix.png"'),
        ('output_dir / "threshold_sweep.png"', 'output_dir / "plots" / "threshold_sweep.png"'),
        ('output_dir / "score_histogram.png"', 'output_dir / "plots" / "score_distribution.png"'),
        ('output_dir / "reconstruction_examples.png"', 'output_dir / "plots" / "reconstruction_examples.png"'),
        ('output_dir / f"failure_examples_{error_type}.png"', 'output_dir / "plots" / f"failure_examples_{error_type}.png"'),
        ('score_ablation_best_model_path = score_ablation_output_root / "best_model.pt"', 'score_ablation_best_model_path = score_ablation_output_root / "checkpoints" / "best_model.pt"'),
        ('score_ablation_output_dir = score_ablation_output_root / "score_ablation"', 'score_ablation_output_dir = score_ablation_output_root / "results" / "score_ablation"'),
        ('save_figure(fig, score_ablation_output_dir / "score_ablation_summary.png")', 'save_figure(fig, score_ablation_output_root / "plots" / "score_ablation_summary.png")'),
        ('base_output_dir / "dropout_sweep_summary.json"', 'base_output_dir / "results" / "dropout_sweep_summary.json"'),
        ('base_output_dir / "dropout_sweep_summary.png"', 'base_output_dir / "plots" / "dropout_sweep_summary.png"'),
        ('run_dir / "history.json"', 'run_dir / "results" / "history.json"'),
        ('run_output_dir / "best_model.pt"', 'run_output_dir / "checkpoints" / "best_model.pt"'),
        ('run_output_dir / "latest_checkpoint.pt"', 'run_output_dir / "checkpoints" / "latest_checkpoint.pt"'),
        ('run_output_dir / f"checkpoint_epoch_{epoch + 1}.pt"', 'run_output_dir / "checkpoints" / f"checkpoint_epoch_{epoch + 1}.pt"'),
        ('run_output_dir / "last_model.pt"', 'run_output_dir / "checkpoints" / "last_model.pt"'),
        ('run_output_dir / "history.json"', 'run_output_dir / "results" / "history.json"'),
        ('run_output_dir / "summary.json"', 'run_output_dir / "results" / "summary.json"'),
    ]
    changed: list[Path] = []
    for path in AUTOENCODER_NOTEBOOKS:
        if replace_in_notebook(path, replacements):
            changed.append(path)
    return changed


def patch_svdd_vae_notebooks() -> list[Path]:
    changed: list[Path] = []
    shared_replacements = [
        ("evaluation_notebook/", "results/evaluation/"),
        ("evaluation_notebook", "results/evaluation"),
        ("EVALUATION_DIR = OUTPUT_DIR / 'evaluation_notebook'", "EVALUATION_DIR = OUTPUT_DIR / 'results' / 'evaluation'"),
        ("history_path = OUTPUT_DIR / 'history.json'", "history_path = OUTPUT_DIR / 'results' / 'history.json'"),
        ("summary_path = OUTPUT_DIR / 'summary.json'", "summary_path = OUTPUT_DIR / 'results' / 'summary.json'"),
        ("best_model_path = OUTPUT_DIR / 'best_model.pt'", "best_model_path = OUTPUT_DIR / 'checkpoints' / 'best_model.pt'"),
    ]
    if replace_in_notebook(SVDD_NOTEBOOK, shared_replacements):
        changed.append(SVDD_NOTEBOOK)
    if replace_in_notebook(VAE_BASE_NOTEBOOK, shared_replacements):
        changed.append(VAE_BASE_NOTEBOOK)

    vae_sweep_replacements = [
        ("`beta_sweep_summary.json`", "`results/beta_sweep_summary.json`"),
        ("summary_path = SWEEP_ROOT / 'beta_sweep_summary.json'", "summary_path = SWEEP_ROOT / 'results' / 'beta_sweep_summary.json'"),
        ("str((run_output_dir / 'best_model.pt').relative_to(REPO_ROOT))", "str((run_output_dir / 'checkpoints' / 'best_model.pt').relative_to(REPO_ROOT))"),
        ("(run_output_dir / 'evaluation' / 'summary.json')", "(run_output_dir / 'results' / 'evaluation' / 'summary.json')"),
        ("summary_payload = json.loads((SWEEP_ROOT / 'beta_sweep_summary.json').read_text(encoding='utf-8'))", "summary_payload = json.loads((SWEEP_ROOT / 'results' / 'beta_sweep_summary.json').read_text(encoding='utf-8'))"),
        ("history_path = SWEEP_ROOT / f'beta_{tag}' / 'history.json'", "history_path = SWEEP_ROOT / f'beta_{tag}' / 'results' / 'history.json'"),
        ("(best_run_dir / 'evaluation' / 'summary.json')", "(best_run_dir / 'results' / 'evaluation' / 'summary.json')"),
        ("(best_run_dir / 'evaluation' / 'test_scores.csv')", "(best_run_dir / 'results' / 'evaluation' / 'test_scores.csv')"),
        ("(best_run_dir / 'evaluation' / 'threshold_sweep.csv')", "(best_run_dir / 'results' / 'evaluation' / 'threshold_sweep.csv')"),
    ]
    if replace_in_notebook(VAE_SWEEP_NOTEBOOK, vae_sweep_replacements):
        changed.append(VAE_SWEEP_NOTEBOOK)
    return changed


def reorganize_autoencoder_run(run_dir: Path) -> None:
    checkpoints_dir = run_dir / "checkpoints"
    results_dir = run_dir / "results"
    plots_dir = run_dir / "plots"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    for path in list(run_dir.glob("*.pt")):
        move_file(path, checkpoints_dir / path.name)
    for path in list(run_dir.glob("*.json")):
        move_file(path, results_dir / path.name)
    for path in list(run_dir.glob("*.csv")):
        move_file(path, results_dir / path.name)
    for path in list(run_dir.glob("*.png")):
        move_file(path, plots_dir / path.name)

    score_ablation_src = run_dir / "score_ablation"
    if score_ablation_src.exists():
        score_ablation_dst = results_dir / "score_ablation"
        if score_ablation_dst.exists():
            move_dir_contents(score_ablation_src, score_ablation_dst)
        else:
            score_ablation_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(score_ablation_src), str(score_ablation_dst))
    score_ablation_plot = results_dir / "score_ablation" / "score_ablation_summary.png"
    move_file(score_ablation_plot, plots_dir / "score_ablation_summary.png")

    move_file(plots_dir / "training_curve.png", plots_dir / "training_curves.png")
    move_file(plots_dir / "score_histogram.png", plots_dir / "score_distribution.png")


def reorganize_autoencoder_family() -> None:
    for run_dir in AUTOENCODER_RUN_DIRS:
        reorganize_autoencoder_run(run_dir)

    (AUTOENCODER_SWEEP_ROOT / "results").mkdir(parents=True, exist_ok=True)
    (AUTOENCODER_SWEEP_ROOT / "plots").mkdir(parents=True, exist_ok=True)
    move_file(
        AUTOENCODER_SWEEP_ROOT / "dropout_sweep_summary.json",
        AUTOENCODER_SWEEP_ROOT / "results" / "dropout_sweep_summary.json",
    )
    move_file(
        AUTOENCODER_SWEEP_ROOT / "dropout_sweep_summary.png",
        AUTOENCODER_SWEEP_ROOT / "plots" / "dropout_sweep_summary.png",
    )


def reorganize_svdd_or_vae_run(run_dir: Path) -> None:
    checkpoints_dir = run_dir / "checkpoints"
    results_dir = run_dir / "results"
    plots_dir = run_dir / "plots"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    for path in list(run_dir.glob("*.pt")):
        move_file(path, checkpoints_dir / path.name)
    for path in list(run_dir.glob("*.json")):
        move_file(path, results_dir / path.name)
    eval_src = run_dir / "evaluation_notebook"
    eval_dst = results_dir / "evaluation"
    if eval_src.exists():
        if eval_dst.exists():
            move_dir_contents(eval_src, eval_dst)
        else:
            eval_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(eval_src), str(eval_dst))


def reorganize_vae_sweep() -> None:
    (VAE_SWEEP_ROOT / "results").mkdir(parents=True, exist_ok=True)
    move_file(VAE_SWEEP_ROOT / "beta_sweep_log.jsonl", VAE_SWEEP_ROOT / "results" / "beta_sweep_log.jsonl")
    move_file(VAE_SWEEP_ROOT / "beta_sweep_summary.json", VAE_SWEEP_ROOT / "results" / "beta_sweep_summary.json")

    for run_dir in VAE_SWEEP_ROOT.glob("beta_*"):
        if not run_dir.is_dir():
            continue
        checkpoints_dir = run_dir / "checkpoints"
        results_dir = run_dir / "results"
        logs_dir = run_dir / "logs"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        for path in list(run_dir.glob("*.pt")):
            move_file(path, checkpoints_dir / path.name)
        for path in list(run_dir.glob("*.json")):
            move_file(path, results_dir / path.name)
        for path in list(run_dir.glob("*.log")):
            move_file(path, logs_dir / path.name)

        eval_src = run_dir / "evaluation"
        eval_dst = results_dir / "evaluation"
        if eval_src.exists():
            if eval_dst.exists():
                move_dir_contents(eval_src, eval_dst)
            else:
                eval_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(eval_src), str(eval_dst))


def main() -> None:
    notebook_changes = []
    notebook_changes.extend(patch_autoencoder_notebooks())
    notebook_changes.extend(patch_svdd_vae_notebooks())

    reorganize_autoencoder_family()
    reorganize_svdd_or_vae_run(SVDD_RUN_DIR)
    reorganize_svdd_or_vae_run(VAE_BASE_RUN_DIR)
    reorganize_vae_sweep()

    print("Patched notebooks:")
    for path in notebook_changes:
        print(f" - {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
