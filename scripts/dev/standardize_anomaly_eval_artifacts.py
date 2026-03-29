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

AUTOENCODER_ARTIFACT_DIRS = [
    REPO_ROOT / "experiments/anomaly_detection/autoencoder/x64/baseline/artifacts/autoencoder_baseline",
    REPO_ROOT / "experiments/anomaly_detection/autoencoder/x64/batchnorm/artifacts/autoencoder_batchnorm",
    REPO_ROOT / "experiments/anomaly_detection/autoencoder/x64/residual/artifacts/autoencoder_residual",
    REPO_ROOT / "experiments/anomaly_detection/autoencoder/x64/batchnorm_dropout/artifacts/autoencoder_batchnorm_dropout/dropout_0p00",
    REPO_ROOT / "experiments/anomaly_detection/autoencoder/x128/baseline/artifacts/autoencoder_baseline",
]


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_notebook(path: Path, notebook: dict) -> None:
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def replace_everywhere(notebook: dict, replacements: list[tuple[str, str]]) -> bool:
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
    return changed


def patch_autoencoder_notebooks() -> list[Path]:
    replacements = [
        ('output_dir / "training_curve.png"', 'output_dir / "plots" / "training_curves.png"'),
        ('output_dir / "confusion_matrix.png"', 'output_dir / "plots" / "confusion_matrix.png"'),
        ('output_dir / "threshold_sweep.png"', 'output_dir / "plots" / "threshold_sweep.png"'),
        ('output_dir / "score_histogram.png"', 'output_dir / "plots" / "score_distribution.png"'),
        ('output_dir / "reconstruction_examples.png"', 'output_dir / "plots" / "reconstruction_examples.png"'),
        ('output_dir / f"failure_examples_{error_type}.png"', 'output_dir / "plots" / f"failure_examples_{error_type}.png"'),
        ('base_output_dir / "dropout_sweep_summary.png"', 'base_output_dir / "plots" / "dropout_sweep_summary.png"'),
    ]
    changed_paths: list[Path] = []
    for path in AUTOENCODER_NOTEBOOKS:
        notebook = load_notebook(path)
        if replace_everywhere(notebook, replacements):
            save_notebook(path, notebook)
            changed_paths.append(path)
    return changed_paths


def patch_exact_code_cell(path: Path, needle: str, new_source: str) -> bool:
    notebook = load_notebook(path)
    changed = False
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if needle in source:
            cell["source"] = new_source.splitlines(keepends=True)
            changed = True
            break
    if changed:
        save_notebook(path, notebook)
    return changed


SVDD_CELL = """focus_low = float(threshold_sweep_df['threshold'].quantile(0.01))
focus_high = float(threshold_sweep_df['threshold'].quantile(0.99))
focus_low = min(focus_low, threshold, float(best_sweep['threshold']))
focus_high = max(focus_high, threshold, float(best_sweep['threshold']))
focus_pad = max((focus_high - focus_low) * 0.1, 1e-6)
focus_low = max(0.0, focus_low - focus_pad)
focus_high = focus_high + focus_pad

cm = evaluation_summary['metrics_at_validation_threshold'].get('confusion_matrix', [[0, 0], [0, 0]])
cm_df = pd.DataFrame(cm, index=['true_normal', 'true_anomaly'], columns=['pred_normal', 'pred_anomaly'])

fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
ax.hist(test_scores_df.loc[test_scores_df['is_anomaly'] == 0, 'score'], bins=30, alpha=0.7, label='normal')
ax.hist(test_scores_df.loc[test_scores_df['is_anomaly'] == 1, 'score'], bins=30, alpha=0.7, label='anomaly')
ax.axvline(threshold, color='red', linestyle='--', label=f'val threshold = {threshold:.4f}')
ax.set_title('Test Score Distribution')
ax.set_xlabel('Anomaly score')
ax.legend()
fig.savefig(PLOTS_DIR / 'score_distribution.png', dpi=160, bbox_inches='tight')
display(fig)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), constrained_layout=True)
for axis in axes:
    axis.plot(threshold_sweep_df['threshold'], threshold_sweep_df['precision'], label='precision')
    axis.plot(threshold_sweep_df['threshold'], threshold_sweep_df['recall'], label='recall')
    axis.plot(threshold_sweep_df['threshold'], threshold_sweep_df['f1'], label='f1')
    axis.axvline(threshold, color='red', linestyle='--', label='val threshold')
    axis.axvline(float(best_sweep['threshold']), color='green', linestyle=':', label='best sweep')
    axis.set_xlabel('Threshold')
axes[0].set_title('Threshold Sweep (Full)')
axes[0].legend()
axes[1].set_title('Threshold Sweep (Zoomed)')
axes[1].set_xlim(focus_low, focus_high)
axes[1].legend()
fig.savefig(PLOTS_DIR / 'threshold_sweep.png', dpi=160, bbox_inches='tight')
display(fig)
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
heatmap = ax.imshow(cm_df.to_numpy(), cmap='Blues')
ax.set_xticks(range(cm_df.shape[1]), cm_df.columns)
ax.set_yticks(range(cm_df.shape[0]), cm_df.index)
ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')
for row_idx in range(cm_df.shape[0]):
    for col_idx in range(cm_df.shape[1]):
        ax.text(col_idx, row_idx, str(int(cm_df.iat[row_idx, col_idx])), ha='center', va='center', color='black')
fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
fig.savefig(PLOTS_DIR / 'confusion_matrix.png', dpi=160, bbox_inches='tight')
display(fig)
plt.close(fig)

fig, axes = plt.subplots(1, 4, figsize=(24, 4.5), constrained_layout=True)
axes[0].hist(test_scores_df.loc[test_scores_df['is_anomaly'] == 0, 'score'], bins=30, alpha=0.7, label='normal')
axes[0].hist(test_scores_df.loc[test_scores_df['is_anomaly'] == 1, 'score'], bins=30, alpha=0.7, label='anomaly')
axes[0].axvline(threshold, color='red', linestyle='--', label=f'val threshold = {threshold:.4f}')
axes[0].set_title('Test Score Distribution')
axes[0].set_xlabel('Anomaly score')
axes[0].legend()
for axis in axes[1:3]:
    axis.plot(threshold_sweep_df['threshold'], threshold_sweep_df['precision'], label='precision')
    axis.plot(threshold_sweep_df['threshold'], threshold_sweep_df['recall'], label='recall')
    axis.plot(threshold_sweep_df['threshold'], threshold_sweep_df['f1'], label='f1')
    axis.axvline(threshold, color='red', linestyle='--', label='val threshold')
    axis.axvline(float(best_sweep['threshold']), color='green', linestyle=':', label='best sweep')
    axis.set_xlabel('Threshold')
axes[1].set_title('Threshold Sweep (Full)')
axes[1].legend()
axes[2].set_title('Threshold Sweep (Zoomed)')
axes[2].set_xlim(focus_low, focus_high)
axes[2].legend()
heatmap = axes[3].imshow(cm_df.to_numpy(), cmap='Blues')
axes[3].set_xticks(range(cm_df.shape[1]), cm_df.columns)
axes[3].set_yticks(range(cm_df.shape[0]), cm_df.index)
axes[3].set_title('Confusion Matrix')
axes[3].set_xlabel('Predicted label')
axes[3].set_ylabel('True label')
for row_idx in range(cm_df.shape[0]):
    for col_idx in range(cm_df.shape[1]):
        axes[3].text(col_idx, row_idx, str(int(cm_df.iat[row_idx, col_idx])), ha='center', va='center', color='black')
fig.colorbar(heatmap, ax=axes[3], fraction=0.046, pad=0.04)
fig.savefig(PLOTS_DIR / 'score_distribution_sweep_confusion.png', dpi=160, bbox_inches='tight')
display(fig)
plt.close(fig)
display(cm_df)
display(threshold_sweep_df.sort_values('f1', ascending=False).head(10))
"""


VAE_BASE_CELL = """cm = evaluation_summary['metrics_at_validation_threshold'].get('confusion_matrix', [[0, 0], [0, 0]])
cm_df = pd.DataFrame(cm, index=['true_normal', 'true_anomaly'], columns=['pred_normal', 'pred_anomaly'])

fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
ax.hist(test_scores_df.loc[test_scores_df['is_anomaly'] == 0, 'score'], bins=30, alpha=0.7, label='normal')
ax.hist(test_scores_df.loc[test_scores_df['is_anomaly'] == 1, 'score'], bins=30, alpha=0.7, label='anomaly')
ax.axvline(threshold, color='red', linestyle='--', label=f'val threshold = {threshold:.4f}')
ax.set_title('Test Score Distribution')
ax.set_xlabel('Anomaly score')
ax.legend()
fig.savefig(PLOTS_DIR / 'score_distribution.png', dpi=160, bbox_inches='tight')
display(fig)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
ax.plot(threshold_sweep_df['threshold'], threshold_sweep_df['precision'], label='precision')
ax.plot(threshold_sweep_df['threshold'], threshold_sweep_df['recall'], label='recall')
ax.plot(threshold_sweep_df['threshold'], threshold_sweep_df['f1'], label='f1')
ax.axvline(threshold, color='red', linestyle='--', label='val threshold')
ax.axvline(float(best_sweep['threshold']), color='green', linestyle=':', label='best sweep')
ax.set_title('Threshold Sweep')
ax.set_xlabel('Threshold')
ax.legend()
fig.savefig(PLOTS_DIR / 'threshold_sweep.png', dpi=160, bbox_inches='tight')
display(fig)
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
heatmap = ax.imshow(cm_df.to_numpy(), cmap='Blues')
ax.set_xticks(range(cm_df.shape[1]), cm_df.columns)
ax.set_yticks(range(cm_df.shape[0]), cm_df.index)
ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')
for row_idx in range(cm_df.shape[0]):
    for col_idx in range(cm_df.shape[1]):
        ax.text(col_idx, row_idx, str(int(cm_df.iat[row_idx, col_idx])), ha='center', va='center', color='black')
fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
fig.savefig(PLOTS_DIR / 'confusion_matrix.png', dpi=160, bbox_inches='tight')
display(fig)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), constrained_layout=True)
axes[0].hist(test_scores_df.loc[test_scores_df['is_anomaly'] == 0, 'score'], bins=30, alpha=0.7, label='normal')
axes[0].hist(test_scores_df.loc[test_scores_df['is_anomaly'] == 1, 'score'], bins=30, alpha=0.7, label='anomaly')
axes[0].axvline(threshold, color='red', linestyle='--', label=f'val threshold = {threshold:.4f}')
axes[0].set_title('Test Score Distribution')
axes[0].set_xlabel('Anomaly score')
axes[0].legend()
axes[1].plot(threshold_sweep_df['threshold'], threshold_sweep_df['precision'], label='precision')
axes[1].plot(threshold_sweep_df['threshold'], threshold_sweep_df['recall'], label='recall')
axes[1].plot(threshold_sweep_df['threshold'], threshold_sweep_df['f1'], label='f1')
axes[1].axvline(threshold, color='red', linestyle='--', label='val threshold')
axes[1].axvline(float(best_sweep['threshold']), color='green', linestyle=':', label='best sweep')
axes[1].set_title('Threshold Sweep')
axes[1].set_xlabel('Threshold')
axes[1].legend()
heatmap = axes[2].imshow(cm_df.to_numpy(), cmap='Blues')
axes[2].set_xticks(range(cm_df.shape[1]), cm_df.columns)
axes[2].set_yticks(range(cm_df.shape[0]), cm_df.index)
axes[2].set_title('Confusion Matrix')
axes[2].set_xlabel('Predicted label')
axes[2].set_ylabel('True label')
for row_idx in range(cm_df.shape[0]):
    for col_idx in range(cm_df.shape[1]):
        axes[2].text(col_idx, row_idx, str(int(cm_df.iat[row_idx, col_idx])), ha='center', va='center', color='black')
fig.colorbar(heatmap, ax=axes[2], fraction=0.046, pad=0.04)
fig.savefig(PLOTS_DIR / 'score_distribution_sweep_confusion.png', dpi=160, bbox_inches='tight')
display(fig)
plt.close(fig)
display(cm_df)
display(threshold_sweep_df.sort_values('f1', ascending=False).head(10))
"""


VAE_SWEEP_CELL = """best_beta_row = beta_sweep_df.iloc[0].to_dict()
best_beta = float(best_beta_row['beta'])
best_tag = str(best_beta).replace('.', 'p')
best_run_dir = SWEEP_ROOT / f'beta_{best_tag}'
best_eval_summary = json.loads((best_run_dir / 'evaluation' / 'summary.json').read_text(encoding='utf-8'))
best_test_scores_df = pd.read_csv(best_run_dir / 'evaluation' / 'test_scores.csv')
best_threshold_sweep_df = pd.read_csv(best_run_dir / 'evaluation' / 'threshold_sweep.csv')
cm = best_eval_summary['metrics_at_validation_threshold'].get('confusion_matrix', [[0, 0], [0, 0]])
cm_df = pd.DataFrame(cm, index=['true_normal', 'true_anomaly'], columns=['pred_normal', 'pred_anomaly'])

print(f'Best beta by validation-threshold F1: {best_beta}')
print(f'Best run directory: {best_run_dir}')
display(pd.DataFrame([best_beta_row]))
display(pd.DataFrame([best_eval_summary['metrics_at_validation_threshold']]))
display(pd.DataFrame([best_eval_summary['best_threshold_sweep']]))

default_threshold = float(best_eval_summary['threshold'])
best_sweep_threshold = float(best_eval_summary['best_threshold_sweep']['threshold'])

fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
ax.hist(best_test_scores_df.loc[best_test_scores_df['is_anomaly'] == 0, 'score'], bins=30, alpha=0.7, label='normal')
ax.hist(best_test_scores_df.loc[best_test_scores_df['is_anomaly'] == 1, 'score'], bins=30, alpha=0.7, label='anomaly')
ax.axvline(default_threshold, color='red', linestyle='--', label=f'val threshold = {default_threshold:.4f}')
ax.set_title(f'Best-Beta Score Distribution (beta={best_beta})')
ax.set_xlabel('Anomaly score')
ax.legend()
fig.savefig(PLOTS_DIR / 'score_distribution.png', dpi=160, bbox_inches='tight')
fig.savefig(PLOTS_DIR / 'best_beta_score_distribution.png', dpi=160, bbox_inches='tight')
display(fig)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
ax.plot(best_threshold_sweep_df['threshold'], best_threshold_sweep_df['precision'], label='precision')
ax.plot(best_threshold_sweep_df['threshold'], best_threshold_sweep_df['recall'], label='recall')
ax.plot(best_threshold_sweep_df['threshold'], best_threshold_sweep_df['f1'], label='f1')
ax.axvline(default_threshold, color='red', linestyle='--', label='val threshold')
ax.axvline(best_sweep_threshold, color='green', linestyle=':', label='best sweep')
ax.set_title('Best-Beta Threshold Sweep')
ax.set_xlabel('Threshold')
ax.legend()
fig.savefig(PLOTS_DIR / 'threshold_sweep.png', dpi=160, bbox_inches='tight')
fig.savefig(PLOTS_DIR / 'best_beta_threshold_sweep.png', dpi=160, bbox_inches='tight')
display(fig)
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
heatmap = ax.imshow(cm_df.to_numpy(), cmap='Blues')
ax.set_xticks(range(cm_df.shape[1]), cm_df.columns)
ax.set_yticks(range(cm_df.shape[0]), cm_df.index)
ax.set_title('Best-Beta Confusion Matrix')
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')
for row_idx in range(cm_df.shape[0]):
    for col_idx in range(cm_df.shape[1]):
        ax.text(col_idx, row_idx, str(int(cm_df.iat[row_idx, col_idx])), ha='center', va='center', color='black')
fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
fig.savefig(PLOTS_DIR / 'confusion_matrix.png', dpi=160, bbox_inches='tight')
fig.savefig(PLOTS_DIR / 'best_beta_confusion_matrix.png', dpi=160, bbox_inches='tight')
display(fig)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), constrained_layout=True)
axes[0].hist(best_test_scores_df.loc[best_test_scores_df['is_anomaly'] == 0, 'score'], bins=30, alpha=0.7, label='normal')
axes[0].hist(best_test_scores_df.loc[best_test_scores_df['is_anomaly'] == 1, 'score'], bins=30, alpha=0.7, label='anomaly')
axes[0].axvline(default_threshold, color='red', linestyle='--', label=f'val threshold = {default_threshold:.4f}')
axes[0].set_title(f'Best-Beta Score Distribution (beta={best_beta})')
axes[0].set_xlabel('Anomaly score')
axes[0].legend()
axes[1].plot(best_threshold_sweep_df['threshold'], best_threshold_sweep_df['precision'], label='precision')
axes[1].plot(best_threshold_sweep_df['threshold'], best_threshold_sweep_df['recall'], label='recall')
axes[1].plot(best_threshold_sweep_df['threshold'], best_threshold_sweep_df['f1'], label='f1')
axes[1].axvline(default_threshold, color='red', linestyle='--', label='val threshold')
axes[1].axvline(best_sweep_threshold, color='green', linestyle=':', label='best sweep')
axes[1].set_title('Best-Beta Threshold Sweep')
axes[1].set_xlabel('Threshold')
axes[1].legend()
heatmap = axes[2].imshow(cm_df.to_numpy(), cmap='Blues')
axes[2].set_xticks(range(cm_df.shape[1]), cm_df.columns)
axes[2].set_yticks(range(cm_df.shape[0]), cm_df.index)
axes[2].set_title('Best-Beta Confusion Matrix')
axes[2].set_xlabel('Predicted label')
axes[2].set_ylabel('True label')
for row_idx in range(cm_df.shape[0]):
    for col_idx in range(cm_df.shape[1]):
        axes[2].text(col_idx, row_idx, str(int(cm_df.iat[row_idx, col_idx])), ha='center', va='center', color='black')
fig.colorbar(heatmap, ax=axes[2], fraction=0.046, pad=0.04)
fig.savefig(PLOTS_DIR / 'best_beta_distribution_sweep_confusion.png', dpi=160, bbox_inches='tight')
display(fig)
plt.close(fig)
display(cm_df)
"""


def patch_svdd_vae_notebooks() -> list[Path]:
    changed_paths: list[Path] = []
    targets = [
        (
            REPO_ROOT / "experiments/anomaly_detection/svdd/x64/baseline/notebook.ipynb",
            "score_distribution_sweep_confusion.png",
            SVDD_CELL,
        ),
        (
            REPO_ROOT / "experiments/anomaly_detection/vae/x64/baseline/notebook.ipynb",
            "score_distribution_sweep_confusion.png",
            VAE_BASE_CELL,
        ),
        (
            REPO_ROOT / "experiments/anomaly_detection/vae/x64/beta_sweep/notebook.ipynb",
            "best_beta_distribution_sweep_confusion.png",
            VAE_SWEEP_CELL,
        ),
    ]
    for path, needle, new_source in targets:
        if patch_exact_code_cell(path, needle, new_source):
            changed_paths.append(path)
    return changed_paths


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def normalize_autoencoder_artifacts() -> None:
    alias_map = {
        "training_curve.png": ["training_curve.png", "training_curves.png"],
        "confusion_matrix.png": ["confusion_matrix.png"],
        "threshold_sweep.png": ["threshold_sweep.png"],
        "score_histogram.png": ["score_histogram.png", "score_distribution.png"],
        "reconstruction_examples.png": ["reconstruction_examples.png"],
    }
    for root in AUTOENCODER_ARTIFACT_DIRS:
        plots_dir = root / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        for source_name, target_names in alias_map.items():
            src = root / source_name
            for target_name in target_names:
                copy_if_exists(src, plots_dir / target_name)
        for src in root.glob("failure_examples_*.png"):
            copy_if_exists(src, plots_dir / src.name)
        sweep_src = root.parent / "dropout_sweep_summary.png"
        if "batchnorm_dropout" in str(root):
            copy_if_exists(sweep_src, root.parent / "plots" / "dropout_sweep_summary.png")


def main() -> None:
    auto_changed = patch_autoencoder_notebooks()
    family_changed = patch_svdd_vae_notebooks()
    normalize_autoencoder_artifacts()

    print("Patched notebooks:")
    for path in [*auto_changed, *family_changed]:
        print(path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
