from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = REPO_ROOT / "experiments" / "anomaly_detection" / "teacher_student" / "resnet18" / "x64" / "main" / "notebook.ipynb"


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_notebook(path: Path, notebook: dict) -> None:
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def source_text(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_code_source(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)


def set_markdown_source(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)


def find_cell(notebook: dict, snippet: str) -> tuple[int, dict]:
    for idx, cell in enumerate(notebook["cells"]):
        if snippet in source_text(cell):
            return idx, cell
    raise ValueError(f"Could not find cell containing snippet: {snippet}")


def insert_after(notebook: dict, idx: int, new_cells: list[dict]) -> None:
    notebook["cells"][idx + 1:idx + 1] = new_cells


def markdown_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def clear_outputs(notebook: dict) -> None:
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []


def main() -> None:
    notebook = load_notebook(NOTEBOOK_PATH)

    _, config_cell = find_cell(notebook, 'CONFIG_NAME = "resnet18_teacher_main"')
    set_code_source(
        config_cell,
        """CONFIG_NAME = "resnet18_teacher_main"
CONFIG_PATH = REPO_ROOT / "experiments/anomaly_detection/teacher_student/resnet18/x64/main/train_config.toml"
THRESHOLD_QUANTILE = 0.95
RUN_TRAINING = False
RUN_DEFAULT_EVALUATION = False

RUN_ABLATION_SWEEP = False
ABLATION_PROFILE = "focused"  # "focused" or "full"
if ABLATION_PROFILE == "focused":
    ABLATION_LAYERS = ["layer1", "layer2"]
    ABLATION_TOPK_RATIOS = [0.15, 0.20, 0.25]
    ABLATION_EPOCHS_OVERRIDE = 15
else:
    ABLATION_LAYERS = ["layer1", "layer2", "layer3"]
    ABLATION_TOPK_RATIOS = [0.10, 0.15, 0.20, 0.25, 0.30]
    ABLATION_EPOCHS_OVERRIDE = 20
ABLATION_REUSE_EXISTING = True

SCORE_SWEEP_WEIGHTS = [
    (1.0, 1.0),
    (1.0, 0.0),
    (0.0, 1.0),
    (2.0, 1.0),
    (1.0, 2.0),
    (1.0, 0.5),
    (0.5, 1.0),
]
SCORE_SWEEP_REDUCTIONS = [
    ("mean", None),
    ("max", None),
    ("topk_mean", 0.01),
    ("topk_mean", 0.05),
    ("topk_mean", 0.10),
    ("topk_mean", 0.20),
]

config = load_toml(CONFIG_PATH)
output_dir = REPO_ROOT / config["run"]["output_dir"]
evaluation_dir = output_dir / "results" / "evaluation"
evaluation_dir.mkdir(parents=True, exist_ok=True)
plots_dir = output_dir / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)
best_model_path = output_dir / "checkpoints" / "best_model.pt"
print(f"Using config: {CONFIG_NAME} -> {CONFIG_PATH}")
config
""",
    )

    _, history_plot_cell = find_cell(notebook, 'fig, axes = plt.subplots(1, 2, figsize=(12, 4))')
    set_code_source(
        history_plot_cell,
        """fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history_df["epoch"], history_df["train_loss"], marker="o", label="train")
axes[0].plot(history_df["epoch"], history_df["val_loss"], marker="o", label="val")
axes[0].set_title("Teacher-Student Total Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(history_df["epoch"], history_df["train_distillation_loss"], marker="o", label="train distillation")
axes[1].plot(history_df["epoch"], history_df["val_distillation_loss"], marker="o", label="val distillation")
axes[1].plot(history_df["epoch"], history_df["train_autoencoder_loss"], marker="o", linestyle="--", label="train feature AE")
axes[1].plot(history_df["epoch"], history_df["val_autoencoder_loss"], marker="o", linestyle="--", label="val feature AE")
axes[1].set_title("Teacher-Student Branch Losses")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

training_curves_path = plots_dir / "training_curves.png"
plt.tight_layout()
plt.savefig(training_curves_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"Saved training curves to {training_curves_path}")
""",
    )

    _, ablation_plot_cell = find_cell(notebook, 'if ablation_df.empty:')
    set_code_source(
        ablation_plot_cell,
        """if ablation_df.empty:
    print("No ablation sweep results available yet.")
else:
    display(ablation_df)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    top_rows = ablation_df.head(8)

    axes[0].bar(top_rows["name"], top_rows["f1"], color="#2a9d8f")
    axes[0].set_title("Top Ablations by F1")
    axes[0].set_ylabel("F1")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(top_rows["name"], top_rows["auroc"], color="#457b9d")
    axes[1].set_title("Top Ablations by AUROC")
    axes[1].set_ylabel("AUROC")
    axes[1].tick_params(axis="x", rotation=45)

    ablation_plot_path = plots_dir / "ablation_summary.png"
    plt.tight_layout()
    plt.savefig(ablation_plot_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved ablation summary plot to {ablation_plot_path}")
""",
    )

    _, comparison_cell = find_cell(notebook, 'default_summary_path = local_evaluation_dir / "summary.json"')
    set_code_source(
        comparison_cell,
        """import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

repo_root = REPO_ROOT if "REPO_ROOT" in globals() else Path.cwd()
if "evaluation_dir" in globals():
    local_evaluation_dir = Path(evaluation_dir)
else:
    candidate_dirs = [
        repo_root / "experiments" / "anomaly_detection" / "teacher_student" / "resnet18" / "x64" / "main" / "artifacts" / "ts_resnet18" / "results" / "evaluation",
    ]
    existing_dirs = [path for path in candidate_dirs if path.exists()]
    if not existing_dirs:
        raise FileNotFoundError("Could not find a TS-Res18 evaluation directory.")
    local_evaluation_dir = existing_dirs[0]

default_summary_path = local_evaluation_dir / "summary.json"
selected_variant_path = local_evaluation_dir / "selected_score_variant.json"
if not default_summary_path.exists():
    raise FileNotFoundError(f"Evaluation summary not found: {default_summary_path}")
if not selected_variant_path.exists():
    raise FileNotFoundError(f"Selected score variant not found: {selected_variant_path}")

default_eval_summary = json.loads(default_summary_path.read_text(encoding="utf-8"))
selected_variant = json.loads(selected_variant_path.read_text(encoding="utf-8"))

default_row = pd.DataFrame(
    [
        {
            "name": "default_config_score",
            "precision": default_eval_summary["metrics_at_validation_threshold"]["precision"],
            "recall": default_eval_summary["metrics_at_validation_threshold"]["recall"],
            "f1": default_eval_summary["metrics_at_validation_threshold"]["f1"],
            "auroc": default_eval_summary["metrics_at_validation_threshold"]["auroc"],
            "auprc": default_eval_summary["metrics_at_validation_threshold"]["auprc"],
            "best_sweep_f1": default_eval_summary["best_threshold_sweep"]["f1"],
        }
    ]
)
selected_row_df = pd.DataFrame([selected_variant])
comparison_df = pd.concat(
    [
        default_row,
        selected_row_df[["name", "precision", "recall", "f1", "auroc", "auprc", "best_sweep_f1"]],
    ],
    ignore_index=True,
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(["default", "best sweep variant"], comparison_df["f1"], color=["#8d99ae", "#2a9d8f"])
axes[0].set_title("Validation-Threshold F1")
axes[0].set_ylabel("F1")

axes[1].bar(["default", "best sweep variant"], comparison_df["auroc"], color=["#8d99ae", "#2a9d8f"])
axes[1].set_title("AUROC")
axes[1].set_ylabel("AUROC")

score_variant_plot_path = plots_dir / "score_variant_comparison.png"
plt.tight_layout()
plt.savefig(score_variant_plot_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"Saved score variant comparison plot to {score_variant_plot_path}")
""",
    )

    defect_idx, _ = find_cell(notebook, "# DEFECT_BREAKDOWN_CELL")
    followup_marker = "## Saved Evaluation Plots"
    if not any(followup_marker in source_text(cell) for cell in notebook["cells"]):
        insert_after(
            notebook,
            defect_idx + 1,
            [
                markdown_cell(
                    """## Saved Evaluation Plots

This section turns the saved validation/test score tables into the standard review plots for the selected deployed checkpoint. It does not retrain the model; it only reads the saved evaluation artifacts and writes figure files into `artifacts/ts_resnet18/plots/`.
"""
                ),
                code_cell(
                    """plot_dir = plots_dir if "plots_dir" in globals() else local_evaluation_dir.parent.parent / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)

val_scores_df = pd.read_csv(local_evaluation_dir / "val_scores.csv")
test_scores_df = pd.read_csv(local_evaluation_dir / "test_scores.csv")
threshold_sweep_df = pd.read_csv(local_evaluation_dir / "threshold_sweep.csv")

fig, ax = plt.subplots(figsize=(8, 4.5))
normal_scores = analysis_df.loc[analysis_df["is_anomaly"] == 0, "score"]
anomaly_scores = analysis_df.loc[analysis_df["is_anomaly"] == 1, "score"]
ax.hist(normal_scores, bins=40, alpha=0.65, label="normal", color="#457b9d")
ax.hist(anomaly_scores, bins=40, alpha=0.65, label="anomaly", color="#e76f51")
ax.axvline(selected_threshold, color="black", linestyle="--", label=f"threshold={selected_threshold:.4f}")
ax.set_title("Selected Score Distribution")
ax.set_xlabel("score")
ax.set_ylabel("count")
ax.legend()
score_distribution_path = plot_dir / "score_distribution.png"
plt.tight_layout()
plt.savefig(score_distribution_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"Saved score distribution plot to {score_distribution_path}")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(threshold_sweep_df["threshold"], threshold_sweep_df["precision"], label="precision", color="#457b9d")
ax.plot(threshold_sweep_df["threshold"], threshold_sweep_df["recall"], label="recall", color="#f4a261")
ax.plot(threshold_sweep_df["threshold"], threshold_sweep_df["f1"], label="f1", color="#2a9d8f")
ax.axvline(selected_threshold, color="black", linestyle="--", label="selected threshold")
ax.set_title("Threshold Sweep")
ax.set_xlabel("threshold")
ax.set_ylabel("metric")
ax.legend()
threshold_sweep_path = plot_dir / "threshold_sweep.png"
plt.tight_layout()
plt.savefig(threshold_sweep_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"Saved threshold sweep plot to {threshold_sweep_path}")

tp = int(((analysis_df["is_anomaly"] == 1) & (analysis_df["predicted_anomaly"] == 1)).sum())
fn = int(((analysis_df["is_anomaly"] == 1) & (analysis_df["predicted_anomaly"] == 0)).sum())
fp = int(((analysis_df["is_anomaly"] == 0) & (analysis_df["predicted_anomaly"] == 1)).sum())
tn = int(((analysis_df["is_anomaly"] == 0) & (analysis_df["predicted_anomaly"] == 0)).sum())
conf_matrix = np.array([[tn, fp], [fn, tp]])

fig, ax = plt.subplots(figsize=(4.8, 4.2))
im = ax.imshow(conf_matrix, cmap="Blues")
for (row_idx, col_idx), value in np.ndenumerate(conf_matrix):
    ax.text(col_idx, row_idx, f"{value}", ha="center", va="center", color="black")
ax.set_xticks([0, 1], ["pred normal", "pred anomaly"])
ax.set_yticks([0, 1], ["true normal", "true anomaly"])
ax.set_title("Confusion Matrix")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
confusion_matrix_path = plot_dir / "confusion_matrix.png"
plt.tight_layout()
plt.savefig(confusion_matrix_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"Saved confusion matrix plot to {confusion_matrix_path}")

fig, ax = plt.subplots(figsize=(9, 4.5))
sorted_breakdown = defect_breakdown_df.sort_values(["recall", "count", "defect_type"], ascending=[True, False, True])
ax.barh(sorted_breakdown["defect_type"], sorted_breakdown["recall"], color="#2a9d8f")
ax.set_xlim(0, 1.0)
ax.set_xlabel("recall")
ax.set_title("Defect-Type Recall")
defect_breakdown_plot_path = plot_dir / "defect_breakdown.png"
plt.tight_layout()
plt.savefig(defect_breakdown_plot_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"Saved defect breakdown plot to {defect_breakdown_plot_path}")
""",
                ),
            ],
        )

    clear_outputs(notebook)
    save_notebook(NOTEBOOK_PATH, notebook)
    print(f"Updated {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
