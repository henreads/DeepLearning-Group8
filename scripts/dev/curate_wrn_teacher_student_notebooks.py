from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

LAYER2_NOTEBOOK = REPO_ROOT / "experiments" / "anomaly_detection" / "teacher_student" / "wideresnet50_2" / "x64" / "layer2_self_contained" / "notebook.ipynb"
MULTILAYER_NOTEBOOK = REPO_ROOT / "experiments" / "anomaly_detection" / "teacher_student" / "wideresnet50_2" / "x64" / "multilayer_self_contained" / "notebook.ipynb"


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_notebook(path: Path, notebook: dict) -> None:
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")


def set_source(cell: dict, source: str) -> None:
    cell["source"] = source.splitlines(keepends=True)


def patch_layer2() -> None:
    nb = load_notebook(LAYER2_NOTEBOOK)

    cell2 = """CONFIG = {
    "run": {
        "output_dir": "experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/layer2_self_contained/artifacts/ts_wideresnet50_layer2",
        "seed": 42,
        "run_training": False,
        "run_score_sweep": False,
    },
    "data": {
        "repo_root": ".",
        "metadata_csv": "data/processed/x64/wm811k/metadata_50k_5pct.csv",
        "image_size": 64,
        "batch_size": 64,
        "num_workers": 8,
    },
    "training": {
        "epochs": 30,
        "learning_rate": 3e-4,
        "weight_decay": 1e-5,
        "device": "cuda",
        "early_stopping_patience": 5,
        "early_stopping_min_delta": 1e-4,
        "checkpoint_every": 5,
    },
    "model": {
        "teacher_backbone": "wideresnet50_2",
        "teacher_layer": "layer2",
        "teacher_pretrained": True,
        "teacher_input_size": 224,
        "normalize_teacher_input": True,
        "feature_autoencoder_hidden_dim": 128,
        "student_weight": 1.0,
        "autoencoder_weight": 1.0,
        "score_student_weight": 1.0,
        "score_autoencoder_weight": 0.0,
        "reduction": "topk_mean",
        "topk_ratio": 0.20,
    },
    "scoring": {
        "threshold_quantile": 0.95,
    }
}

cwd = Path.cwd().resolve()
candidate_roots = [cwd, *cwd.parents]
REPO_ROOT = None
for candidate in candidate_roots:
    if (candidate / "src" / "wafer_defect").exists() and (candidate / "configs").exists():
        REPO_ROOT = candidate
        break

if REPO_ROOT is None:
    raise RuntimeError("Could not locate repo root containing src/wafer_defect and configs/")

RUN_TRAINING = bool(CONFIG["run"].get("run_training", False))
RUN_SCORE_SWEEP = bool(CONFIG["run"].get("run_score_sweep", False))

image_size = int(CONFIG["data"]["image_size"])
batch_size = int(CONFIG["data"]["batch_size"])
num_workers = int(CONFIG["data"]["num_workers"])
requested_device_name = str(CONFIG["training"]["device"])
device = torch.device("cuda" if requested_device_name == "auto" and torch.cuda.is_available() else requested_device_name)

seed = int(CONFIG["run"]["seed"])
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

CONFIG
"""
    set_source(nb["cells"][2], cell2)

    cell8 = """# -----------------------------
# Build model and optimizer
# -----------------------------
output_dir = REPO_ROOT / CONFIG["run"]["output_dir"]
checkpoint_dir = output_dir / "checkpoints"
results_dir = output_dir / "results"
evaluation_dir = results_dir / "evaluation"
plots_dir = output_dir / "plots"

for path in [output_dir, checkpoint_dir, results_dir, evaluation_dir, plots_dir]:
    path.mkdir(parents=True, exist_ok=True)

# Save config for reproducibility
with (results_dir / "config.json").open("w", encoding="utf-8") as f:
    json.dump(CONFIG, f, indent=2)

print("Experiment outputs saved to:", output_dir)

model = WideTeacherTSDistillationModel(CONFIG, image_size=image_size).to(device)
optimizer = torch.optim.Adam(
    (p for p in model.parameters() if p.requires_grad),
    lr=float(CONFIG["training"]["learning_rate"]),
    weight_decay=float(CONFIG["training"]["weight_decay"]),
)

print("Model device:", next(model.parameters()).device)
print("Teacher layer:", CONFIG["model"]["teacher_layer"])
print("Feature dim:", model.feature_dim)
"""
    set_source(nb["cells"][8], cell8)

    cell10 = """if not history_df.empty:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history_df["epoch"], history_df["train_loss"], marker="o", label="train")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], marker="o", label="val")
    axes[0].set_title("Wide TS total loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["train_distillation_loss"], marker="o", label="train student")
    axes[1].plot(history_df["epoch"], history_df["val_distillation_loss"], marker="o", label="val student")
    axes[1].plot(history_df["epoch"], history_df["train_autoencoder_loss"], marker="o", label="train auto")
    axes[1].plot(history_df["epoch"], history_df["val_autoencoder_loss"], marker="o", label="val auto")
    axes[1].set_title("Component losses")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(plots_dir / "training_curves.png", dpi=200, bbox_inches="tight")
    plt.show()
else:
    print("No training history to plot.")
"""
    set_source(nb["cells"][10], cell10)

    cell13 = """fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(val_scores_df["score"], bins=30, alpha=0.8, color="#4d908e")
axes[0].axvline(default_eval_summary["threshold"], color="red", linestyle="--", label=f"threshold={default_eval_summary['threshold']:.4f}")
axes[0].set_title("Validation score distribution")
axes[0].set_xlabel("Anomaly score")
axes[0].set_ylabel("Count")
axes[0].legend()

axes[1].hist(test_scores_df.loc[test_scores_df["is_anomaly"] == 0, "score"], bins=30, alpha=0.7, label="normal")
axes[1].hist(test_scores_df.loc[test_scores_df["is_anomaly"] == 1, "score"], bins=30, alpha=0.7, label="anomaly")
axes[1].axvline(default_eval_summary["threshold"], color="red", linestyle="--", label=f"threshold={default_eval_summary['threshold']:.4f}")
axes[1].set_title("Test score distribution")
axes[1].set_xlabel("Anomaly score")
axes[1].set_ylabel("Count")
axes[1].legend()

plt.tight_layout()
fig.savefig(plots_dir / "score_distribution.png", dpi=200, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(8, 4.5))
threshold_sweep_df.plot(x="threshold", y=["precision", "recall", "f1"], ax=ax)
ax.axvline(default_eval_summary["threshold"], color="red", linestyle="--", linewidth=1.75, label="validation threshold")
ax.set_title("WideResNet50-2 TS threshold sweep")
ax.set_ylabel("metric")
ax.legend()
plt.tight_layout()
fig.savefig(plots_dir / "threshold_sweep.png", dpi=200, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1], labels=["pred_normal", "pred_anomaly"])
ax.set_yticks([0, 1], labels=["true_normal", "true_anomaly"])
ax.set_title("Confusion matrix")
for row in range(cm.shape[0]):
    for col in range(cm.shape[1]):
        ax.text(col, row, int(cm[row, col]), ha="center", va="center", color="#111111")
plt.tight_layout()
fig.savefig(plots_dir / "confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.show()
"""
    set_source(nb["cells"][13], cell13)

    cell16 = """if score_sweep_df.empty:
    print("No score sweep results available.")
else:
    default_row = pd.DataFrame([{
        "name": "default_config_score",
        "precision": default_eval_summary["metrics_at_validation_threshold"]["precision"],
        "recall": default_eval_summary["metrics_at_validation_threshold"]["recall"],
        "f1": default_eval_summary["metrics_at_validation_threshold"]["f1"],
        "auroc": default_eval_summary["metrics_at_validation_threshold"]["auroc"],
        "auprc": default_eval_summary["metrics_at_validation_threshold"]["auprc"],
        "best_sweep_f1": default_eval_summary["best_threshold_sweep"]["f1"],
    }])

    plot_df = pd.concat([
        default_row,
        score_sweep_df[["name", "precision", "recall", "f1", "auroc", "auprc", "best_sweep_f1"]]
    ], ignore_index=True).sort_values("f1", ascending=False).head(12)

    display(plot_df)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(plot_df["name"][::-1], plot_df["f1"][::-1])
    ax.set_xlabel("Validation-threshold F1")
    ax.set_title("WideResNet50-2 TS score sweep")
    plt.tight_layout()
    fig.savefig(plots_dir / "score_sweep_comparison.png", dpi=200, bbox_inches="tight")
    plt.show()
"""
    set_source(nb["cells"][16], cell16)

    cell18 = "".join(nb["cells"][18]["source"])
    if 'defect_breakdown_df.to_csv(output_path, index=False)' in cell18 and 'defect_breakdown.png' not in cell18:
        cell18 += """

fig, ax = plt.subplots(figsize=(10, 4.8))
ax.bar(defect_breakdown_df["defect_type"], defect_breakdown_df["recall"], color="#457b9d")
ax.set_ylim(0.0, 1.05)
ax.set_title("Recall by defect type for selected score variant")
ax.set_ylabel("recall")
ax.tick_params(axis="x", rotation=25)
plt.tight_layout()
fig.savefig(plots_dir / "defect_breakdown.png", dpi=200, bbox_inches="tight")
plt.show()
"""
        set_source(nb["cells"][18], cell18)

    save_notebook(LAYER2_NOTEBOOK, nb)


def patch_multilayer() -> None:
    nb = load_notebook(MULTILAYER_NOTEBOOK)

    cell0 = """# WideResNet50-2 Teacher-Student Distillation (Multilayer Self-Contained)

This notebook is **self-contained**: it inlines the dataset loader, scoring helpers, model definition, training loop, evaluation, and optional score sweep.
You do **not** need extra helper modules or TOML files.

It is set up to be safe for submission review:

- by default it creates the local artifact folder structure
- it only trains when you explicitly enable `run_training`
- it only runs the score sweep when you explicitly enable `run_score_sweep`
- if no checkpoint exists yet, the notebook explains that a local rerun is required instead of crashing immediately
"""
    set_source(nb["cells"][0], cell0)

    cell2 = """CONFIG = {
    "run": {
        "output_dir": "experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/multilayer_self_contained/artifacts/ts_wideresnet50_multilayer",
        "seed": 42,
        "run_training": False,
        "run_score_sweep": False,
    },
    "data": {
        "repo_root": ".",
        "metadata_csv": "data/processed/x64/wm811k/metadata_50k_5pct.csv",
        "image_size": 64,
        "batch_size": 64,
        "num_workers": 8,
    },
    "training": {
        "epochs": 30,
        "learning_rate": 3e-4,
        "weight_decay": 1e-5,
        "device": "cuda",
        "early_stopping_patience": 5,
        "early_stopping_min_delta": 1e-4,
        "checkpoint_every": 5,
    },
    "model": {
        "teacher_backbone": "wideresnet50_2",
        "teacher_layers": ["layer2", "layer3"],
        "teacher_pretrained": True,
        "teacher_input_size": 224,
        "normalize_teacher_input": True,
        "feature_autoencoder_hidden_dim": 128,
        "student_weight": 1.0,
        "autoencoder_weight": 1.0,
        "score_student_weight": 2.0,
        "score_autoencoder_weight": 1.0,
        "reduction": "topk_mean",
        "topk_ratio": 0.25,
    },
    "scoring": {
        "threshold_quantile": 0.95,
    }
}

cwd = Path.cwd().resolve()
candidate_roots = [cwd, *cwd.parents]
REPO_ROOT = None
for candidate in candidate_roots:
    if (candidate / "src" / "wafer_defect").exists() and (candidate / "configs").exists():
        REPO_ROOT = candidate
        break

if REPO_ROOT is None:
    raise RuntimeError("Could not locate repo root containing src/wafer_defect and configs/")

RUN_TRAINING = bool(CONFIG["run"].get("run_training", False))
RUN_SCORE_SWEEP = bool(CONFIG["run"].get("run_score_sweep", False))

image_size = int(CONFIG["data"]["image_size"])
batch_size = int(CONFIG["data"]["batch_size"])
num_workers = int(CONFIG["data"]["num_workers"])
requested_device_name = str(CONFIG["training"]["device"])
device = torch.device("cuda" if requested_device_name == "auto" and torch.cuda.is_available() else requested_device_name)

seed = int(CONFIG["run"]["seed"])
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

CONFIG
"""
    set_source(nb["cells"][2], cell2)

    cell4 = """# =========================
# HELPER FUNCTIONS + PREPARE DATASET FROM LSWMD.pkl
# same protocol as the repo benchmark builder
# =========================

import os
import sys
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.core.indexes as core_indexes
import torch
import torch.nn.functional as F

LABEL_NORMAL = "none"
LABEL_DEFECT = "pattern"

RAW_PICKLE = str(REPO_ROOT / "data" / "raw" / "LSWMD.pkl")
OUTPUT_ROOT = str(REPO_ROOT)
IMAGE_SIZE = 64
NORMAL_LIMIT = 50000
TEST_DEFECT_FRACTION = 0.05
SEED = 42

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def read_legacy_pickle(path: Path) -> pd.DataFrame:
    sys.modules["pandas.indexes"] = core_indexes
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")

def unwrap_legacy_value(value):
    if value is None:
        return ""
    if hasattr(value, "size") and getattr(value, "size") == 0:
        return ""
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return str(value).strip()

def normalize_map(wafer_map: np.ndarray, image_size: int) -> np.ndarray:
    wafer_map = np.asarray(wafer_map, dtype=np.float32)
    wafer_map = wafer_map / 2.0
    tensor = torch.from_numpy(wafer_map).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=(image_size, image_size), mode="nearest")
    return resized.squeeze(0).numpy().astype(np.float32)
"""
    set_source(nb["cells"][4], cell4)

    cell8 = """# -----------------------------
# Build model and optimizer
# -----------------------------
output_dir = REPO_ROOT / CONFIG["run"]["output_dir"]
checkpoint_dir = output_dir / "checkpoints"
results_dir = output_dir / "results"
evaluation_dir = results_dir / "evaluation"
plots_dir = output_dir / "plots"

for path in [output_dir, checkpoint_dir, results_dir, evaluation_dir, plots_dir]:
    path.mkdir(parents=True, exist_ok=True)

# Save config for reproducibility
with (results_dir / "config.json").open("w", encoding="utf-8") as f:
    json.dump(CONFIG, f, indent=2)

print("Experiment outputs saved to:", output_dir)

model = WideTeacherTSDistillationModel(CONFIG, image_size=image_size).to(device)
optimizer = torch.optim.Adam(
    (p for p in model.parameters() if p.requires_grad),
    lr=float(CONFIG["training"]["learning_rate"]),
    weight_decay=float(CONFIG["training"]["weight_decay"]),
)

checkpoint_path = checkpoint_dir / "best_model.pt"
HAS_CHECKPOINT = checkpoint_path.exists()
if not HAS_CHECKPOINT:
    print(f"No saved multilayer checkpoint found yet at {checkpoint_path}. Leave the notebook in review mode or set run_training=True to generate artifacts.")

print("Model device:", next(model.parameters()).device)
print("Teacher layers:", CONFIG["model"]["teacher_layers"])
print("Feature dims:", model.feature_dims)
"""
    set_source(nb["cells"][8], cell8)

    cell10 = """if not history_df.empty:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history_df["epoch"], history_df["train_loss"], marker="o", label="train")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], marker="o", label="val")
    axes[0].set_title("Wide TS total loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["train_distillation_loss"], marker="o", label="train student")
    axes[1].plot(history_df["epoch"], history_df["val_distillation_loss"], marker="o", label="val student")
    axes[1].plot(history_df["epoch"], history_df["train_autoencoder_loss"], marker="o", label="train auto")
    axes[1].plot(history_df["epoch"], history_df["val_autoencoder_loss"], marker="o", label="val auto")
    axes[1].set_title("Component losses")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(plots_dir / "training_curves.png", dpi=200, bbox_inches="tight")
    plt.show()
else:
    print("No training history to plot yet. Run training first if you want to generate these curves.")
"""
    set_source(nb["cells"][10], cell10)

    cell11 = """# -----------------------------
# Evaluate default configured score
# -----------------------------
checkpoint = None
default_eval_summary = None
val_scores_df = pd.DataFrame()
test_scores_df = pd.DataFrame()
threshold_sweep_df = pd.DataFrame()

if not checkpoint_path.exists():
    print(f"No saved multilayer checkpoint found at {checkpoint_path}. Set run_training=True to create checkpoints and evaluation artifacts.")
else:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    def collect_scores(model: nn.Module, dataloader: DataLoader, device: torch.device, desc: str):
        scores = []
        labels = []
        with torch.inference_mode():
            iterator = tqdm(dataloader, desc=desc)
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if device.type == "cuda"
                else torch.autocast(device_type="cpu", enabled=False)
            )
            with autocast_context:
                for inputs, batch_labels in iterator:
                    inputs = inputs.to(device, non_blocking=True)
                    batch_scores = model(inputs)
                    scores.append(batch_scores.detach().cpu())
                    labels.append(batch_labels.cpu())
        return torch.cat(scores, dim=0).numpy(), torch.cat(labels, dim=0).numpy()

    student_scale, auto_scale = estimate_ts_error_scales(model, val_loader, device)
    model.set_error_scales(student_scale, auto_scale)

    val_scores, val_labels = collect_scores(model, val_loader, device, desc="Val scores")
    test_scores, test_labels = collect_scores(model, test_loader, device, desc="Test scores")

    val_scores_df = pd.DataFrame({"score": val_scores, "is_anomaly": val_labels.astype(int)})
    test_scores_df = pd.DataFrame({"score": test_scores, "is_anomaly": test_labels.astype(int)})

    threshold_quantile = float(CONFIG["scoring"].get("threshold_quantile", 0.95))
    val_normal_scores = val_scores_df.loc[val_scores_df["is_anomaly"] == 0, "score"]
    threshold = float(val_normal_scores.quantile(threshold_quantile))

    metrics = summarize_threshold_metrics(test_labels.astype(int), test_scores, threshold)
    threshold_sweep_df, best_sweep = sweep_threshold_metrics(test_labels.astype(int), test_scores)

    val_scores_df.to_csv(evaluation_dir / "val_scores.csv", index=False)
    test_scores_df.to_csv(evaluation_dir / "test_scores.csv", index=False)
    threshold_sweep_df.to_csv(evaluation_dir / "threshold_sweep.csv", index=False)

    default_eval_summary = {
        "model_type": "ts_distillation",
        "threshold_quantile": threshold_quantile,
        "threshold": threshold,
        "metrics_at_validation_threshold": metrics,
        "best_threshold_sweep": best_sweep,
    }
    (evaluation_dir / "summary.json").write_text(json.dumps(default_eval_summary, indent=2), encoding="utf-8")
"""
    set_source(nb["cells"][11], cell11)

    cell12 = """if default_eval_summary is None:
    print("Default multilayer evaluation is not available yet. Generate a checkpoint first to see metrics.")
else:
    metrics_df = pd.DataFrame([
        {"metric": "precision", "value": default_eval_summary["metrics_at_validation_threshold"]["precision"]},
        {"metric": "recall", "value": default_eval_summary["metrics_at_validation_threshold"]["recall"]},
        {"metric": "f1", "value": default_eval_summary["metrics_at_validation_threshold"]["f1"]},
        {"metric": "auroc", "value": default_eval_summary["metrics_at_validation_threshold"]["auroc"]},
        {"metric": "auprc", "value": default_eval_summary["metrics_at_validation_threshold"]["auprc"]},
        {"metric": "threshold", "value": default_eval_summary["threshold"]},
    ])
    display(metrics_df)

    cm = np.array(default_eval_summary["metrics_at_validation_threshold"]["confusion_matrix"])
    cm_df = pd.DataFrame(cm, index=["true_normal", "true_anomaly"], columns=["pred_normal", "pred_anomaly"])
    display(cm_df)

    best = default_eval_summary["best_threshold_sweep"]
    print(
        f"Best sweep threshold: {best['threshold']:.6f} | "
        f"precision={best['precision']:.4f}, recall={best['recall']:.4f}, f1={best['f1']:.4f}"
    )
"""
    set_source(nb["cells"][12], cell12)

    cell13 = """if default_eval_summary is None:
    print("No saved multilayer evaluation plots yet. Generate a checkpoint first to populate this section.")
else:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(val_scores_df["score"], bins=30, alpha=0.8, color="#4d908e")
    axes[0].axvline(default_eval_summary["threshold"], color="red", linestyle="--", label=f"threshold={default_eval_summary['threshold']:.4f}")
    axes[0].set_title("Validation score distribution")
    axes[0].set_xlabel("Anomaly score")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    axes[1].hist(test_scores_df.loc[test_scores_df["is_anomaly"] == 0, "score"], bins=30, alpha=0.7, label="normal")
    axes[1].hist(test_scores_df.loc[test_scores_df["is_anomaly"] == 1, "score"], bins=30, alpha=0.7, label="anomaly")
    axes[1].axvline(default_eval_summary["threshold"], color="red", linestyle="--", label=f"threshold={default_eval_summary['threshold']:.4f}")
    axes[1].set_title("Test score distribution")
    axes[1].set_xlabel("Anomaly score")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(plots_dir / "score_distribution.png", dpi=200, bbox_inches="tight")
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    threshold_sweep_df.plot(x="threshold", y=["precision", "recall", "f1"], ax=ax)
    ax.axvline(default_eval_summary["threshold"], color="red", linestyle="--", linewidth=1.75, label="validation threshold")
    ax.set_title("WideResNet50-2 multilayer threshold sweep")
    ax.set_ylabel("metric")
    ax.legend()
    plt.tight_layout()
    fig.savefig(plots_dir / "threshold_sweep.png", dpi=200, bbox_inches="tight")
    plt.show()

    cm = np.array(default_eval_summary["metrics_at_validation_threshold"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["pred_normal", "pred_anomaly"])
    ax.set_yticks([0, 1], labels=["true_normal", "true_anomaly"])
    ax.set_title("Confusion matrix")
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(col, row, int(cm[row, col]), ha="center", va="center", color="#111111")
    plt.tight_layout()
    fig.savefig(plots_dir / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.show()
"""
    set_source(nb["cells"][13], cell13)

    cell16 = """if score_sweep_df.empty:
    print("No score sweep results available. Generate a checkpoint and enable run_score_sweep=True to populate this section.")
else:
    default_row = pd.DataFrame([{
        "name": "default_config_score",
        "precision": default_eval_summary["metrics_at_validation_threshold"]["precision"],
        "recall": default_eval_summary["metrics_at_validation_threshold"]["recall"],
        "f1": default_eval_summary["metrics_at_validation_threshold"]["f1"],
        "auroc": default_eval_summary["metrics_at_validation_threshold"]["auroc"],
        "auprc": default_eval_summary["metrics_at_validation_threshold"]["auprc"],
        "best_sweep_f1": default_eval_summary["best_threshold_sweep"]["f1"],
    }])

    plot_df = pd.concat([
        default_row,
        score_sweep_df[["name", "precision", "recall", "f1", "auroc", "auprc", "best_sweep_f1"]]
    ], ignore_index=True).sort_values("f1", ascending=False).head(12)

    display(plot_df)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(plot_df["name"][::-1], plot_df["f1"][::-1])
    ax.set_xlabel("Validation-threshold F1")
    ax.set_title("WideResNet50-2 multilayer score sweep")
    plt.tight_layout()
    fig.savefig(plots_dir / "score_sweep_comparison.png", dpi=200, bbox_inches="tight")
    plt.show()
"""
    set_source(nb["cells"][16], cell16)

    cell18 = """# DEFECT_BREAKDOWN_CELL
score_sweep_path = evaluation_dir / "score_sweep.csv"
if ("score_sweep_df" not in globals()) or score_sweep_df.empty:
    if score_sweep_path.exists():
        score_sweep_df = pd.read_csv(score_sweep_path)
    else:
        score_sweep_df = pd.DataFrame()

if score_sweep_df.empty:
    print("No score sweep file is available yet for the multilayer branch. Run training and the score sweep first to generate a defect breakdown.")
else:
    selected_variant = score_sweep_df.iloc[0].to_dict()

    required_globals = ["model", "val_loader", "test_loader", "device", "test_dataset"]
    missing_globals = [name for name in required_globals if name not in globals()]
    if missing_globals:
        raise RuntimeError(
            "No retraining needed. Rerun the notebook setup/evaluation cells so these objects exist: "
            + ", ".join(missing_globals)
        )

    if any(name not in globals() for name in ["val_student_maps", "val_auto_maps", "val_labels", "test_student_maps", "test_auto_maps", "test_labels"]):
        val_student_maps, val_auto_maps, val_labels = collect_normalized_maps(model, val_loader, device, desc="Val maps (defect breakdown)")
        test_student_maps, test_auto_maps, test_labels = collect_normalized_maps(model, test_loader, device, desc="Test maps (defect breakdown)")

    student_weight = float(selected_variant["student_weight"])
    auto_weight = float(selected_variant["auto_weight"])
    reduction = str(selected_variant["reduction"])
    topk_ratio_value = selected_variant.get("topk_ratio", np.nan)
    topk_ratio = None if pd.isna(topk_ratio_value) else float(topk_ratio_value)
    selected_threshold = float(selected_variant["threshold"])

    selected_test_map = student_weight * test_student_maps + auto_weight * test_auto_maps
    selected_test_scores = np.asarray(reduce_map_np(selected_test_map, reduction, topk_ratio)).reshape(-1)

    analysis_df = test_dataset.metadata.reset_index(drop=True).copy()
    analysis_df["score"] = selected_test_scores
    analysis_df["predicted_anomaly"] = (analysis_df["score"] > selected_threshold).astype(int)

    defect_breakdown_df = (
        analysis_df.loc[analysis_df["is_anomaly"] == 1]
        .groupby("defect_type")
        .agg(
            count=("defect_type", "size"),
            detected=("predicted_anomaly", "sum"),
            mean_score=("score", "mean"),
            median_score=("score", "median"),
        )
        .reset_index()
    )
    defect_breakdown_df["detected"] = defect_breakdown_df["detected"].astype(int)
    defect_breakdown_df["missed"] = defect_breakdown_df["count"] - defect_breakdown_df["detected"]
    defect_breakdown_df["recall"] = defect_breakdown_df["detected"] / defect_breakdown_df["count"]
    defect_breakdown_df = defect_breakdown_df.sort_values(["recall", "count", "defect_type"], ascending=[True, False, True]).reset_index(drop=True)

    display(defect_breakdown_df)
    output_path = evaluation_dir / f"{selected_variant['name']}_defect_breakdown.csv"
    defect_breakdown_df.to_csv(output_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(defect_breakdown_df["defect_type"], defect_breakdown_df["recall"], color="#457b9d")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Recall by defect type for selected score variant")
    ax.set_ylabel("recall")
    ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    fig.savefig(plots_dir / "defect_breakdown.png", dpi=200, bbox_inches="tight")
    plt.show()
"""
    set_source(nb["cells"][18], cell18)

    save_notebook(MULTILAYER_NOTEBOOK, nb)


def main() -> None:
    patch_layer2()
    patch_multilayer()


if __name__ == "__main__":
    main()
