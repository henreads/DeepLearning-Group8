from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SEED07_NOTEBOOK = REPO_ROOT / "experiments/classifier/multiclass/x64/seed07/notebook.ipynb"


def code_cell(source: str, *, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(source: str, *, cell_id: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def main() -> int:
    notebook = json.loads(SEED07_NOTEBOOK.read_text(encoding="utf-8"))

    notebook["cells"] = [
        markdown_cell(
            """# Multiclass Classifier Seed07 Workflow

This notebook keeps the `seed07` multiclass classifier workflow together in one place:
- optionally train the `seed07` classifier run from the local config in this folder
- optionally generate unlabeled pseudo-label outputs from the same checkpoint
- review held-out test metrics and pseudo-label summaries
- recreate the saved `10a_style` UMAP plots from exported coordinate CSVs
""",
            cell_id="seed07_overview",
        ),
        code_cell(
            """from pathlib import Path
import json
import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

cwd = Path.cwd().resolve()
candidate_roots = [cwd, *cwd.parents]
REPO_ROOT = None
for candidate in candidate_roots:
    if (candidate / "src" / "wafer_defect").exists() and (candidate / "configs").exists():
        REPO_ROOT = candidate
        break

if REPO_ROOT is None:
    raise RuntimeError("Could not locate repo root containing src/wafer_defect and configs/")

SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

NOTEBOOK_DIR = REPO_ROOT / "experiments/classifier/multiclass/x64/seed07"

from wafer_defect.config import load_toml
""",
            cell_id="seed07_imports",
        ),
        code_cell(
            """data_config_path = NOTEBOOK_DIR / "data_config.toml"
train_config_path = NOTEBOOK_DIR / "train_config.toml"
data_config = load_toml(data_config_path)
train_config = load_toml(train_config_path)

seed07_artifact_dir = NOTEBOOK_DIR / "artifacts" / Path(train_config["training"]["output_dir"]).name
checkpoint_path = seed07_artifact_dir / "best_model.pt"
history_path = seed07_artifact_dir / "history.csv"
metrics_path = seed07_artifact_dir / "metrics.json"
test_predictions_path = seed07_artifact_dir / "test_predictions.csv"
unlabeled_predictions_path = seed07_artifact_dir / "unlabeled_predictions.csv"
unlabeled_summary_path = seed07_artifact_dir / "unlabeled_predictions.summary.json"
unlabeled_predictions_raw_path = seed07_artifact_dir / "unlabeled_predictions.seed07.raw.csv"
unlabeled_predictions_accepted_path = seed07_artifact_dir / "unlabeled_predictions.seed07.accepted.csv"

RUN_TRAINING = False
RUN_PSEUDOLABEL_INFERENCE = False
confidence_threshold = 0.98

seed07_artifact_dir.mkdir(parents=True, exist_ok=True)

print("Data config:", data_config_path)
print("Seed07 config:", train_config_path)
print("Seed07 artifact dir:", seed07_artifact_dir)
print("Checkpoint:", checkpoint_path)
print("Unlabeled predictions path:", unlabeled_predictions_path)
print("Confidence threshold:", confidence_threshold)
print("Run training:", RUN_TRAINING)
print("Run pseudo-label inference:", RUN_PSEUDOLABEL_INFERENCE)
""",
            cell_id="seed07_config",
        ),
        code_cell(
            """required_training_paths = [checkpoint_path, metrics_path, test_predictions_path]

if RUN_TRAINING:
    train_command = [
        sys.executable,
        str(REPO_ROOT / "scripts/classifier/train_multiclass_classifier.py"),
        "--config",
        str(train_config_path),
    ]
    print("Running:", " ".join(train_command))
    subprocess.run(train_command, cwd=REPO_ROOT, check=True)
else:
    missing_training_paths = [path for path in required_training_paths if not path.exists()]
    if missing_training_paths:
        raise FileNotFoundError(
            "RUN_TRAINING is False and the saved seed07 training artifacts are missing:\\n"
            + "\\n".join(str(path) for path in missing_training_paths)
        )
    print(f"RUN_TRAINING is False. Reusing saved training artifacts from {seed07_artifact_dir}")
""",
            cell_id="seed07_training",
        ),
        code_cell(
            """if RUN_PSEUDOLABEL_INFERENCE:
    predict_command = [
        sys.executable,
        str(REPO_ROOT / "scripts/classifier/predict_unlabeled_multiclass.py"),
        "--config",
        str(data_config_path),
        "--checkpoint",
        str(checkpoint_path),
        "--output-csv",
        str(unlabeled_predictions_path),
        "--min-confidence",
        str(confidence_threshold),
    ]
    print("Running:", " ".join(predict_command))
    subprocess.run(predict_command, cwd=REPO_ROOT, check=True)
else:
    print(f"RUN_PSEUDOLABEL_INFERENCE is False. Reusing saved predictions from {unlabeled_predictions_path}")
""",
            cell_id="seed07_pseudolabel_inference",
        ),
        code_cell(
            """history = pd.read_csv(history_path) if history_path.exists() else None
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
test_predictions = pd.read_csv(test_predictions_path)
unlabeled_predictions = pd.read_csv(unlabeled_predictions_path) if unlabeled_predictions_path.exists() else None
unlabeled_predictions_raw = pd.read_csv(unlabeled_predictions_raw_path) if unlabeled_predictions_raw_path.exists() else None
unlabeled_predictions_accepted = (
    pd.read_csv(unlabeled_predictions_accepted_path)
    if unlabeled_predictions_accepted_path.exists()
    else None
)
unlabeled_summary = json.loads(unlabeled_summary_path.read_text(encoding="utf-8")) if unlabeled_summary_path.exists() else None

print("Test accuracy:", metrics["test"]["accuracy"])
print("Test balanced accuracy:", metrics["test"]["balanced_accuracy"])
if history is not None:
    print("Best validation balanced accuracy:", history["val_balanced_accuracy"].max())
if unlabeled_predictions is not None and "accepted_for_pseudo_label" in unlabeled_predictions.columns:
    accepted_fraction = (
        unlabeled_predictions["accepted_for_pseudo_label"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
        .mean()
    )
    print("Accepted pseudo-label fraction:", accepted_fraction)
elif unlabeled_predictions_accepted is not None and unlabeled_predictions_raw is not None:
    print("Accepted pseudo-label fraction:", len(unlabeled_predictions_accepted) / len(unlabeled_predictions_raw))
else:
    print("No saved unlabeled predictions found yet.")

if history is not None:
    display(history.tail())
display(pd.DataFrame(metrics["test"]["classification_report"]).T)
display(test_predictions.head())
if unlabeled_predictions is not None:
    display(unlabeled_predictions.head())
if unlabeled_predictions_raw is not None:
    display(unlabeled_predictions_raw.head())
if unlabeled_predictions_accepted is not None:
    display(unlabeled_predictions_accepted.head())
if unlabeled_summary is not None:
    display(pd.json_normalize(unlabeled_summary, sep="."))
""",
            cell_id="seed07_artifact_review",
        ),
        code_cell(
            """if unlabeled_predictions_raw is not None:
    raw_label_col = "pseudo_label" if "pseudo_label" in unlabeled_predictions_raw.columns else None
    accepted_flag_col = "accepted_for_pseudo_label" if "accepted_for_pseudo_label" in unlabeled_predictions_raw.columns else None

    if raw_label_col is None:
        print("Raw pseudo-label CSV does not contain a pseudo_label column.")
    else:
        raw_counts = (
            unlabeled_predictions_raw[raw_label_col]
            .fillna("<missing>")
            .value_counts(dropna=False)
            .rename_axis("pseudo_label")
            .rename("raw_count")
            .to_frame()
        )

        if accepted_flag_col is not None:
            accepted_mask = unlabeled_predictions_raw[accepted_flag_col].astype(str).str.lower().eq("true")
            accepted_counts = (
                unlabeled_predictions_raw.loc[accepted_mask, raw_label_col]
                .fillna("<missing>")
                .value_counts(dropna=False)
                .rename_axis("pseudo_label")
                .rename("accepted_count")
                .to_frame()
            )
        elif unlabeled_predictions_accepted is not None and raw_label_col in unlabeled_predictions_accepted.columns:
            accepted_counts = (
                unlabeled_predictions_accepted[raw_label_col]
                .fillna("<missing>")
                .value_counts(dropna=False)
                .rename_axis("pseudo_label")
                .rename("accepted_count")
                .to_frame()
            )
        else:
            accepted_counts = pd.DataFrame(columns=["accepted_count"])

        pseudo_label_summary = raw_counts.join(accepted_counts, how="outer").fillna(0)
        pseudo_label_summary[["raw_count", "accepted_count"]] = pseudo_label_summary[["raw_count", "accepted_count"]].astype(int)
        pseudo_label_summary["accepted_rate"] = (
            pseudo_label_summary["accepted_count"]
            .div(pseudo_label_summary["raw_count"].where(pseudo_label_summary["raw_count"] > 0))
            .fillna(0.0)
        )

        display(pseudo_label_summary.sort_values(["accepted_count", "raw_count"], ascending=[False, False]))
else:
    print("No raw pseudo-label CSV found, so class-level pseudo-label counts are unavailable.")
""",
            cell_id="seed07_pseudolabel_summary",
        ),
        markdown_cell(
            """## UMAP Review

These cells recreate the saved `10a_style` UMAP plots from exported coordinate CSVs. They prefer a local `seed07/upload_artifacts` copy when it exists and otherwise fall back to the legacy `umap/upload_artifacts` folder.
""",
            cell_id="seed07_umap_heading",
        ),
        code_cell(
            """local_umap_dir = NOTEBOOK_DIR / "upload_artifacts" / "umap_10a_style"
legacy_umap_dir = NOTEBOOK_DIR.parent / "umap" / "upload_artifacts" / "umap_10a_style"
umap_points_candidates = [
    local_umap_dir / "embedding_umap_points_10a_style.csv",
    legacy_umap_dir / "embedding_umap_points_10a_style.csv",
]
UMAP_POINTS_CSV = next((path for path in umap_points_candidates if path.exists()), umap_points_candidates[0])
UMAP_PLOTS_DIR = local_umap_dir

if not UMAP_POINTS_CSV.exists():
    raise FileNotFoundError(
        "Missing saved seed07 UMAP points CSV. Checked:\\n" + "\\n".join(str(path) for path in umap_points_candidates)
    )

umap_df = pd.read_csv(UMAP_POINTS_CSV)
umap_df["score"] = pd.to_numeric(
    umap_df.get("score", pd.Series(index=umap_df.index, dtype=float)),
    errors="coerce",
)
umap_df["pseudo_label_confidence"] = pd.to_numeric(
    umap_df.get("pseudo_label_confidence", pd.Series(index=umap_df.index, dtype=float)),
    errors="coerce",
)
accepted_series = umap_df.get("accepted_for_pseudo_label", pd.Series(index=umap_df.index, dtype=object))
umap_df["accepted_for_pseudo_label"] = accepted_series.astype(str).str.lower().map({"true": True, "false": False})

print("Loaded UMAP points from:", UMAP_POINTS_CSV)
print("Rows:", len(umap_df))
display(umap_df.head())
display(umap_df["split_label"].value_counts(dropna=False).rename("count").to_frame())
""",
            cell_id="seed07_umap_load",
        ),
        code_cell(
            """UMAP_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def save_split_plot(umap_df: pd.DataFrame, figure_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    style_map = {
        "labeled_normal": dict(s=10, alpha=0.30, label="labeled_normal", color="#4d908e"),
        "labeled_defect": dict(s=12, alpha=0.50, label="labeled_defect", color="#f3722c"),
        "pseudo_unlabeled": dict(s=14, alpha=0.65, label="pseudo_unlabeled", color="#577590"),
    }

    for split_name, group in umap_df.groupby("split_label", sort=False):
        style = style_map.get(split_name, dict(s=12, alpha=0.50, label=str(split_name), color="#6b7280"))
        ax.scatter(group["umap_1"], group["umap_2"], **style)

    ax.set_title("10A-style UMAP of Seed07 Classifier Embeddings")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

def save_score_plot(umap_df: pd.DataFrame, figure_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    score_mask = umap_df["split_label"] == "pseudo_unlabeled"
    scored_points = umap_df.loc[score_mask & umap_df["score"].notna()].copy()

    ax.scatter(
        umap_df.loc[~score_mask, "umap_1"],
        umap_df.loc[~score_mask, "umap_2"],
        s=8,
        alpha=0.12,
        color="#9ca3af",
        label="labeled reference",
    )

    if scored_points.empty:
        ax.scatter(
            umap_df.loc[score_mask, "umap_1"],
            umap_df.loc[score_mask, "umap_2"],
            s=18,
            alpha=0.65,
            color="#577590",
            label="pseudo_unlabeled",
        )
    else:
        sc = ax.scatter(
            scored_points["umap_1"],
            scored_points["umap_2"],
            c=scored_points["score"],
            cmap="viridis",
            s=18,
            alpha=0.85,
            label="pseudo_unlabeled",
        )
        fig.colorbar(sc, ax=ax, label="pseudo-label confidence")

    ax.set_title("10A-style UMAP Colored by Pseudo-Label Confidence")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

def save_pseudo_label_plot(umap_df: pd.DataFrame, figure_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    score_mask = umap_df["split_label"] == "pseudo_unlabeled"

    ax.scatter(
        umap_df.loc[~score_mask, "umap_1"],
        umap_df.loc[~score_mask, "umap_2"],
        s=8,
        alpha=0.10,
        color="#d1d5db",
        label="labeled reference",
    )

    palette = plt.get_cmap("tab10")
    pseudo_points = umap_df.loc[score_mask].copy()
    pseudo_points = pseudo_points[pseudo_points["pseudo_label"].notna()]
    for idx, (pseudo_label, group) in enumerate(pseudo_points.groupby("pseudo_label", sort=True)):
        ax.scatter(
            group["umap_1"],
            group["umap_2"],
            s=18,
            alpha=0.70,
            label=str(pseudo_label),
            color=palette(idx % palette.N),
        )

    ax.set_title("10A-style UMAP with Pseudo Labels Highlighted")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

split_plot_path = UMAP_PLOTS_DIR / "umap_by_split_10a_style.notebook.png"
score_plot_path = UMAP_PLOTS_DIR / "umap_by_score_10a_style.notebook.png"
pseudo_label_plot_path = UMAP_PLOTS_DIR / "umap_by_pseudo_label_10a_style.notebook.png"

save_split_plot(umap_df=umap_df, figure_path=split_plot_path)
save_score_plot(umap_df=umap_df, figure_path=score_plot_path)
save_pseudo_label_plot(umap_df=umap_df, figure_path=pseudo_label_plot_path)

print("Saved:")
print(" -", split_plot_path)
print(" -", score_plot_path)
print(" -", pseudo_label_plot_path)
""",
            cell_id="seed07_umap_plots",
        ),
    ]

    SEED07_NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
