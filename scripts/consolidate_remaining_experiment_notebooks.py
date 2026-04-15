from __future__ import annotations

import json
from pathlib import Path
from textwrap import indent
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parents[1]

SEED07_NOTEBOOK = REPO_ROOT / "experiments/classifier/multiclass/x64/seed07/notebook.ipynb"
SEED07_UMAP_NOTEBOOK = REPO_ROOT / "experiments/classifier/multiclass/x64/umap/notebook.ipynb"

EFFNET_MAIN_NOTEBOOK = REPO_ROOT / "experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main_one_layer/notebook.ipynb"
EFFNET_UMAP_NOTEBOOK = REPO_ROOT / "experiments/anomaly_detection/patchcore/efficientnet_b1/x240/umap_followup/notebook.ipynb"

RESNET18_MAIN_NOTEBOOK = REPO_ROOT / "experiments/anomaly_detection/patchcore/resnet18/x64/main/notebook.ipynb"
RESNET18_UMAP_NOTEBOOK = REPO_ROOT / "experiments/anomaly_detection/patchcore/resnet18/x64/holdout70k_3p5k_umap_followup/notebook.ipynb"

SUPERVISED_MAIN_NOTEBOOK = REPO_ROOT / "experiments/anomaly_detection_defect/supervised_sweep/vit_b16/x224/main/notebook.ipynb"
SUPERVISED_UMAP_NOTEBOOK = REPO_ROOT / "experiments/anomaly_detection_defect/supervised_sweep/vit_b16/x224/main/umap/notebook.ipynb"

WRN50_MAIN_NOTEBOOK = REPO_ROOT / "experiments/anomaly_detection/patchcore/wideresnet50/x64/labeled_120k/notebook.ipynb"
WRN50_RESULTS_NOTEBOOK = REPO_ROOT / "experiments/anomaly_detection/patchcore/wideresnet50/x64/labeled_120k/results_review.ipynb"
WRN50_THRESHOLD_NOTEBOOK = REPO_ROOT / "experiments/anomaly_detection/patchcore/wideresnet50/x64/labeled_120k/threshold_policies.ipynb"
WRN50_DATASET_NOTEBOOK = REPO_ROOT / "experiments/anomaly_detection/patchcore/wideresnet50/x64/labeled_120k/dataset_helper.ipynb"


def _read_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_notebook(path: Path, notebook: dict) -> None:
    for cell in notebook.get("cells", []):
        cell.setdefault("id", uuid4().hex[:8])
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_source(cell: dict) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def _dump_source(text: str, original: object) -> object:
    if isinstance(original, list):
        return text.splitlines(keepends=True)
    return text


def _set_source(cell: dict, text: str) -> None:
    cell["source"] = _dump_source(text, cell.get("source", ""))


def _code_cell(source: str, *, cell_id: str | None = None) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id or uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def _markdown_cell(source: str, *, cell_id: str | None = None) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id or uuid4().hex[:8],
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def _wrap_with_guard(source: str, *, condition: str, warning: str) -> str:
    body = indent(source.rstrip() + "\n", "    ")
    return (
        f"if {condition}:\n"
        f"{body}"
        f"else:\n"
        f"    print({warning!r})\n"
    )


def _delete_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def patch_seed07() -> None:
    notebook = _read_notebook(SEED07_NOTEBOOK)

    _set_source(
        notebook["cells"][3],
        """required_training_paths = [checkpoint_path, metrics_path, test_predictions_path]
TRAINING_ARTIFACTS_AVAILABLE = True

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
        TRAINING_ARTIFACTS_AVAILABLE = False
        print("[WARNING] RUN_TRAINING is False and the saved seed07 training artifacts are missing.")
        for missing_path in missing_training_paths:
            print("  -", missing_path)
        print("[WARNING] Training metrics and prediction review cells will be skipped until those artifacts exist.")
    else:
        print(f"RUN_TRAINING is False. Reusing saved training artifacts from {seed07_artifact_dir}")
""",
    )

    _set_source(
        notebook["cells"][5],
        """history = pd.read_csv(history_path) if history_path.exists() else None
metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None
test_predictions = pd.read_csv(test_predictions_path) if test_predictions_path.exists() else None
unlabeled_predictions = pd.read_csv(unlabeled_predictions_path) if unlabeled_predictions_path.exists() else None
unlabeled_predictions_raw = pd.read_csv(unlabeled_predictions_raw_path) if unlabeled_predictions_raw_path.exists() else None
unlabeled_predictions_accepted = (
    pd.read_csv(unlabeled_predictions_accepted_path)
    if unlabeled_predictions_accepted_path.exists()
    else None
)
unlabeled_summary = json.loads(unlabeled_summary_path.read_text(encoding="utf-8")) if unlabeled_summary_path.exists() else None

if metrics is None or test_predictions is None:
    print("[WARNING] The saved seed07 evaluation artifacts are not available yet, so the review tables are being skipped.")
else:
    print("Test accuracy:", metrics["test"]["accuracy"])
    print("Test balanced accuracy:", metrics["test"]["balanced_accuracy"])
    if history is not None and "val_balanced_accuracy" in history.columns:
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
    )

    _set_source(
        notebook["cells"][8],
        """local_umap_dir = NOTEBOOK_DIR / "upload_artifacts" / "umap_10a_style"
legacy_umap_dir = NOTEBOOK_DIR.parent / "umap" / "upload_artifacts" / "umap_10a_style"
umap_points_candidates = [
    local_umap_dir / "embedding_umap_points_10a_style.csv",
    legacy_umap_dir / "embedding_umap_points_10a_style.csv",
]
UMAP_POINTS_CSV = next((path for path in umap_points_candidates if path.exists()), umap_points_candidates[0])
UMAP_PLOTS_DIR = local_umap_dir
umap_df = None

if not UMAP_POINTS_CSV.exists():
    print("[WARNING] Missing saved seed07 UMAP points CSV.")
    for candidate in umap_points_candidates:
        print("  -", candidate)
    print("[WARNING] The UMAP review cells will be skipped until the coordinate CSV is available.")
else:
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
    )

    _set_source(
        notebook["cells"][9],
        _wrap_with_guard(
            _load_source(notebook["cells"][9]),
            condition="umap_df is not None and not umap_df.empty",
            warning="Missing seed07 UMAP coordinates, so no UMAP figures were regenerated in this notebook run.",
        ),
    )

    _write_notebook(SEED07_NOTEBOOK, notebook)
    _delete_if_exists(SEED07_UMAP_NOTEBOOK)


def patch_effnet_main() -> None:
    notebook = _read_notebook(EFFNET_MAIN_NOTEBOOK)

    _set_source(
        notebook["cells"][3],
        """# -- LOAD_FROM_PROCESSED FLAG --------------------------------------------------
# True  -> load from data/processed/x240/wm811k/metadata_50k_5pct.csv + shared .npy arrays.
#         Requires running data/dataset/x240/benchmark_50k_5pct/notebook.ipynb first.
#         On first use the arrays are read and cached locally as usual; subsequent
#         runs load from the local .npz cache and are equally fast.
# False -> rebuild directly from raw LSWMD.pkl
LOAD_FROM_PROCESSED = True
# -----------------------------------------------------------------------------

output_dir = prepare_output_dir(CONFIG)
cache_path = dataset_cache_path(CONFIG)
raw_pickle_path = None
dataset = None
DATASET_READY = False

PROCESSED_META = PROJECT_ROOT / "data" / "processed" / "x240" / "wm811k" / "metadata_50k_5pct.csv"

def load_from_processed_csv(meta_path: Path) -> dict[str, np.ndarray]:
    # Load the pre-built x240 split from the shared metadata CSV + individual .npy files.
    meta = pd.read_csv(meta_path)
    result: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        rows = meta[meta["split"] == split].reset_index(drop=True)
        arrays = np.stack([
            np.load(PROJECT_ROOT / row["array_path"])
            for _, row in tqdm(rows.iterrows(), total=len(rows), desc=f"load_{split}", leave=True)
        ])
        labels = rows["is_anomaly"].to_numpy(dtype=np.int64)
        defect_types = rows["defect_type"].fillna("unlabeled").replace("", "unlabeled").to_numpy()
        result[f"{split}_x"] = arrays
        result[f"{split}_y"] = labels
        result[f"{split}_defect_types"] = defect_types
    return result

if LOAD_FROM_PROCESSED and PROCESSED_META.exists():
    dataset = load_from_processed_csv(PROCESSED_META)
    DATASET_READY = True
    print(f"Loaded processed split from {PROCESSED_META}")
else:
    if LOAD_FROM_PROCESSED:
        print(f"[WARNING] Processed data not found at {PROCESSED_META}.")
        print("  -> Falling back to raw LSWMD rebuild when the raw pickle is available.")
    try:
        raw_pickle_path = auto_find_raw_pickle(CONFIG["run"]["raw_pickle"])
    except FileNotFoundError as exc:
        print(f"[WARNING] {exc}")
        print("[WARNING] Dataset-dependent cells can still be reviewed in artifact-only mode, but retraining or UMAP regeneration from embeddings will be skipped.")
    else:
        dataset = load_or_build_dataset(CONFIG, cache_path=cache_path, raw_pickle_path=raw_pickle_path)
        DATASET_READY = True

if DATASET_READY:
    display(pd.DataFrame([
        {"split": "train", "count": len(dataset["train_y"]), "anomalies": int((dataset["train_y"] == 1).sum())},
        {"split": "val", "count": len(dataset["val_y"]), "anomalies": int((dataset["val_y"] == 1).sum())},
        {"split": "test", "count": len(dataset["test_y"]), "anomalies": int((dataset["test_y"] == 1).sum())},
    ]))
else:
    print("[WARNING] Dataset previews are unavailable because neither processed data nor the raw pickle could be resolved.")
""",
    )

    _set_source(
        notebook["cells"][4],
        """# -- RETRAIN FLAG --------------------------------------------------------------
# True  -> rebuild the memory bank and re-run all evaluations from scratch.
# False -> load the saved checkpoint and pre-computed CSVs from disk (no GPU needed).
RETRAIN = False
# -----------------------------------------------------------------------------

RESULT_READY = False
paths = artifact_layout(output_dir)
checkpoint_path = paths["checkpoints_dir"] / "best_model.pt"
required_eval_paths = [
    checkpoint_path,
    paths["results_dir"] / "variant_summary.csv",
    paths["evaluation_dir"] / "val_scores.csv",
    paths["evaluation_dir"] / "test_scores.csv",
    paths["evaluation_dir"] / "threshold_sweep.csv",
    paths["evaluation_dir"] / "defect_breakdown.csv",
]

if RETRAIN:
    if not DATASET_READY:
        print("[WARNING] Dataset inputs are unavailable, so RETRAIN=True cannot rebuild this experiment right now.")
        result = None
    else:
        result = run_and_save_variant(dataset, CONFIG)
        RESULT_READY = result is not None
else:
    missing_eval_paths = [path for path in required_eval_paths if not path.exists()]
    if missing_eval_paths:
        print("[WARNING] Saved EfficientNet-B1 main-phase artifacts are missing, so evaluation cells will be skipped.")
        for missing_path in missing_eval_paths:
            print("  -", missing_path)
        result = None
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        result = {
            "summary": checkpoint["summary"],
            "score_df": pd.read_csv(paths["results_dir"] / "variant_summary.csv"),
            "val_scores_df": pd.read_csv(paths["evaluation_dir"] / "val_scores.csv"),
            "test_scores_df": pd.read_csv(paths["evaluation_dir"] / "test_scores.csv"),
            "threshold_sweep_df": pd.read_csv(paths["evaluation_dir"] / "threshold_sweep.csv"),
            "defect_breakdown_df": pd.read_csv(paths["evaluation_dir"] / "defect_breakdown.csv"),
            "checkpoint": checkpoint,
        }
        RESULT_READY = True
        print(f"Loaded checkpoint from {checkpoint_path}")

if RESULT_READY:
    display(result["score_df"])
    display(result["defect_breakdown_df"].head(12))
else:
    print("[WARNING] The main benchmark summary tables are unavailable until this experiment's local artifacts are created.")
""",
    )

    _set_source(
        notebook["cells"][5],
        """paths = artifact_layout(output_dir)
print(f"Best-row CSV: {paths['results_dir'] / 'best_row.csv'}")
print(f"Validation-score CSV: {paths['evaluation_dir'] / 'val_scores.csv'}")
print(f"Test-score CSV: {paths['evaluation_dir'] / 'test_scores.csv'}")
print(f"Threshold-sweep CSV: {paths['evaluation_dir'] / 'threshold_sweep.csv'}")
print(f"Defect-breakdown CSV: {paths['evaluation_dir'] / 'defect_breakdown.csv'}")
print(f"Checkpoint: {paths['checkpoints_dir'] / 'best_model.pt'}")

if RESULT_READY:
    try:
        summary_df = load_saved_summary(CONFIG)
    except FileNotFoundError as exc:
        print(f"[WARNING] {exc}")
        print("[WARNING] The saved summary table is unavailable, but the notebook will continue.")
    else:
        display(summary_df)
else:
    print("[WARNING] Saved summary review is unavailable because the main result bundle is missing.")
""",
    )

    for idx, warning in [
        (6, "Main benchmark result artifacts are missing, so plot regeneration is being skipped."),
        (9, "Reference-fit UMAP artifacts are missing, so the joint-fit comparison view is being skipped."),
        (10, "UMAP source plots were not available, so the flat plot-copy step was skipped."),
    ]:
        _set_source(
            notebook["cells"][idx],
            _wrap_with_guard(
                _load_source(notebook["cells"][idx]),
                condition="RESULT_READY",
                warning=warning,
            ),
        )

    _set_source(
        notebook["cells"][8],
        """# -- REGENERATE_UMAP FLAG ------------------------------------------------------
# True  -> collect embeddings from the loaded model and recompute the reference-fit UMAP.
# False -> load previously saved embeddings and UMAP summary from disk; skip recomputation.
REGENERATE_UMAP = False
# -----------------------------------------------------------------------------

if not RESULT_READY:
    print("[WARNING] Reference-fit UMAP review is unavailable because the main result bundle did not load.")
else:
    import matplotlib.pyplot as plt
    import umap.umap_ as umap

    evaluation_dir = artifact_layout(output_dir)["umap_reference_dir"]
    plots_dir = evaluation_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    def collect_embeddings_and_scores(model, loader, device, desc):
        embeddings_all, labels_all, scores_all = [], [], []
        model.eval()
        progress = tqdm(loader, desc=desc, total=len(loader), leave=True)
        with torch.inference_mode():
            for inputs, labels in progress:
                inputs = inputs.to(device, non_blocking=device.type == "cuda")
                patch_embeddings = model.patch_embeddings(inputs)
                image_embeddings = patch_embeddings.mean(dim=1)
                patch_scores = model.nearest_patch_distances(patch_embeddings)
                scores = model.reduce_patch_distances(patch_scores)
                embeddings_all.append(image_embeddings.cpu().numpy())
                labels_all.extend(labels.cpu().tolist())
                scores_all.extend(scores.cpu().tolist())
        return (
            np.concatenate(embeddings_all, axis=0),
            np.asarray(labels_all, dtype=np.int64),
            np.asarray(scores_all, dtype=np.float32),
        )

    if REGENERATE_UMAP:
        if not DATASET_READY:
            print("[WARNING] Dataset inputs are unavailable, so REGENERATE_UMAP=True cannot rebuild embeddings right now.")
        else:
            device = resolve_device(str(CONFIG["model"]["device"]))
            main_loaders = make_loaders(dataset, CONFIG, device)

            umap_model = EfficientNetB1OneLayerPatchCoreModel(
                model_input_size=int(CONFIG["model"]["model_input_size"]),
                feature_idx=int(CONFIG["model"]["effnet_feature_idx"]),
                patch_embed_dim=int(CONFIG["model"]["patch_embed_dim"]),
                topk_ratio=float(CONFIG["model"]["topk_patch_ratio"]),
                nn_k=int(CONFIG["model"]["patchcore_nn_k"]),
                query_chunk_size=int(CONFIG["model"]["score_chunk"]),
                amp_enabled=bool(CONFIG["model"]["amp"]),
            ).to(device).eval()

            umap_state = result["checkpoint"]["model_state_dict"]
            if "memory_bank" in umap_state:
                umap_model.set_memory_bank(umap_state["memory_bank"])
                umap_model.load_state_dict(umap_state, strict=False)
            else:
                umap_model.load_state_dict(umap_state)

            train_embeddings, train_labels, _ = collect_embeddings_and_scores(umap_model, main_loaders["train"], device, "umap_train")
            val_embeddings, val_labels, val_scores = collect_embeddings_and_scores(umap_model, main_loaders["val"], device, "umap_val")
            test_embeddings, test_labels, test_scores = collect_embeddings_and_scores(umap_model, main_loaders["test"], device, "umap_test")

            np.save(evaluation_dir / "train_embeddings.npy", train_embeddings)
            np.save(evaluation_dir / "val_embeddings.npy", val_embeddings)
            np.save(evaluation_dir / "test_embeddings.npy", test_embeddings)
            np.save(evaluation_dir / "train_labels.npy", train_labels)
            np.save(evaluation_dir / "val_labels.npy", val_labels)
            np.save(evaluation_dir / "test_labels.npy", test_labels)
            np.save(evaluation_dir / "val_scores.npy", val_scores)
            np.save(evaluation_dir / "test_scores.npy", test_scores)

            umap_bundle = export_reference_umap_bundle(
                output_dir=evaluation_dir,
                umap_module=umap,
                train_normal_embeddings=train_embeddings,
                val_embeddings=val_embeddings,
                val_labels=val_labels,
                test_embeddings=test_embeddings,
                test_labels=test_labels,
                val_model_scores=val_scores,
                test_model_scores=test_scores,
                threshold_quantile=float(CONFIG["scoring"]["threshold_quantile"]),
                random_state=42,
                pca_components=50,
                n_neighbors=15,
                min_dist=0.1,
                knn_k=15,
                metric="euclidean",
                max_train_reference=5000,
                max_val_normal=5000,
                max_test_normal=5000,
                max_test_anomaly=250,
                title_prefix="EfficientNet-B1 One-Layer PatchCore",
                points_filename="umap_points.csv",
                split_plot_filename="plots/umap_by_split.png",
                score_plot_filename="plots/umap_by_score.png",
                summary_filename="umap_summary.json",
                sweep_filename="umap_knn_threshold_sweep.csv",
            )
            display(pd.json_normalize(umap_bundle["summary"], sep="."))
            print(f"Saved reference-fit UMAP artifacts to {evaluation_dir}")
    else:
        points_path = evaluation_dir / "umap_points.csv"
        if not points_path.exists():
            print(f"[WARNING] UMAP points CSV not found at {points_path}.")
            print("[WARNING] Set REGENERATE_UMAP=True to collect embeddings and compute UMAP from scratch.")
        else:
            ref_df = pd.read_csv(points_path)
            summary_path = evaluation_dir / "umap_summary.json"
            if summary_path.exists():
                display(pd.json_normalize(json.loads(summary_path.read_text(encoding="utf-8")), sep="."))
            print(f"Loaded reference-fit UMAP points from {points_path}")

            group_colors = {
                "val_normal": "#1f77b4",
                "test_normal": "#2ca02c",
                "test_anomaly": "#d62728",
            }
            plot_df = ref_df[ref_df["split_label"].isin(group_colors)]

            plt.figure(figsize=(9, 7))
            for group, color in group_colors.items():
                subset = plot_df[plot_df["split_label"] == group]
                if len(subset):
                    plt.scatter(subset["umap_1"], subset["umap_2"], s=10, alpha=0.55, label=group, c=color)
            plt.xlabel("UMAP-1")
            plt.ylabel("UMAP-2")
            plt.title("EfficientNet-B1 One-Layer PatchCore: Reference-Fit UMAP by Split")
            plt.legend(frameon=False)
            plt.tight_layout()
            plt.savefig(plots_dir / "umap_by_split.png", dpi=220, bbox_inches="tight")
            plt.show()

            plt.figure(figsize=(9, 7))
            scatter = plt.scatter(
                plot_df["umap_1"],
                plot_df["umap_2"],
                c=plot_df["model_score"],
                cmap="viridis",
                s=10,
                alpha=0.65,
            )
            plt.xlabel("UMAP-1")
            plt.ylabel("UMAP-2")
            plt.title("EfficientNet-B1 One-Layer PatchCore: Reference-Fit UMAP by Model Score")
            plt.colorbar(scatter, label="Model Score")
            plt.tight_layout()
            plt.savefig(plots_dir / "umap_by_score.png", dpi=220, bbox_inches="tight")
            plt.show()
            print(f"Reference-fit UMAP plots saved to {plots_dir}")
""",
    )

    _set_source(
        notebook["cells"][12],
        """# UMAP interpretation helper: reads saved artifacts and prints a compact summary.
from pathlib import Path
import json
import numpy as np
import pandas as pd

def _safe_ratio(numerator, denominator):
    return float(numerator) / float(denominator) if float(denominator) != 0.0 else float("nan")

def _euclidean_xy(df: pd.DataFrame, origin_x: float, origin_y: float) -> np.ndarray:
    dx = df["umap_1"].to_numpy(dtype=float) - float(origin_x)
    dy = df["umap_2"].to_numpy(dtype=float) - float(origin_y)
    return np.sqrt(dx * dx + dy * dy)

evaluation_dir = artifact_layout(output_dir)["umap_reference_dir"]
points_path = evaluation_dir / "umap_points.csv"
summary_path = evaluation_dir / "umap_summary.json"

if not points_path.exists():
    print(f"[WARNING] UMAP points are not available at {points_path}, so the interpretation summary is being skipped.")
else:
    points_df = pd.read_csv(points_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    val_normal = points_df[points_df["split_label"] == "val_normal"].copy()
    test_normal = points_df[points_df["split_label"] == "test_normal"].copy()
    test_anomaly = points_df[points_df["split_label"] == "test_anomaly"].copy()

    if len(val_normal) == 0:
        print("[WARNING] The saved UMAP CSV does not contain val_normal rows, so no interpretation summary was generated.")
    else:
        val_center_x = float(val_normal["umap_1"].mean())
        val_center_y = float(val_normal["umap_2"].mean())
        val_radius = _euclidean_xy(val_normal, val_center_x, val_center_y)
        base_radius = float(np.percentile(val_radius, 95)) if len(val_radius) else float("nan")

        normal_shift = float("nan")
        anomaly_shift = float("nan")
        normal_outside = float("nan")
        anomaly_outside = float("nan")
        score_overlap = float("nan")

        if len(test_normal):
            test_normal_center_x = float(test_normal["umap_1"].mean())
            test_normal_center_y = float(test_normal["umap_2"].mean())
            normal_shift = _safe_ratio(np.hypot(test_normal_center_x - val_center_x, test_normal_center_y - val_center_y), base_radius)
            normal_outside = float((_euclidean_xy(test_normal, val_center_x, val_center_y) > base_radius).mean())

        if len(test_anomaly):
            test_anomaly_center_x = float(test_anomaly["umap_1"].mean())
            test_anomaly_center_y = float(test_anomaly["umap_2"].mean())
            anomaly_shift = _safe_ratio(np.hypot(test_anomaly_center_x - val_center_x, test_anomaly_center_y - val_center_y), base_radius)
            anomaly_outside = float((_euclidean_xy(test_anomaly, val_center_x, val_center_y) > base_radius).mean())

        if len(test_normal) and len(test_anomaly):
            normal_q90 = float(test_normal["model_score"].quantile(0.90))
            anomaly_q10 = float(test_anomaly["model_score"].quantile(0.10))
            score_overlap = normal_q90 - anomaly_q10

        interpretation = []

        if not np.isnan(normal_shift):
            if normal_shift <= 0.35:
                interpretation.append("test normals stay close to the validation-normal reference cloud")
            elif normal_shift <= 0.75:
                interpretation.append("test normals show a mild shift away from the validation-normal cloud")
            else:
                interpretation.append("test normals are noticeably shifted, so distribution drift is likely")

        if not np.isnan(anomaly_outside):
            if anomaly_outside >= 0.75:
                interpretation.append("most anomalies land outside the normal reference radius, which is a strong separation sign")
            elif anomaly_outside >= 0.45:
                interpretation.append("anomalies are only partly separated; some defect modes still overlap normals")
            else:
                interpretation.append("many anomalies remain inside the normal cloud, so feature separation is weak")

        if not np.isnan(score_overlap):
            if score_overlap < 0:
                interpretation.append("model scores separate normals and anomalies cleanly at the distribution tails")
            else:
                interpretation.append("model-score tails overlap, so thresholding will still involve trade-offs")

        summary_rows = [
            {"metric": "val_normal_count", "value": int(len(val_normal))},
            {"metric": "test_normal_count", "value": int(len(test_normal))},
            {"metric": "test_anomaly_count", "value": int(len(test_anomaly))},
            {"metric": "val_normal_radius_p95", "value": base_radius},
            {"metric": "test_normal_centroid_shift_vs_val_radius", "value": normal_shift},
            {"metric": "test_anomaly_centroid_shift_vs_val_radius", "value": anomaly_shift},
            {"metric": "test_normal_fraction_outside_val_radius", "value": normal_outside},
            {"metric": "test_anomaly_fraction_outside_val_radius", "value": anomaly_outside},
            {"metric": "score_tail_overlap_q90normal_minus_q10anomaly", "value": score_overlap},
            {"metric": "umap_knn_threshold", "value": summary.get("umap_knn_threshold", float("nan"))},
        ]

        summary_df = pd.DataFrame(summary_rows)
        display(summary_df)

        print("UMAP reading guide for this run:")
        for line in interpretation:
            print(f"- {line}")

        if not interpretation:
            print("- Not enough groups were present to generate an interpretation summary.")
""",
    )

    _write_notebook(EFFNET_MAIN_NOTEBOOK, notebook)
    _delete_if_exists(EFFNET_UMAP_NOTEBOOK)


def patch_resnet18_main() -> None:
    notebook = _read_notebook(RESNET18_MAIN_NOTEBOOK)

    cell11_source = _load_source(notebook["cells"][11])
    cell11_source = cell11_source.replace(
        """    if not all([summary_path, val_scores_path, test_scores_path, threshold_sweep_path]):
        missing = {
            "summary_path": summary_path,
            "val_scores_path": val_scores_path,
            "test_scores_path": test_scores_path,
            "threshold_sweep_path": threshold_sweep_path,
        }
        raise FileNotFoundError(f"Missing cached files for variant {variant_name}: {missing}")
""",
        """    if not all([summary_path, val_scores_path, test_scores_path, threshold_sweep_path]):
        missing = {
            "summary_path": summary_path,
            "val_scores_path": val_scores_path,
            "test_scores_path": test_scores_path,
            "threshold_sweep_path": threshold_sweep_path,
        }
        print(f"[WARNING] Missing cached files for variant {variant_name}: {missing}")
        return None
""",
    )
    _set_source(notebook["cells"][11], cell11_source)

    cell13_source = _load_source(notebook["cells"][13])
    cell13_source = (
        "selected_variant_name = None\nselected_variant = None\n" + cell13_source
    )
    cell13_source = cell13_source.replace(
        """    selected_variant = load_variant_outputs(selected_variant_name)
    variant_outputs[selected_variant_name] = selected_variant
    print(f"Loaded cached PatchCore sweep results from {sweep_results_path}")
""",
        """    selected_variant = load_variant_outputs(selected_variant_name)
    if selected_variant is None:
        print("[WARNING] Cached sweep outputs exist, but the selected variant bundle is incomplete.")
        sweep_results_df = pd.DataFrame()
        sweep_summary = {}
        selected_variant_name = None
    else:
        variant_outputs[selected_variant_name] = selected_variant
        print(f"Loaded cached PatchCore sweep results from {sweep_results_path}")
""",
    )
    cell13_source = cell13_source.replace(
        "else:\n    sweep_rows = []\n",
        "elif RETRAIN:\n    sweep_rows = []\n",
    )
    cell13_source = cell13_source.replace(
        "display(sweep_results_df)\n",
        """else:
    sweep_results_df = pd.DataFrame()
    sweep_summary = {}
    selected_variant_name = None
    print("[WARNING] No cached PatchCore sweep summary was found and RETRAIN=False, so the sweep review is being skipped.")

if sweep_results_df.empty:
    print("[WARNING] No PatchCore sweep table is available to display in this notebook run.")
else:
    display(sweep_results_df)
""",
    )
    _set_source(notebook["cells"][13], cell13_source)

    for idx, warning in [
        (15, "Selected-variant metrics are unavailable because the cached sweep artifacts are missing."),
        (17, "Selected-variant plots are unavailable because the cached sweep artifacts are missing."),
        (19, "Failure-analysis tables are unavailable because the cached sweep artifacts are missing."),
        (21, "Failure-analysis plots are unavailable because the cached sweep artifacts are missing."),
        (25, "Saved-output summary is unavailable because the selected variant did not load."),
    ]:
        _set_source(
            notebook["cells"][idx],
            _wrap_with_guard(
                _load_source(notebook["cells"][idx]),
                condition="selected_variant_name is not None and selected_variant is not None",
                warning=warning,
            ),
        )

    _set_source(
        notebook["cells"][23],
        """if selected_variant_name is None or selected_variant is None or sweep_results_df.empty:
    rendered_variants_df = pd.DataFrame(columns=["variant_name", "plots_dir", "evaluation_dir"])
    print("[WARNING] Cached variant rendering is being skipped because the sweep outputs are incomplete.")
else:
    variant_names_to_render = resolve_variant_names_to_render(sweep_results_df, selected_variant_name)
    rendered_variant_rows = []

    for variant_name in variant_names_to_render:
        if variant_name not in variant_outputs:
            variant_outputs[variant_name] = load_variant_outputs(variant_name)
        variant_payload = variant_outputs.get(variant_name)
        if variant_payload is None:
            print(f"[WARNING] Skipping variant rendering for {variant_name} because its cached files are incomplete.")
            continue
        render_info = render_variant_artifacts(variant_name, variant_payload)
        rendered_variant_rows.append(
            {
                "variant_name": variant_name,
                "plots_dir": render_info["plots_dir"],
                "evaluation_dir": render_info["evaluation_dir"],
            }
        )

    rendered_variants_df = pd.DataFrame(rendered_variant_rows)
    display(rendered_variants_df)
""",
    )

    notebook["cells"].extend(
        [
            _markdown_cell(
                """## UMAP Review

This merged section keeps the saved UMAP evaluation alongside the main PatchCore sweep notebook. It loads the selected variant's exported UMAP points when they exist and otherwise logs a warning instead of stopping the notebook.
""",
                cell_id="resnet18_umap_review",
            ),
            _code_cell(
                """selected_umap_df = None
selected_umap_path = None

if selected_variant_name is None or selected_variant is None:
    print("[WARNING] The selected variant did not load, so the saved UMAP review is unavailable.")
else:
    selected_eval_dir = selected_variant["output_dir"] / "results" / "evaluation"
    umap_candidates = [
        selected_eval_dir / "plots" / "embedding_umap_points.csv",
        selected_eval_dir / "embedding_umap_points.csv",
    ]
    selected_umap_path = next((path for path in umap_candidates if path.exists()), None)
    if selected_umap_path is None:
        print("[WARNING] No saved ResNet18 PatchCore UMAP CSV was found for the selected variant.")
        for candidate in umap_candidates:
            print("  -", candidate)
    else:
        selected_umap_df = pd.read_csv(selected_umap_path)
        print(f"Loaded selected-variant UMAP points from {selected_umap_path}")
        display(selected_umap_df.head())
        display(selected_umap_df["split_label"].value_counts(dropna=False).rename("count").to_frame())
""",
                cell_id="resnet18_umap_load",
            ),
            _code_cell(
                """embedded_anomalies = pd.DataFrame()
selected_umap_test_df = None

if selected_umap_df is None:
    print("[WARNING] UMAP neighborhood analysis is unavailable because no saved UMAP CSV was found.")
else:
    from sklearn.neighbors import NearestNeighbors

    selected_umap_test_df = selected_umap_df[selected_umap_df["split_label"].isin(["test_normal", "test_anomaly"])].copy()
    if selected_umap_test_df.empty:
        print("[WARNING] The saved UMAP CSV does not contain test_normal/test_anomaly rows.")
    else:
        coordinates = selected_umap_test_df[["umap_1", "umap_2"]].to_numpy()
        labels = selected_umap_test_df["is_anomaly"].to_numpy()
        neighbor_count = min(16, len(selected_umap_test_df))
        if neighbor_count <= 1:
            print("[WARNING] Not enough UMAP points are available for neighborhood analysis.")
        else:
            nbrs = NearestNeighbors(n_neighbors=neighbor_count, metric="euclidean")
            nbrs.fit(coordinates)
            _, indices = nbrs.kneighbors(coordinates)
            neighbor_idx = indices[:, 1:]
            neighbor_labels = labels[neighbor_idx]
            selected_umap_test_df["anomaly_neighbor_ratio"] = neighbor_labels.mean(axis=1)
            selected_umap_test_df["normal_neighbor_ratio"] = 1.0 - selected_umap_test_df["anomaly_neighbor_ratio"]

            embedded_anomaly_threshold = 0.8
            embedded_anomalies = selected_umap_test_df[
                (selected_umap_test_df["is_anomaly"] == 1)
                & (selected_umap_test_df["normal_neighbor_ratio"] >= embedded_anomaly_threshold)
            ].copy()

            print(f"Total test anomalies: {(selected_umap_test_df['is_anomaly'] == 1).sum()}")
            print(f"Embedded anomalies (>= {embedded_anomaly_threshold:.0%} normal neighbors): {len(embedded_anomalies)}")
            display(selected_umap_test_df.head())
""",
                cell_id="resnet18_umap_overlap",
            ),
            _code_cell(
                """if selected_umap_test_df is None or selected_umap_test_df.empty:
    print("[WARNING] No UMAP scatter plot was generated because the saved UMAP test view is unavailable.")
else:
    fig, ax = plt.subplots(figsize=(9, 7))

    normals = selected_umap_test_df[selected_umap_test_df["is_anomaly"] == 0]
    anomalies = selected_umap_test_df[selected_umap_test_df["is_anomaly"] == 1]
    ax.scatter(normals["umap_1"], normals["umap_2"], s=10, alpha=0.18, label="test_normal")
    ax.scatter(anomalies["umap_1"], anomalies["umap_2"], s=14, alpha=0.45, label="test_anomaly")

    if not embedded_anomalies.empty:
        ax.scatter(
            embedded_anomalies["umap_1"],
            embedded_anomalies["umap_2"],
            s=28,
            alpha=0.95,
            marker="x",
            label="embedded_anomaly",
        )

    ax.set_title("UMAP with Embedded Anomalies Highlighted")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.close(fig)
""",
                cell_id="resnet18_umap_plot",
            ),
        ]
    )

    _write_notebook(RESNET18_MAIN_NOTEBOOK, notebook)
    _delete_if_exists(RESNET18_UMAP_NOTEBOOK)


def patch_supervised_main() -> None:
    notebook = _read_notebook(SUPERVISED_MAIN_NOTEBOOK)

    _set_source(
        notebook["cells"][6],
        """summary_path = RESULTS_DIR / "sweep_summary.json"
SWEEP_RESULTS_AVAILABLE = False
sweep_summary = {}
cfg = {}
holdout_types = []
results = []
df = pd.DataFrame()

if not summary_path.exists():
    print(f"[WARNING] sweep_summary.json was not found at {summary_path}.")
    print("[WARNING] The saved sweep review will be skipped until local artifacts are available.")
else:
    sweep_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cfg = sweep_summary["config"]
    holdout_types = cfg["holdout_defect_types"]
    results = sweep_summary["sweep_results"]
    df = pd.DataFrame([
        {
            "fraction": r["fraction"],
            "n_defects_train": r["n_defects_train"],
            "auroc": r["auroc"],
            "auprc": r["auprc"],
            "f1": r["f1"],
            "precision": r["precision"],
            "recall": r["recall"],
            "best_sweep_f1": r["best_sweep_f1"],
        }
        for r in results
    ])
    SWEEP_RESULTS_AVAILABLE = not df.empty

if SWEEP_RESULTS_AVAILABLE:
    print(f"Holdout classes (unseen during training): {holdout_types}")
    print(f"Train defect pool size: {cfg['train_defect_pool_size']}")
    print(f"Total labeled defects: {cfg['total_labeled_defects']}")
    print()
    display(df.set_index("n_defects_train").round(3))
else:
    print("[WARNING] No supervised sweep rows are available to display in this notebook run.")
""",
    )

    for idx, warning in [
        (8, "Sweep performance plots are unavailable because the saved supervised sweep summary is missing."),
        (10, "Per-class recall tables are unavailable because the saved supervised sweep summary is missing."),
        (12, "Seen-vs-holdout recall plots are unavailable because the saved supervised sweep summary is missing."),
        (14, "Largest-sweep per-class recall plots are unavailable because the saved supervised sweep summary is missing."),
        (16, "Summary tables are unavailable because the saved supervised sweep summary is missing."),
        (18, "Saved-plot listing is unavailable because the saved supervised sweep summary is missing."),
    ]:
        _set_source(
            notebook["cells"][idx],
            _wrap_with_guard(
                _load_source(notebook["cells"][idx]),
                condition="SWEEP_RESULTS_AVAILABLE",
                warning=warning,
            ),
        )

    notebook["cells"].extend(
        [
            _markdown_cell(
                """## UMAP Review

This notebook now keeps the saved CLS-embedding UMAP review together with the main supervised sweep analysis. If the UMAP exports are not present locally, the section logs a warning and moves on.
""",
                cell_id="supervised_umap_heading",
            ),
            _code_cell(
                """UMAP_OUTPUT_DIR = ARTIFACT_DIR / "umap"
LEGACY_UMAP_OUTPUT_DIR = EXPERIMENT_DIR / "umap" / "artifacts"
umap_coords_candidates = [
    UMAP_OUTPUT_DIR / "umap_coords.csv",
    LEGACY_UMAP_OUTPUT_DIR / "umap_coords.csv",
]
UMAP_COORDS_PATH = next((path for path in umap_coords_candidates if path.exists()), None)
supervised_umap_df = None

if UMAP_COORDS_PATH is None:
    print("[WARNING] No saved supervised-sweep UMAP coordinate CSV was found.")
    for candidate in umap_coords_candidates:
        print("  -", candidate)
else:
    supervised_umap_df = pd.read_csv(UMAP_COORDS_PATH)
    print(f"Loaded UMAP coordinates from {UMAP_COORDS_PATH}")
    display(supervised_umap_df.head())
    display(supervised_umap_df["group"].value_counts(dropna=False).rename("count").to_frame())
""",
                cell_id="supervised_umap_load",
            ),
            _code_cell(
                """if supervised_umap_df is None or supervised_umap_df.empty:
    print("[WARNING] No UMAP-by-role plot was generated because the saved UMAP coordinates are unavailable.")
else:
    role_styles = {
        "train_normal": dict(color="#d1d5db", s=4, alpha=0.25, zorder=1, label="Train normal"),
        "test_normal": dict(color="#4d908e", s=8, alpha=0.40, zorder=2, label="Test normal"),
        "seen_defect": dict(color="#277da1", s=14, alpha=0.65, zorder=3, label="Seen defect"),
        "holdout_scratch": dict(color="#e63946", s=22, alpha=0.85, zorder=5, label="Holdout - Scratch"),
        "holdout_loc": dict(color="#f4a261", s=22, alpha=0.85, zorder=4, label="Holdout - Loc"),
    }

    fig, ax = plt.subplots(figsize=(10, 7))
    for group_name in ["train_normal", "test_normal", "seen_defect", "holdout_loc", "holdout_scratch"]:
        group_df = supervised_umap_df[supervised_umap_df["group"] == group_name]
        if group_df.empty:
            continue
        ax.scatter(group_df["umap_1"], group_df["umap_2"], **role_styles[group_name])

    ax.set_title("Supervised ViT-B/16 CLS embeddings by role")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "supervised_umap_by_role.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print("Saved:", PLOTS_DIR / "supervised_umap_by_role.png")
""",
                cell_id="supervised_umap_role_plot",
            ),
            _code_cell(
                """if supervised_umap_df is None or supervised_umap_df.empty:
    print("[WARNING] No UMAP defect-type review was generated because the saved UMAP coordinates are unavailable.")
else:
    fig, ax = plt.subplots(figsize=(10, 7))
    normal_df = supervised_umap_df[supervised_umap_df["defect_type"] == "normal"]
    ax.scatter(normal_df["umap_1"], normal_df["umap_2"], s=4, alpha=0.20, color="#d1d5db", label="Normal")

    palette = plt.get_cmap("tab10")
    defect_types = [dtype for dtype in sorted(supervised_umap_df["defect_type"].dropna().unique()) if dtype != "normal"]
    for index, defect_type in enumerate(defect_types):
        defect_df = supervised_umap_df[supervised_umap_df["defect_type"] == defect_type]
        if defect_df.empty:
            continue
        marker = "*" if defect_type in {"Scratch", "Loc"} else "o"
        size = 48 if marker == "*" else 16
        ax.scatter(
            defect_df["umap_1"],
            defect_df["umap_2"],
            s=size,
            alpha=0.75,
            marker=marker,
            label=defect_type,
            color=palette(index % palette.N),
        )

    ax.set_title("Supervised ViT-B/16 CLS embeddings by defect type")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(frameon=False, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "supervised_umap_by_defect_type.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print("Saved:", PLOTS_DIR / "supervised_umap_by_defect_type.png")
""",
                cell_id="supervised_umap_type_plot",
            ),
        ]
    )

    _write_notebook(SUPERVISED_MAIN_NOTEBOOK, notebook)
    _delete_if_exists(SUPERVISED_UMAP_NOTEBOOK)


def patch_wrn50_main() -> None:
    notebook = _read_notebook(WRN50_MAIN_NOTEBOOK)

    cell1_source = _load_source(notebook["cells"][1])
    cell1_source = cell1_source.replace(
        "from IPython.display import display\nfrom torch.utils.data import DataLoader\n",
        """from IPython.display import Markdown, display
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
""",
    )
    cell1_source = cell1_source.replace(
        """from patchcore_wrn50_local import (
    DEFAULT_SPLIT_CONFIG,
    DEFAULT_VARIANTS,
    WaferArrayDataset,
    attach_scores_to_metadata,
    auto_find_raw_pickle,
    defect_type_summary,
    load_wafer_array,
    prepare_dataset,
    resolve_data_root,
    resolve_device,
    resolve_output_root,
    run_patchcore_variant,
    set_seed,
    split_summary_wide,
)
""",
        """from patchcore_wrn50_local import (
    DEFAULT_SPLIT_CONFIG,
    DEFAULT_VARIANTS,
    WaferArrayDataset,
    attach_scores_to_metadata,
    auto_find_raw_pickle,
    defect_type_summary,
    load_wafer_array,
    metadata_paths,
    prepare_dataset,
    resolve_data_root,
    resolve_device,
    resolve_output_root,
    run_patchcore_variant,
    set_seed,
    split_summary,
    split_summary_wide,
)
from helpers.patchcore_threshold_tools import (
    build_review_policy_summary,
    build_single_threshold_policy_table,
    build_threshold_sweep,
    load_variant_artifacts,
)
""",
    )
    _set_source(notebook["cells"][1], cell1_source)

    _set_source(
        notebook["cells"][2],
        """RAW_PICKLE = os.environ.get("WM811K_RAW_PICKLE")
IMAGE_SIZE = 64
SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = 0
DEVICE = "auto"
TEACHER_LAYERS = ["layer2", "layer3"]
PRETRAINED = True
FREEZE_BACKBONE = True
NORMALIZE_IMAGENET = True
BACKBONE_INPUT_SIZE = 224
QUERY_CHUNK_SIZE = 1024
MEMORY_CHUNK_SIZE = 4096
THRESHOLD_QUANTILE = 0.95
THRESHOLD_STRATEGY = "validation_f1"
MAX_VALIDATION_FALSE_POSITIVE_RATE = None
SPLIT_CONFIG = DEFAULT_SPLIT_CONFIG.copy()
SWEEP_VARIANTS = DEFAULT_VARIANTS.copy()
DATA_ROOT = resolve_data_root(BUNDLE_ROOT)
OUTPUT_ROOT = resolve_output_root(BUNDLE_ROOT)
ARTIFACT_DIR = OUTPUT_ROOT / "patchcore_wrn50_multilayer_120k_5pct"
PLOTS_DIR = ARTIFACT_DIR / "plots"
RUN_VARIANT_SWEEP = False
VARIANT_NAME_OVERRIDE = None
MIN_RECALL = 0.70
MAX_FALSE_POSITIVE_RATE = 0.03
MAX_AUTO_NORMAL_ANOMALY_RATE = 0.01
MIN_AUTO_ANOMALY_PRECISION = 0.60

set_seed(SEED)
device = resolve_device(DEVICE)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

metadata = None
metadata_path = None
raw_pickle = None
arrays_dir = None
DATASET_READY = False

try:
    raw_pickle = auto_find_raw_pickle(RAW_PICKLE)
    metadata_path = prepare_dataset(raw_pickle, DATA_ROOT, IMAGE_SIZE, SPLIT_CONFIG, seed=SEED, overwrite=False)
    metadata = pd.read_csv(metadata_path)
    _, arrays_dir = metadata_paths(DATA_ROOT, IMAGE_SIZE, SPLIT_CONFIG)
    DATASET_READY = True
except FileNotFoundError as exc:
    print(f"[WARNING] {exc}")
    print("[WARNING] Dataset previews will be skipped until the local WM811K source pickle is available.")

if DATASET_READY:
    display(split_summary_wide(metadata))
    display(defect_type_summary(metadata).head(18))

print("Bundle root:", BUNDLE_ROOT)
print("Data root:", DATA_ROOT)
print("Output root:", OUTPUT_ROOT)
print("Artifact dir:", ARTIFACT_DIR)
print("Raw pickle:", raw_pickle)
print("Metadata path:", metadata_path)
print("Using device:", device)
print("Run variant sweep:", RUN_VARIANT_SWEEP)
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
""",
    )

    _set_source(
        notebook["cells"][3],
        """ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

selected_variant_name = None
selected_result = None
bundle_summary = {}
sweep_results_df = pd.DataFrame()
val_scores_df = None
test_scores_df = None
threshold_sweep_df = None
pred_df = None
threshold = None
best_sweep = None
EVALUATION_READY = False

bundle_summary_path = ARTIFACT_DIR / "bundle_summary.json"
sweep_results_path = ARTIFACT_DIR / "patchcore_sweep_results.csv"

if RUN_VARIANT_SWEEP:
    if not DATASET_READY:
        print("[WARNING] Dataset inputs are unavailable, so the WRN50 PatchCore sweep cannot be rebuilt in this notebook run.")
    else:
        train_dataset = WaferArrayDataset(metadata_path, split="train", data_root=DATA_ROOT)
        val_dataset = WaferArrayDataset(metadata_path, split="val", data_root=DATA_ROOT)
        test_dataset = WaferArrayDataset(metadata_path, split="test", data_root=DATA_ROOT)

        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        variant_results = {}
        rows = []
        for variant in SWEEP_VARIANTS:
            print(f"\\n=== Running variant: {variant['name']} ===")
            result = run_patchcore_variant(
                variant,
                train_dataset=train_dataset,
                val_loader=val_loader,
                test_loader=test_loader,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                device=device,
                output_dir=ARTIFACT_DIR,
                seed=SEED,
                teacher_layers=TEACHER_LAYERS,
                pretrained=PRETRAINED,
                freeze_backbone=FREEZE_BACKBONE,
                backbone_input_size=BACKBONE_INPUT_SIZE,
                normalize_imagenet=NORMALIZE_IMAGENET,
                threshold_quantile=THRESHOLD_QUANTILE,
                threshold_strategy=THRESHOLD_STRATEGY,
                max_validation_false_positive_rate=MAX_VALIDATION_FALSE_POSITIVE_RATE,
                query_chunk_size=QUERY_CHUNK_SIZE,
                memory_chunk_size=MEMORY_CHUNK_SIZE,
            )
            variant_results[variant["name"]] = result
            rows.append(result["row"])

        sweep_results_df = pd.DataFrame(rows).sort_values(["f1", "auroc", "auprc"], ascending=False).reset_index(drop=True)
        sweep_results_df.to_csv(sweep_results_path, index=False)
        selected_variant_name = str(VARIANT_NAME_OVERRIDE or sweep_results_df.iloc[0]["name"])
        selected_result = variant_results[selected_variant_name]
        threshold = float(selected_result["threshold"])
        val_scores_df = selected_result["val_scores_df"]
        test_scores_df = selected_result["test_scores_df"]
        threshold_sweep_df = selected_result["threshold_sweep_df"]
        best_sweep = selected_result["best_sweep"]

        bundle_summary = {
            "selected_variant": selected_variant_name,
            "split_config": SPLIT_CONFIG,
            "raw_pickle": str(raw_pickle),
            "metadata_path": str(metadata_path),
            "threshold_strategy": THRESHOLD_STRATEGY,
            "variants": [variant["name"] for variant in SWEEP_VARIANTS],
        }
        bundle_summary_path.write_text(json.dumps(bundle_summary, indent=2), encoding="utf-8")

        pred_df = attach_scores_to_metadata(
            metadata[metadata["split"] == "test"].reset_index(drop=True),
            selected_result["test_scores_df"],
            threshold,
        )
        pred_df.to_csv(ARTIFACT_DIR / f"{selected_variant_name}_test_predictions.csv", index=False)
        EVALUATION_READY = True
else:
    if not bundle_summary_path.exists() or not sweep_results_path.exists():
        print("[WARNING] No saved WRN50 PatchCore artifact bundle was found.")
        print("[WARNING] Keep RUN_VARIANT_SWEEP=False for review-only mode, or set it to True to rebuild the local artifacts.")
    else:
        bundle_summary = json.loads(bundle_summary_path.read_text(encoding="utf-8"))
        sweep_results_df = pd.read_csv(sweep_results_path).sort_values("f1", ascending=False).reset_index(drop=True)
        selected_variant_name = str(VARIANT_NAME_OVERRIDE or bundle_summary["selected_variant"])
        selected_row = sweep_results_df.loc[sweep_results_df["name"] == selected_variant_name].iloc[0]
        variant_dir = ARTIFACT_DIR / selected_variant_name

        try:
            summary, val_scores_df, test_scores_df = load_variant_artifacts(ARTIFACT_DIR, selected_variant_name)
        except FileNotFoundError as exc:
            print(f"[WARNING] {exc}")
            print("[WARNING] Saved evaluation review will be skipped until the selected variant files exist.")
        else:
            threshold = float(summary["threshold"])
            threshold_sweep_path = variant_dir / "threshold_sweep.csv"
            threshold_sweep_df = (
                pd.read_csv(threshold_sweep_path)
                if threshold_sweep_path.exists()
                else build_threshold_sweep(test_scores_df)
            )
            best_sweep = summary.get("best_sweep") or {
                "threshold": float(selected_row.get("best_sweep_threshold", threshold)),
                "precision": float(selected_row.get("best_sweep_precision", summary.get("precision", 0.0))),
                "recall": float(selected_row.get("best_sweep_recall", summary.get("recall", 0.0))),
                "f1": float(selected_row.get("best_sweep_f1", summary.get("f1", 0.0))),
            }
            selected_result = {
                "metrics": summary.get(
                    "metrics_at_validation_threshold",
                    {
                        "precision": float(summary.get("precision", 0.0)),
                        "recall": float(summary.get("recall", 0.0)),
                        "f1": float(summary.get("f1", 0.0)),
                        "auroc": float(summary.get("auroc", 0.0)),
                        "auprc": float(summary.get("auprc", 0.0)),
                        "confusion_matrix": summary.get("confusion_matrix", [[0, 0], [0, 0]]),
                    },
                ),
                "best_sweep": best_sweep,
                "threshold": threshold,
                "val_scores_df": val_scores_df,
                "test_scores_df": test_scores_df,
            }

            predictions_path = ARTIFACT_DIR / f"{selected_variant_name}_test_predictions.csv"
            if predictions_path.exists():
                pred_df = pd.read_csv(predictions_path)
            elif DATASET_READY:
                pred_df = attach_scores_to_metadata(
                    metadata[metadata["split"] == "test"].reset_index(drop=True),
                    test_scores_df,
                    threshold,
                )
                pred_df.to_csv(predictions_path, index=False)
            else:
                pred_df = None

            EVALUATION_READY = True

if EVALUATION_READY:
    print(f"Selected variant: {selected_variant_name}")
    display(sweep_results_df)
else:
    print("[WARNING] No WRN50 evaluation tables are available to display in this notebook run.")
""",
    )

    for idx, warning in [
        (4, "WRN50 selected-variant metrics are unavailable because the local artifact bundle is missing."),
        (5, "WRN50 sweep plots are unavailable because the local artifact bundle is missing."),
        (7, "WRN50 failure-analysis outputs are unavailable because the local artifact bundle is missing."),
    ]:
        _set_source(
            notebook["cells"][idx],
            _wrap_with_guard(
                _load_source(notebook["cells"][idx]),
                condition="EVALUATION_READY and selected_result is not None",
                warning=warning,
            ),
        )

    notebook["cells"].extend(
        [
            _markdown_cell(
                """## Dataset Sanity View

The old dataset-helper notebook is folded into the main experiment flow here so the split summary and a few representative wafer maps live next to the training and evaluation cells.
""",
                cell_id="wrn50_dataset_heading",
            ),
            _code_cell(
                """if not DATASET_READY:
    print("[WARNING] Dataset sanity plots are unavailable because the raw dataset inputs could not be prepared locally.")
else:
    display(split_summary(metadata))
    display(split_summary_wide(metadata))

    sample_specs = [
        ("train", 0, "Train normal"),
        ("train", 1, "Train anomaly"),
        ("val", 0, "Val normal"),
        ("val", 1, "Val anomaly"),
        ("test", 0, "Test normal"),
        ("test", 1, "Test anomaly"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (split_name, is_anomaly, title) in zip(axes.ravel(), sample_specs):
        rows = metadata[(metadata["split"] == split_name) & (metadata["is_anomaly"] == is_anomaly)]
        if rows.empty:
            ax.axis("off")
            continue
        row = rows.iloc[0]
        wafer = load_wafer_array(DATA_ROOT, row["array_path"])
        ax.imshow(wafer, cmap="viridis")
        ax.set_title(f"{title}\\n{row['defect_type']}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    plt.close(fig)
""",
                cell_id="wrn50_dataset_view",
            ),
            _markdown_cell(
                """## Threshold Policies

The saved threshold-policy review now lives directly in this notebook so we can compare alternate operating points without jumping to a second file.
""",
                cell_id="wrn50_threshold_heading",
            ),
            _code_cell(
                """if not EVALUATION_READY or selected_result is None or val_scores_df is None or test_scores_df is None:
    print("[WARNING] Threshold-policy analysis is unavailable because the selected WRN50 score bundle is missing.")
else:
    single_threshold_df = build_single_threshold_policy_table(
        val_scores_df,
        test_scores_df,
        current_threshold=float(threshold),
        min_recall=MIN_RECALL,
        max_false_positive_rate=MAX_FALSE_POSITIVE_RATE,
    )
    display(single_threshold_df.sort_values("test_f1", ascending=False).reset_index(drop=True).round(4))

    val_sweep_df = build_threshold_sweep(val_scores_df)
    validation_f1_threshold = float(single_threshold_df.loc[single_threshold_df["policy"] == "validation_f1", "threshold"].iloc[0])
    fp_cap_threshold = float(single_threshold_df.loc[single_threshold_df["policy"] == f"fp_cap_{MAX_FALSE_POSITIVE_RATE:.2%}", "threshold"].iloc[0])
    recall_floor_threshold = float(single_threshold_df.loc[single_threshold_df["policy"] == f"recall_floor_{MIN_RECALL:.2f}", "threshold"].iloc[0])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(val_sweep_df["threshold"], val_sweep_df["precision"], label="Validation precision", linewidth=2)
    ax.plot(val_sweep_df["threshold"], val_sweep_df["recall"], label="Validation recall", linewidth=2)
    ax.plot(val_sweep_df["threshold"], val_sweep_df["f1"], label="Validation F1", linewidth=2)
    ax.axvline(float(threshold), color="black", linestyle="--", linewidth=1.5, label="Current threshold")
    ax.axvline(validation_f1_threshold, color="tab:green", linestyle=":", linewidth=1.5, label="Validation F1 threshold")
    ax.axvline(fp_cap_threshold, color="tab:red", linestyle=":", linewidth=1.5, label="FP-cap threshold")
    ax.axvline(recall_floor_threshold, color="tab:blue", linestyle=":", linewidth=1.5, label="Recall-floor threshold")
    ax.set_title(f"Validation Threshold Sweep: {selected_variant_name}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric value")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()
    plt.close(fig)

    review_policy_df = build_review_policy_summary(
        val_scores_df,
        test_scores_df,
        max_auto_normal_anomaly_rate=MAX_AUTO_NORMAL_ANOMALY_RATE,
        min_auto_anomaly_precision=MIN_AUTO_ANOMALY_PRECISION,
    )
    display(review_policy_df.round(4))

    best_f1_row = single_threshold_df.loc[single_threshold_df["policy"] == "validation_f1"].iloc[0]
    recall_row = single_threshold_df.loc[single_threshold_df["policy"] == f"recall_floor_{MIN_RECALL:.2f}"].iloc[0]
    review_row = review_policy_df.iloc[0]
    message = f\"\"\"
### Working Guidance

- **Reduce false negatives:** use the recall-floor threshold near `{recall_row['threshold']:.6f}`. On the saved test split it reaches recall `{recall_row['test_recall']:.3f}`, but false positives rise to `{int(recall_row['test_fp'])}`.
- **Reduce false defect calls:** use the FP-capped or validation-F1 threshold. The validation-F1 threshold `{best_f1_row['threshold']:.6f}` cuts test false positives to `{int(best_f1_row['test_fp'])}` and lifts test F1 to `{best_f1_row['test_f1']:.3f}`.
- **Reduce both kinds of automatic mistakes:** use the review band. With the current defaults, scores below `{review_row['low_threshold']:.6f}` are auto-normal, scores at or above `{review_row['high_threshold']:.6f}` are auto-anomaly, and the middle goes to review.
- With that review band on the saved test split, auto-normal wafers have anomaly rate `{review_row['test_auto_normal_anomaly_rate']:.3%}`, auto-anomaly wafers have precision `{review_row['test_auto_anomaly_precision']:.3%}`, and `{review_row['test_review_rate']:.2%}` of wafers go to review.
\"\"\"
    display(Markdown(message))
""",
                cell_id="wrn50_threshold_review",
            ),
        ]
    )

    _write_notebook(WRN50_MAIN_NOTEBOOK, notebook)
    for extra_path in [WRN50_RESULTS_NOTEBOOK, WRN50_THRESHOLD_NOTEBOOK, WRN50_DATASET_NOTEBOOK]:
        _delete_if_exists(extra_path)


def main() -> int:
    patch_seed07()
    patch_effnet_main()
    patch_resnet18_main()
    patch_supervised_main()
    patch_wrn50_main()
    print("Consolidated the remaining experiment notebook candidates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
