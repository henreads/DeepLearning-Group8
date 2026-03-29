from __future__ import annotations

import json
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = REPO_ROOT / "experiments" / "anomaly_detection" / "fastflow" / "x64" / "main" / "notebook.ipynb"


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(source).lstrip("\n").splitlines(True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).lstrip("\n").splitlines(True),
    }


cells: list[dict] = []

cells.extend(
    [
        md_cell(
            """
            # FastFlow x64 Variant Sweep

            This notebook is the curated submission-facing entry point for the FastFlow family on the shared `x64` anomaly benchmark.
            It compares three Wide ResNet50-2 feature variants while defaulting to saved artifacts instead of retraining.
            """
        ),
        md_cell(
            """
            ## Submission Context

            - Dataset notebook: `data/dataset/x64/benchmark_50k_5pct/notebook.ipynb`
            - Dataset config: `data/dataset/x64/benchmark_50k_5pct/data_config.toml`
            - Experiment config: `experiments/anomaly_detection/fastflow/x64/main/train_config.toml`
            - Artifact root: `experiments/anomaly_detection/fastflow/x64/main/artifacts/fastflow_variant_sweep/`
            - Default behavior: reuse saved CSV artifacts first; only train missing variants when you explicitly opt in
            - Historical note: the original prototype run saved CSV summaries but not checkpoints; future training runs from this curated notebook now save `*_best_model.pt`
            """
        ),
        md_cell(
            """
            ### Setup And Imports

            This cell resolves the repo root, exposes `src/` on `PYTHONPATH`, and imports the shared dataset loader plus the plotting and PyTorch utilities used later in the notebook.
            """
        ),
        code_cell(
            """
            from __future__ import annotations

            import json
            import math
            import random
            import shutil
            import sys
            from pathlib import Path
            from typing import Any

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import torch
            import torch.nn.functional as F
            import tomllib
            from sklearn.metrics import average_precision_score, roc_auc_score
            from torch import nn
            from torch.utils.data import DataLoader
            from tqdm.auto import tqdm

            try:
                from IPython.display import display
            except ImportError:
                def display(obj: object) -> None:
                    print(obj)

            REPO_ROOT = Path.cwd().resolve()
            if not (REPO_ROOT / "src" / "wafer_defect").exists():
                for candidate in [REPO_ROOT, *REPO_ROOT.parents]:
                    if (candidate / "src" / "wafer_defect").exists():
                        REPO_ROOT = candidate
                        break

            SRC_ROOT = REPO_ROOT / "src"
            if str(SRC_ROOT) not in sys.path:
                sys.path.insert(0, str(SRC_ROOT))

            from wafer_defect.data.wm811k import WaferMapDataset

            torch.backends.cudnn.benchmark = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
            """
        ),
        md_cell(
            """
            ### Load The Curated Experiment Config

            This cell reads the local TOML config and exposes the small set of runtime flags we will use. The notebook defaults to artifact reuse, so missing checkpoints do not trigger retraining unless you explicitly request it.
            """
        ),
        code_cell(
            """
            CONFIG_PATH = REPO_ROOT / "experiments" / "anomaly_detection" / "fastflow" / "x64" / "main" / "train_config.toml"
            with CONFIG_PATH.open("rb") as handle:
                config = tomllib.load(handle)

            OUTPUT_DIR = REPO_ROOT / config["run"]["output_dir"]
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            RESULTS_DIR = OUTPUT_DIR / "results"
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            PLOTS_DIR = OUTPUT_DIR / "plots"
            PLOTS_DIR.mkdir(parents=True, exist_ok=True)

            FORCE_RETRAIN_VARIANTS = False
            RUN_MISSING_VARIANTS = False
            FORCE_REBUILD_PLOTS = False
            QUALITATIVE_VARIANT = "wrn50_l23_s4"
            QUALITATIVE_MAX_EXAMPLES = 6

            config
            """
        ),
        md_cell(
            """
            ### Migrate Legacy Artifacts And Inspect Current Coverage

            This cell normalizes the old typo'd artifact folder if it still exists, then shows which variants already have reusable CSV outputs and which ones already have checkpoints.
            """
        ),
        code_cell(
            """
            LEGACY_OUTPUT_DIR = REPO_ROOT / "experiments" / "anomaly_detection" / "fastflow" / "x64" / "main" / "artifacts" / "fatsflow"
            if LEGACY_OUTPUT_DIR.exists():
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                for old_path in LEGACY_OUTPUT_DIR.iterdir():
                    target = OUTPUT_DIR / old_path.name
                    if not target.exists():
                        shutil.move(str(old_path), str(target))
                if not any(LEGACY_OUTPUT_DIR.iterdir()):
                    LEGACY_OUTPUT_DIR.rmdir()

            def variant_root(output_dir: Path, variant_name: str) -> Path:
                root = output_dir / variant_name
                (root / "checkpoints").mkdir(parents=True, exist_ok=True)
                (root / "results").mkdir(parents=True, exist_ok=True)
                return root

            def variant_paths(output_dir: Path, variant_name: str) -> dict[str, Path]:
                root = variant_root(output_dir, variant_name)
                return {
                    "root": root,
                    "checkpoints_dir": root / "checkpoints",
                    "results_dir": root / "results",
                    "history": root / "results" / "history.csv",
                    "scores": root / "results" / "scores.csv",
                    "defect_breakdown": root / "results" / "defect_breakdown.csv",
                    "best_row": root / "results" / "best_row.csv",
                    "best_model": root / "checkpoints" / "best_model.pt",
                    "latest_checkpoint": root / "checkpoints" / "latest_checkpoint.pt",
                    "summary_json": root / "results" / "summary.json",
                }

            legacy_suffix_map = {
                "history.csv": ("results", "history.csv"),
                "scores.csv": ("results", "scores.csv"),
                "defect_breakdown.csv": ("results", "defect_breakdown.csv"),
                "best_row.csv": ("results", "best_row.csv"),
                "summary.json": ("results", "summary.json"),
                "best_model.pt": ("checkpoints", "best_model.pt"),
                "latest_checkpoint.pt": ("checkpoints", "latest_checkpoint.pt"),
            }
            for variant in config["variants"]:
                variant_name = variant["name"]
                paths = variant_paths(OUTPUT_DIR, variant_name)
                for suffix, (folder_name, target_name) in legacy_suffix_map.items():
                    legacy_path = OUTPUT_DIR / f"{variant_name}_{suffix}"
                    target_path = paths["root"] / folder_name / target_name
                    if legacy_path.exists() and not target_path.exists():
                        shutil.move(str(legacy_path), str(target_path))
                for epoch_checkpoint in OUTPUT_DIR.glob(f"{variant_name}_checkpoint_epoch_*.pt"):
                    target_path = paths["checkpoints_dir"] / epoch_checkpoint.name.replace(f"{variant_name}_", "", 1)
                    if not target_path.exists():
                        shutil.move(str(epoch_checkpoint), str(target_path))

            for shared_name in ["fastflow_variant_summary.csv", "run_manifest.json"]:
                shared_path = OUTPUT_DIR / shared_name
                if shared_path.exists():
                    target_path = RESULTS_DIR / shared_name
                    if not target_path.exists():
                        shutil.move(str(shared_path), str(target_path))

            artifact_rows = []
            for variant in config["variants"]:
                paths = variant_paths(OUTPUT_DIR, variant["name"])
                artifact_rows.append(
                    {
                        "variant": variant["name"],
                        "history_csv": paths["history"].exists(),
                        "score_csv": paths["scores"].exists(),
                        "defect_breakdown_csv": paths["defect_breakdown"].exists(),
                        "best_row_csv": paths["best_row"].exists(),
                        "best_model_pt": paths["best_model"].exists(),
                        "latest_checkpoint_pt": paths["latest_checkpoint"].exists(),
                    }
                )

            artifact_status_df = pd.DataFrame(artifact_rows)
            display(artifact_status_df)
            """
        ),
        md_cell(
            """
            ### Load The Shared Processed Dataset

            This cell uses the curated processed metadata CSV instead of rebuilding the benchmark split from the raw pickle. It also gives a quick preview of the split counts so a grader can confirm the notebook is pointed at the intended benchmark.
            """
        ),
        code_cell(
            """
            metadata_path = REPO_ROOT / config["data"]["metadata_csv"]
            metadata_df = pd.read_csv(metadata_path)

            train_dataset = WaferMapDataset(metadata_path, split="train", image_size=int(config["data"]["image_size"]))
            val_dataset = WaferMapDataset(metadata_path, split="val", image_size=int(config["data"]["image_size"]))
            test_dataset = WaferMapDataset(metadata_path, split="test", image_size=int(config["data"]["image_size"]))

            split_summary_df = (
                metadata_df.groupby(["split", "is_anomaly"])
                .size()
                .reset_index(name="count")
                .sort_values(["split", "is_anomaly"])
                .reset_index(drop=True)
            )
            print(f"Metadata CSV: {metadata_path}")
            display(split_summary_df)
            display(metadata_df.head(5))
            """
        ),
    ]
)

fastflow_helpers_insert_at = len(cells)

cells.extend(
    [
        md_cell(
            """
            ### Training, Checkpointing, And Reuse Helpers

            This cell defines the loader construction, training loop, checkpoint saving, saved-artifact loading, and checkpoint restoration helpers that power the artifact-first workflow.
            """
        ),
        code_cell(
            """
            def make_loaders(train_ds: WaferMapDataset, val_ds: WaferMapDataset, test_ds: WaferMapDataset, config: dict[str, Any], device: torch.device) -> dict[str, DataLoader]:
                batch_size = int(config["data"]["batch_size"])
                num_workers = int(config["data"]["num_workers"])
                pin_memory = device.type == "cuda"
                return {
                    "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory),
                    "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
                    "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
                }


            def run_epoch(
                model: nn.Module,
                loader: DataLoader,
                device: torch.device,
                optimizer: torch.optim.Optimizer | None,
                scaler: torch.amp.GradScaler | None,
                amp_enabled: bool,
                grad_clip_norm: float | None,
                desc: str,
            ) -> float:
                is_train = optimizer is not None
                model.train(is_train)
                total_loss = 0.0
                sample_count = 0
                autocast_enabled = amp_enabled and device.type == "cuda"
                iterator = tqdm(loader, desc=desc, leave=False)
                for batch_x, _ in iterator:
                    batch_x = batch_x.to(device, non_blocking=True)
                    if is_train:
                        optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                        zs, logdets = model(batch_x)
                        loss = compute_loss(zs, logdets)
                    if is_train:
                        if scaler is not None and autocast_enabled:
                            scaler.scale(loss).backward()
                            if grad_clip_norm is not None:
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            if grad_clip_norm is not None:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
                            optimizer.step()
                    batch_size = batch_x.shape[0]
                    total_loss += float(loss.detach().cpu()) * batch_size
                    sample_count += batch_size
                return total_loss / max(1, sample_count)


            @torch.no_grad()
            def collect_maps(model: FastFlowModel, loader: DataLoader, device: torch.device, output_size: int, desc: str) -> tuple[torch.Tensor, np.ndarray]:
                model.eval()
                map_batches = []
                label_batches = []
                iterator = tqdm(loader, desc=desc, leave=False)
                for batch_x, batch_y in iterator:
                    batch_x = batch_x.to(device, non_blocking=True)
                    maps = model.anomaly_map(batch_x, output_size=output_size)
                    map_batches.append(maps.cpu())
                    label_batches.append(batch_y.cpu().numpy())
                return torch.cat(map_batches, dim=0), np.concatenate(label_batches, axis=0)


            def train_and_evaluate_variant(
                variant: dict[str, Any],
                config: dict[str, Any],
                train_ds: WaferMapDataset,
                val_ds: WaferMapDataset,
                test_ds: WaferMapDataset,
            ) -> dict[str, Any]:
                set_seed(int(config["run"]["seed"]))
                device = resolve_device(str(config["training"]["device"]))
                loaders = make_loaders(train_ds, val_ds, test_ds, config, device)
                model = FastFlowModel(
                    variant=variant,
                    image_size=int(config["data"]["image_size"]),
                    pretrained_backbone=bool(config["model"].get("backbone_pretrained", True)),
                ).to(device)
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=float(config["training"]["learning_rate"]),
                    weight_decay=float(config["training"]["weight_decay"]),
                )
                amp_enabled = bool(config["training"].get("amp", False))
                scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled and device.type == "cuda")
                best_state_dict = None
                best_val_loss = float("inf")
                best_epoch = 0
                stale_epochs = 0
                history_rows = []
                paths = variant_paths(OUTPUT_DIR, variant["name"])

                for epoch in tqdm(range(1, int(config["training"]["epochs"]) + 1), desc=f"epochs:{variant['name']}", leave=True):
                    train_loss = run_epoch(
                        model=model,
                        loader=loaders["train"],
                        device=device,
                        optimizer=optimizer,
                        scaler=scaler,
                        amp_enabled=amp_enabled,
                        grad_clip_norm=float(config["training"]["grad_clip_norm"]),
                        desc=f"train:{variant['name']}:e{epoch}",
                    )
                    val_loss = run_epoch(
                        model=model,
                        loader=loaders["val"],
                        device=device,
                        optimizer=None,
                        scaler=None,
                        amp_enabled=False,
                        grad_clip_norm=None,
                        desc=f"val:{variant['name']}:e{epoch}",
                    )
                    history_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
                    improved = val_loss < best_val_loss
                    if improved:
                        best_val_loss = val_loss
                        best_epoch = epoch
                        best_state_dict = clone_state_dict(model)
                        stale_epochs = 0
                        torch.save(
                            {
                                "epoch": epoch,
                                "best_epoch": best_epoch,
                                "best_val_loss": best_val_loss,
                                "variant": dict(variant),
                                "config": config,
                                "model_state_dict": best_state_dict,
                                "optimizer_state_dict": optimizer.state_dict(),
                                "history": history_rows,
                            },
                            paths["best_model"],
                        )
                    else:
                        stale_epochs += 1

                    latest_payload = {
                        "epoch": epoch,
                        "best_epoch": best_epoch,
                        "best_val_loss": best_val_loss,
                        "variant": dict(variant),
                        "config": config,
                        "model_state_dict": clone_state_dict(model),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "history": history_rows,
                    }
                    torch.save(latest_payload, paths["latest_checkpoint"])
                    checkpoint_every = int(config["training"].get("checkpoint_every", 0))
                    if checkpoint_every > 0 and epoch % checkpoint_every == 0:
                        torch.save(latest_payload, paths["checkpoints_dir"] / f"checkpoint_epoch_{epoch}.pt")

                    if stale_epochs >= int(config["training"]["early_stopping_patience"]):
                        break

                if best_state_dict is None:
                    best_state_dict = clone_state_dict(model)
                model.load_state_dict(best_state_dict)

                val_maps, _ = collect_maps(model, loaders["val"], device, int(config["data"]["image_size"]), desc=f"maps_val:{variant['name']}")
                test_maps, test_labels = collect_maps(model, loaders["test"], device, int(config["data"]["image_size"]), desc=f"maps_test:{variant['name']}")

                rows = []
                for score_variant in tqdm(config["scoring"]["sweep_variants"], desc=f"scores:{variant['name']}", leave=False):
                    val_scores = reduce_scores(val_maps, score_variant["reduction"], float(score_variant["topk_ratio"]))
                    test_scores = reduce_scores(test_maps, score_variant["reduction"], float(score_variant["topk_ratio"]))
                    threshold = float(np.quantile(val_scores, float(config["scoring"]["threshold_quantile"])))
                    metrics = summarize_threshold_metrics(test_labels.astype(int), test_scores, threshold)
                    rows.append(
                        {
                            "variant": variant["name"],
                            "score_name": score_variant["name"],
                            "reduction": score_variant["reduction"],
                            "topk_ratio": float(score_variant["topk_ratio"]),
                            **metrics,
                        }
                    )

                score_df = pd.DataFrame(rows).sort_values(["f1", "recall", "precision"], ascending=False).reset_index(drop=True)
                best_row = score_df.iloc[0].to_dict()
                best_scores = reduce_scores(test_maps, best_row["reduction"], float(best_row["topk_ratio"]))
                defect_breakdown_df = build_defect_breakdown(
                    defect_types=test_ds.metadata["defect_type"].to_numpy(),
                    labels=test_labels.astype(int),
                    scores=best_scores,
                    threshold=float(best_row["threshold"]),
                )

                history_df = pd.DataFrame(history_rows)
                score_df.to_csv(paths["scores"], index=False)
                history_df.to_csv(paths["history"], index=False)
                defect_breakdown_df.to_csv(paths["defect_breakdown"], index=False)
                pd.DataFrame([best_row]).to_csv(paths["best_row"], index=False)
                paths["summary_json"].write_text(
                    json.dumps(
                        {
                            "variant": variant["name"],
                            "best_epoch": int(best_epoch),
                            "best_val_loss": float(best_val_loss),
                            "best_row": best_row,
                            "checkpoint": str(paths["best_model"].relative_to(REPO_ROOT)) if paths["best_model"].exists() else "",
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                return {
                    "history": history_df,
                    "score_df": score_df,
                    "best": best_row,
                    "defect_breakdown_df": defect_breakdown_df,
                    "checkpoint_path": paths["best_model"] if paths["best_model"].exists() else None,
                }


            def load_saved_variant_result(variant_name: str) -> dict[str, Any]:
                paths = variant_paths(OUTPUT_DIR, variant_name)
                return {
                    "history": pd.read_csv(paths["history"]),
                    "score_df": pd.read_csv(paths["scores"]),
                    "best": pd.read_csv(paths["best_row"]).iloc[0].to_dict(),
                    "defect_breakdown_df": pd.read_csv(paths["defect_breakdown"]),
                    "checkpoint_path": paths["best_model"] if paths["best_model"].exists() else None,
                }


            def build_model_from_checkpoint(checkpoint_path: Path, variant: dict[str, Any], config: dict[str, Any]) -> FastFlowModel:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                checkpoint_variant = dict(checkpoint.get("variant", variant))
                checkpoint_config = dict(checkpoint.get("config", config))
                model = FastFlowModel(
                    variant=checkpoint_variant,
                    image_size=int(checkpoint_config["data"]["image_size"]),
                    pretrained_backbone=bool(checkpoint_config["model"].get("backbone_pretrained", True)),
                )
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()
                return model
            """
        ),
        md_cell(
            """
            ### Reuse Saved Results Or Train Missing Variants

            This cell is the main control point. By default it loads the saved CSV outputs already present in the repo. If you set `RUN_MISSING_VARIANTS = True` it will train any missing variants and, unlike the old prototype notebook, save reusable checkpoints.
            """
        ),
        code_cell(
            """
            variant_results: dict[str, dict[str, Any]] = {}
            missing_variants: list[str] = []

            for variant in config["variants"]:
                paths = variant_paths(OUTPUT_DIR, variant["name"])
                artifacts_ready = all(paths[key].exists() for key in ["history", "scores", "defect_breakdown", "best_row"])
                if artifacts_ready and not FORCE_RETRAIN_VARIANTS:
                    variant_results[variant["name"]] = load_saved_variant_result(variant["name"])
                    continue

                if not FORCE_RETRAIN_VARIANTS and not RUN_MISSING_VARIANTS:
                    missing_variants.append(variant["name"])
                    continue

                result = train_and_evaluate_variant(
                    variant=variant,
                    config=config,
                    train_ds=train_dataset,
                    val_ds=val_dataset,
                    test_ds=test_dataset,
                )
                variant_results[variant["name"]] = result

            if missing_variants:
                print("Missing variants were left untouched because artifact-first mode is enabled:")
                print(missing_variants)
                print("Set RUN_MISSING_VARIANTS = True or FORCE_RETRAIN_VARIANTS = True if you want to train them.")

            if not variant_results:
                raise RuntimeError("No FastFlow variant artifacts were loaded. Enable RUN_MISSING_VARIANTS to generate them.")

            summary_df = (
                pd.DataFrame([result["best"] for result in variant_results.values()])
                .sort_values(["f1", "recall", "precision"], ascending=False)
                .reset_index(drop=True)
            )
            summary_df.to_csv(RESULTS_DIR / "fastflow_variant_summary.csv", index=False)
            run_manifest = {
                "best_variant": str(summary_df.iloc[0]["variant"]),
                "variants": list(summary_df["variant"]),
                "artifact_root": str(OUTPUT_DIR.relative_to(REPO_ROOT)),
            }
            (RESULTS_DIR / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
            display(summary_df)
            """
        ),
        md_cell(
            """
            ### Plot The Saved Results

            This cell rebuilds the main report-style figures from the saved CSV artifacts so the notebook both displays them inline and saves them into the experiment artifact directory.
            """
        ),
        code_cell(
            """
            best_variant_name = str(summary_df.iloc[0]["variant"])
            best_result = variant_results[best_variant_name]

            def save_and_show(fig: plt.Figure, output_path: Path) -> None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(output_path, dpi=160, bbox_inches="tight")
                display(fig)
                plt.close(fig)


            metric_columns = ["f1", "recall", "precision", "auroc", "auprc"]
            summary_plot_df = summary_df[["variant", *metric_columns]].copy()
            fig, axes = plt.subplots(1, len(metric_columns), figsize=(4.0 * len(metric_columns), 4.5), constrained_layout=True)
            for axis, metric_name in zip(axes, metric_columns):
                axis.bar(summary_plot_df["variant"], summary_plot_df[metric_name], color="#1f77b4")
                axis.set_title(metric_name.upper())
                axis.set_ylim(0.0, 1.0)
                axis.tick_params(axis="x", rotation=25)
            save_and_show(fig, PLOTS_DIR / "variant_comparison_metrics.png")

            fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
            for variant_name, result in variant_results.items():
                history_df = result["history"]
                ax.plot(history_df["epoch"], history_df["train_loss"], label=f"{variant_name} train")
                ax.plot(history_df["epoch"], history_df["val_loss"], linestyle="--", label=f"{variant_name} val")
            ax.set_title("FastFlow Training And Validation Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend(loc="best", fontsize=8)
            save_and_show(fig, PLOTS_DIR / "training_curves.png")

            fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
            score_df = best_result["score_df"].copy()
            ax.bar(score_df["score_name"], score_df["f1"], color="#2ca02c")
            ax.set_title(f"Score Reduction Sweep For {best_variant_name}")
            ax.set_xlabel("Score reduction")
            ax.set_ylabel("F1")
            ax.set_ylim(0.0, 1.0)
            save_and_show(fig, PLOTS_DIR / "best_variant_score_sweep_f1.png")

            fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
            defect_df = best_result["defect_breakdown_df"].copy().head(8)
            ax.bar(defect_df["defect_type"], defect_df["recall"], color="#d62728")
            ax.set_title(f"Per-Defect Recall For {best_variant_name}")
            ax.set_xlabel("Defect type")
            ax.set_ylabel("Recall")
            ax.set_ylim(0.0, 1.0)
            ax.tick_params(axis="x", rotation=30)
            save_and_show(fig, PLOTS_DIR / "best_variant_defect_breakdown.png")

            best_variant_paths = variant_paths(OUTPUT_DIR, best_variant_name)
            print(f"Best variant: {best_variant_name}")
            print(f"Best-row CSV: {best_variant_paths['best_row']}")
            print(f"Defect-breakdown CSV: {best_variant_paths['defect_breakdown']}")
            print(f"Checkpoint available: {best_variant_paths['best_model'].exists()}")
            display(best_result["score_df"])
            display(best_result["defect_breakdown_df"].head(12))
            """
        ),
        md_cell(
            """
            ### Optional Qualitative Heatmaps From A Saved Checkpoint

            This cell only runs when a checkpoint exists for the selected variant. It loads the saved model, scores a handful of test examples, and saves a qualitative figure. If no checkpoint is present yet, the notebook skips the step cleanly and keeps the rest of the artifact-first workflow usable.
            """
        ),
        code_cell(
            """
            checkpoint_variant = next((variant for variant in config["variants"] if variant["name"] == QUALITATIVE_VARIANT), None)
            if checkpoint_variant is None:
                raise KeyError(f"Unknown QUALITATIVE_VARIANT: {QUALITATIVE_VARIANT}")

            checkpoint_path = variant_paths(OUTPUT_DIR, QUALITATIVE_VARIANT)["best_model"]
            if not checkpoint_path.exists():
                print(f"No checkpoint saved yet for {QUALITATIVE_VARIANT}.")
                print("To generate one, set RUN_MISSING_VARIANTS = True or FORCE_RETRAIN_VARIANTS = True and rerun the training cell.")
            else:
                device = resolve_device(str(config["training"]["device"]))
                model = build_model_from_checkpoint(checkpoint_path, checkpoint_variant, config).to(device)
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=8,
                    shuffle=False,
                    num_workers=int(config["data"]["num_workers"]),
                    pin_memory=device.type == "cuda",
                )

                example_images = []
                example_maps = []
                example_labels = []
                with torch.no_grad():
                    for batch_x, batch_y in test_loader:
                        batch_x = batch_x.to(device, non_blocking=True)
                        batch_maps = model.anomaly_map(batch_x, output_size=int(config["data"]["image_size"])).cpu()
                        example_images.extend(batch_x.cpu())
                        example_maps.extend(batch_maps)
                        example_labels.extend(batch_y.cpu().tolist())
                        if len(example_images) >= QUALITATIVE_MAX_EXAMPLES:
                            break

                n_examples = min(QUALITATIVE_MAX_EXAMPLES, len(example_images))
                fig, axes = plt.subplots(n_examples, 2, figsize=(8, 3 * n_examples), constrained_layout=True)
                if n_examples == 1:
                    axes = np.array([axes])
                for row_idx in range(n_examples):
                    image = example_images[row_idx].squeeze().numpy()
                    heatmap = example_maps[row_idx].squeeze().numpy()
                    label = int(example_labels[row_idx])
                    axes[row_idx, 0].imshow(image, cmap="gray")
                    axes[row_idx, 0].set_title(f"Wafer map #{row_idx} | label={label}")
                    axes[row_idx, 0].axis("off")
                    axes[row_idx, 1].imshow(heatmap, cmap="inferno")
                    axes[row_idx, 1].set_title("FastFlow anomaly map")
                    axes[row_idx, 1].axis("off")
                save_and_show(fig, PLOTS_DIR / f"{QUALITATIVE_VARIANT}_qualitative_heatmaps.png")
            """
        ),
    ]
)

fastflow_utility_cells = [
        md_cell(
            """
            ### Define FastFlow Utilities

            This cell contains the FastFlow backbone, training loop, checkpoint serialization, and evaluation helpers. The backbone import is lazy so artifact-only notebook runs do not require `torchvision` unless we actually need to train or load a checkpoint.
            """
        ),
        code_cell(
            """
            LOG_2PI = math.log(2.0 * math.pi)


            def set_seed(seed: int) -> None:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)


            def resolve_device(device_name: str) -> torch.device:
                if device_name == "auto":
                    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
                return torch.device(device_name)


            def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
                return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


            class ImageNetWaferPreprocessor(nn.Module):
                def __init__(self, input_size: int) -> None:
                    super().__init__()
                    self.input_size = int(input_size)
                    self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
                    self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    x = x.repeat(1, 3, 1, 1)
                    x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
                    return (x - self.mean) / self.std


            class FrozenWideResNet50Backbone(nn.Module):
                def __init__(self, layers: list[str], input_size: int, pretrained: bool = True) -> None:
                    super().__init__()
                    try:
                        from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
                        from torchvision.models.feature_extraction import create_feature_extractor
                    except Exception as exc:
                        raise RuntimeError("torchvision is required for FastFlow training or checkpoint inference.") from exc

                    weights = Wide_ResNet50_2_Weights.DEFAULT if pretrained else None
                    backbone = wide_resnet50_2(weights=weights)
                    self.preprocess = ImageNetWaferPreprocessor(input_size)
                    self.extractor = create_feature_extractor(backbone, return_nodes={layer: layer for layer in layers})
                    self.extractor.eval()
                    for parameter in self.extractor.parameters():
                        parameter.requires_grad_(False)

                def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
                    x = self.preprocess(x)
                    with torch.no_grad():
                        return self.extractor(x)


            class Orthogonal1x1Conv(nn.Module):
                def __init__(self, channels: int) -> None:
                    super().__init__()
                    self.weight_raw = nn.Parameter(torch.eye(channels, dtype=torch.float32))

                def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                    q, _ = torch.linalg.qr(self.weight_raw)
                    return F.conv2d(x, q.view(q.shape[0], q.shape[1], 1, 1)), torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)


            class AffineCoupling(nn.Module):
                def __init__(self, channels: int, hidden_channels: int, scale_clamp: float) -> None:
                    super().__init__()
                    half = channels // 2
                    self.scale_clamp = float(scale_clamp)
                    self.net = nn.Sequential(
                        nn.Conv2d(half, hidden_channels, 3, padding=1),
                        nn.GELU(),
                        nn.Conv2d(hidden_channels, hidden_channels, 1),
                        nn.GELU(),
                        nn.Conv2d(hidden_channels, channels, 3, padding=1),
                    )
                    nn.init.zeros_(self.net[-1].weight)
                    nn.init.zeros_(self.net[-1].bias)

                def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                    x_a, x_b = torch.chunk(x, 2, dim=1)
                    shift, log_scale = torch.chunk(self.net(x_a), 2, dim=1)
                    log_scale = torch.tanh(log_scale) * self.scale_clamp
                    y_b = x_b * torch.exp(log_scale) + shift
                    return torch.cat([x_a, y_b], dim=1), log_scale.flatten(1).sum(dim=1)


            class FastFlowStage(nn.Module):
                def __init__(self, channels: int, flow_steps: int, hidden_channels: int, scale_clamp: float) -> None:
                    super().__init__()
                    self.mixers = nn.ModuleList([Orthogonal1x1Conv(channels) for _ in range(flow_steps)])
                    self.couplings = nn.ModuleList([AffineCoupling(channels, hidden_channels, scale_clamp) for _ in range(flow_steps)])

                def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                    total_logdet = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
                    z = x
                    for mix, coupling in zip(self.mixers, self.couplings):
                        z, mix_logdet = mix(z)
                        z, coupling_logdet = coupling(z)
                        total_logdet = total_logdet + mix_logdet + coupling_logdet
                    return z, total_logdet


            class FastFlowModel(nn.Module):
                def __init__(self, variant: dict[str, Any], image_size: int, pretrained_backbone: bool = True) -> None:
                    super().__init__()
                    self.backbone = FrozenWideResNet50Backbone(
                        layers=list(variant["layers"]),
                        input_size=int(variant["input_size"]),
                        pretrained=pretrained_backbone,
                    )
                    with torch.no_grad():
                        dummy = torch.zeros(1, 1, int(image_size), int(image_size))
                        feature_shapes = self.backbone(dummy)
                    self.stages = nn.ModuleDict(
                        {
                            name: FastFlowStage(
                                channels=feature.shape[1],
                                flow_steps=int(variant["flow_steps"]),
                                hidden_channels=int(variant["hidden_channels"]),
                                scale_clamp=float(variant["scale_clamp"]),
                            )
                            for name, feature in feature_shapes.items()
                        }
                    )

                def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
                    features = self.backbone(x)
                    zs, logdets = [], []
                    for name, feature in features.items():
                        z, logdet = self.stages[name](feature)
                        zs.append(z)
                        logdets.append(logdet)
                    return zs, logdets

                def anomaly_map(self, x: torch.Tensor, output_size: int) -> torch.Tensor:
                    zs, _ = self.forward(x)
                    maps = [
                        F.interpolate(
                            0.5 * torch.nan_to_num(z, nan=0.0, posinf=1e6, neginf=-1e6).pow(2).mean(dim=1, keepdim=True),
                            size=(output_size, output_size),
                            mode="bilinear",
                            align_corners=False,
                        )
                        for z in zs
                    ]
                    return torch.nan_to_num(torch.stack(maps).mean(dim=0), nan=0.0, posinf=1e6, neginf=0.0)


            def compute_loss(zs: list[torch.Tensor], logdets: list[torch.Tensor]) -> torch.Tensor:
                losses = []
                for z, logdet in zip(zs, logdets):
                    nll = 0.5 * (z.pow(2) + LOG_2PI).flatten(1).sum(dim=1) - logdet
                    losses.append((nll / z[0].numel()).mean())
                return torch.stack(losses).mean()


            def reduce_scores(maps: torch.Tensor, reduction: str, topk_ratio: float) -> np.ndarray:
                maps = torch.nan_to_num(maps.detach().float(), nan=0.0, posinf=1e6, neginf=0.0)
                flat = maps.flatten(1)
                if reduction == "mean":
                    return flat.mean(dim=1).cpu().numpy()
                if reduction == "max":
                    return flat.max(dim=1).values.cpu().numpy()
                k = max(1, int(math.ceil(flat.shape[1] * float(topk_ratio))))
                return flat.topk(k=k, dim=1).values.mean(dim=1).cpu().numpy()


            def summarize_threshold_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
                labels = labels.astype(int)
                scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
                predicted = (scores > threshold).astype(int)
                tp = int(((predicted == 1) & (labels == 1)).sum())
                fp = int(((predicted == 1) & (labels == 0)).sum())
                tn = int(((predicted == 0) & (labels == 0)).sum())
                fn = int(((predicted == 0) & (labels == 1)).sum())
                precision = tp / max(1, tp + fp)
                recall = tp / max(1, tp + fn)
                f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
                balanced_accuracy = 0.5 * ((tp / max(1, tp + fn)) + (tn / max(1, tn + fp)))
                return {
                    "threshold": float(threshold),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "auroc": float(roc_auc_score(labels, scores)),
                    "auprc": float(average_precision_score(labels, scores)),
                    "balanced_accuracy": float(balanced_accuracy),
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                }


            def build_defect_breakdown(defect_types: np.ndarray, labels: np.ndarray, scores: np.ndarray, threshold: float) -> pd.DataFrame:
                frame = pd.DataFrame(
                    {
                        "defect_type": defect_types,
                        "label": labels.astype(int),
                        "score": np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0),
                    }
                )
                frame = frame[frame["label"] == 1].copy()
                frame["detected"] = (frame["score"] > threshold).astype(int)
                summary = (
                    frame.groupby("defect_type")
                    .agg(
                        count=("defect_type", "size"),
                        detected=("detected", "sum"),
                        mean_score=("score", "mean"),
                        median_score=("score", "median"),
                    )
                    .reset_index()
                )
                summary["missed"] = summary["count"] - summary["detected"]
                summary["recall"] = summary["detected"] / summary["count"].clip(lower=1)
                return summary.sort_values(["recall", "count", "mean_score"], ascending=[True, False, False]).reset_index(drop=True)
            """
        ),
    ]

cells[fastflow_helpers_insert_at:fastflow_helpers_insert_at] = fastflow_utility_cells


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH}")
