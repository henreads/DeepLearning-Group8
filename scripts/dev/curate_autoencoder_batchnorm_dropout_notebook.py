from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def md_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.split("\n")]}


DESCRIPTIONS = [
    "### Imports\n\nThis cell loads the libraries, repo-local modules, and path helpers used by the notebook.",
    "### Run Controls\n\nThis cell defines the experiment config path, the dropout sweep values, and the rerun flags. Leave both flags `False` to reuse saved artifacts when they already exist.",
    "### Reproducibility And Helpers\n\nThis cell sets the random seed, resolves the execution device, and defines a helper for saving figures.",
    "### Metadata Check\n\nThis cell loads the configured metadata CSV so we can verify the split before building loaders.",
    "### Data Loaders\n\nThis cell builds the train, validation, and test loaders used throughout the notebook.",
    "### Model Factory\n\nThis cell defines how each dropout variant is named on disk and how the corresponding autoencoder is constructed.",
    "### Dropout Sweep Or Artifact Reuse\n\nThis cell either reuses the saved sweep artifacts or reruns the dropout sweep when explicitly requested.",
    "### Sweep Diagnostics\n\nThis cell visualizes the saved sweep histories, highlights the selected dropout setting, and saves the summary figure.",
    "### Persist Sweep Outputs\n\nThis cell refreshes the sweep summary files so the artifact directory reflects the current selected run and paths.",
    "### Load Best Checkpoint And Score Test Split\n\nThis cell loads the best checkpoint from the selected dropout run and computes anomaly scores on the test split.",
    "### Validation Threshold\n\nThis cell computes the deployment threshold from validation-normal scores.",
    "### Metrics\n\nThis cell applies the validation-derived threshold, computes evaluation metrics, and saves the score table and metric summary.",
    "### Threshold Sweep Plot\n\nThis cell compares precision, recall, and F1 across score thresholds, then saves both the table and the figure.",
    "### Score Distribution Plot\n\nThis cell visualizes the test-score distribution for normal and anomalous wafers and saves the histogram figure.",
    "### Reconstruction Examples\n\nThis cell shows a small set of input and reconstruction pairs and saves the figure.",
    "### Failure Tables\n\nThis cell builds the error-analysis table and saves the detailed failure-analysis CSVs for later reference.",
    "### Failure Examples\n\nThis cell visualizes representative false positives, false negatives, true positives, and true negatives and saves each figure.",
    "### Score Ablation Run\n\nThis cell runs the score-ablation helper only when its outputs are missing or rerun is explicitly requested.",
    "### Score Ablation Results\n\nThis cell loads the saved score-ablation outputs so they can be inspected without rerunning the script.",
    "### Score Ablation Plot\n\nThis cell visualizes the score-ablation comparison and saves the summary plot.",
]


TITLE = """# Autoencoder Training Notebook (BatchNorm + Dropout Sweep)

This notebook compares several BatchNorm + dropout variants on the curated x64 benchmark split. By default it reuses the saved sweep artifacts from its local artifact folder, selects the best saved dropout run, and only reruns the sweep when `FORCE_RERUN_DROPOUT_SWEEP = True`."""


SUBMISSION_CONTEXT = """## Submission Context

- Dataset notebook: `data/dataset/x64/benchmark_50k_5pct/notebook.ipynb`
- Dataset config: `data/dataset/x64/benchmark_50k_5pct/data_config.toml`
- Experiment config: `experiments/anomaly_detection/autoencoder/x64/batchnorm_dropout/train_config.toml`
- Artifact root: `experiments/anomaly_detection/autoencoder/x64/batchnorm_dropout/artifacts/autoencoder_batchnorm_dropout`
- Default behavior: load the saved sweep summary, selected checkpoint, and score-ablation outputs if they already exist; only rerun the sweep when explicitly requested."""


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_notebook(path: Path, notebook: dict) -> None:
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")


def replace_code_cells(template_nb: dict, new_sources: list[list[str]]) -> dict:
    code_cells = [cell for cell in template_nb["cells"] if cell["cell_type"] == "code"]
    if len(code_cells) != len(new_sources):
        raise RuntimeError(f"Expected {len(new_sources)} code cells, found {len(code_cells)}")
    for cell, source in zip(code_cells, new_sources):
        cell["source"] = source
    return template_nb


def build_training_cell() -> list[str]:
    return [
        "epochs = int(config[\"training\"][\"epochs\"])\n",
        "patience = int(config[\"training\"].get(\"early_stopping_patience\", 0))\n",
        "min_delta = float(config[\"training\"].get(\"early_stopping_min_delta\", 0.0))\n",
        "checkpoint_every = int(config[\"training\"].get(\"checkpoint_every\", 5))\n",
        "base_output_dir = REPO_ROOT / config[\"run\"][\"output_dir\"]\n",
        "base_output_dir.mkdir(parents=True, exist_ok=True)\n",
        "sweep_results_path = base_output_dir / \"dropout_sweep_results.csv\"\n",
        "sweep_summary_path = base_output_dir / \"dropout_sweep_summary.json\"\n",
        "training_ran = False\n",
        "resume_from = \"\"\n",
        "sweep_histories = {}\n",
        "history = []\n",
        "best_state_dict = None\n",
        "best_epoch = 0\n",
        "best_val_loss = float(\"inf\")\n",
        "stale_epochs = 0\n",
        "\n",
        "def resolve_run_output_dir(dropout_prob: float, stored_relative: str | None = None) -> Path:\n",
        "    expected = base_output_dir / dropout_slug(dropout_prob)\n",
        "    if expected.exists() or not stored_relative:\n",
        "        return expected\n",
        "    candidate = REPO_ROOT / Path(stored_relative)\n",
        "    if candidate.exists():\n",
        "        return candidate\n",
        "    return expected\n",
        "\n",
        "artifacts_ready = False\n",
        "if not FORCE_RERUN_DROPOUT_SWEEP and sweep_results_path.exists() and sweep_summary_path.exists():\n",
        "    sweep_results_df = pd.read_csv(sweep_results_path)\n",
        "    sweep_summary = json.loads(sweep_summary_path.read_text(encoding=\"utf-8\"))\n",
        "    best_dropout = float(sweep_summary.get(\"selected_dropout\", sweep_results_df.sort_values(\"best_val_loss\").iloc[0][\"dropout_prob\"]))\n",
        "    normalized_rows = []\n",
        "    for row in sweep_results_df.to_dict(orient=\"records\"):\n",
        "        dropout_prob = float(row[\"dropout_prob\"])\n",
        "        run_dir = resolve_run_output_dir(dropout_prob, row.get(\"output_dir\"))\n",
        "        row[\"dropout_prob\"] = dropout_prob\n",
        "        row[\"output_dir\"] = run_dir.relative_to(REPO_ROOT).as_posix()\n",
        "        normalized_rows.append(row)\n",
        "        history_candidate = run_dir / \"history.json\"\n",
        "        if history_candidate.exists():\n",
        "            sweep_histories[dropout_prob] = json.loads(history_candidate.read_text(encoding=\"utf-8\"))\n",
        "    sweep_results_df = pd.DataFrame(normalized_rows).sort_values([\"best_val_loss\", \"epochs_ran\"]).reset_index(drop=True)\n",
        "    output_dir = resolve_run_output_dir(best_dropout, sweep_summary.get(\"selected_output_dir\"))\n",
        "    history_path = output_dir / \"history.json\"\n",
        "    best_model_path = output_dir / \"best_model.pt\"\n",
        "    artifacts_ready = history_path.exists() and best_model_path.exists()\n",
        "    if artifacts_ready:\n",
        "        history = json.loads(history_path.read_text(encoding=\"utf-8\"))\n",
        "        config[\"model\"][\"dropout_prob\"] = best_dropout\n",
        "        config[\"run\"][\"output_dir\"] = output_dir.relative_to(REPO_ROOT).as_posix()\n",
        "        best_checkpoint = torch.load(best_model_path, map_location=device)\n",
        "        best_epoch = int(best_checkpoint.get(\"best_epoch\", best_checkpoint.get(\"epoch\", 0)))\n",
        "        best_val_loss = float(best_checkpoint.get(\"best_val_loss\", float(\"nan\")))\n",
        "        stale_epochs = int(best_checkpoint.get(\"stale_epochs\", 0))\n",
        "        model = build_autoencoder(best_dropout)\n",
        "        model.load_state_dict(best_checkpoint[\"model_state_dict\"])\n",
        "        optimizer = torch.optim.Adam(\n",
        "            model.parameters(),\n",
        "            lr=float(config[\"training\"][\"learning_rate\"]),\n",
        "            weight_decay=float(config[\"training\"][\"weight_decay\"]),\n",
        "        )\n",
        "        best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}\n",
        "        print(f\"Found existing sweep artifacts in {base_output_dir}. Skipping sweep. Set FORCE_RERUN_DROPOUT_SWEEP = True to recompute.\")\n",
        "    else:\n",
        "        print(\"Found sweep metadata, but the selected run artifacts are incomplete. Rerunning the sweep.\")\n",
        "\n",
        "if FORCE_RERUN_DROPOUT_SWEEP or not artifacts_ready:\n",
        "    print({\"epochs\": epochs, \"anomaly_score\": ANOMALY_SCORE_NAME, \"topk_ratio\": TOPK_RATIO, \"dropout_sweep\": DROPOUT_SWEEP})\n",
        "    sweep_rows = []\n",
        "    sweep_histories = {}\n",
        "    best_run = None\n",
        "    for dropout_prob in DROPOUT_SWEEP:\n",
        "        run_output_dir = base_output_dir / dropout_slug(dropout_prob)\n",
        "        run_output_dir.mkdir(parents=True, exist_ok=True)\n",
        "        run_config = json.loads(json.dumps(config))\n",
        "        run_config[\"model\"][\"dropout_prob\"] = float(dropout_prob)\n",
        "        run_config[\"run\"][\"output_dir\"] = run_output_dir.relative_to(REPO_ROOT).as_posix()\n",
        "        model = build_autoencoder(dropout_prob)\n",
        "        optimizer = torch.optim.Adam(\n",
        "            model.parameters(),\n",
        "            lr=float(run_config[\"training\"][\"learning_rate\"]),\n",
        "            weight_decay=float(run_config[\"training\"][\"weight_decay\"]),\n",
        "        )\n",
        "        history = []\n",
        "        run_best_val_loss = float(\"inf\")\n",
        "        run_best_epoch = 0\n",
        "        run_best_state_dict = None\n",
        "        run_stale_epochs = 0\n",
        "        print(f\"\\n=== Dropout {dropout_prob:.2f} | output={run_output_dir} ===\")\n",
        "        for epoch in range(epochs):\n",
        "            train_metrics = run_autoencoder_epoch(model, train_loader, device, optimizer)\n",
        "            val_metrics = run_autoencoder_epoch(model, val_loader, device)\n",
        "            record = {\"epoch\": epoch + 1, \"train_loss\": train_metrics.loss, \"val_loss\": val_metrics.loss}\n",
        "            history.append(record)\n",
        "            print({\"dropout\": dropout_prob, **record})\n",
        "            improved = (run_best_val_loss - val_metrics.loss) > min_delta\n",
        "            if improved:\n",
        "                run_best_val_loss = val_metrics.loss\n",
        "                run_best_epoch = epoch + 1\n",
        "                run_best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}\n",
        "                run_stale_epochs = 0\n",
        "                torch.save(\n",
        "                    {\n",
        "                        \"epoch\": epoch + 1,\n",
        "                        \"model_state_dict\": run_best_state_dict,\n",
        "                        \"optimizer_state_dict\": optimizer.state_dict(),\n",
        "                        \"config\": run_config,\n",
        "                        \"best_epoch\": run_best_epoch,\n",
        "                        \"best_val_loss\": run_best_val_loss,\n",
        "                        \"stale_epochs\": run_stale_epochs,\n",
        "                        \"history\": history,\n",
        "                    },\n",
        "                    run_output_dir / \"best_model.pt\",\n",
        "                )\n",
        "            else:\n",
        "                run_stale_epochs += 1\n",
        "            latest_checkpoint = {\n",
        "                \"epoch\": epoch + 1,\n",
        "                \"model_state_dict\": model.state_dict(),\n",
        "                \"optimizer_state_dict\": optimizer.state_dict(),\n",
        "                \"config\": run_config,\n",
        "                \"best_epoch\": run_best_epoch,\n",
        "                \"best_val_loss\": run_best_val_loss,\n",
        "                \"stale_epochs\": run_stale_epochs,\n",
        "                \"history\": history,\n",
        "            }\n",
        "            torch.save(latest_checkpoint, run_output_dir / \"latest_checkpoint.pt\")\n",
        "            if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:\n",
        "                torch.save(latest_checkpoint, run_output_dir / f\"checkpoint_epoch_{epoch + 1}.pt\")\n",
        "            if patience > 0 and run_stale_epochs >= patience:\n",
        "                print(f\"Early stopping at epoch {epoch + 1}. Best epoch: {run_best_epoch}, best val loss: {run_best_val_loss:.6f}\")\n",
        "                break\n",
        "        if run_best_state_dict is None:\n",
        "            run_best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}\n",
        "        torch.save(\n",
        "            {\n",
        "                \"epoch\": len(history),\n",
        "                \"model_state_dict\": model.state_dict(),\n",
        "                \"optimizer_state_dict\": optimizer.state_dict(),\n",
        "                \"config\": run_config,\n",
        "                \"best_epoch\": run_best_epoch,\n",
        "                \"best_val_loss\": run_best_val_loss,\n",
        "                \"stale_epochs\": run_stale_epochs,\n",
        "                \"history\": history,\n",
        "            },\n",
        "            run_output_dir / \"last_model.pt\",\n",
        "        )\n",
        "        (run_output_dir / \"history.json\").write_text(json.dumps(history, indent=2), encoding=\"utf-8\")\n",
        "        run_summary = {\n",
        "            \"dropout_prob\": float(dropout_prob),\n",
        "            \"best_epoch\": run_best_epoch,\n",
        "            \"best_val_loss\": run_best_val_loss,\n",
        "            \"epochs_ran\": len(history),\n",
        "            \"output_dir\": run_output_dir.relative_to(REPO_ROOT).as_posix(),\n",
        "        }\n",
        "        (run_output_dir / \"summary.json\").write_text(json.dumps(run_summary, indent=2), encoding=\"utf-8\")\n",
        "        sweep_histories[float(dropout_prob)] = history\n",
        "        sweep_rows.append(run_summary)\n",
        "        if best_run is None or run_best_val_loss < best_run[\"best_val_loss\"]:\n",
        "            best_run = {\n",
        "                \"dropout_prob\": float(dropout_prob),\n",
        "                \"best_epoch\": run_best_epoch,\n",
        "                \"best_val_loss\": run_best_val_loss,\n",
        "                \"history\": history,\n",
        "                \"output_dir\": run_output_dir,\n",
        "                \"config\": run_config,\n",
        "            }\n",
        "    sweep_results_df = pd.DataFrame(sweep_rows).sort_values([\"best_val_loss\", \"epochs_ran\"]).reset_index(drop=True)\n",
        "    best_dropout = float(best_run[\"dropout_prob\"])\n",
        "    config = best_run[\"config\"]\n",
        "    output_dir = best_run[\"output_dir\"]\n",
        "    history = best_run[\"history\"]\n",
        "    history_path = output_dir / \"history.json\"\n",
        "    best_epoch = int(best_run[\"best_epoch\"])\n",
        "    best_val_loss = float(best_run[\"best_val_loss\"])\n",
        "    best_model_path = output_dir / \"best_model.pt\"\n",
        "    model = build_autoencoder(best_dropout)\n",
        "    best_checkpoint = torch.load(best_model_path, map_location=device)\n",
        "    model.load_state_dict(best_checkpoint[\"model_state_dict\"])\n",
        "    optimizer = torch.optim.Adam(\n",
        "        model.parameters(),\n",
        "        lr=float(config[\"training\"][\"learning_rate\"]),\n",
        "        weight_decay=float(config[\"training\"][\"weight_decay\"]),\n",
        "    )\n",
        "    best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}\n",
        "    training_ran = True\n",
        "    print(f\"\\nSelected dropout={best_dropout:.2f} from sweep\")\n",
        "\n",
        "display(sweep_results_df)\n",
    ]


def build_sweep_diagnostics_cell() -> list[str]:
    return [
        "history_df = pd.DataFrame(history)\n",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 4))\n",
        "for dropout_prob, run_history in sweep_histories.items():\n",
        "    run_history_df = pd.DataFrame(run_history)\n",
        "    if not run_history_df.empty:\n",
        "        axes[0].plot(run_history_df[\"epoch\"], run_history_df[\"val_loss\"], marker=\"o\", label=f\"dropout={dropout_prob:.2f}\")\n",
        "axes[0].set_title(\"Validation Loss Across Dropout Sweep\")\n",
        "axes[0].set_xlabel(\"Epoch\")\n",
        "axes[0].set_ylabel(\"Val Loss\")\n",
        "axes[0].grid(True, alpha=0.3)\n",
        "axes[0].legend()\n",
        "axes[1].bar(sweep_results_df[\"dropout_prob\"].astype(str), sweep_results_df[\"best_val_loss\"])\n",
        "axes[1].set_title(\"Best Validation Loss by Dropout\")\n",
        "axes[1].set_xlabel(\"Dropout\")\n",
        "axes[1].set_ylabel(\"Best Val Loss\")\n",
        "plt.tight_layout()\n",
        "save_figure(fig, base_output_dir / \"dropout_sweep_summary.png\")\n",
        "plt.show()\n",
        "print(f\"Selected dropout for downstream evaluation: {best_dropout:.2f}\")\n",
        "history_df.tail()\n",
    ]


def build_persist_cell() -> list[str]:
    return [
        "normalized_rows = []\n",
        "for row in sweep_results_df.to_dict(orient=\"records\"):\n",
        "    dropout_prob = float(row[\"dropout_prob\"])\n",
        "    run_dir = base_output_dir / dropout_slug(dropout_prob)\n",
        "    row[\"dropout_prob\"] = dropout_prob\n",
        "    row[\"output_dir\"] = run_dir.relative_to(REPO_ROOT).as_posix()\n",
        "    normalized_rows.append(row)\n",
        "sweep_results_df = pd.DataFrame(normalized_rows).sort_values([\"best_val_loss\", \"epochs_ran\"]).reset_index(drop=True)\n",
        "sweep_summary = {\n",
        "    \"selected_dropout\": best_dropout,\n",
        "    \"selected_output_dir\": output_dir.relative_to(REPO_ROOT).as_posix(),\n",
        "    \"selection_metric\": \"best_val_loss\",\n",
        "    \"runs\": sweep_results_df.to_dict(orient=\"records\"),\n",
        "}\n",
        "sweep_results_df.to_csv(base_output_dir / \"dropout_sweep_results.csv\", index=False)\n",
        "with (base_output_dir / \"dropout_sweep_summary.json\").open(\"w\", encoding=\"utf-8\") as handle:\n",
        "    json.dump(sweep_summary, handle, indent=2)\n",
        "history_df.to_csv(output_dir / \"history.csv\", index=False)\n",
        "if training_ran:\n",
        "    print(f\"Saved sweep outputs to {base_output_dir}\")\n",
        "else:\n",
        "    print(f\"Confirmed existing sweep artifacts and refreshed metadata paths in {base_output_dir}\")\n",
        "print(f\"Selected run directory: {output_dir}\")\n",
        "sweep_summary\n",
    ]


def build_load_checkpoint_cell() -> list[str]:
    return [
        "best_model_path = output_dir / \"best_model.pt\"\n",
        "if not best_model_path.exists():\n",
        "    raise FileNotFoundError(f\"Best checkpoint not found at {best_model_path}\")\n",
        "model = build_autoencoder(best_dropout)\n",
        "best_checkpoint = torch.load(best_model_path, map_location=device)\n",
        "model.load_state_dict(best_checkpoint[\"model_state_dict\"])\n",
        "print(f\"Loaded best_model.pt from epoch {best_checkpoint.get('best_epoch', 'unknown')} for dropout={best_dropout:.2f}\")\n",
        "model.eval()\n",
        "\n",
        "def reconstruction_error(inputs: torch.Tensor, outputs: torch.Tensor, score_name: str = ANOMALY_SCORE_NAME) -> torch.Tensor:\n",
        "    if score_name == \"mse_mean\":\n",
        "        return spatial_mean(squared_error_map(inputs, outputs))\n",
        "    if score_name == \"topk_abs_mean\":\n",
        "        return topk_spatial_mean(absolute_error_map(inputs, outputs), topk_ratio=TOPK_RATIO)\n",
        "    raise ValueError(f\"Unsupported score_name: {score_name}\")\n",
        "\n",
        "test_scores = []\n",
        "with torch.no_grad():\n",
        "    for inputs, labels in test_loader:\n",
        "        inputs = inputs.to(device)\n",
        "        outputs = model(inputs)\n",
        "        scores = reconstruction_error(inputs, outputs, score_name=ANOMALY_SCORE_NAME).cpu().numpy()\n",
        "        labels = labels.cpu().numpy()\n",
        "        for score, label in zip(scores, labels):\n",
        "            test_scores.append({\"score\": float(score), \"is_anomaly\": int(label)})\n",
        "score_df = pd.DataFrame(test_scores)\n",
        "print({\"selected_dropout\": best_dropout, \"evaluation_score\": ANOMALY_SCORE_NAME, \"topk_ratio\": TOPK_RATIO if ANOMALY_SCORE_NAME == \"topk_abs_mean\" else None})\n",
        "score_df.head()\n",
    ]


def build_code_sources(template_nb: dict) -> list[list[str]]:
    template_code_cells = [cell["source"] for cell in template_nb["cells"] if cell["cell_type"] == "code"]
    if len(template_code_cells) != 20:
        raise RuntimeError(f"Expected 20 code cells in template, found {len(template_code_cells)}")

    template_code_cells[1] = [
        "CONFIG_PATH = REPO_ROOT / \"experiments/anomaly_detection/autoencoder/x64/batchnorm_dropout/train_config.toml\"\n",
        "EPOCHS_OVERRIDE = None\n",
        "FORCE_RERUN_DROPOUT_SWEEP = False\n",
        "FORCE_SCORE_ABLATION_RERUN = False\n",
        "ANOMALY_SCORE_NAME = \"topk_abs_mean\"\n",
        "TOPK_RATIO = 0.01\n",
        "DROPOUT_SWEEP = [0.0, 0.05, 0.1, 0.2]\n",
        "config = load_toml(CONFIG_PATH)\n",
        "if EPOCHS_OVERRIDE is not None:\n",
        "    config[\"training\"][\"epochs\"] = int(EPOCHS_OVERRIDE)\n",
        "config\n",
    ]
    template_code_cells[5] = [
        "def dropout_slug(dropout_prob: float) -> str:\n",
        "    return f\"dropout_{dropout_prob:.2f}\".replace(\".\", \"p\")\n",
        "\n",
        "def build_autoencoder(dropout_prob: float) -> ConvAutoencoder:\n",
        "    return ConvAutoencoder(\n",
        "        latent_dim=int(config[\"model\"][\"latent_dim\"]),\n",
        "        image_size=image_size,\n",
        "        use_batchnorm=bool(config[\"model\"].get(\"use_batchnorm\", False)),\n",
        "        dropout_prob=float(dropout_prob),\n",
        "    ).to(device)\n",
        "\n",
        "print({\"dropout_sweep\": DROPOUT_SWEEP, \"artifact_root\": str(REPO_ROOT / config[\"run\"][\"output_dir\"])})\n",
        "build_autoencoder(DROPOUT_SWEEP[0])\n",
    ]
    template_code_cells[6] = build_training_cell()
    template_code_cells[7] = build_sweep_diagnostics_cell()
    template_code_cells[8] = build_persist_cell()
    template_code_cells[9] = build_load_checkpoint_cell()
    return template_code_cells


def rebuild_cells(template_nb: dict) -> list[dict]:
    code_cells = [cell for cell in template_nb["cells"] if cell["cell_type"] == "code"]
    new_cells = [md_cell(TITLE), md_cell(SUBMISSION_CONTEXT)]
    for description, code_cell in zip(DESCRIPTIONS, code_cells):
        new_cells.append(md_cell(description))
        new_cells.append(code_cell)
    return new_cells


def main() -> None:
    template_path = REPO_ROOT / "experiments/anomaly_detection/autoencoder/x64/batchnorm/notebook.ipynb"
    target_path = REPO_ROOT / "experiments/anomaly_detection/autoencoder/x64/batchnorm_dropout/notebook.ipynb"
    template_nb = load_notebook(template_path)
    replace_code_cells(template_nb, build_code_sources(template_nb))
    template_nb["cells"] = rebuild_cells(template_nb)
    write_notebook(target_path, template_nb)
    print(f"Updated {target_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
