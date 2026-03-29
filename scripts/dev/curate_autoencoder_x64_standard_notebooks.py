from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def md_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.split("\n")]}


DESCRIPTIONS = [
    "### Imports\n\nThis cell loads the libraries, repo-local modules, and path helpers used by the notebook.",
    "### Run Controls\n\nThis cell defines the experiment config path and the main rerun flags. Leave `FORCE_RETRAIN = False` to reuse saved artifacts when they already exist.",
    "### Reproducibility And Helpers\n\nThis cell sets the random seed, resolves the execution device, and defines a helper for saving figures.",
    "### Metadata Check\n\nThis cell loads the configured metadata CSV so we can verify the split before building loaders.",
    "### Data Loaders\n\nThis cell builds the train, validation, and test loaders used throughout the notebook.",
    "### Model Setup\n\nThis cell constructs the model and optimizer that will be used either for training or for loading an existing checkpoint.",
    "### Training Or Artifact Reuse\n\nThis cell either trains the model or reuses the existing checkpoint and history files when they are already present.",
    "### Training Curve\n\nThis cell displays the saved training history and exports the training-curve figure to the artifact folder.",
    "### Persist Training Outputs\n\nThis cell writes training outputs only when a fresh training run was executed. If artifacts were reused, it reports that nothing was overwritten.",
    "### Load Best Checkpoint And Score Test Split\n\nThis cell loads the best checkpoint and computes anomaly scores on the test split.",
    "### Validation Threshold\n\nThis cell computes the deployment threshold from validation-normal scores.",
    "### Metrics\n\nThis cell applies the validation-derived threshold, computes evaluation metrics, and saves the score table and metric summary.",
    "### Threshold Sweep Plot\n\nThis cell compares precision, recall, and F1 across score thresholds, then saves both the table and the figure.",
    "### Score Distribution Plot\n\nThis cell visualizes the test-score distribution for normal and anomalous wafers and saves the histogram figure.",
    "### Reconstruction Examples\n\nThis cell shows a small set of input and reconstruction pairs and saves the figure.",
    "### Failure Tables\n\nThis cell builds the error-analysis table and saves the detailed failure-analysis CSV for later reference.",
    "### Failure Examples\n\nThis cell visualizes representative false positives, false negatives, true positives, and true negatives and saves each figure.",
    "### Score Ablation Run\n\nThis cell runs the score-ablation helper only when its outputs are missing or rerun is explicitly requested.",
    "### Score Ablation Results\n\nThis cell loads the saved score-ablation outputs so they can be inspected without rerunning the script.",
    "### Score Ablation Plot\n\nThis cell visualizes the score-ablation comparison and saves the summary plot.",
]


TITLES = {
    "baseline": "# Autoencoder Training Notebook\n\nThis notebook runs the baseline convolutional autoencoder on the curated x64 benchmark split. By default it reuses saved artifacts from its local artifact folder and only retrains when `FORCE_RETRAIN = True`.",
    "batchnorm": "# Autoencoder Training Notebook (BatchNorm Variant)\n\nThis notebook runs the BatchNorm autoencoder variant on the curated x64 benchmark split. By default it reuses saved artifacts from its local artifact folder and only retrains when `FORCE_RETRAIN = True`.",
    "residual": "# Autoencoder Training Notebook (Residual Backbone Variant)\n\nThis notebook runs the residual autoencoder variant on the curated x64 benchmark split. By default it reuses saved artifacts from its local artifact folder and only retrains when `FORCE_RETRAIN = True`.",
}


SUBMISSION_CONTEXTS = {
    "baseline": """## Submission Context

- Dataset notebook: `data/dataset/x64/benchmark_50k_5pct/notebook.ipynb`
- Dataset config: `data/dataset/x64/benchmark_50k_5pct/data_config.toml`
- Experiment config: `experiments/anomaly_detection/autoencoder/x64/baseline/train_config.toml`
- Artifact root: `experiments/anomaly_detection/autoencoder/x64/baseline/artifacts/autoencoder_baseline`
- Default behavior: load the saved checkpoint, history, and score-ablation outputs if they already exist; only retrain when explicitly requested.""",
    "batchnorm": """## Submission Context

- Dataset notebook: `data/dataset/x64/benchmark_50k_5pct/notebook.ipynb`
- Dataset config: `data/dataset/x64/benchmark_50k_5pct/data_config.toml`
- Experiment config: `experiments/anomaly_detection/autoencoder/x64/batchnorm/train_config.toml`
- Artifact root: `experiments/anomaly_detection/autoencoder/x64/batchnorm/artifacts/autoencoder_batchnorm`
- Default behavior: load the saved checkpoint, history, and score-ablation outputs if they already exist; only retrain when explicitly requested.""",
    "residual": """## Submission Context

- Dataset notebook: `data/dataset/x64/benchmark_50k_5pct/notebook.ipynb`
- Dataset config: `data/dataset/x64/benchmark_50k_5pct/data_config.toml`
- Experiment config: `experiments/anomaly_detection/autoencoder/x64/residual/train_config.toml`
- Artifact root: `experiments/anomaly_detection/autoencoder/x64/residual/artifacts/autoencoder_residual`
- Default behavior: load the saved checkpoint, history, and score-ablation outputs if they already exist; only retrain when explicitly requested.""",
}


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_notebook(path: Path, notebook: dict) -> None:
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")


def ensure_imports(source: list[str]) -> list[str]:
    text = "".join(source)
    if "from IPython.display import display\n" not in text:
        text = text.replace("import torch\n", "import torch\nfrom IPython.display import display\n")
    if "from wafer_defect.models.autoencoder import ConvAutoencoder, build_autoencoder_from_config\n" not in text:
        text = text.replace(
            "from wafer_defect.models.autoencoder import ConvAutoencoder\n",
            "from wafer_defect.models.autoencoder import ConvAutoencoder, build_autoencoder_from_config\n",
        )
    return text.splitlines(keepends=True)


def training_cell_source() -> list[str]:
    return [
        "history = []\n",
        "epochs = int(config[\"training\"][\"epochs\"])\n",
        "patience = int(config[\"training\"].get(\"early_stopping_patience\", 0))\n",
        "min_delta = float(config[\"training\"].get(\"early_stopping_min_delta\", 0.0))\n",
        "checkpoint_every = int(config[\"training\"].get(\"checkpoint_every\", 5))\n",
        "resume_from = str(config[\"training\"].get(\"resume_from\", \"\")).strip()\n",
        "best_val_loss = float(\"inf\")\n",
        "best_epoch = 0\n",
        "best_state_dict = None\n",
        "stale_epochs = 0\n",
        "start_epoch = 0\n",
        "training_ran = False\n",
        "output_dir = REPO_ROOT / config[\"run\"][\"output_dir\"]\n",
        "output_dir.mkdir(parents=True, exist_ok=True)\n",
        "history_path = output_dir / \"history.json\"\n",
        "best_model_path = output_dir / \"best_model.pt\"\n",
        "\n",
        "artifacts_ready = best_model_path.exists() and history_path.exists()\n",
        "if not FORCE_RETRAIN and artifacts_ready:\n",
        "    with history_path.open(\"r\", encoding=\"utf-8\") as handle:\n",
        "        history = json.load(handle)\n",
        "    best_checkpoint = torch.load(best_model_path, map_location=device)\n",
        "    model.load_state_dict(best_checkpoint[\"model_state_dict\"])\n",
        "    best_epoch = int(best_checkpoint.get(\"best_epoch\", best_checkpoint.get(\"epoch\", 0)))\n",
        "    best_val_loss = float(best_checkpoint.get(\"best_val_loss\", float(\"nan\")))\n",
        "    stale_epochs = int(best_checkpoint.get(\"stale_epochs\", 0))\n",
        "    best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}\n",
        "    print(f\"Found existing artifacts in {output_dir}. Skipping training. Set FORCE_RETRAIN = True to retrain.\")\n",
        "else:\n",
        "    if resume_from:\n",
        "        resume_path = Path(resume_from)\n",
        "        if not resume_path.is_absolute():\n",
        "            resume_path = REPO_ROOT / resume_path\n",
        "        checkpoint = torch.load(resume_path, map_location=device)\n",
        "        model.load_state_dict(checkpoint[\"model_state_dict\"])\n",
        "        optimizer.load_state_dict(checkpoint[\"optimizer_state_dict\"])\n",
        "        start_epoch = int(checkpoint.get(\"epoch\", 0))\n",
        "        best_val_loss = float(checkpoint.get(\"best_val_loss\", best_val_loss))\n",
        "        best_epoch = int(checkpoint.get(\"best_epoch\", best_epoch))\n",
        "        stale_epochs = int(checkpoint.get(\"stale_epochs\", stale_epochs))\n",
        "        history = checkpoint.get(\"history\", [])\n",
        "        best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}\n",
        "        print(f\"Resumed from {resume_path} at epoch {start_epoch}\")\n",
        "\n",
        "    print({\"epochs\": epochs, \"anomaly_score\": ANOMALY_SCORE_NAME, \"topk_ratio\": TOPK_RATIO})\n",
        "\n",
        "    for epoch in range(start_epoch, epochs):\n",
        "        train_metrics = run_autoencoder_epoch(model, train_loader, device, optimizer)\n",
        "        val_metrics = run_autoencoder_epoch(model, val_loader, device)\n",
        "        record = {\"epoch\": epoch + 1, \"train_loss\": train_metrics.loss, \"val_loss\": val_metrics.loss}\n",
        "        history.append(record)\n",
        "        print(record)\n",
        "\n",
        "        improved = (best_val_loss - val_metrics.loss) > min_delta\n",
        "        if improved:\n",
        "            best_val_loss = val_metrics.loss\n",
        "            best_epoch = epoch + 1\n",
        "            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}\n",
        "            stale_epochs = 0\n",
        "            torch.save({\"epoch\": epoch + 1, \"model_state_dict\": best_state_dict, \"optimizer_state_dict\": optimizer.state_dict(), \"config\": config, \"best_epoch\": best_epoch, \"best_val_loss\": best_val_loss, \"stale_epochs\": stale_epochs, \"history\": history}, output_dir / \"best_model.pt\")\n",
        "        else:\n",
        "            stale_epochs += 1\n",
        "\n",
        "        latest_checkpoint = {\"epoch\": epoch + 1, \"model_state_dict\": model.state_dict(), \"optimizer_state_dict\": optimizer.state_dict(), \"config\": config, \"best_epoch\": best_epoch, \"best_val_loss\": best_val_loss, \"stale_epochs\": stale_epochs, \"history\": history}\n",
        "        torch.save(latest_checkpoint, output_dir / \"latest_checkpoint.pt\")\n",
        "        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:\n",
        "            torch.save(latest_checkpoint, output_dir / f\"checkpoint_epoch_{epoch + 1}.pt\")\n",
        "        if patience > 0 and stale_epochs >= patience:\n",
        "            print(f\"Early stopping at epoch {epoch + 1}. Best epoch: {best_epoch}, best val loss: {best_val_loss:.6f}\")\n",
        "            break\n",
        "\n",
        "    if best_state_dict is None:\n",
        "        best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}\n",
        "    training_ran = True\n",
    ]


def training_curve_cell_source() -> list[str]:
    return [
        "if not history:\n",
        "    with history_path.open(\"r\", encoding=\"utf-8\") as handle:\n",
        "        history = json.load(handle)\n",
        "history_df = pd.DataFrame(history)\n",
        "fig, ax = plt.subplots(figsize=(8, 4))\n",
        "ax.plot(history_df[\"epoch\"], history_df[\"train_loss\"], marker=\"o\", label=\"train\")\n",
        "ax.plot(history_df[\"epoch\"], history_df[\"val_loss\"], marker=\"o\", label=\"val\")\n",
        "ax.set_title(\"Autoencoder Training Curve\")\n",
        "ax.set_xlabel(\"Epoch\")\n",
        "ax.set_ylabel(\"MSE Loss\")\n",
        "ax.grid(True, alpha=0.3)\n",
        "ax.legend()\n",
        "plt.tight_layout()\n",
        "save_figure(fig, output_dir / \"training_curve.png\")\n",
        "plt.show()\n",
        "history_df.tail()\n",
    ]


def persist_outputs_cell_source() -> list[str]:
    return [
        "if training_ran:\n",
        "    torch.save({\"epoch\": len(history), \"model_state_dict\": model.state_dict(), \"optimizer_state_dict\": optimizer.state_dict(), \"config\": config, \"best_epoch\": best_epoch, \"best_val_loss\": best_val_loss, \"stale_epochs\": stale_epochs, \"history\": history}, output_dir / \"last_model.pt\")\n",
        "    if best_state_dict is not None:\n",
        "        torch.save({\"epoch\": best_epoch, \"model_state_dict\": best_state_dict, \"optimizer_state_dict\": optimizer.state_dict(), \"config\": config, \"best_epoch\": best_epoch, \"best_val_loss\": best_val_loss, \"stale_epochs\": stale_epochs, \"history\": history}, output_dir / \"best_model.pt\")\n",
        "    with history_path.open(\"w\", encoding=\"utf-8\") as handle:\n",
        "        json.dump(history, handle, indent=2)\n",
        "    summary = {\"best_epoch\": best_epoch, \"best_val_loss\": best_val_loss, \"epochs_ran\": len(history), \"resumed_from\": resume_from, \"training_ran\": True}\n",
        "    with (output_dir / \"summary.json\").open(\"w\", encoding=\"utf-8\") as handle:\n",
        "        json.dump(summary, handle, indent=2)\n",
        "    print(f\"Saved outputs to {output_dir}\")\n",
        "else:\n",
        "    print(\"Reused existing training artifacts; no training files were rewritten.\")\n",
        "    summary_path = output_dir / \"summary.json\"\n",
        "    summary = json.loads(summary_path.read_text(encoding=\"utf-8\")) if summary_path.exists() else {\"best_epoch\": best_epoch, \"best_val_loss\": best_val_loss, \"epochs_ran\": len(history), \"resumed_from\": resume_from, \"training_ran\": False}\n",
        "summary\n",
    ]


def score_ablation_run_cell_source() -> list[str]:
    return [
        "import os\n",
        "import subprocess\n",
        "\n",
        "score_ablation_config = config if \"config\" in globals() else load_toml(CONFIG_PATH)\n",
        "score_ablation_output_root = REPO_ROOT / score_ablation_config[\"run\"][\"output_dir\"]\n",
        "score_ablation_best_model_path = score_ablation_output_root / \"best_model.pt\"\n",
        "if not score_ablation_best_model_path.exists():\n",
        "    raise FileNotFoundError(f\"Best autoencoder checkpoint not found: {score_ablation_best_model_path}\")\n",
        "score_ablation_output_dir = score_ablation_output_root / \"score_ablation\"\n",
        "score_ablation_output_dir.mkdir(parents=True, exist_ok=True)\n",
        "score_ablation_csv_path = score_ablation_output_dir / \"score_summary.csv\"\n",
        "score_ablation_json_path = score_ablation_output_dir / \"score_summary.json\"\n",
        "score_ablation_cmd = [sys.executable, \"scripts/evaluate_autoencoder_scores.py\", \"--checkpoint\", str(score_ablation_best_model_path.relative_to(REPO_ROOT)), \"--config\", str(CONFIG_PATH.relative_to(REPO_ROOT)), \"--output-dir\", str(score_ablation_output_dir.relative_to(REPO_ROOT))]\n",
        "score_ablation_env = os.environ.copy()\n",
        "src_path = str(REPO_ROOT / \"src\")\n",
        "score_ablation_env[\"PYTHONPATH\"] = src_path if not score_ablation_env.get(\"PYTHONPATH\") else src_path + os.pathsep + score_ablation_env[\"PYTHONPATH\"]\n",
        "if FORCE_SCORE_ABLATION_RERUN or not (score_ablation_csv_path.exists() and score_ablation_json_path.exists()):\n",
        "    print(\"Running:\")\n",
        "    print(\" \".join(score_ablation_cmd))\n",
        "    subprocess.run(score_ablation_cmd, cwd=REPO_ROOT, env=score_ablation_env, check=True)\n",
        "else:\n",
        "    print(f\"Found existing score ablation outputs in {score_ablation_output_dir}. Skipping rerun. Set FORCE_SCORE_ABLATION_RERUN = True to recompute.\")\n",
    ]


def load_checkpoint_cell_source() -> list[str]:
    return [
        "best_model_path = output_dir / \"best_model.pt\"\n",
        "if not best_model_path.exists():\n",
        "    raise FileNotFoundError(f\"Best autoencoder checkpoint not found: {best_model_path}\")\n",
        "best_checkpoint = torch.load(best_model_path, map_location=device)\n",
        "model.load_state_dict(best_checkpoint[\"model_state_dict\"])\n",
        "print(f\"Loaded best_model.pt from epoch {best_checkpoint.get('best_epoch', 'unknown')}\")\n",
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
        "print({\"evaluation_score\": ANOMALY_SCORE_NAME, \"topk_ratio\": TOPK_RATIO if ANOMALY_SCORE_NAME == \"topk_abs_mean\" else None})\n",
        "score_df.head()\n",
    ]


def model_setup_cell_source(variant: str) -> list[str]:
    if variant == "residual":
        return [
            "model = build_autoencoder_from_config(config, image_size=image_size).to(device)\n",
            "optimizer = torch.optim.Adam(\n",
            "    model.parameters(),\n",
            "    lr=float(config[\"training\"][\"learning_rate\"]),\n",
            "    weight_decay=float(config[\"training\"][\"weight_decay\"]),\n",
            ")\n",
            "\n",
            "model\n",
        ]
    return [
        "model = ConvAutoencoder(\n",
        "    latent_dim=int(config[\"model\"][\"latent_dim\"]),\n",
        "    image_size=image_size,\n",
        "    use_batchnorm=bool(config[\"model\"].get(\"use_batchnorm\", False)),\n",
        "    dropout_prob=float(config[\"model\"].get(\"dropout_prob\", 0.0)),\n",
        ").to(device)\n",
        "optimizer = torch.optim.Adam(\n",
        "    model.parameters(),\n",
        "    lr=float(config[\"training\"][\"learning_rate\"]),\n",
        "    weight_decay=float(config[\"training\"][\"weight_decay\"]),\n",
        ")\n",
        "\n",
        "model\n",
    ]


def build_code_sources(template_nb: dict, variant: str) -> list[list[str]]:
    template_code_cells = [cell["source"] for cell in template_nb["cells"] if cell["cell_type"] == "code"]
    if len(template_code_cells) != 20:
        raise RuntimeError(f"Expected 20 code cells in template, found {len(template_code_cells)}")

    code_sources = [list(source) for source in template_code_cells]
    code_sources[0] = ensure_imports(code_sources[0])
    code_sources[1] = [
        f"CONFIG_PATH = REPO_ROOT / \"experiments/anomaly_detection/autoencoder/x64/{variant}/train_config.toml\"\n",
        "EPOCHS_OVERRIDE = None\n",
        "FORCE_RETRAIN = False\n",
        "FORCE_SCORE_ABLATION_RERUN = False\n",
        "ANOMALY_SCORE_NAME = \"topk_abs_mean\"\n",
        "TOPK_RATIO = 0.01\n",
        "config = load_toml(CONFIG_PATH)\n",
        "if EPOCHS_OVERRIDE is not None:\n",
        "    config[\"training\"][\"epochs\"] = int(EPOCHS_OVERRIDE)\n",
        "config\n",
    ]
    code_sources[5] = model_setup_cell_source(variant)
    code_sources[6] = training_cell_source()
    code_sources[7] = training_curve_cell_source()
    code_sources[8] = persist_outputs_cell_source()
    code_sources[9] = load_checkpoint_cell_source()

    code_sources[11] = code_sources[11] + [
        "\n",
        "score_df.to_csv(output_dir / \"test_scores.csv\", index=False)\n",
        "metrics_df.to_csv(output_dir / \"metrics.csv\", index=False)\n",
    ]
    code_sources[12] = [
        "precision_curve, recall_curve, pr_thresholds = precision_recall_curve(score_df[\"is_anomaly\"], score_df[\"score\"])\n",
        "threshold_sweep_df = pd.DataFrame({\"threshold\": pr_thresholds, \"precision\": precision_curve[:-1], \"recall\": recall_curve[:-1]})\n",
        "threshold_sweep_df[\"f1\"] = 2 * threshold_sweep_df[\"precision\"] * threshold_sweep_df[\"recall\"] / (threshold_sweep_df[\"precision\"] + threshold_sweep_df[\"recall\"] + 1e-12)\n",
        "threshold_sweep_df[\"predicted_anomalies\"] = [int((score_df[\"score\"] > t).sum()) for t in threshold_sweep_df[\"threshold\"]]\n",
        "best_f1_row = threshold_sweep_df.loc[threshold_sweep_df[\"f1\"].idxmax()]\n",
        "threshold_sweep_df.to_csv(output_dir / \"threshold_sweep.csv\", index=False)\n",
        "display(threshold_sweep_df.sort_values(\"f1\", ascending=False).head(10))\n",
        "print(f\"Best F1 threshold: {best_f1_row['threshold']:.6f} | precision={best_f1_row['precision']:.4f}, recall={best_f1_row['recall']:.4f}, f1={best_f1_row['f1']:.4f}\")\n",
        "fig, ax = plt.subplots(figsize=(8, 4))\n",
        "ax.plot(threshold_sweep_df[\"threshold\"], threshold_sweep_df[\"precision\"], label=\"precision\")\n",
        "ax.plot(threshold_sweep_df[\"threshold\"], threshold_sweep_df[\"recall\"], label=\"recall\")\n",
        "ax.plot(threshold_sweep_df[\"threshold\"], threshold_sweep_df[\"f1\"], label=\"f1\")\n",
        "ax.axvline(threshold, color=\"red\", linestyle=\"--\", label=f\"validation threshold = {threshold:.4f}\")\n",
        "ax.axvline(best_f1_row[\"threshold\"], color=\"green\", linestyle=\":\", label=f\"best f1 threshold = {best_f1_row['threshold']:.4f}\")\n",
        "ax.set_xlabel(\"Anomaly-score threshold\")\n",
        "ax.set_ylabel(\"Metric value\")\n",
        "ax.set_title(f\"Threshold Sweep on Test Split ({ANOMALY_SCORE_NAME})\")\n",
        "ax.legend()\n",
        "save_figure(fig, output_dir / \"threshold_sweep.png\")\n",
        "plt.show()\n",
    ]
    code_sources[13] = [
        "fig, ax = plt.subplots(figsize=(8, 4))\n",
        "ax.hist(score_df[score_df[\"is_anomaly\"] == 0][\"score\"], bins=30, alpha=0.7, label=\"normal\")\n",
        "ax.hist(score_df[score_df[\"is_anomaly\"] == 1][\"score\"], bins=30, alpha=0.7, label=\"anomaly\")\n",
        "ax.axvline(threshold, color=\"red\", linestyle=\"--\", label=f\"threshold={threshold:.4f}\")\n",
        "ax.set_title(f\"Anomaly Score on Test Split ({ANOMALY_SCORE_NAME})\")\n",
        "ax.set_xlabel(f\"Per-sample score: {ANOMALY_SCORE_NAME}\")\n",
        "ax.set_ylabel(\"Count\")\n",
        "ax.legend()\n",
        "plt.tight_layout()\n",
        "save_figure(fig, output_dir / \"score_histogram.png\")\n",
        "plt.show()\n",
    ]
    code_sources[15] = code_sources[15] + [
        "\n",
        "analysis_df.to_csv(output_dir / \"failure_analysis.csv\", index=False)\n",
        "error_summary_df.to_csv(output_dir / \"failure_error_summary.csv\")\n",
        "defect_recall_df.to_csv(output_dir / \"failure_defect_recall.csv\")\n",
        "fp_defect_df.to_csv(output_dir / \"failure_false_positive_breakdown.csv\")\n",
    ]
    code_sources[17] = score_ablation_run_cell_source()
    return code_sources


def build_notebook(variant: str) -> dict:
    template_path = REPO_ROOT / "experiments/anomaly_detection/autoencoder/x64/batchnorm_dropout/notebook.ipynb"
    template_nb = load_notebook(template_path)
    code_sources = build_code_sources(template_nb, variant)
    code_cells = [cell for cell in template_nb["cells"] if cell["cell_type"] == "code"]
    for cell, source in zip(code_cells, code_sources):
        cell["source"] = source
    cells = [md_cell(TITLES[variant]), md_cell(SUBMISSION_CONTEXTS[variant])]
    for description, code_cell in zip(DESCRIPTIONS, code_cells):
        cells.append(md_cell(description))
        cells.append(code_cell)
    template_nb["cells"] = cells
    return template_nb


def main() -> None:
    for variant in ["baseline", "batchnorm", "residual"]:
        notebook = build_notebook(variant)
        target_path = REPO_ROOT / f"experiments/anomaly_detection/autoencoder/x64/{variant}/notebook.ipynb"
        write_notebook(target_path, notebook)
        print(f"Updated {target_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
