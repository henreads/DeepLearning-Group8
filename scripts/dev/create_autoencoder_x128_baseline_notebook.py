from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    source_path = REPO_ROOT / "experiments/anomaly_detection/autoencoder/x64/baseline/notebook.ipynb"
    target_path = REPO_ROOT / "experiments/anomaly_detection/autoencoder/x128/baseline/notebook.ipynb"

    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    notebook["cells"][0]["source"] = [
        "# Autoencoder Training Notebook (X128 Baseline)\n",
        "\n",
        "This notebook runs the baseline convolutional autoencoder on the curated x128 benchmark split. By default it reuses saved artifacts from its local artifact folder and only retrains when `FORCE_RETRAIN = True`.\n",
    ]
    notebook["cells"][1]["source"] = [
        "## Submission Context\n",
        "\n",
        "- Dataset metadata: `data/processed/x128/wm811k/metadata_50k_5pct.csv`\n",
        "- Reference data config: `configs/data/data_128.toml`\n",
        "- Experiment config: `experiments/anomaly_detection/autoencoder/x128/baseline/train_config.toml`\n",
        "- Artifact root: `experiments/anomaly_detection/autoencoder/x128/baseline/artifacts/autoencoder_baseline`\n",
        "- Default behavior: load the saved checkpoint, history, and score-ablation outputs if they already exist; only retrain when explicitly requested.\n",
    ]
    notebook["cells"][5]["source"] = [
        "CONFIG_PATH = REPO_ROOT / \"experiments/anomaly_detection/autoencoder/x128/baseline/train_config.toml\"\n",
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

    target_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Created {target_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
