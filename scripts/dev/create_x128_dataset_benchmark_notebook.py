from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    source_path = REPO_ROOT / "data/dataset/x64/benchmark_50k_5pct/notebook.ipynb"
    target_path = REPO_ROOT / "data/dataset/x128/benchmark_50k_5pct/notebook.ipynb"

    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    notebook["cells"][0]["source"] = [
        "# x128 Benchmark 50k 5pct Dataset Build\n",
        "\n",
        "This notebook prepares and validates the curated `128 x 128` WM-811K benchmark split used by the `x128` autoencoder experiments.\n",
        "\n",
        "It is meant to answer two questions clearly:\n",
        "\n",
        "- can we regenerate the processed arrays and metadata from the raw pickle?\n",
        "- do the outputs written under `data/processed/` match the intended split configuration?\n",
    ]
    notebook["cells"][1]["source"] = [
        "## Notebook Flow\n",
        "\n",
        "Run the notebook from top to bottom.\n",
        "\n",
        "1. load the dataset config for this benchmark branch\n",
        "2. confirm the raw pickle exists and inspect the raw label distribution\n",
        "3. run the shared preparation script that writes arrays and metadata into `data/processed/x128/wm811k/`\n",
        "4. validate the generated CSV and array files\n",
        "5. inspect a few example wafers from the exported split\n",
    ]
    cell2 = "".join(notebook["cells"][2]["source"]).replace(
        'DATASET_DIR = REPO_ROOT / "data" / "dataset" / "x64" / "benchmark_50k_5pct"',
        'DATASET_DIR = REPO_ROOT / "data" / "dataset" / "x128" / "benchmark_50k_5pct"',
    )
    notebook["cells"][2]["source"] = cell2.splitlines(keepends=True)
    notebook["cells"][6]["source"] = [
        "## Generate Processed Outputs\n",
        "\n",
        "This cell calls the shared preparation script used by the rest of the repository. It writes the metadata CSV and the `.npy` wafer arrays under `data/processed/x128/wm811k/` according to the config in this folder.\n",
    ]
    notebook["cells"][13]["source"] = [
        "## Adapting This Pattern\n",
        "\n",
        "When we curate more dataset branches, they should follow the same notebook structure:\n",
        "\n",
        "- local `data_config.toml`\n",
        "- one generation cell that calls the shared prep script\n",
        "- one validation block that confirms the CSV and arrays were written correctly\n",
        "\n",
        "For `x128`, this notebook is the dataset-side companion to the `x128` autoencoder baseline experiment.\n",
    ]

    target_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Created {target_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
