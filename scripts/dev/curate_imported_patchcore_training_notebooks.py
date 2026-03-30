from __future__ import annotations

from pathlib import Path
import re

import nbformat
from nbformat.v4 import new_markdown_cell


REPO_ROOT = Path(__file__).resolve().parents[2]

BRANCHES = [
    (
        "experiments/anomaly_detection/patchcore/efficientnet_b1/x240/one_layer_no_defect_tuning",
        "PatchCore EfficientNet-B1 One-Layer No-Defect-Tuning (`x240`)",
        "This notebook is the canonical training and evaluation workflow for the one-layer EfficientNet-B1 PatchCore experiment without defect-aware threshold tuning.",
    ),
    (
        "experiments/anomaly_detection/patchcore/efficientnet_b1/x240/one_layer",
        "PatchCore EfficientNet-B1 One-Layer (`x240`)",
        "This notebook is the canonical training and evaluation workflow for the one-layer EfficientNet-B1 PatchCore experiment.",
    ),
    (
        "experiments/anomaly_detection/patchcore/efficientnet_b1/x240/layer3_5",
        "PatchCore EfficientNet-B1 Layer 3/5 (`x240`)",
        "This notebook is the canonical training and evaluation workflow for the two-layer EfficientNet-B1 PatchCore experiment using layers 3 and 5.",
    ),
    (
        "experiments/anomaly_detection/patchcore/efficientnet_b1/x240/layer3_5_no_defect_tuning",
        "PatchCore EfficientNet-B1 Layer 3/5 No-Defect-Tuning (`x240`)",
        "This notebook is the canonical training and evaluation workflow for the EfficientNet-B1 PatchCore experiment without defect-aware threshold tuning.",
    ),
    (
        "experiments/anomaly_detection/patchcore/vit_b16/x224/one_layer_defect_tuning",
        "PatchCore ViT-B/16 One-Layer Defect-Tuning (`x224`)",
        "This notebook is the canonical training and evaluation workflow for the one-layer ViT-B/16 PatchCore experiment with defect-aware threshold tuning.",
    ),
    (
        "experiments/anomaly_detection/patchcore/vit_b16/x224/one_layer_no_defect_tuning",
        "PatchCore ViT-B/16 One-Layer No-Defect-Tuning (`x224`)",
        "This notebook is the canonical training and evaluation workflow for the one-layer ViT-B/16 PatchCore experiment without defect-aware threshold tuning.",
    ),
    (
        "experiments/anomaly_detection/patchcore/vit_b16/x224/two_block",
        "PatchCore ViT-B/16 Two-Block (`x224`)",
        "This notebook is the canonical training and evaluation workflow for the two-block ViT-B/16 PatchCore experiment.",
    ),
    (
        "experiments/anomaly_detection/patchcore/vit_b16/x224/two_block_no_defect_tuning",
        "PatchCore ViT-B/16 Two-Block No-Defect-Tuning (`x224`)",
        "This notebook is the canonical training and evaluation workflow for the two-block ViT-B/16 PatchCore experiment without defect-aware threshold tuning.",
    ),
]


def make_overview(title: str, summary: str, branch_rel: str) -> str:
    return (
        f"# {title}\n\n"
        f"{summary}\n\n"
        "Run this notebook from top to bottom to reproduce the experiment locally. "
        "The notebook prepares the data split, trains the model when needed, evaluates the trained model, "
        "and writes outputs into the experiment-local artifact folders.\n\n"
        "Artifacts written by this notebook:\n"
        f"- `[{branch_rel}/artifacts/checkpoints]({branch_rel}/artifacts/checkpoints)` for model checkpoints\n"
        f"- `[{branch_rel}/artifacts/plots]({branch_rel}/artifacts/plots)` for saved figures\n"
        f"- `[{branch_rel}/artifacts/results]({branch_rel}/artifacts/results)` for metrics, score files, and CSV outputs\n"
    )


def first_meaningful_line(source: str) -> str:
    for raw in source.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            line = line.lstrip("#").strip(" -\t")
        return line
    return ""


def normalize_title(text: str) -> str:
    text = text.replace("??", " ")
    text = text.replace("->", " to ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:1].upper() + text[1:] if text else "Notebook Step"


def describe_cell(source: str, cell_idx: int) -> tuple[str, str]:
    first = first_meaningful_line(source)
    lower = source.lower()

    if "install dependencies" in lower or "import importlib" in lower:
        return "Setup", "This cell installs or checks optional notebook dependencies before the rest of the workflow runs."
    if "core imports" in lower or "imports" in lower:
        return "Imports", "This cell imports the Python packages used across training, evaluation, plotting, and artifact export."
    if "project_root" in lower or "artifact_dir" in lower or "configuration" in lower:
        return "Configuration", "This cell resolves the repo root, defines experiment settings, and points all outputs to the local artifact folders."
    if "read_legacy_pickle" in lower or "load and clean labels" in lower:
        return "Load Dataset", "This cell loads the legacy WM-811K pickle, cleans labels, and prepares the dataframe used for the experiment split."
    if "required_normals" in lower or "enforce requested split sizes" in lower:
        return "Create Split", "This cell builds the exact train, tuning, and test split used for this experiment protocol."
    if "wafer_to_tensor" in lower:
        return "Preprocess Images", "This cell converts wafer maps into normalized tensor images for the backbone model."
    if "dataloaders" in lower or "loader_kwargs" in lower:
        return "Build DataLoaders", "This cell wraps the prepared datasets into DataLoaders for feature extraction and evaluation."
    if "patchfeatureextractor" in lower or "patchcore feature extractor" in lower or "vit block" in lower:
        return "Define Model", "This cell defines the PatchCore feature extractor and supporting model components for this branch."
    if "memory bank" in lower or "sampled_patches" in lower or "score_and_return_maps" in lower:
        return "Train and Score", "This cell builds the PatchCore memory bank, computes anomaly scores, and saves the score bundle needed for later evaluation."
    if "with np.load" in lower or "scores.npz" in lower:
        return "Reload Scores", "This cell reloads the saved score bundle so threshold selection and downstream evaluation use the persisted results."
    if "final evaluation" in lower or "y_true" in lower:
        return "Evaluate", "This cell computes the final evaluation metrics on the held-out split using the selected decision threshold."
    if "per-class breakdown" in lower or "manual inspection" in lower:
        return "Failure Analysis", "This cell summarizes defect-level behavior and writes per-sample analysis outputs for qualitative review."
    if "umap" in lower:
        return "UMAP Visualization", "This cell projects saved embeddings for qualitative visualization and saves the resulting UMAP outputs."
    if "torch.save" in lower or "to_json" in lower:
        return "Save Outputs", "This cell exports the trained model artifact, metrics, and any final bookkeeping files for reproducibility."
    if "clear memory" in lower or "vars_to_clear" in lower:
        return "Cleanup", "This cell releases large tensors and datasets so the notebook leaves the runtime in a clean state."

    title = normalize_title(first) if first else f"Notebook Step {cell_idx}"
    return title, "This cell continues the experiment workflow and writes its outputs into the local artifact folders."


def curate_notebook(branch_dir: Path, title: str, summary: str) -> None:
    notebook_path = branch_dir / "notebook.ipynb"
    nb = nbformat.read(notebook_path, as_version=4)

    new_cells = []
    branch_rel = branch_dir.relative_to(REPO_ROOT).as_posix()
    new_cells.append(new_markdown_cell(make_overview(title, summary, branch_rel)))

    code_idx = 0
    for cell in nb.cells:
        if cell.cell_type == "code":
            code_idx += 1
            if "if str(SRC_ROOT) not in sys.path" in cell.source and "import sys" not in cell.source:
                cell.source = cell.source.replace("from pathlib import Path\n\n", "from pathlib import Path\nimport sys\n\n")
            cell.outputs = []
            cell.execution_count = None
            step_title, step_body = describe_cell(cell.source, code_idx)
            new_cells.append(new_markdown_cell(f"## {step_title}\n\n{step_body}"))
            new_cells.append(cell)
        else:
            # Drop imported markdown/output framing so the curated notebook has one consistent structure.
            continue

    nb.cells = new_cells
    nbformat.write(nb, notebook_path)


def update_readme(branch_dir: Path, title: str, summary: str) -> None:
    readme_path = branch_dir / "README.md"
    content = (
        f"# {title}\n\n"
        f"{summary}\n\n"
        "Files:\n"
        "- `notebook.ipynb`: the canonical training and evaluation notebook for this experiment\n"
        "- `artifacts/checkpoints/`: saved model checkpoints written by the notebook\n"
        "- `artifacts/plots/`: figures saved by the notebook\n"
        "- `artifacts/results/`: metrics, score files, exported CSVs, and supporting outputs\n\n"
        "Notes:\n"
        "- The notebook uses repo-local dataset paths and writes outputs back into the local artifact folders.\n"
        "- The canonical expectation for this branch is that the files kept under `artifacts/` are generated by running `notebook.ipynb`.\n"
    )
    readme_path.write_text(content, encoding="utf-8")


def main() -> None:
    for branch_rel, title, summary in BRANCHES:
        branch_dir = REPO_ROOT / branch_rel
        curate_notebook(branch_dir, title, summary)
        update_readme(branch_dir, title, summary)
    print(f"Curated {len(BRANCHES)} imported PatchCore notebooks into the canonical notebook.ipynb format.")


if __name__ == "__main__":
    main()
