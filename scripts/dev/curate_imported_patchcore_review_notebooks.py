from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


REPO_ROOT = Path(__file__).resolve().parents[2]

BRANCHES = [
    {
        "folder": REPO_ROOT / "experiments" / "anomaly_detection" / "patchcore" / "efficientnet_b1" / "x240" / "one_layer",
        "title": "PatchCore Review Notebook (EfficientNet-B1 One-Layer, x240)",
        "family_note": "Imported external EfficientNet-B1 single-layer PatchCore run.",
    },
    {
        "folder": REPO_ROOT / "experiments" / "anomaly_detection" / "patchcore" / "efficientnet_b1" / "x240" / "layer3_5",
        "title": "PatchCore Review Notebook (EfficientNet-B1 Layer3+5, x240)",
        "family_note": "Imported external EfficientNet-B1 multi-layer PatchCore run.",
    },
    {
        "folder": REPO_ROOT / "experiments" / "anomaly_detection" / "patchcore" / "efficientnet_b1" / "x240" / "layer3_5_no_defect_tuning",
        "title": "PatchCore Review Notebook (EfficientNet-B1 Layer3+5 No Defect Tuning, x240)",
        "family_note": "Imported external EfficientNet-B1 multi-layer PatchCore run without defect-tuned thresholding.",
    },
    {
        "folder": REPO_ROOT / "experiments" / "anomaly_detection" / "patchcore" / "vit_b16" / "x224" / "one_layer_defect_tuning",
        "title": "PatchCore Review Notebook (ViT-B/16 One-Layer Defect Tuning, x224)",
        "family_note": "Imported external ViT-B/16 one-layer PatchCore run with defect-tuned thresholding.",
    },
    {
        "folder": REPO_ROOT / "experiments" / "anomaly_detection" / "patchcore" / "vit_b16" / "x224" / "one_layer_no_defect_tuning",
        "title": "PatchCore Review Notebook (ViT-B/16 One-Layer No Defect Tuning, x224)",
        "family_note": "Imported external ViT-B/16 one-layer PatchCore run without defect-tuned thresholding.",
    },
    {
        "folder": REPO_ROOT / "experiments" / "anomaly_detection" / "patchcore" / "vit_b16" / "x224" / "two_block",
        "title": "PatchCore Review Notebook (ViT-B/16 Two-Block, x224)",
        "family_note": "Imported external ViT-B/16 two-block PatchCore run.",
    },
    {
        "folder": REPO_ROOT / "experiments" / "anomaly_detection" / "patchcore" / "vit_b16" / "x224" / "two_block_no_defect_tuning",
        "title": "PatchCore Review Notebook (ViT-B/16 Two-Block No Defect Tuning, x224)",
        "family_note": "Imported external ViT-B/16 two-block PatchCore run without defect-tuned thresholding.",
    },
]


def archive_source_notebook(folder: Path) -> None:
    notebook = folder / "notebook.ipynb"
    source_notebook = folder / "source_notebook.ipynb"
    if notebook.exists() and not source_notebook.exists():
        shutil.copy2(notebook, source_notebook)


def build_review_notebook(folder: Path, title: str, family_note: str) -> nbformat.NotebookNode:
    folder_rel = folder.relative_to(REPO_ROOT).as_posix()
    artifact_root_rel = f"{folder_rel}/artifacts"
    source_notebook_rel = f"{folder_rel}/source_notebook.ipynb"

    cells = [
        new_markdown_cell(
            textwrap.dedent(
                f"""\
                # {title}

                This curated notebook reviews the extracted outputs from an imported external run rather than retraining the model.

                {family_note}
                """
            )
        ),
        new_markdown_cell(
            textwrap.dedent(
                f"""\
                ## Submission Context

                - Imported source notebook: `{source_notebook_rel}`
                - Artifact root: `{artifact_root_rel}`
                - Checkpoint folder: `{artifact_root_rel}/checkpoints`
                - Plot folder: `{artifact_root_rel}/plots`
                - Results folder: `{artifact_root_rel}/results`
                - Mode: extracted-output review
                """
            )
        ),
        new_markdown_cell("## Imports and Paths\n\nThis cell resolves the artifact folders and loads the extracted-output manifest for the imported notebook."),
        new_code_cell(
            textwrap.dedent(
                f"""\
                from pathlib import Path
                import json

                import matplotlib.pyplot as plt
                import pandas as pd
                from IPython.display import Image, Markdown, display

                cwd = Path.cwd().resolve()
                candidate_roots = [cwd, *cwd.parents]
                REPO_ROOT = None
                for candidate in candidate_roots:
                    if (candidate / "src" / "wafer_defect").exists() and (candidate / "configs").exists():
                        REPO_ROOT = candidate
                        break

                if REPO_ROOT is None:
                    raise RuntimeError("Could not locate repo root containing src/wafer_defect and configs/")

                BRANCH_DIR = REPO_ROOT / "{folder_rel}"
                ARTIFACT_ROOT = BRANCH_DIR / "artifacts"
                CHECKPOINTS_DIR = ARTIFACT_ROOT / "checkpoints"
                PLOTS_DIR = ARTIFACT_ROOT / "plots"
                RESULTS_DIR = ARTIFACT_ROOT / "results"
                EXTRACTED_DIR = RESULTS_DIR / "extracted_notebook_outputs"
                MANIFEST_PATH = RESULTS_DIR / "summary.json"
                manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
                manifest
                """
            )
        ),
        new_markdown_cell("## Artifact Inventory\n\nThis cell summarizes the extracted outputs and checkpoint notes saved for this imported branch."),
        new_code_cell(
            textwrap.dedent(
                """\
                artifact_inventory = {
                    "checkpoint_files": sorted(path.name for path in CHECKPOINTS_DIR.glob("*")),
                    "plot_files": sorted(path.name for path in PLOTS_DIR.glob("*")),
                    "result_files": sorted(path.name for path in RESULTS_DIR.glob("*") if path.is_file()),
                    "extracted_output_files": sorted(path.name for path in EXTRACTED_DIR.glob("*")),
                }
                artifact_inventory
                """
            )
        ),
        new_markdown_cell("## Extracted Text Outputs\n\nThis cell loads the text outputs extracted from the saved notebook cells and highlights the ones that mention metrics, thresholds, checkpoints, or defect recall."),
        new_code_cell(
            textwrap.dedent(
                """\
                text_paths = sorted(EXTRACTED_DIR.glob("*.txt"))
                text_rows = []
                keyword_hits = []

                for path in text_paths:
                    content = path.read_text(encoding="utf-8", errors="replace")
                    text_rows.append({"file": path.name, "chars": len(content)})
                    lowered = content.lower()
                    if any(token in lowered for token in ["roc-auc", "threshold", "precision", "recall", "f1-score", "confusion", "saved model artifact", "per-class defect recall"]):
                        keyword_hits.append((path.name, content))

                text_index_df = pd.DataFrame(text_rows)
                display(text_index_df)

                for name, content in keyword_hits:
                    display(Markdown(f"### `{name}`"))
                    print(content)
                """
            )
        ),
        new_markdown_cell("## Extracted Plots\n\nThis cell displays the PNG figures extracted from the saved notebook cells."),
        new_code_cell(
            textwrap.dedent(
                """\
                plot_paths = sorted(PLOTS_DIR.glob("*.png"))
                display(pd.DataFrame({"plot_file": [path.name for path in plot_paths]}))

                for path in plot_paths:
                    display(Markdown(f"### `{path.name}`"))
                    display(Image(filename=str(path)))
                """
            )
        ),
        new_markdown_cell("## Checkpoint Notes\n\nThis cell shows what the saved notebook outputs said about model artifacts or checkpoint files."),
        new_code_cell(
            textwrap.dedent(
                """\
                checkpoint_note_path = CHECKPOINTS_DIR / "MISSING_CHECKPOINT.txt"
                checkpoint_note = checkpoint_note_path.read_text(encoding="utf-8", errors="replace")
                print(checkpoint_note)
                """
            )
        ),
        new_markdown_cell("## Saved Outputs\n\nThis cell returns the canonical paths for the curated review branch."),
        new_code_cell(
            textwrap.dedent(
                """\
                saved_outputs = {
                    "branch_dir": str(BRANCH_DIR),
                    "artifact_root": str(ARTIFACT_ROOT),
                    "checkpoints_dir": str(CHECKPOINTS_DIR),
                    "plots_dir": str(PLOTS_DIR),
                    "results_dir": str(RESULTS_DIR),
                    "source_notebook": str(BRANCH_DIR / "source_notebook.ipynb"),
                }
                saved_outputs
                """
            )
        ),
    ]

    notebook = new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata["language_info"] = {"name": "python", "version": "3.x"}
    return notebook


def update_readme(folder: Path) -> None:
    path = folder / "README.md"
    content = path.read_text(encoding="utf-8").rstrip()
    addition = (
        "\n\nCurated review notebook:\n"
        "- `notebook.ipynb`: lightweight review notebook that loads the extracted outputs directly\n"
        "- `source_notebook.ipynb`: preserved imported training notebook\n"
    )
    if "source_notebook.ipynb" not in content:
        path.write_text(content + addition + "\n", encoding="utf-8")


def main() -> None:
    for item in BRANCHES:
        folder = item["folder"]
        archive_source_notebook(folder)
        nbformat.write(build_review_notebook(folder, item["title"], item["family_note"]), folder / "notebook.ipynb")
        update_readme(folder)
    print(f"Curated {len(BRANCHES)} imported PatchCore review notebooks.")


if __name__ == "__main__":
    main()
