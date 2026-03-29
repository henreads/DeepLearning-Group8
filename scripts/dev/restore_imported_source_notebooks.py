from __future__ import annotations

from pathlib import Path

import nbformat


REPO_ROOT = Path(__file__).resolve().parents[2]

BRANCHES = [
    ("experiments/anomaly_detection/patchcore/efficientnet_b1/x240/one_layer", "patchcore_efficientnet_b1_one_layer"),
    ("experiments/anomaly_detection/patchcore/efficientnet_b1/x240/layer3_5", "patchcore_efficientnet_b1_layer3_5"),
    ("experiments/anomaly_detection/patchcore/efficientnet_b1/x240/layer3_5_no_defect_tuning", "patchcore_efficientnet_b1_layer3_5_no_defect_tuning"),
    ("experiments/anomaly_detection/patchcore/vit_b16/x224/one_layer_defect_tuning", "patchcore_vit_b16_one_layer_defect_tuning"),
    ("experiments/anomaly_detection/patchcore/vit_b16/x224/one_layer_no_defect_tuning", "patchcore_vit_b16_one_layer_no_defect_tuning"),
    ("experiments/anomaly_detection/patchcore/vit_b16/x224/two_block", "patchcore_vit_b16_two_block"),
    ("experiments/anomaly_detection/patchcore/vit_b16/x224/two_block_no_defect_tuning", "patchcore_vit_b16_two_block_no_defect_tuning"),
]


def patch_source_notebook(branch_dir: Path, artifact_name: str) -> None:
    source_path = branch_dir / "source_notebook.ipynb"
    review_path = branch_dir / "review_notebook.ipynb"
    notebook_path = branch_dir / "notebook.ipynb"

    if not source_path.exists():
        return

    if notebook_path.exists() and notebook_path.read_bytes() != source_path.read_bytes():
        if not review_path.exists():
            notebook_path.replace(review_path)

    nb = nbformat.read(source_path, as_version=4)
    branch_rel = branch_dir.relative_to(REPO_ROOT).as_posix()
    artifact_root = f"{branch_rel}/artifacts"

    for cell in nb.cells:
        if cell.cell_type != "code":
            continue

        source = cell.source

        if "DATA_PATH" in source and "ARTIFACT_DIR" in source:
            cell.source = source
            cell.source = (
                "from pathlib import Path\n\n"
                "cwd = Path.cwd().resolve()\n"
                "candidate_roots = [cwd, *cwd.parents]\n"
                "PROJECT_ROOT = None\n"
                "for candidate in candidate_roots:\n"
                "    if (candidate / 'src' / 'wafer_defect').exists() and (candidate / 'configs').exists():\n"
                "        PROJECT_ROOT = candidate\n"
                "        break\n\n"
                "if PROJECT_ROOT is None:\n"
                "    raise RuntimeError('Could not locate repo root containing src/wafer_defect and configs/')\n\n"
                "SRC_ROOT = PROJECT_ROOT / 'src'\n"
                "if str(SRC_ROOT) not in sys.path:\n"
                "    sys.path.insert(0, str(SRC_ROOT))\n\n"
            ) + source
            cell.source = cell.source.replace("DATA_PATH = '../../data/raw/LSWMD.pkl'", "DATA_PATH = str(PROJECT_ROOT / 'data' / 'raw' / 'LSWMD.pkl')")
            cell.source = cell.source.replace("DATA_PATH  = '../data/raw/LSWMD.pkl'   # ← adjust to your path", "DATA_PATH  = str(PROJECT_ROOT / 'data' / 'raw' / 'LSWMD.pkl')")
            cell.source = cell.source.replace("DATA_PATH  = '../../data/raw/LSWMD.pkl'   # ← adjust to your path", "DATA_PATH  = str(PROJECT_ROOT / 'data' / 'raw' / 'LSWMD.pkl')")

            replacement = (
                f"ARTIFACT_DIR = str(PROJECT_ROOT / '{artifact_root}')\n"
                "CHECKPOINTS_DIR = os.path.join(ARTIFACT_DIR, 'checkpoints')\n"
                "PLOTS_DIR = os.path.join(ARTIFACT_DIR, 'plots')\n"
                "RESULTS_DIR = os.path.join(ARTIFACT_DIR, 'results')\n"
                "MODEL_EXPORT_PATH = os.path.join(CHECKPOINTS_DIR, 'patchcore_model.pt')\n"
                "METRICS_EXPORT_PATH = os.path.join(RESULTS_DIR, 'evaluation_metrics.json')\n"
                "os.makedirs(CHECKPOINTS_DIR, exist_ok=True)\n"
                "os.makedirs(PLOTS_DIR, exist_ok=True)\n"
                "os.makedirs(RESULTS_DIR, exist_ok=True)\n"
                "print('Artifacts will be saved to:', ARTIFACT_DIR)"
            )
            lines = []
            for line in cell.source.splitlines():
                if line.startswith("ARTIFACT_DIR"):
                    lines.append(replacement)
                    continue
                if line.startswith("MODEL_EXPORT_PATH") or line.startswith("METRICS_EXPORT_PATH"):
                    continue
                if line.startswith("os.makedirs(ARTIFACT_DIR") or line.startswith("print('Artifacts will be saved to:'") or line.startswith("print(f'Artifacts →"):
                    continue
                lines.append(line)
            cell.source = "\n".join(lines)

        cell.source = cell.source.replace("os.path.join(ARTIFACT_DIR, 'scores.npz')", "os.path.join(RESULTS_DIR, 'scores.npz')")
        cell.source = cell.source.replace("umap_png = os.path.join(ARTIFACT_DIR, 'umap_test_embeddings.png')", "umap_png = os.path.join(PLOTS_DIR, 'umap_test_embeddings.png')")
        cell.source = cell.source.replace("umap_csv = os.path.join(ARTIFACT_DIR, 'umap_test_embeddings.csv')", "umap_csv = os.path.join(RESULTS_DIR, 'umap_test_embeddings.csv')")
        cell.source = cell.source.replace("df = pd.read_pickle(DATA_PATH)", "from wafer_defect.data.legacy_pickle import read_legacy_pickle\n\ndf = read_legacy_pickle(DATA_PATH)")

    nbformat.write(nb, notebook_path)


def update_readme(branch_dir: Path) -> None:
    readme_path = branch_dir / "README.md"
    content = readme_path.read_text(encoding="utf-8").rstrip()
    content = content.replace(
        "- `notebook.ipynb` is now a curated review notebook that loads the extracted outputs directly",
        "- `notebook.ipynb` is the runnable imported source notebook with repo-local data and artifact paths",
    )
    if "review_notebook.ipynb" not in content:
        content += (
            "\n\nNotebook roles:\n"
            "- `notebook.ipynb`: runnable imported source notebook with patched repo-local paths\n"
            "- `review_notebook.ipynb`: lightweight review notebook for the extracted outputs\n"
        )
    readme_path.write_text(content + "\n", encoding="utf-8")


def main() -> None:
    for branch_rel, artifact_name in BRANCHES:
        branch_dir = REPO_ROOT / branch_rel
        patch_source_notebook(branch_dir, artifact_name)
        update_readme(branch_dir)
    print(f"Restored runnable source notebooks for {len(BRANCHES)} imported branches.")


if __name__ == "__main__":
    main()
