from __future__ import annotations

from pathlib import Path

import nbformat


REPO_ROOT = Path(__file__).resolve().parents[2]

BRANCHES = [
    "experiments/anomaly_detection/patchcore/efficientnet_b1/x240/one_layer",
    "experiments/anomaly_detection/patchcore/efficientnet_b1/x240/layer3_5",
    "experiments/anomaly_detection/patchcore/efficientnet_b1/x240/layer3_5_no_defect_tuning",
    "experiments/anomaly_detection/patchcore/vit_b16/x224/one_layer_defect_tuning",
    "experiments/anomaly_detection/patchcore/vit_b16/x224/one_layer_no_defect_tuning",
    "experiments/anomaly_detection/patchcore/vit_b16/x224/two_block",
    "experiments/anomaly_detection/patchcore/vit_b16/x224/two_block_no_defect_tuning",
]


def flatten_branch(branch_rel: str) -> None:
    branch_dir = REPO_ROOT / branch_rel
    art_dir = branch_dir / "artifacts"

    # Remove any redundant named run folders left behind from the earlier import pass.
    for child in art_dir.iterdir():
        if child.is_dir() and child.name not in {"checkpoints", "plots", "results"}:
            if any(p.is_file() for p in child.rglob("*")):
                raise RuntimeError(f"Refusing to remove non-empty nested artifact folder: {child}")
            for subdir in sorted((p for p in child.rglob("*") if p.is_dir()), reverse=True):
                subdir.rmdir()
            child.rmdir()

    notebook_path = branch_dir / "notebook.ipynb"
    nb = nbformat.read(notebook_path, as_version=4)
    target = f"ARTIFACT_DIR = str(PROJECT_ROOT / '{branch_rel.replace('\\', '/')}/artifacts')"

    changed = False
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if "ARTIFACT_DIR = str(PROJECT_ROOT /" in cell.source:
            lines = cell.source.splitlines()
            new_lines = []
            for line in lines:
                if line.startswith("ARTIFACT_DIR = str(PROJECT_ROOT / "):
                    line = target
                    changed = True
                new_lines.append(line)
            cell.source = "\n".join(new_lines)

    if changed:
        nbformat.write(nb, notebook_path)


def main() -> None:
    for branch in BRANCHES:
        flatten_branch(branch)
    print(f"Flattened redundant artifact nesting for {len(BRANCHES)} imported PatchCore branches.")


if __name__ == "__main__":
    main()
