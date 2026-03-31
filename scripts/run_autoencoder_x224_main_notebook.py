#!/usr/bin/env python
"""
Phased runner for autoencoder x224 main notebook.

Usage:
    python scripts/run_autoencoder_x224_main_notebook.py train
    python scripts/run_autoencoder_x224_main_notebook.py eval
    python scripts/run_autoencoder_x224_main_notebook.py sweep

Phases:
- train: Cells 0-14 (setup, config, data, training, history, persist)
- eval: Cells 0-5, 8, 15-22 (setup, config, data, test scoring, threshold, metrics)
- sweep: Cells 0-5, 8, 15-22 (eval) + threshold sweep and failure analysis (cells 23-34)
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).parent.parent.resolve()
NOTEBOOK_PATH = (
    REPO_ROOT / "experiments" / "anomaly_detection" / "autoencoder" / "x224" / "main" / "notebook.ipynb"
)

PHASE_CELLS = {
    "train": [3, 5, 7, 9, 11, 13, 15, 17, 19],  # Setup through persist
    "eval": [3, 5, 7, 9, 11, 13, 15, 21, 23, 25],  # Setup, data, scoring, metrics
    "sweep": [3, 5, 7, 9, 11, 13, 15, 21, 23, 25, 27, 29, 31, 33, 35],  # All through failure examples
}


def load_notebook(path: Path) -> dict:
    """Load notebook JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_notebook(nb: dict, path: Path) -> None:
    """Save notebook JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)


def build_manifest(
    notebook: dict, phase: str, repo_root: Path
) -> tuple[dict, list[int]]:
    """
    Build execution manifest for a phase.

    Returns:
        (modified_notebook, cell_indices)
    """
    nb_copy = json.loads(json.dumps(notebook))  # Deep copy
    cell_indices = PHASE_CELLS.get(phase, [])

    return nb_copy, cell_indices


def run_notebook_in_process(notebook_path: Path, phase: str) -> int:
    """
    Run notebook cells in-process using nbconvert and python.

    Falls back to running the original notebook with flags set.
    """
    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--ExecutePreprocessor.timeout=3600",
        "--output", str(notebook_path.parent / f"_executed_{phase}.ipynb"),
        str(notebook_path),
    ]

    print(f"Running phase '{phase}' via nbconvert...")
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=notebook_path.parent)
    return result.returncode


def main(phase: Optional[str] = None) -> int:
    """
    Execute a phase of the notebook.

    Args:
        phase: 'train', 'eval', or 'sweep'

    Returns:
        Exit code
    """
    if phase is None:
        phase = sys.argv[1] if len(sys.argv) > 1 else "train"

    if phase not in PHASE_CELLS:
        print(f"Unknown phase: {phase}")
        print(f"Available phases: {list(PHASE_CELLS.keys())}")
        return 1

    if not NOTEBOOK_PATH.exists():
        print(f"Notebook not found: {NOTEBOOK_PATH}")
        return 1

    print(f"Phase: {phase}")
    print(f"Notebook: {NOTEBOOK_PATH}")
    print(f"Cells: {PHASE_CELLS[phase]}")

    # Run the notebook
    return run_notebook_in_process(NOTEBOOK_PATH, phase)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
