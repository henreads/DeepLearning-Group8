"""Runner: aligned ViT-B/16 + EfficientNet-B1 ensemble re-scoring.

Executes only the four aligned-re-scoring cells from the ensemble notebook on a
shared test set so that per-class recall can be computed correctly.

Usage (from repo root):
    python scripts/run_ensemble_vit_effnetb1_aligned_x224.py \
        --raw-pickle data/raw/LSWMD.pkl \
        --output-dir experiments/anomaly_detection/ensemble/x224/vit_effnetb1_ensemble/artifacts \
        --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import shutil
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

NOTEBOOK_PATH = Path(
    "experiments/anomaly_detection/ensemble/x224/vit_effnetb1_ensemble/notebook.ipynb"
)

# IDs of the four aligned re-scoring cells added to the notebook.
ALIGNED_CELL_IDS = {"bfb9d5da", "0ba59007", "d06edb91", "29c0fa2a"}

TAG = "[ensemble-vit-effnetb1-aligned]"


def _build_bootstrap_globals(repo_root: Path, artifact_dir: Path) -> dict[str, Any]:
    """Pre-populate the globals dict the aligned cells depend on."""
    import json as _json
    import warnings as _warnings
    import numpy as _np
    import pandas as _pd
    from pathlib import Path as _Path
    from IPython.display import display as _display
    from sklearn.metrics import (
        roc_auc_score as _roc_auc_score,
        average_precision_score as _avg_prec,
        precision_recall_fscore_support as _prf,
    )

    return {
        "__name__": "__main__",
        "json":      _json,
        "warnings":  _warnings,
        "np":        _np,
        "pd":        _pd,
        "Path":      _Path,
        "display":   _display,
        "roc_auc_score":                   _roc_auc_score,
        "average_precision_score":         _avg_prec,
        "precision_recall_fscore_support": _prf,
        "REPO_ROOT":    repo_root,
        "ARTIFACT_DIR": artifact_dir,
    }


def execute_aligned_cells(
    *,
    notebook_path: Path,
    repo_root: Path,
    artifact_dir: Path,
    batch_size: int,
) -> dict[str, Any]:
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
    aligned_cells = [c for c in code_cells if c.get("id", "") in ALIGNED_CELL_IDS]

    if len(aligned_cells) != 4:
        found = [c.get("id") for c in aligned_cells]
        raise RuntimeError(
            f"Expected 4 aligned re-scoring cells, found {len(aligned_cells)}: {found}"
        )

    g = _build_bootstrap_globals(repo_root, artifact_dir)

    # Allow batch-size override in the scoring cell
    for idx, cell in enumerate(aligned_cells, start=1):
        cell_id = cell.get("id", f"aligned_cell_{idx}")
        code = "".join(cell.get("source", []))

        # Override BATCH_SIZE if set by runner arg
        if "BATCH_SIZE = 64" in code or "BATCH_SIZE=" in code:
            code = code.replace("BATCH_SIZE = 64", f"BATCH_SIZE = {batch_size}")

        print(f"{TAG} cell {idx}/4  ({cell_id})", flush=True)
        exec(compile(code, f"{notebook_path.name}::{cell_id}", "exec"), g)
        print(f"{TAG} cell {idx}/4 done", flush=True)

    summary_path = artifact_dir / "aligned_ensemble_summary.json"
    breakdown_path = artifact_dir / "aligned_defect_breakdown.csv"

    manifest: dict[str, Any] = {
        "status": "complete",
        "output_dir": str(artifact_dir),
        "summary_path": str(summary_path),
        "breakdown_path": str(breakdown_path),
    }
    if summary_path.exists():
        manifest["metrics"] = json.loads(summary_path.read_text(encoding="utf-8"))

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Aligned ViT + EffNet ensemble re-scoring")
    parser.add_argument("--raw-pickle", required=True, help="Path to LSWMD.pkl")
    parser.add_argument("--output-dir",  required=True, help="Artifact output directory")
    parser.add_argument("--batch-size",  type=int, default=64)
    args = parser.parse_args()

    raw_pickle = Path(args.raw_pickle).resolve()
    if not raw_pickle.exists():
        raise FileNotFoundError(f"Raw pickle not found: {raw_pickle}")

    # Runner scripts are expected to run from repo root; ensure LSWMD.pkl is in place.
    expected_raw = Path("data/raw/LSWMD.pkl").resolve()
    expected_raw.parent.mkdir(parents=True, exist_ok=True)
    if raw_pickle != expected_raw:
        print(f"{TAG} copying LSWMD.pkl → {expected_raw}", flush=True)
        shutil.copy2(raw_pickle, expected_raw)

    repo_root = Path.cwd().resolve()
    artifact_dir = Path(args.output_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    notebook_path = (repo_root / NOTEBOOK_PATH).resolve()
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    manifest = execute_aligned_cells(
        notebook_path=notebook_path,
        repo_root=repo_root,
        artifact_dir=artifact_dir,
        batch_size=args.batch_size,
    )

    manifest_path = artifact_dir / "aligned_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
