"""Run the EfficientNet-B1 x240 PatchCore notebook headlessly."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any
import zipfile

import pandas as pd
import torch


NOTEBOOK_PATH = Path("experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main/notebook.ipynb")
ARTIFACT_OUTPUT_DIR = Path(
    "experiments/anomaly_detection/patchcore/efficientnet_b1/x240/main/artifacts/patchcore_efficientnet_b1_one_layer"
)
MAIN_CELL_INDICES = [2, 3, 4, 5]
EXTRA_CELL_INDICES = [6, 7, 8, 10]


def resolve_repo_root() -> Path:
    script_path = Path(__file__).resolve()
    candidates = [script_path.parent, *script_path.parents, Path.cwd().resolve(), *Path.cwd().resolve().parents]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        notebook_exists = (candidate / NOTEBOOK_PATH).exists()
        project_layout = (candidate / "experiments").exists() and (candidate / "scripts").exists()
        if notebook_exists and project_layout:
            return candidate
    raise FileNotFoundError("Could not locate repo root for the EfficientNet-B1 x240 PatchCore runner.")


def display(obj: object) -> None:
    print(obj, flush=True)


def _cleanup_corrupt_dataset_cache(output_dir_path: Path) -> None:
    cache_path = output_dir_path / "dataset_cache.npz"
    if not cache_path.exists():
        return
    if zipfile.is_zipfile(cache_path):
        return
    print(
        f"[patchcore-effb1-x240] removing corrupt dataset cache before run: {cache_path}",
        flush=True,
    )
    cache_path.unlink()


def _load_saved_main_result(output_dir_path: Path, variant_name: str) -> dict[str, Any]:
    summary_path = output_dir_path / f"{variant_name}_summary.json"
    checkpoint_path = output_dir_path / f"{variant_name}_best_model.pt"
    best_row_path = output_dir_path / f"{variant_name}_best_row.csv"
    defect_breakdown_path = output_dir_path / f"{variant_name}_defect_breakdown.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"Main-phase summary not found: {summary_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Main-phase checkpoint not found: {checkpoint_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    score_df = pd.read_csv(best_row_path) if best_row_path.exists() else pd.DataFrame()
    defect_breakdown_df = pd.read_csv(defect_breakdown_path) if defect_breakdown_path.exists() else pd.DataFrame()
    return {
        "summary": summary,
        "checkpoint": checkpoint,
        "score_df": score_df,
        "defect_breakdown_df": defect_breakdown_df,
    }


def execute_phase(
    notebook_path: Path,
    *,
    raw_pickle: str,
    output_dir: str,
    num_workers: int,
    phase: str,
) -> dict[str, Any]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    globals_dict: dict[str, Any] = {"__name__": "__main__", "display": display}
    output_dir_path = Path(output_dir).resolve()
    _cleanup_corrupt_dataset_cache(output_dir_path)
    phase_cells = MAIN_CELL_INDICES if phase == "main" else [2, 3, *EXTRA_CELL_INDICES]
    print(
        f"[patchcore-effb1-x240] executing {len(phase_cells)} code cells for phase={phase} from {notebook_path}",
        flush=True,
    )

    for step, cell_index in enumerate(phase_cells, start=1):
        cell = notebook["cells"][cell_index]
        source = "".join(cell.get("source", []))
        print(
            f"[patchcore-effb1-x240] starting code cell {step}/{len(phase_cells)} "
            f"(phase={phase}, notebook index {cell_index})",
            flush=True,
        )
        code = compile(source, f"{notebook_path.name}::cell_{cell_index}", "exec")
        exec(code, globals_dict)

        if cell_index == 2:
            config = globals_dict["CONFIG"]
            config["run"]["raw_pickle"] = raw_pickle
            config["run"]["output_dir"] = output_dir
            config["model"]["num_workers"] = int(num_workers)
            config["model"]["persistent_workers"] = bool(num_workers > 0)
            if phase != "main":
                variant_name = str(config["run"]["variant_name"])
                globals_dict["result"] = _load_saved_main_result(output_dir_path, variant_name)

        print(
            f"[patchcore-effb1-x240] finished code cell {step}/{len(phase_cells)} (phase={phase})",
            flush=True,
        )

    output_dir_path = Path(globals_dict["output_dir"]).resolve()
    result = globals_dict.get("result", {})
    summary = result.get("summary", {})
    best_variant = str(summary.get("name", ""))
    manifest = {
        "output_dir": str(output_dir_path),
        "best_variant": best_variant,
        "raw_pickle": raw_pickle,
        "num_workers": int(num_workers),
        "phase": phase,
        "executed_cells": phase_cells,
    }
    manifest_name = "run_manifest.json" if phase == "extras" else f"{phase}_phase_manifest.json"
    (output_dir_path / manifest_name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-pickle", default="/root/project/data/raw/LSWMD.pkl")
    parser.add_argument("--output-dir", default=str(ARTIFACT_OUTPUT_DIR))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--phase", choices=["main", "extras"], default="main")
    args = parser.parse_args()

    repo_root = resolve_repo_root()
    os.chdir(repo_root)
    os.environ.setdefault("MPLBACKEND", "Agg")
    print(
        f"[patchcore-effb1-x240] repo root: {repo_root}; raw_pickle={args.raw_pickle}; "
        f"output_dir={args.output_dir}; num_workers={args.num_workers}; phase={args.phase}",
        flush=True,
    )

    manifest = execute_phase(
        (repo_root / NOTEBOOK_PATH).resolve(),
        raw_pickle=args.raw_pickle,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        phase=args.phase,
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
