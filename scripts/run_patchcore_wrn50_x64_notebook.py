"""Run the WRN50 x64 PatchCore notebook headlessly."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


NOTEBOOK_PATH = Path("experiments/anomaly_detection/patchcore/wideresnet50/x64/main/notebook.ipynb")
ARTIFACT_OUTPUT_DIR = Path(
    "experiments/anomaly_detection/patchcore/wideresnet50/x64/main/artifacts/patchcore_wideresnet50_multilayer"
)
CODE_CELL_INDICES = [2, 3, 4, 5, 6, 7, 9]


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
    raise FileNotFoundError("Could not locate repo root for the WRN50 x64 PatchCore runner.")


def display(obj: object) -> None:
    print(obj, flush=True)


def execute_notebook(
    notebook_path: Path,
    *,
    raw_pickle: str,
    output_dir: str,
    num_workers: int,
) -> dict[str, Any]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    globals_dict: dict[str, Any] = {"__name__": "__main__", "display": display}
    print(f"[patchcore-wrn50-x64] executing {len(CODE_CELL_INDICES)} code cells from {notebook_path}", flush=True)

    for step, cell_index in enumerate(CODE_CELL_INDICES, start=1):
        cell = notebook["cells"][cell_index]
        source = "".join(cell.get("source", []))
        print(
            f"[patchcore-wrn50-x64] starting code cell {step}/{len(CODE_CELL_INDICES)} "
            f"(notebook index {cell_index})",
            flush=True,
        )
        code = compile(source, f"{notebook_path.name}::cell_{cell_index}", "exec")
        exec(code, globals_dict)

        if cell_index == 2:
            globals_dict["RAW_PICKLE"] = raw_pickle
            globals_dict["OUTPUT_DIR"] = output_dir
            globals_dict["NUM_WORKERS"] = int(num_workers)

        print(f"[patchcore-wrn50-x64] finished code cell {step}/{len(CODE_CELL_INDICES)}", flush=True)

    output_dir_path = Path(globals_dict["output_dir"]).resolve()
    best_row = globals_dict.get("best_row", {})
    best_variant = str(best_row.get("name", ""))
    manifest = {
        "output_dir": str(output_dir_path),
        "best_variant": best_variant,
        "raw_pickle": raw_pickle,
        "num_workers": int(num_workers),
    }
    (output_dir_path / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-pickle", default="/root/project/data/raw/LSWMD.pkl")
    parser.add_argument("--output-dir", default=str(ARTIFACT_OUTPUT_DIR))
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    repo_root = resolve_repo_root()
    os.chdir(repo_root)
    os.environ.setdefault("MPLBACKEND", "Agg")
    print(
        f"[patchcore-wrn50-x64] repo root: {repo_root}; raw_pickle={args.raw_pickle}; "
        f"output_dir={args.output_dir}; num_workers={args.num_workers}",
        flush=True,
    )

    manifest = execute_notebook(
        (repo_root / NOTEBOOK_PATH).resolve(),
        raw_pickle=args.raw_pickle,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
