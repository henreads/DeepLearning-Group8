"""Run the curated FastFlow x64 notebook headlessly from the command line."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


NOTEBOOK_PATH = Path("experiments/anomaly_detection/fastflow/x64/main/notebook.ipynb")


def resolve_repo_root() -> Path:
    script_path = Path(__file__).resolve()
    candidates = [script_path.parent, *script_path.parents, Path.cwd().resolve(), *Path.cwd().resolve().parents]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        notebook_exists = (candidate / NOTEBOOK_PATH).exists()
        local_package_layout = (candidate / "src" / "wafer_defect").exists()
        modal_package_layout = (candidate / "scripts").exists() and (candidate / "experiments").exists()
        if notebook_exists and (local_package_layout or modal_package_layout):
            return candidate
    raise FileNotFoundError("Could not locate repo root for the FastFlow notebook runner.")


def execute_notebook(
    notebook_path: Path,
    *,
    run_missing_variants: bool,
    force_retrain_variants: bool,
    qualitative_variant: str,
    qualitative_max_examples: int,
) -> dict[str, Any]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    globals_dict: dict[str, Any] = {"__name__": "__main__"}
    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    print(f"[fastflow-runner] executing {len(code_cells)} code cells from {notebook_path}", flush=True)

    executed_code_cells = 0
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        executed_code_cells += 1
        print(f"[fastflow-runner] starting code cell {executed_code_cells}/{len(code_cells)} (notebook index {index})", flush=True)
        source = "".join(cell.get("source", []))
        code = compile(source, f"{notebook_path.name}::cell_{index}", "exec")
        exec(code, globals_dict)

        if "RUN_MISSING_VARIANTS" in globals_dict:
            globals_dict["RUN_MISSING_VARIANTS"] = run_missing_variants
        if "FORCE_RETRAIN_VARIANTS" in globals_dict:
            globals_dict["FORCE_RETRAIN_VARIANTS"] = force_retrain_variants
        if "QUALITATIVE_VARIANT" in globals_dict:
            globals_dict["QUALITATIVE_VARIANT"] = qualitative_variant
        if "QUALITATIVE_MAX_EXAMPLES" in globals_dict:
            globals_dict["QUALITATIVE_MAX_EXAMPLES"] = qualitative_max_examples
        print(f"[fastflow-runner] finished code cell {executed_code_cells}/{len(code_cells)}", flush=True)

    output_dir = Path(globals_dict["OUTPUT_DIR"]).resolve()
    summary_df = globals_dict["summary_df"]
    best_variant = str(summary_df.iloc[0]["variant"])
    manifest = {
        "output_dir": str(output_dir),
        "best_variant": best_variant,
        "run_missing_variants": run_missing_variants,
        "force_retrain_variants": force_retrain_variants,
        "qualitative_variant": qualitative_variant,
        "qualitative_max_examples": qualitative_max_examples,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-missing-variants", action="store_true")
    parser.add_argument("--force-retrain-variants", action="store_true")
    parser.add_argument("--qualitative-variant", default="wrn50_l23_s4")
    parser.add_argument("--qualitative-max-examples", type=int, default=6)
    args = parser.parse_args()

    repo_root = resolve_repo_root()
    os.chdir(repo_root)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("WM811K_REPO_ROOT", str(repo_root))
    print(f"[fastflow-runner] repo root: {repo_root}", flush=True)
    print(
        f"[fastflow-runner] options: run_missing_variants={args.run_missing_variants}, "
        f"force_retrain_variants={args.force_retrain_variants}, "
        f"qualitative_variant={args.qualitative_variant}, "
        f"qualitative_max_examples={args.qualitative_max_examples}",
        flush=True,
    )

    manifest = execute_notebook(
        (repo_root / NOTEBOOK_PATH).resolve(),
        run_missing_variants=args.run_missing_variants,
        force_retrain_variants=args.force_retrain_variants,
        qualitative_variant=args.qualitative_variant,
        qualitative_max_examples=args.qualitative_max_examples,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
