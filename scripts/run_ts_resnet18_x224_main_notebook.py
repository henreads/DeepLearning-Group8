"""Run the teacher-student ResNet18 x224 main notebook headlessly in phases."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


NOTEBOOK_PATH = Path("experiments/anomaly_detection/teacher_student/resnet18/x224/main/notebook.ipynb")
ARTIFACT_OUTPUT_DIR = Path(
    "experiments/anomaly_detection/teacher_student/resnet18/x224/main/artifacts/ts_resnet18_x224"
)

PHASE_CELLS = {
    "train": [2, 3, 4, 5, 6],
    "eval": [2, 3, 4, 5, 7, 9],
    "sweep": [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18, 20],
}


def resolve_repo_root() -> Path:
    script_path = Path(__file__).resolve()
    candidates = [script_path.parent, *script_path.parents, Path.cwd().resolve(), *Path.cwd().resolve().parents]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / NOTEBOOK_PATH).exists() and (candidate / "experiments").exists() and (candidate / "scripts").exists():
            return candidate
    raise FileNotFoundError("Could not locate repo root for the TS ResNet18 x224 main runner.")


def display(obj: object) -> None:
    print(obj, flush=True)


def execute_phase(notebook_path: Path, *, phase: str, output_dir: str) -> dict[str, Any]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    globals_dict: dict[str, Any] = {"__name__": "__main__", "display": display}
    repo_root = Path.cwd().resolve()
    phase_cells = PHASE_CELLS[phase]
    print(
        f"[ts-resnet18-x224-main] executing {len(phase_cells)} code cells for phase={phase} from {notebook_path}",
        flush=True,
    )

    for step, cell_index in enumerate(phase_cells, start=1):
        cell = notebook["cells"][cell_index]
        # Skip markdown cells
        if cell.get("cell_type") != "code":
            print(
                f"[ts-resnet18-x224-main] skipping non-code cell {step}/{len(phase_cells)} "
                f"(phase={phase}, notebook index {cell_index})",
                flush=True,
            )
            continue
        source = "".join(cell.get("source", []))
        if cell_index == 2:
            source = source.replace(
                """cwd = Path.cwd().resolve()
candidate_roots = [cwd, *cwd.parents]
REPO_ROOT = None
for candidate in candidate_roots:
    if (candidate / "src" / "wafer_defect").exists() and (candidate / "configs").exists():
        REPO_ROOT = candidate
        break

if REPO_ROOT is None:
    raise RuntimeError("Could not locate repo root containing src/wafer_defect and configs/")
""",
                f'REPO_ROOT = Path(r"{repo_root.as_posix()}")\n',
            )
        print(
            f"[ts-resnet18-x224-main] starting code cell {step}/{len(phase_cells)} "
            f"(phase={phase}, notebook index {cell_index})",
            flush=True,
        )
        code = compile(source, f"{notebook_path.name}::cell_{cell_index}", "exec")
        exec(code, globals_dict)

        if cell_index == 3:
            artifact_dir = Path(output_dir).resolve()
            checkpoint_dir = artifact_dir / "checkpoints"
            results_dir = artifact_dir / "results"
            evaluation_dir = results_dir / "evaluation"
            plots_dir = artifact_dir / "plots"
            globals_dict["ARTIFACT_DIR"] = artifact_dir
            globals_dict["CHECKPOINT_DIR"] = checkpoint_dir
            globals_dict["RESULTS_DIR"] = results_dir
            globals_dict["EVALUATION_DIR"] = evaluation_dir
            globals_dict["PLOTS_DIR"] = plots_dir
            globals_dict["CHECKPOINT_PATH"] = checkpoint_dir / "best_model.pt"
            globals_dict["RUN_LOCAL_TRAINING"] = phase == "train"
            globals_dict["RUN_DEFAULT_EVALUATION"] = phase == "eval"
            globals_dict["RUN_SCORE_SWEEP"] = phase == "sweep"
            for path in [checkpoint_dir, results_dir, evaluation_dir, plots_dir]:
                path.mkdir(parents=True, exist_ok=True)

        print(
            f"[ts-resnet18-x224-main] finished code cell {step}/{len(phase_cells)} (phase={phase})",
            flush=True,
        )

    artifact_dir = Path(globals_dict["ARTIFACT_DIR"]).resolve()
    manifest = {
        "phase": phase,
        "artifact_dir": str(artifact_dir),
        "checkpoint_exists": bool((artifact_dir / "checkpoints" / "best_model.pt").exists()),
        "evaluation_summary_exists": bool((artifact_dir / "results" / "evaluation" / "summary.json").exists()),
        "score_sweep_exists": bool((artifact_dir / "results" / "evaluation" / "score_sweep_summary.csv").exists()),
    }
    manifest_name = "run_manifest.json" if phase == "sweep" else f"{phase}_phase_manifest.json"
    (artifact_dir / manifest_name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ARTIFACT_OUTPUT_DIR))
    parser.add_argument("--phase", choices=sorted(PHASE_CELLS), required=True)
    args = parser.parse_args()

    repo_root = resolve_repo_root()
    os.chdir(repo_root)
    os.environ.setdefault("MPLBACKEND", "Agg")
    print(
        f"[ts-resnet18-x224-main] repo root: {repo_root}; output_dir={args.output_dir}; phase={args.phase}",
        flush=True,
    )
    manifest = execute_phase((repo_root / NOTEBOOK_PATH).resolve(), phase=args.phase, output_dir=args.output_dir)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
