"""Run the teacher-student WideResNet50-2 multilayer x64 notebook headlessly in phases."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


NOTEBOOK_PATH = Path(
    "experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/multilayer_self_contained/notebook.ipynb"
)
ARTIFACT_OUTPUT_DIR = Path(
    "experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/multilayer_self_contained/artifacts/ts_wideresnet50_multilayer"
)

PHASE_CELLS = {
    "train": [1, 2, 4, 5, 6, 7, 8, 9, 10],
    "eval": [1, 2, 4, 5, 6, 7, 8, 11, 12, 13],
    "sweep": [1, 2, 4, 5, 6, 7, 8, 11, 15, 16, 18],
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
    raise FileNotFoundError("Could not locate repo root for the TS WideResNet50-2 multilayer runner.")


def display(obj: object) -> None:
    print(obj, flush=True)


def execute_phase(notebook_path: Path, *, phase: str, output_dir: str, num_workers: int) -> dict[str, Any]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    globals_dict: dict[str, Any] = {"__name__": "__main__", "display": display}
    phase_cells = PHASE_CELLS[phase]
    print(
        f"[ts-wrn50-x64-multilayer] executing {len(phase_cells)} code cells for phase={phase} from {notebook_path}",
        flush=True,
    )

    for step, cell_index in enumerate(phase_cells, start=1):
        cell = notebook["cells"][cell_index]
        source = "".join(cell.get("source", []))
        print(
            f"[ts-wrn50-x64-multilayer] starting code cell {step}/{len(phase_cells)} "
            f"(phase={phase}, notebook index {cell_index})",
            flush=True,
        )
        code = compile(source, f"{notebook_path.name}::cell_{cell_index}", "exec")
        exec(code, globals_dict)

        if cell_index == 2:
            config = globals_dict["CONFIG"]
            config["run"]["output_dir"] = output_dir
            config["run"]["run_training"] = phase == "train"
            config["run"]["run_score_sweep"] = phase == "sweep"
            config["data"]["num_workers"] = int(num_workers)
            globals_dict["RUN_TRAINING"] = phase == "train"
            globals_dict["RUN_SCORE_SWEEP"] = phase == "sweep"

        print(
            f"[ts-wrn50-x64-multilayer] finished code cell {step}/{len(phase_cells)} (phase={phase})",
            flush=True,
        )

    artifact_dir = Path(output_dir).resolve()
    manifest = {
        "phase": phase,
        "artifact_dir": str(artifact_dir),
        "checkpoint_exists": bool((artifact_dir / "checkpoints" / "best_model.pt").exists()),
        "evaluation_summary_exists": bool((artifact_dir / "results" / "evaluation" / "summary.json").exists()),
        "score_sweep_exists": bool((artifact_dir / "results" / "evaluation" / "score_sweep.csv").exists()),
    }
    manifest_name = "run_manifest.json" if phase == "sweep" else f"{phase}_phase_manifest.json"
    (artifact_dir / manifest_name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ARTIFACT_OUTPUT_DIR))
    parser.add_argument("--phase", choices=sorted(PHASE_CELLS), required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    repo_root = resolve_repo_root()
    os.chdir(repo_root)
    os.environ.setdefault("MPLBACKEND", "Agg")
    print(
        f"[ts-wrn50-x64-multilayer] repo root: {repo_root}; output_dir={args.output_dir}; "
        f"phase={args.phase}; num_workers={args.num_workers}",
        flush=True,
    )
    manifest = execute_phase(
        (repo_root / NOTEBOOK_PATH).resolve(),
        phase=args.phase,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
