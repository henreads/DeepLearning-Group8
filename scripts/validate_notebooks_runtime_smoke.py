from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any


DEFAULT_ROOTS = ["data", "experiments", "kaggle_upload", "uploads"]
DEFAULT_TIMEOUT_SECONDS = 45
DATASET_TIMEOUT_SECONDS = 120
REPORT_DIR = Path("outputs/notebook_validation")
REPORT_PREFIX = "all_notebooks_runtime_smoke"
MODE = "runtime_smoke_subprocess"


def list_notebooks(roots: list[str]) -> list[Path]:
    notebook_paths: list[Path] = []
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        notebook_paths.extend(path for path in root_path.rglob("*.ipynb") if path.is_file())
    return sorted({path.resolve() for path in notebook_paths})


def timeout_for(notebook_path: Path, repo_root: Path) -> int:
    relative_path = notebook_path.relative_to(repo_root)
    if relative_path.parts[:2] == ("data", "dataset"):
        return DATASET_TIMEOUT_SECONDS
    return DEFAULT_TIMEOUT_SECONDS


def trim_output(text: str | None, *, limit: int = 12000) -> str | None:
    if not text:
        return None
    text = text.strip()
    if len(text) <= limit:
        return text
    head = text[: limit - 200]
    tail = text[-160:]
    return f"{head}\n...\n{tail}"


def sanitize_code(source: str) -> str:
    cleaned_lines: list[str] = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("%") or stripped.startswith("!"):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines) + ("\n" if source.endswith("\n") else "")


def execute_notebook_worker(notebook_path: Path, repo_root: Path) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib

        matplotlib.use("Agg")
    except Exception:
        pass

    sys.path.insert(0, str(repo_root))
    src_path = repo_root / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    scope: dict[str, Any] = {
        "__name__": "__main__",
        "__file__": str(notebook_path),
    }

    def _display(*objects: Any, **_: Any) -> None:
        for obj in objects:
            print(repr(obj))

    class _DummyShell:
        def run_line_magic(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def run_cell_magic(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def system(self, *_args: Any, **_kwargs: Any) -> int:
            return 0

    scope["display"] = _display
    scope["get_ipython"] = lambda: _DummyShell()

    code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    for index, cell in enumerate(code_cells, start=1):
        cell_id = cell.get("id", f"cell-{index}")
        code = sanitize_code("".join(cell.get("source", [])))
        if not code.strip():
            continue
        compiled = compile(code, f"{notebook_path}#cell-{cell_id}", "exec")
        exec(compiled, scope)


def run_one_notebook(notebook_path: Path, repo_root: Path, timeout_seconds: int) -> dict[str, Any]:
    start = time.perf_counter()
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["IPYTHONDIR"] = str((repo_root / ".ipython_validation" / "runtime_smoke").resolve())

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        str(notebook_path),
        "--repo-root",
        str(repo_root),
    ]

    try:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
        duration = round(time.perf_counter() - start, 2)
        if completed.returncode == 0:
            return {
                "path": str(notebook_path.relative_to(repo_root)).replace("/", "\\"),
                "status": "passed",
                "duration_seconds": duration,
                "timeout_seconds": timeout_seconds,
                "error_type": None,
                "error_message": None,
            }

        error_message = trim_output(completed.stderr or completed.stdout)
        if not error_message:
            error_message = f"Notebook subprocess exited with code {completed.returncode}"
        return {
            "path": str(notebook_path.relative_to(repo_root)).replace("/", "\\"),
            "status": "failed",
            "duration_seconds": duration,
            "timeout_seconds": timeout_seconds,
            "error_type": "RuntimeError",
            "error_message": error_message,
        }
    except subprocess.TimeoutExpired as exc:
        duration = round(time.perf_counter() - start, 2)
        output = ""
        if exc.stdout:
            output += exc.stdout
        if exc.stderr:
            output += exc.stderr
        error_message = trim_output(output) or f"Notebook exceeded timeout of {timeout_seconds}s"
        return {
            "path": str(notebook_path.relative_to(repo_root)).replace("/", "\\"),
            "status": "timed_out",
            "duration_seconds": duration,
            "timeout_seconds": timeout_seconds,
            "error_type": "TimeoutExpired",
            "error_message": error_message,
        }


def run_controller(roots: list[str], report_path: Path | None) -> Path:
    repo_root = Path.cwd().resolve()
    notebook_paths = list_notebooks(roots)
    if not notebook_paths:
        raise FileNotFoundError(f"No notebooks found under: {roots}")

    results: list[dict[str, Any]] = []
    for index, notebook_path in enumerate(notebook_paths, start=1):
        timeout_seconds = timeout_for(notebook_path, repo_root)
        relative_path = notebook_path.relative_to(repo_root)
        print(
            f"[runtime-smoke] {index}/{len(notebook_paths)}  {relative_path}  (timeout={timeout_seconds}s)",
            flush=True,
        )
        result = run_one_notebook(notebook_path, repo_root, timeout_seconds)
        print(
            f"[runtime-smoke] -> {result['status']} in {result['duration_seconds']}s",
            flush=True,
        )
        results.append(result)

    generated_at = datetime.now().isoformat()
    report = {
        "generated_at": generated_at,
        "roots": roots,
        "mode": MODE,
        "total": len(results),
        "passed": sum(1 for item in results if item["status"] == "passed"),
        "failed": sum(1 for item in results if item["status"] == "failed"),
        "timed_out": sum(1 for item in results if item["status"] == "timed_out"),
        "results": results,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    final_report_path = report_path
    if final_report_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_report_path = REPORT_DIR / f"{REPORT_PREFIX}_{timestamp}.json"
    final_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[runtime-smoke] report: {final_report_path}", flush=True)
    return final_report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runtime smoke-check notebooks in this repository.")
    parser.add_argument("--roots", nargs="*", default=DEFAULT_ROOTS)
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--repo-root", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        notebook_path = Path(args.worker).resolve()
        repo_root = Path(args.repo_root).resolve() if args.repo_root else Path.cwd().resolve()
        try:
            execute_notebook_worker(notebook_path, repo_root)
        except Exception:
            traceback.print_exc()
            raise
        return

    report_path = Path(args.report_path).resolve() if args.report_path else None
    run_controller(list(args.roots), report_path)


if __name__ == "__main__":
    main()
