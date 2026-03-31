from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = REPO_ROOT / "REPORT.md"
NOTEBOOK_PATH = REPO_ROOT / "experiments/anomaly_detection/fastflow/x64/main/notebook.ipynb"


def print_report_window(start: int, end: int) -> None:
    lines = REPORT_PATH.read_text(encoding="utf-8").splitlines()
    print(f"--- REPORT {start}:{end} ---")
    for lineno in range(start, min(end, len(lines)) + 1):
        print(f"{lineno}: {lines[lineno - 1]}")


def print_notebook_cells() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    print("--- NOTEBOOK CELLS ---")
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        first_line = source.splitlines()[0] if source.splitlines() else ""
        print(f"{index}: {cell['cell_type']} : {first_line[:120]}")


if __name__ == "__main__":
    print_report_window(2018, 2115)
    print_notebook_cells()
