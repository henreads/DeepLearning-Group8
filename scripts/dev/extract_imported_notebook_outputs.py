from __future__ import annotations

import base64
import json
from pathlib import Path

import nbformat


REPO_ROOT = Path(__file__).resolve().parents[2]

NOTEBOOK_BRANCHES = [
    REPO_ROOT / "experiments" / "anomaly_detection" / "patchcore" / "efficientnet_b1" / "x240" / "one_layer",
    REPO_ROOT / "experiments" / "anomaly_detection" / "patchcore" / "efficientnet_b1" / "x240" / "layer3_5",
    REPO_ROOT / "experiments" / "anomaly_detection" / "patchcore" / "efficientnet_b1" / "x240" / "layer3_5_no_defect_tuning",
    REPO_ROOT / "experiments" / "anomaly_detection" / "patchcore" / "vit_b16" / "x224" / "one_layer_defect_tuning",
    REPO_ROOT / "experiments" / "anomaly_detection" / "patchcore" / "vit_b16" / "x224" / "one_layer_no_defect_tuning",
    REPO_ROOT / "experiments" / "anomaly_detection" / "patchcore" / "vit_b16" / "x224" / "two_block",
    REPO_ROOT / "experiments" / "anomaly_detection" / "patchcore" / "vit_b16" / "x224" / "two_block_no_defect_tuning",
]


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def sanitize_text(text: str) -> str:
    return (
        text.replace("→", "->")
        .replace("—", "-")
        .replace("–", "-")
    )


def extract_branch(branch_dir: Path) -> dict[str, object]:
    notebook_path = branch_dir / "notebook.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    artifact_root = branch_dir / "artifacts"
    legacy_output_root = artifact_root / "extracted_notebook_outputs"
    checkpoints_dir = artifact_root / "checkpoints"
    plots_dir = artifact_root / "plots"
    results_dir = artifact_root / "results"
    output_root = results_dir / "extracted_notebook_outputs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    extracted_files: list[str] = []
    checkpoint_mentions: list[str] = []

    for cell_index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        outputs = cell.get("outputs", [])
        if not outputs:
            continue

        for output_index, output in enumerate(outputs, start=1):
            record: dict[str, object] = {
                "cell_index": cell_index,
                "output_index": output_index,
                "output_type": output.get("output_type"),
                "files": [],
            }
            stem = f"cell_{cell_index:03d}_output_{output_index:02d}"

            if output.get("output_type") == "stream":
                path = output_root / f"{stem}.txt"
                text = sanitize_text(output.get("text", ""))
                write_text(path, text)
                record["files"].append(path.name)
                extracted_files.append(str(path.relative_to(branch_dir)).replace("\\", "/"))
                for line in text.splitlines():
                    if ".pt" in line or "Saved model artifact to:" in line or "checkpoint" in line.lower():
                        checkpoint_mentions.append(line)

            elif output.get("output_type") in {"display_data", "execute_result"}:
                data = output.get("data", {})

                if "image/png" in data:
                    path = plots_dir / f"{stem}.png"
                    write_bytes(path, base64.b64decode(data["image/png"]))
                    record["files"].append(path.name)
                    extracted_files.append(str(path.relative_to(branch_dir)).replace("\\", "/"))

                if "text/plain" in data:
                    path = output_root / f"{stem}.txt"
                    text_data = data["text/plain"]
                    if isinstance(text_data, list):
                        text_data = "".join(text_data)
                    text_data = sanitize_text(str(text_data))
                    write_text(path, text_data)
                    record["files"].append(path.name)
                    extracted_files.append(str(path.relative_to(branch_dir)).replace("\\", "/"))
                    for line in text_data.splitlines():
                        if ".pt" in line or "Saved model artifact to:" in line or "checkpoint" in line.lower():
                            checkpoint_mentions.append(line)

                if "text/html" in data:
                    path = output_root / f"{stem}.html"
                    html_data = data["text/html"]
                    if isinstance(html_data, list):
                        html_data = "".join(html_data)
                    write_text(path, str(html_data))
                    record["files"].append(path.name)
                    extracted_files.append(str(path.relative_to(branch_dir)).replace("\\", "/"))

            elif output.get("output_type") == "error":
                path = output_root / f"{stem}_error.txt"
                lines = output.get("traceback") or [f"{output.get('ename')}: {output.get('evalue')}"]
                write_text(path, "\n".join(lines))
                record["files"].append(path.name)
                extracted_files.append(str(path.relative_to(branch_dir)).replace("\\", "/"))

            if record["files"]:
                manifest.append(record)

    summary = {
        "branch": str(branch_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
        "notebook": str(notebook_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "artifact_root": str(artifact_root.relative_to(REPO_ROOT)).replace("\\", "/"),
        "checkpoints_dir": str(checkpoints_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
        "plots_dir": str(plots_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
        "results_dir": str(results_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
        "output_root": str(output_root.relative_to(REPO_ROOT)).replace("\\", "/"),
        "total_cells": len(notebook.cells),
        "extracted_output_entries": len(manifest),
        "extracted_files": extracted_files,
        "checkpoint_mentions": checkpoint_mentions,
        "manifest": manifest,
    }
    write_text(results_dir / "summary.json", json.dumps(summary, indent=2))
    write_text(output_root / "summary.json", json.dumps(summary, indent=2))
    checkpoint_note = (
        "No checkpoint binary was extracted from the notebook cell outputs.\n"
        "This folder only contains a note plus any checkpoint/model-path mentions found in the saved notebook text outputs.\n"
    )
    if checkpoint_mentions:
        checkpoint_note += "\nMentions found in notebook outputs:\n- " + "\n- ".join(checkpoint_mentions)
    write_text(checkpoints_dir / "MISSING_CHECKPOINT.txt", checkpoint_note)
    if legacy_output_root.exists():
        for path in legacy_output_root.iterdir():
            if path.is_file():
                path.unlink()
        legacy_output_root.rmdir()
    return summary


def update_readme(branch_dir: Path, summary: dict[str, object]) -> None:
    readme_path = branch_dir / "README.md"
    existing = readme_path.read_text(encoding="utf-8").rstrip() if readme_path.exists() else ""
    old_block = (
        "\n\nExtracted notebook outputs:\n"
        "- `artifacts/extracted_notebook_outputs/`: plots and text outputs extracted directly from the saved notebook cells\n"
        "- `artifacts/extracted_notebook_outputs/summary.json`: manifest of extracted files\n"
    )
    if old_block in existing:
        existing = existing.replace(old_block, "")
    addition = (
        "\n\nExtracted notebook outputs:\n"
        "- `artifacts/plots/`: PNG figures extracted directly from the saved notebook cells\n"
        "- `artifacts/results/extracted_notebook_outputs/`: text and HTML outputs extracted from the saved notebook cells\n"
        "- `artifacts/results/summary.json`: manifest of extracted files\n"
        "- `artifacts/checkpoints/MISSING_CHECKPOINT.txt`: note describing checkpoint mentions found in the notebook outputs\n"
    )
    if "artifacts/results/extracted_notebook_outputs/" not in existing:
        readme_path.write_text(existing + addition + "\n", encoding="utf-8")


def main() -> None:
    summaries = [extract_branch(branch_dir) for branch_dir in NOTEBOOK_BRANCHES]
    for branch_dir, summary in zip(NOTEBOOK_BRANCHES, summaries):
        update_readme(branch_dir, summary)

    out_path = REPO_ROOT / "experiments" / "anomaly_detection" / "patchcore" / "imported_notebook_output_extraction_summary.json"
    write_text(out_path, json.dumps(summaries, indent=2))
    print(f"Extracted outputs for {len(summaries)} imported notebooks.")


if __name__ == "__main__":
    main()
