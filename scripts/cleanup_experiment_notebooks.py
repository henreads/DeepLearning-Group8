from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


FLAG_NAMES = [
    "RUN_TRAINING",
    "RETRAIN",
    "FORCE_RETRAIN",
    "FORCE_RERUN",
    "FORCE_REBUILD_SCORES",
    "FORCE_RESCORE",
    "RUN_HOLDOUT",
    "RUN_HOLDOUT_EVALUATION",
    "FORCE_HOLDOUT_RERUN",
    "TRAIN_MISSING",
    "REGENERATE_UMAP",
    "RERUN_PROJECTION",
    "RERUN_SCORE_ABLATION",
    "RUN_SCORE_SWEEP",
    "RUN_DEFAULT_EVALUATION",
    "RUN_LOCAL_TRAINING",
    "RUN_VARIANT_SWEEP",
    "RUN_PSEUDOLABEL_INFERENCE",
    "RERUN",
    "FORCE_RERUN_UMAP",
]

FLAG_ASSIGNMENT_RE = re.compile(
    rf"(?m)^(?P<indent>\s*)(?P<name>{'|'.join(re.escape(name) for name in FLAG_NAMES)})\s*=\s*True\b"
)

INLINE_INSTRUCTION_COMMENT_RE = re.compile(
    r"(?m)^(?P<code>\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*(?:True|False|[A-Za-z_][A-Za-z0-9_]*))\s+#\s*(?:Set|Run with|Enable)\b.*$"
)

STANDALONE_INSTRUCTION_COMMENT_RE = re.compile(
    r"^\s*#\s*(?:Set|Run with|Enable|Controlled by)\b.*$", re.IGNORECASE
)

INLINE_TRUE_TAIL_RE = re.compile(
    r"(?P<lead>[.!?])\s+(?:Set|Enable|Run with)\s+[^.\n\"']*?=\s*True\b[^.\n\"']*\.?",
    re.IGNORECASE,
)

INLINE_FALSE_TAIL_RE = re.compile(
    r"(?P<lead>[.!?])\s+(?:Set|Enable|Run with)\s+[^.\n\"']*?=\s*False\b[^.\n\"']*\.?",
    re.IGNORECASE,
)

GENERAL_LINE_DROP_PATTERNS = [
    re.compile(r"^\s*[-*]?\s*Set\b.*\bTrue\b.*$", re.IGNORECASE),
    re.compile(r"^\s*[-*]?\s*Run with\b.*\bTrue\b.*$", re.IGNORECASE),
    re.compile(r"^\s*[-*]?\s*Enable\b.*\bTrue\b.*$", re.IGNORECASE),
    re.compile(r"^\s*Controlled by .*True/False", re.IGNORECASE),
    re.compile(r"^\s*Requires .*run all cells above first", re.IGNORECASE),
]

INLINE_FRAGMENT_DROP_PATTERNS = [
    re.compile(r"\s+Set\s+[A-Za-z_`][A-Za-z0-9_` ]*=\s*True\b[^.\n]*\.?", re.IGNORECASE),
    re.compile(r"\s+Set\s+[A-Za-z_`][A-Za-z0-9_` ]*=\s*False\b[^.\n]*\.?", re.IGNORECASE),
    re.compile(r"\s+Run with\s+[A-Za-z_`][A-Za-z0-9_` ]*=\s*True\b[^.\n]*\.?", re.IGNORECASE),
    re.compile(r"\s+Leave the notebook in review mode or set\s+[A-Za-z_`][A-Za-z0-9_` ]*=\s*True\b[^.\n]*\.?", re.IGNORECASE),
]

UNICODE_REPLACEMENTS = str.maketrans(
    {
        "\u00a0": " ",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u2192": "->",
        "\u2500": "-",
        "\u00d7": "x",
        "\u2248": "approx.",
        "\u2713": "",
        "\u2714": "",
        "\u26a0": "Warning",
        "\ufe0f": "",
    }
)

TEXT_REPLACEMENTS = [
    ("\u2714\ufe0f", ""),
    ("\u26a0\ufe0f", "Warning"),
]

EMOJI_RE = re.compile(r"[\u2705\U0001F680\U0001F449\U0001F642\U0001F3AF]")


def _load_text(source: object) -> tuple[str, bool]:
    if isinstance(source, list):
        return "".join(source), True
    return str(source), False


def _dump_text(text: str, was_list: bool) -> object:
    if was_list:
        return text.splitlines(keepends=True)
    return text


def _normalize_common_text(text: str, *, drop_inline_fragments: bool) -> str:
    for old, new in TEXT_REPLACEMENTS:
        text = text.replace(old, new)
    text = text.translate(UNICODE_REPLACEMENTS)
    text = EMOJI_RE.sub("", text)
    text = text.replace("??keep", "keep")
    text = INLINE_TRUE_TAIL_RE.sub(r"\g<lead>", text)
    text = INLINE_FALSE_TAIL_RE.sub(r"\g<lead>", text)
    if drop_inline_fragments:
        for pattern in INLINE_FRAGMENT_DROP_PATTERNS:
            text = pattern.sub("", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _clean_markdown(text: str) -> str:
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        line = _normalize_common_text(line, drop_inline_fragments=True)
        stripped = line.strip()
        if stripped and any(pattern.search(stripped) for pattern in GENERAL_LINE_DROP_PATTERNS):
            continue
        if "in the next cell" in stripped.lower():
            continue
        if "run all cells above first" in stripped.lower():
            continue
        cleaned_lines.append(line.rstrip())

    text = "\n".join(cleaned_lines).strip("\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _clean_code(text: str) -> str:
    text = _normalize_common_text(text, drop_inline_fragments=False)
    text = FLAG_ASSIGNMENT_RE.sub(r"\g<indent>\g<name> = False", text)
    text = INLINE_INSTRUCTION_COMMENT_RE.sub(r"\g<code>", text)

    cleaned_lines: list[str] = []
    for line in text.splitlines():
        if STANDALONE_INSTRUCTION_COMMENT_RE.match(line):
            continue
        cleaned_lines.append(line.rstrip())

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _clear_outputs(cell: dict) -> bool:
    changed = False
    if cell.get("cell_type") == "code":
        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True
    return changed


def clean_notebook(path: Path) -> bool:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    changed = False

    for cell in notebook.get("cells", []):
        changed |= _clear_outputs(cell)

        source, was_list = _load_text(cell.get("source", ""))
        if cell.get("cell_type") == "markdown":
            cleaned = _clean_markdown(source)
        elif cell.get("cell_type") == "code":
            cleaned = _clean_code(source)
        else:
            cleaned = source

        if cleaned != source:
            cell["source"] = _dump_text(cleaned, was_list)
            changed = True

    if changed:
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    return changed


def iter_target_notebooks(root: Path) -> list[Path]:
    notebooks = sorted(root.rglob("*.ipynb"))
    filtered: list[Path] = []
    for path in notebooks:
        path_posix = path.as_posix()
        if "experiments/anomaly_detection/patchcore/vit_b16/" in path_posix:
            continue
        filtered.append(path)
    return filtered


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply a cleanup pass to experiment notebooks.")
    parser.add_argument("--root", default="experiments", help="Notebook root to scan.")
    parser.add_argument("--dry-run", action="store_true", help="Report files that would change without rewriting them.")
    args = parser.parse_args()

    root = Path(args.root)
    notebooks = iter_target_notebooks(root)
    changed_paths: list[Path] = []

    for path in notebooks:
        original = path.read_text(encoding="utf-8")
        if args.dry_run:
            notebook = json.loads(original)
            changed = False
            for cell in notebook.get("cells", []):
                changed |= _clear_outputs(cell)
                source, was_list = _load_text(cell.get("source", ""))
                if cell.get("cell_type") == "markdown":
                    cleaned = _clean_markdown(source)
                elif cell.get("cell_type") == "code":
                    cleaned = _clean_code(source)
                else:
                    cleaned = source
                if cleaned != source:
                    cell["source"] = _dump_text(cleaned, was_list)
                    changed = True
            if changed:
                changed_paths.append(path)
            continue

        if clean_notebook(path):
            changed_paths.append(path)

    print(f"Scanned {len(notebooks)} notebooks.")
    print(f"Changed {len(changed_paths)} notebooks.")
    for path in changed_paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
