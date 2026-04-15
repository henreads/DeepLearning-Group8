from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from textwrap import indent
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parents[1]

CONTROL_FLAG_NAMES = [
    "RUN_TRAINING",
    "RETRAIN",
    "FORCE_RETRAIN",
    "FORCE_RERUN",
    "FORCE_RERUN_SWEEP",
    "FORCE_EVALUATION_RERUN",
    "RERUN_EVALUATION",
    "RUN_LOCAL_TRAINING",
    "RUN_DEFAULT_EVALUATION",
    "RUN_BLOCK_SWEEP",
    "RUN_SCORE_SWEEP",
    "RUN_VARIANT_SWEEP",
    "RUN_LATENT_DIM_SWEEP",
    "RUN_BETA_SWEEP",
    "TRAIN_MISSING",
    "FORCE_HOLDOUT_RERUN",
    "RUN_HOLDOUT",
    "RUN_HOLDOUT_EVALUATION",
    "FORCE_REBUILD_SCORES",
    "FORCE_RESCORE",
    "RERUN_SCORE_ABLATION",
    "RERUN_PROJECTION",
    "REGENERATE_UMAP",
    "FORCE_RERUN_UMAP",
    "RUN_PSEUDOLABEL_INFERENCE",
]

FLAG_ASSIGNMENT_RE = re.compile(
    rf"(?m)^(?P<indent>\s*)(?P<name>{'|'.join(re.escape(name) for name in CONTROL_FLAG_NAMES)})\s*=\s*True\b.*$"
)

DYNAMIC_FLAG_ASSIGNMENT_RE = re.compile(
    rf"(?m)^(?P<indent>\s*)(?P<name>{'|'.join(re.escape(name) for name in CONTROL_FLAG_NAMES)})\s*=\s*bool\(.+\)\s*$"
)

EXISTING_FLAG_RE = re.compile(rf"\b({'|'.join(re.escape(name) for name in CONTROL_FLAG_NAMES)})\b")

TRAINING_CELL_HINTS = (
    "optimizer.step(",
    "loss.backward(",
    "for epoch in range(",
    "train_and_evaluate_variant(",
    "scripts/train_",
    "train_loader",
)

ARTIFACT_RISK_TOKENS = (
    "artifact",
    "artifacts",
    "checkpoint",
    "history",
    "summary",
    "score",
    "evaluation",
    "umap",
    "embedding",
    "prediction",
    "metadata",
    "variant",
    "holdout",
    "plot",
    "result",
)

RISKY_SOURCE_MARKERS = (
    "raise FileNotFoundError",
    "torch.load(",
    "pd.read_csv(",
    "np.load(",
    "read_text(",
    "subprocess.run(",
    "stream_command(",
    "Popen(",
    "IPImage(",
    "Image(",
    "history_df",
    "metrics_path",
    "checkpoint_path",
    "CHECKPOINT_PATH",
    "summary_df",
    "score_df",
    "selected_variant",
    "val_scores_df",
    "test_scores_df",
    "threshold_sweep",
    "score_sweep",
    "defect_breakdown",
    "classification_report(",
    "confusion_matrix(",
    "history.plot(",
    "y_true",
    "y_pred",
    "y_prob",
    "test_loader",
    "val_loader",
    "artifacts_dir",
    "ARTIFACT_DIR",
    "results_dir",
    "EVALUATION_DIR",
    "UMAP_DIR",
    "PLOTS_DIR",
)


def _load_source(cell: dict) -> tuple[str, bool]:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source), True
    return str(source), False


def _dump_source(text: str, was_list: bool) -> object:
    if was_list:
        return text.splitlines(keepends=True)
    return text


def _code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


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


def _normalize_flag_assignments(source: str) -> str:
    source = FLAG_ASSIGNMENT_RE.sub(r"\g<indent>\g<name> = False", source)
    source = DYNAMIC_FLAG_ASSIGNMENT_RE.sub(r"\g<indent>\g<name> = False", source)
    return source


def _needs_inserted_training_flag(notebook: dict) -> bool:
    has_flag = False
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source, _ = _load_source(cell)
        if EXISTING_FLAG_RE.search(source):
            has_flag = True
    return not has_flag


def _ensure_training_flag_cell(notebook: dict) -> bool:
    if not _needs_inserted_training_flag(notebook):
        return False

    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") == "code":
            notebook["cells"].insert(
                index + 1,
                _code_cell(
                    "RUN_TRAINING = False\n"
                    "print(f'RUN_TRAINING = {RUN_TRAINING}')\n"
                ),
            )
            return True
    return False


def _is_flag_name(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id in CONTROL_FLAG_NAMES


def _extract_negated_expr(node: ast.AST) -> ast.expr | None:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return node.operand
    return None


def _build_warning_body() -> list[ast.stmt]:
    return ast.parse(
        "print('[WARNING] The rerun/training flags are False and the saved artifacts for this section are missing. Skipping this section.')"
    ).body


def _make_or_expr(values: list[ast.expr]) -> ast.expr:
    if len(values) == 1:
        return values[0]
    return ast.BoolOp(op=ast.Or(), values=values)


def _match_run_first_pattern(test: ast.AST) -> tuple[ast.expr, ast.expr] | None:
    if not isinstance(test, ast.BoolOp) or not isinstance(test.op, ast.Or):
        return None
    flags: list[ast.expr] = []
    ready_expr: ast.expr | None = None
    for value in test.values:
        if _is_flag_name(value):
            flags.append(value)
            continue
        negated = _extract_negated_expr(value)
        if negated is not None and ready_expr is None:
            ready_expr = negated
            continue
        return None
    if not flags or ready_expr is None:
        return None
    return _make_or_expr(flags), ready_expr


def _match_load_first_pattern(test: ast.AST) -> tuple[ast.expr, ast.expr] | None:
    if not isinstance(test, ast.BoolOp) or not isinstance(test.op, ast.And) or len(test.values) != 2:
        return None
    left, right = test.values
    if isinstance(left, ast.UnaryOp) and isinstance(left.op, ast.Not) and _is_flag_name(left.operand):
        return left.operand, right
    if isinstance(right, ast.UnaryOp) and isinstance(right.op, ast.Not) and _is_flag_name(right.operand):
        return right.operand, left
    return None


class ControlFlowTransformer(ast.NodeTransformer):
    def visit_If(self, node: ast.If) -> ast.AST:
        node = self.generic_visit(node)
        if not isinstance(node, ast.If) or not node.orelse:
            return node

        run_first = _match_run_first_pattern(node.test)
        if run_first is not None:
            flags_expr, ready_expr = run_first
            new_node = ast.If(
                test=flags_expr,
                body=node.body,
                orelse=[
                    ast.If(
                        test=ready_expr,
                        body=node.orelse,
                        orelse=_build_warning_body(),
                    )
                ],
            )
            return ast.fix_missing_locations(new_node)

        load_first = _match_load_first_pattern(node.test)
        if load_first is not None:
            flag_expr, _ready_expr = load_first
            new_node = ast.If(
                test=node.test,
                body=node.body,
                orelse=[
                    ast.If(
                        test=flag_expr,
                        body=node.orelse,
                        orelse=_build_warning_body(),
                    )
                ],
            )
            return ast.fix_missing_locations(new_node)

        return node


def _transform_control_flow(source: str) -> str:
    if "%%" in source or source.lstrip().startswith("!"):
        return source
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    transformed = ControlFlowTransformer().visit(tree)
    ast.fix_missing_locations(transformed)
    try:
        updated = ast.unparse(transformed)
    except Exception:
        return source
    if source.endswith("\n"):
        updated += "\n"
    return updated


def _should_wrap_for_missing_artifacts(source: str) -> bool:
    if source.lstrip().startswith("try:\n"):
        return False
    if "from __future__ import" in source:
        return False
    lowered = source.lower()
    return any(marker.lower() in lowered for marker in RISKY_SOURCE_MARKERS)


def _wrap_missing_artifact_cell(source: str) -> str:
    body = indent(source.rstrip() + "\n", "    ")
    artifact_tokens = ", ".join(repr(token) for token in ARTIFACT_RISK_TOKENS)
    source_preview = json.dumps(source.lower()[:4000])
    return (
        "try:\n"
        f"{body}"
        "except Exception as exc:\n"
        "    _codex_msg = str(exc).lower()\n"
        f"    _codex_source = {source_preview}\n"
        f"    _codex_tokens = ({artifact_tokens})\n"
        "    if isinstance(exc, FileNotFoundError):\n"
        "        print(f'[WARNING] {exc}')\n"
        "    elif isinstance(exc, NameError):\n"
        "        print(f'[WARNING] Skipping this cell because earlier artifact-dependent outputs are unavailable: {exc}')\n"
        "    elif isinstance(exc, (RuntimeError, KeyError, IndexError, ValueError, AttributeError)):\n"
        "        if any(token in _codex_msg for token in _codex_tokens) or any(token in _codex_source for token in _codex_tokens):\n"
        "            print(f'[WARNING] Skipping this cell because prerequisite artifacts or cached outputs are missing or incomplete: {exc}')\n"
        "        else:\n"
        "            raise\n"
        "    else:\n"
        "        raise\n"
    )


def _wrap_training_cell_without_flag(source: str) -> str:
    if "RUN_TRAINING" in source:
        return source
    body = indent(source.rstrip() + "\n", "    ")
    return (
        "if RUN_TRAINING:\n"
        f"{body}"
        "else:\n"
        "    print('[WARNING] RUN_TRAINING is False. Skipping this training section.')\n"
    )


def _patch_code_source(source: str, *, notebook_has_inserted_training_flag: bool) -> str:
    updated = _normalize_flag_assignments(source)
    updated = _transform_control_flow(updated)
    updated = re.sub(r"(?m)^(\s*)if RETRAIN or not artifacts_ready:\s*$", r"\1if RETRAIN:", updated)
    updated = re.sub(r"(?m)^(\s*)if FORCE_RETRAIN or not artifacts_ready:\s*$", r"\1if FORCE_RETRAIN:", updated)
    updated = re.sub(r"(?m)^(\s*)if RUN_TRAINING or not artifacts_ready:\s*$", r"\1if RUN_TRAINING:", updated)
    updated = re.sub(r"(?m)^(\s*)if RERUN_EVALUATION or not evaluation_ready:\s*$", r"\1if RERUN_EVALUATION:", updated)
    updated = re.sub(r"(?m)^(\s*)if FORCE_EVALUATION_RERUN or not evaluation_ready:\s*$", r"\1if FORCE_EVALUATION_RERUN:", updated)

    if notebook_has_inserted_training_flag and any(hint in updated for hint in TRAINING_CELL_HINTS):
        updated = _wrap_training_cell_without_flag(updated)

    if _should_wrap_for_missing_artifacts(updated):
        updated = _wrap_missing_artifact_cell(updated)

    updated = re.sub(r"\n{3,}", "\n\n", updated)
    return updated


def patch_notebook(path: Path) -> bool:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    changed = _ensure_training_flag_cell(notebook)
    inserted_training_flag = changed

    for cell in notebook.get("cells", []):
        changed |= _clear_outputs(cell)
        if cell.get("cell_type") != "code":
            continue
        source, was_list = _load_source(cell)
        updated = _patch_code_source(source, notebook_has_inserted_training_flag=inserted_training_flag)
        if updated != source:
            cell["source"] = _dump_source(updated, was_list)
            changed = True

    if changed:
        for cell in notebook.get("cells", []):
            cell.setdefault("id", uuid4().hex[:8])
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return changed


def main() -> int:
    notebooks = sorted(REPO_ROOT.rglob("*.ipynb"))
    changed_paths: list[Path] = []
    for path in notebooks:
        if patch_notebook(path):
            changed_paths.append(path)

    print(f"Scanned {len(notebooks)} notebooks.")
    print(f"Changed {len(changed_paths)} notebooks.")
    for path in changed_paths:
        print(path.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
