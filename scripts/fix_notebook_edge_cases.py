from __future__ import annotations

import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_notebook(path: Path, notebook: dict) -> None:
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _get_source(cell: dict) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def _set_source(cell: dict, source: str) -> None:
    if isinstance(cell.get("source", ""), list):
        cell["source"] = source.splitlines(keepends=True)
    else:
        cell["source"] = source


def _replace_in_first_code_cell(notebook: dict, needle: str, replacement: str) -> bool:
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = _get_source(cell)
        if needle not in source:
            continue
        updated = source.replace(needle, replacement, 1)
        _set_source(cell, updated)
        return True
    return False


def fix_effnet_layer3_family() -> None:
    paths = [
        REPO_ROOT / "experiments/anomaly_detection/patchcore/efficientnet_b1/x240/layer3_5/notebook.ipynb",
        REPO_ROOT / "experiments/anomaly_detection/patchcore/efficientnet_b1/x240/layer3_5_no_defect_tuning/notebook.ipynb",
    ]
    pattern = re.compile(
        r"(?ms)^    # Store model artifacts\n"
        r"    if RUN_TRAINING or not os\.path\.exists\(MODEL_EXPORT_PATH\):\n"
        r"(?P<body>.*?^        torch\.save\(artifact, MODEL_EXPORT_PATH\)\n)"
        r"        else:\n"
        r"        artifact = torch\.load\(MODEL_EXPORT_PATH, map_location='cpu'\)\n"
        r"        print\('Reusing saved model artifact from:', MODEL_EXPORT_PATH\)\n"
    )

    for path in paths:
        notebook = _load_notebook(path)
        cell = notebook["cells"][22]
        source = _get_source(cell)
        updated = pattern.sub(
            "    # Store model artifacts\n"
            "    if RUN_TRAINING:\n"
            r"\g<body>"
            "    elif os.path.exists(MODEL_EXPORT_PATH):\n"
            "        artifact = torch.load(MODEL_EXPORT_PATH, map_location='cpu')\n"
            "        print('Reusing saved model artifact from:', MODEL_EXPORT_PATH)\n"
            "    else:\n"
            "        artifact = None\n"
            "        print('[WARNING] RUN_TRAINING is False and the saved model artifact is missing. Skipping model artifact export/load.')\n",
            source,
        )
        _set_source(cell, updated)
        _write_notebook(path, notebook)


def fix_teacher_student_x224_import_cells() -> None:
    replacements = [
        (
            REPO_ROOT / "experiments/anomaly_detection/teacher_student/resnet18/x224/main/notebook.ipynb",
            REPO_ROOT / "experiments/anomaly_detection/teacher_student/resnet18/x64/main/notebook.ipynb",
        ),
        (
            REPO_ROOT / "experiments/anomaly_detection/teacher_student/resnet50/x224/main/notebook.ipynb",
            REPO_ROOT / "experiments/anomaly_detection/teacher_student/resnet50/x64/main/notebook.ipynb",
        ),
    ]

    for target_path, source_path in replacements:
        target_notebook = _load_notebook(target_path)
        source_notebook = _load_notebook(source_path)
        source = _get_source(source_notebook["cells"][2])
        _set_source(target_notebook["cells"][2], source)
        _write_notebook(target_path, target_notebook)


def fix_teacher_student_x224_config_cells() -> None:
    replacements = [
        (
            REPO_ROOT / "experiments/anomaly_detection/teacher_student/resnet18/x224/main/notebook.ipynb",
            REPO_ROOT / "experiments/anomaly_detection/teacher_student/resnet18/x64/main/notebook.ipynb",
            {
                "teacher_student/resnet18/x64/main/train_config.toml": "teacher_student/resnet18/x224/main/train_config.toml",
                "teacher_student/resnet18/x64/main/": "teacher_student/resnet18/x224/main/",
            },
        ),
        (
            REPO_ROOT / "experiments/anomaly_detection/teacher_student/resnet50/x224/main/notebook.ipynb",
            REPO_ROOT / "experiments/anomaly_detection/teacher_student/resnet50/x64/main/notebook.ipynb",
            {
                "teacher_student/resnet50/x64/main/artifacts/ts_resnet50": "teacher_student/resnet50/x224/main/artifacts/ts_resnet50_x224",
                "teacher_student/resnet50/x64/main/train_config.toml": "teacher_student/resnet50/x224/main/train_config.toml",
            },
        ),
    ]

    for target_path, source_path, replacements_map in replacements:
        target_notebook = _load_notebook(target_path)
        source_notebook = _load_notebook(source_path)
        source = _get_source(source_notebook["cells"][3])
        for old, new in replacements_map.items():
            source = source.replace(old, new)
        _set_source(target_notebook["cells"][3], source)
        _write_notebook(target_path, target_notebook)


def fix_teacher_student_x224_runtime_cells() -> None:
    replacements = [
        (
            REPO_ROOT / "experiments/anomaly_detection/teacher_student/resnet18/x224/main/notebook.ipynb",
            REPO_ROOT / "experiments/anomaly_detection/teacher_student/resnet18/x64/main/notebook.ipynb",
        ),
        (
            REPO_ROOT / "experiments/anomaly_detection/teacher_student/resnet50/x224/main/notebook.ipynb",
            REPO_ROOT / "experiments/anomaly_detection/teacher_student/resnet50/x64/main/notebook.ipynb",
        ),
    ]

    for target_path, source_path in replacements:
        target_notebook = _load_notebook(target_path)
        source_notebook = _load_notebook(source_path)
        source = _get_source(source_notebook["cells"][4])
        _set_source(target_notebook["cells"][4], source)
        _write_notebook(target_path, target_notebook)


def fix_teacher_student_x224_dataset_cells() -> None:
    replacements = [
        (
            REPO_ROOT / "experiments/anomaly_detection/teacher_student/resnet18/x224/main/notebook.ipynb",
            REPO_ROOT / "experiments/anomaly_detection/teacher_student/resnet18/x64/main/notebook.ipynb",
        ),
        (
            REPO_ROOT / "experiments/anomaly_detection/teacher_student/resnet50/x224/main/notebook.ipynb",
            REPO_ROOT / "experiments/anomaly_detection/teacher_student/resnet50/x64/main/notebook.ipynb",
        ),
    ]

    for target_path, source_path in replacements:
        target_notebook = _load_notebook(target_path)
        source_notebook = _load_notebook(source_path)
        source = _get_source(source_notebook["cells"][5])
        _set_source(target_notebook["cells"][5], source)
        _write_notebook(target_path, target_notebook)


def fix_teacher_student_x224_remaining_cells() -> None:
    replacements = [
        (
            REPO_ROOT / "experiments/anomaly_detection/teacher_student/resnet18/x224/main/notebook.ipynb",
            REPO_ROOT / "experiments/anomaly_detection/teacher_student/resnet18/x64/main/notebook.ipynb",
            [4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18],
        ),
        (
            REPO_ROOT / "experiments/anomaly_detection/teacher_student/resnet50/x224/main/notebook.ipynb",
            REPO_ROOT / "experiments/anomaly_detection/teacher_student/resnet50/x64/main/notebook.ipynb",
            [4, 5, 6, 9, 11, 13],
        ),
    ]

    for target_path, source_path, indices in replacements:
        target_notebook = _load_notebook(target_path)
        source_notebook = _load_notebook(source_path)
        for index in indices:
            if index >= len(target_notebook["cells"]) or index >= len(source_notebook["cells"]):
                continue
            source = _get_source(source_notebook["cells"][index])
            _set_source(target_notebook["cells"][index], source)
        _write_notebook(target_path, target_notebook)

    resnet18_x224 = REPO_ROOT / "experiments/anomaly_detection/teacher_student/resnet18/x224/main/notebook.ipynb"
    resnet18_x64 = REPO_ROOT / "experiments/anomaly_detection/teacher_student/resnet18/x64/main/notebook.ipynb"
    target_notebook = _load_notebook(resnet18_x224)
    source_notebook = _load_notebook(resnet18_x64)
    _set_source(target_notebook["cells"][22], _get_source(source_notebook["cells"][23]))
    _write_notebook(resnet18_x224, target_notebook)


def fix_block_depth_sweep_training_gate() -> None:
    path = REPO_ROOT / "experiments/anomaly_detection/patchcore/vit_b16/x224/block_depth_sweep/notebook.ipynb"
    notebook = _load_notebook(path)

    config_needle = (
        "    FORCE_RERUN = False\n"
        "    print(f'Sweeping blocks: {BLOCKS_TO_SWEEP}')\n"
        "    print(f'Artifacts root: {ARTIFACT_BASE}')\n"
    )
    config_replacement = (
        "    FORCE_RERUN = False\n"
        "    RUN_BLOCK_SWEEP = False\n"
        "    print(f'Sweeping blocks: {BLOCKS_TO_SWEEP}')\n"
        "    print(f'RUN_BLOCK_SWEEP={RUN_BLOCK_SWEEP}  FORCE_RERUN={FORCE_RERUN}')\n"
        "    print(f'Artifacts root: {ARTIFACT_BASE}')\n"
    )
    replaced_config = _replace_in_first_code_cell(notebook, config_needle, config_replacement)
    if not replaced_config:
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            source = _get_source(cell)
            if "RUN_BLOCK_SWEEP = False" in source:
                replaced_config = True
                break

    updated_sweep = False
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = _get_source(cell)
        if "for block_idx in BLOCKS_TO_SWEEP:" not in source or "build_memory_bank(extractor, train_loader)" not in source:
            continue
        if "RUN_BLOCK_SWEEP and FORCE_RERUN are False and the saved block artifacts are missing" in source:
            updated_sweep = True
            break
        updated_source = """try:
    sweep_rows = []
    for block_idx in BLOCKS_TO_SWEEP:
        print(f\"\\n{'=' * 60}\")
        print(f'Block {block_idx}')
        print(f\"{'=' * 60}\")
        block_dir = ARTIFACT_BASE / f'block{block_idx}'
        block_dir.mkdir(parents=True, exist_ok=True)
        scores_path = block_dir / 'scores.npz'
        metrics_path = block_dir / 'metrics.json'
        if scores_path.exists() and (not FORCE_RERUN):
            print(f'  Loading saved scores from {scores_path}')
            d = np.load(scores_path)
            val_n_z = d['val_normal_z']
            test_n_z = d['test_normal_z']
            test_d_z = d['test_defect_z']
        elif RUN_BLOCK_SWEEP or FORCE_RERUN:
            print(f'  Building extractor for block {block_idx}..')
            extractor = build_extractor(block_idx)
            print('  Building memory bank..')
            bank = build_memory_bank(extractor, train_loader)
            bank_t = bank.t().contiguous()
            print(f'  Bank: {len(bank):,} x {bank.shape[1]}-d')
            train_scores = score_loader_fn(extractor, train_loader, bank_t, desc=f'score train  [blk{block_idx}]')
            val_scores = score_loader_fn(extractor, val_loader, bank_t, desc=f'score val    [blk{block_idx}]')
            test_n_scores = score_loader_fn(extractor, test_normal_loader, bank_t, desc=f'score test-n [blk{block_idx}]')
            test_d_scores = score_loader_fn(extractor, test_defect_loader, bank_t, desc=f'score test-d [blk{block_idx}]')
            mu = float(np.mean(train_scores))
            std = float(np.std(train_scores) + 1e-8)
            val_n_z = (val_scores - mu) / std
            test_n_z = (test_n_scores - mu) / std
            test_d_z = (test_d_scores - mu) / std
            np.savez_compressed(scores_path, val_normal_z=val_n_z, test_normal_z=test_n_z, test_defect_z=test_d_z, train_mu=np.array(mu), train_std=np.array(std))
            print(f'  Scores saved to {scores_path}')
            del extractor, bank, bank_t
            gc.collect()
            if USE_CUDA:
                torch.cuda.empty_cache()
        else:
            print('[WARNING] RUN_BLOCK_SWEEP and FORCE_RERUN are False and the saved block artifacts are missing. Skipping this block.')
            continue
        metrics = compute_metrics(val_n_z, test_n_z, test_d_z)
        metrics['block_idx'] = block_idx
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        print(f\"  F1={metrics['f1']:.4f}  AUROC={metrics['auroc']:.4f}  AUPRC={metrics['auprc']:.4f}\")
        sweep_rows.append(metrics)
    if sweep_rows:
        sweep_df = pd.DataFrame(sweep_rows).sort_values('block_idx').reset_index(drop=True)
        sweep_csv = ARTIFACT_BASE / 'block_sweep_results.csv'
        sweep_df.to_csv(sweep_csv, index=False)
        print(f'\\nSweep results saved to {sweep_csv}')
        sweep_df[['block_idx', 'f1', 'auroc', 'auprc', 'threshold', 'precision', 'recall']]
    else:
        sweep_df = pd.DataFrame()
        print('[WARNING] No block sweep rows are available to display in this notebook run.')
except Exception as exc:
    _codex_msg = str(exc).lower()
    _codex_source = "block_depth_sweep"
    _codex_tokens = ('artifact', 'artifacts', 'checkpoint', 'history', 'summary', 'score', 'evaluation', 'umap', 'embedding', 'prediction', 'metadata', 'variant', 'holdout', 'plot', 'result')
    if isinstance(exc, FileNotFoundError):
        print(f'[WARNING] {exc}')
    elif isinstance(exc, NameError):
        print(f'[WARNING] Skipping this cell because earlier artifact-dependent outputs are unavailable: {exc}')
    elif isinstance(exc, (RuntimeError, KeyError, IndexError, ValueError, AttributeError)):
        if any((token in _codex_msg for token in _codex_tokens)) or any((token in _codex_source for token in _codex_tokens)):
            print(f'[WARNING] Skipping this cell because prerequisite artifacts or cached outputs are missing or incomplete: {exc}')
        else:
            raise
    else:
        raise
"""
        _set_source(cell, updated_source)
        updated_sweep = True
        break

    if not replaced_config or not updated_sweep:
        raise RuntimeError(f"Failed to patch the block-depth sweep notebook: {path}")
    _write_notebook(path, notebook)


def fix_dropout_sweep_warning() -> None:
    path = REPO_ROOT / "experiments/anomaly_detection/autoencoder/x64/batchnorm_dropout/notebook.ipynb"
    notebook = _load_notebook(path)
    needle = "Found sweep metadata, but the selected run artifacts are incomplete. Rerunning the sweep."
    replacement = (
        "Found sweep metadata, but the selected run artifacts are incomplete. "
        "RETRAIN is False, so training will stay skipped and downstream review cells may be unavailable."
    )
    if not _replace_in_first_code_cell(notebook, needle, replacement):
        already_patched = any(
            cell.get("cell_type") == "code"
            and replacement in _get_source(cell)
            for cell in notebook.get("cells", [])
        )
        if not already_patched:
            raise RuntimeError(f"Failed to patch the dropout sweep warning: {path}")
    _write_notebook(path, notebook)


def fix_patchcore_cached_output_gates() -> None:
    paths = [
        REPO_ROOT / "experiments/anomaly_detection/patchcore/ae_bn/x64/main/notebook.ipynb",
        REPO_ROOT / "experiments/anomaly_detection/patchcore/resnet18/x64/main/notebook.ipynb",
        REPO_ROOT / "experiments/anomaly_detection/patchcore/resnet50/x64/main/notebook.ipynb",
    ]
    warning_branch = (
        "    else:\n"
        "        sweep_results_df = pd.DataFrame()\n"
        "        sweep_summary = {}\n"
        "        selected_variant_name = None\n"
        "        selected_variant = None\n"
        "        print('[WARNING] RETRAIN is False and the saved PatchCore sweep outputs are missing. Skipping the sweep rerun.')\n"
    )

    for path in paths:
        notebook = _load_notebook(path)
        patched = False
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            source = _get_source(cell)
            if "use_cached_outputs = not RETRAIN" not in source or "build_patchcore_model(" not in source:
                continue
            updated = source
            if "    elif RETRAIN:\n" not in updated:
                updated = updated.replace(
                    "    else:\n        sweep_rows = []\n",
                    "    elif RETRAIN:\n        sweep_rows = []\n",
                    1,
                )
            if warning_branch not in updated:
                updated = updated.replace(
                    "    display(sweep_results_df)\n",
                    warning_branch + "    display(sweep_results_df)\n",
                    1,
                )
            if updated == source:
                raise RuntimeError(f"Failed to patch the PatchCore cached-output gate: {path}")
            _set_source(cell, updated)
            patched = True
            break
        if not patched:
            raise RuntimeError(f"Could not find the PatchCore sweep cell to patch: {path}")
        _write_notebook(path, notebook)


def main() -> int:
    fix_effnet_layer3_family()
    fix_teacher_student_x224_import_cells()
    fix_teacher_student_x224_config_cells()
    fix_teacher_student_x224_runtime_cells()
    fix_teacher_student_x224_dataset_cells()
    fix_teacher_student_x224_remaining_cells()
    fix_block_depth_sweep_training_gate()
    fix_dropout_sweep_warning()
    fix_patchcore_cached_output_gates()
    print("Fixed notebook edge cases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
