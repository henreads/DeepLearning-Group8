from __future__ import annotations

import json
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_notebook(path: Path, notebook: dict) -> None:
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def set_source(notebook: dict, index: int, source: str) -> None:
    notebook["cells"][index]["source"] = source.splitlines(keepends=True)


def wrap_cell_with_warning(notebook: dict, index: int, condition: str, warning: str) -> None:
    body = "".join(notebook["cells"][index]["source"])
    set_source(
        notebook,
        index,
        f"if not ({condition}):\n"
        f"    print('[WARNING] {warning}')\n"
        "else:\n"
        + textwrap.indent(body, "    "),
    )


def wrap_cells_with_warning(notebook: dict, indices: list[int], condition: str, warning: str) -> None:
    for index in indices:
        if index < len(notebook["cells"]):
            wrap_cell_with_warning(notebook, index, condition, warning)


def patch_autoencoder_x224(path: Path) -> None:
    notebook = load_notebook(path)
    set_source(
        notebook,
        6,
        textwrap.dedent(
            """
            def set_seed(seed: int) -> None:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

            def resolve_device(device_name: str) -> torch.device:
                if device_name == 'auto':
                    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                return torch.device(device_name)

            def save_figure(fig: plt.Figure, path: Path) -> Path:
                path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(path, bbox_inches='tight', dpi=150)
                print(f'Saved figure to {path}')
                return path

            def warn_skip(message: str) -> None:
                print(f'[WARNING] {message}')

            set_seed(int(config['run']['seed']))
            device = resolve_device(config['training']['device'])
            device
            """
        ).lstrip(),
    )
    set_source(
        notebook,
        8,
        textwrap.dedent(
            """
            metadata_path = REPO_ROOT / config['data']['metadata_csv']
            image_size = int(config['data'].get('image_size', 64))
            metadata_available = metadata_path.exists()
            metadata = pd.DataFrame()
            if metadata_available:
                metadata = pd.read_csv(metadata_path)
                display(metadata.head())
                display(metadata['split'].value_counts().rename_axis('split').to_frame('count'))
                display(metadata['is_anomaly'].value_counts().rename_axis('is_anomaly').to_frame('count'))
            else:
                warn_skip(f'Metadata CSV not found: {metadata_path}. Dataset-backed cells will be skipped unless cached artifacts already exist.')
            """
        ).lstrip(),
    )
    set_source(
        notebook,
        10,
        textwrap.dedent(
            """
            train_dataset = None
            val_dataset = None
            test_dataset = None
            train_loader = None
            val_loader = None
            test_loader = None
            if metadata_available:
                train_dataset = WaferMapDataset(metadata_path, split='train', image_size=image_size)
                val_dataset = WaferMapDataset(metadata_path, split='val', image_size=image_size)
                test_dataset = WaferMapDataset(metadata_path, split='test', image_size=image_size)
                train_loader = DataLoader(train_dataset, batch_size=int(config['data']['batch_size']), shuffle=True, num_workers=int(config['data']['num_workers']))
                val_loader = DataLoader(val_dataset, batch_size=int(config['data']['batch_size']), shuffle=False, num_workers=int(config['data']['num_workers']))
                test_loader = DataLoader(test_dataset, batch_size=int(config['data']['batch_size']), shuffle=False, num_workers=int(config['data']['num_workers']))
                print(f'train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}')
            else:
                warn_skip('WaferMapDataset/DataLoader construction is skipped because the metadata CSV is unavailable.')
            """
        ).lstrip(),
    )
    set_source(
        notebook,
        14,
        textwrap.dedent(
            """
            history = []
            output_dir = REPO_ROOT / config['run']['output_dir']
            output_dir.mkdir(parents=True, exist_ok=True)
            history_path = output_dir / 'results' / 'history.json'
            best_model_path = output_dir / 'checkpoints' / 'best_model.pt'
            resume_from = ''
            best_epoch = 0
            best_val_loss = float('nan')
            stale_epochs = 0
            best_state_dict = None
            training_ran = False
            evaluation_ready = False
            artifacts_ready = best_model_path.exists() and history_path.exists()
            if not RETRAIN and artifacts_ready:
                history = json.loads(history_path.read_text(encoding='utf-8'))
                best_checkpoint = torch.load(best_model_path, map_location=device)
                model.load_state_dict(best_checkpoint['model_state_dict'])
                best_epoch = int(best_checkpoint.get('best_epoch', best_checkpoint.get('epoch', 0)))
                best_val_loss = float(best_checkpoint.get('best_val_loss', float('nan')))
                stale_epochs = int(best_checkpoint.get('stale_epochs', 0))
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                print(f'Found existing artifacts in {output_dir}. Skipping training.')
            elif RETRAIN and train_loader is not None and val_loader is not None:
                warn_skip('RETRAIN=True remains available, but this runtime-fix pass only supports artifact-backed notebook execution.')
            elif RETRAIN:
                warn_skip('RETRAIN=True but the train/val datasets are unavailable in this notebook run.')
            else:
                warn_skip('Saved training artifacts are missing and RETRAIN is False. Skipping training-backed cells.')
            """
        ).lstrip(),
    )
    set_source(
        notebook,
        16,
        textwrap.dedent(
            """
            if not history and history_path.exists():
                history = json.loads(history_path.read_text(encoding='utf-8'))
            history_df = pd.DataFrame(history)
            if history_df.empty:
                warn_skip('Training curves are unavailable because history.json is missing or empty.')
            else:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(history_df['epoch'], history_df['train_loss'], marker='o', label='train')
                ax.plot(history_df['epoch'], history_df['val_loss'], marker='o', label='val')
                ax.set_title('Autoencoder Training Curve')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('MSE Loss')
                ax.grid(True, alpha=0.3)
                ax.legend()
                plt.tight_layout()
                save_figure(fig, output_dir / 'plots' / 'training_curves.png')
                plt.show()
                display(history_df.tail())
            """
        ).lstrip(),
    )
    set_source(
        notebook,
        20,
        textwrap.dedent(
            """
            best_model_path = output_dir / 'checkpoints' / 'best_model.pt'
            score_df = pd.DataFrame()
            if not best_model_path.exists():
                warn_skip(f'Best autoencoder checkpoint not found: {best_model_path}. Skipping evaluation-backed cells.')
                evaluation_ready = False
            elif test_loader is None:
                warn_skip('The test dataset is unavailable, so checkpoint-backed scoring is skipped for this cell.')
                evaluation_ready = False
            else:
                best_checkpoint = torch.load(best_model_path, map_location=device)
                model.load_state_dict(best_checkpoint['model_state_dict'])
                model.eval()

                def reconstruction_error(inputs: torch.Tensor, outputs: torch.Tensor, score_name: str=ANOMALY_SCORE_NAME) -> torch.Tensor:
                    if score_name == 'mse_mean':
                        return spatial_mean(squared_error_map(inputs, outputs))
                    if score_name == 'topk_abs_mean':
                        return topk_spatial_mean(absolute_error_map(inputs, outputs), topk_ratio=TOPK_RATIO)
                    raise ValueError(f'Unsupported score_name: {score_name}')

                test_scores = []
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs = inputs.to(device)
                        outputs = model(inputs)
                        scores = reconstruction_error(inputs, outputs, score_name=ANOMALY_SCORE_NAME).cpu().numpy()
                        labels = labels.cpu().numpy()
                        for score, label in zip(scores, labels):
                            test_scores.append({'score': float(score), 'is_anomaly': int(label)})
                score_df = pd.DataFrame(test_scores)
                evaluation_ready = not score_df.empty
                display(score_df.head())
            """
        ).lstrip(),
    )
    set_source(
        notebook,
        18,
        textwrap.dedent(
            """
            if training_ran:
                torch.save({'epoch': len(history), 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'config': config, 'best_epoch': best_epoch, 'best_val_loss': best_val_loss, 'stale_epochs': stale_epochs, 'history': history}, output_dir / 'checkpoints' / 'last_model.pt')
                if best_state_dict is not None:
                    torch.save({'epoch': best_epoch, 'model_state_dict': best_state_dict, 'optimizer_state_dict': optimizer.state_dict(), 'config': config, 'best_epoch': best_epoch, 'best_val_loss': best_val_loss, 'stale_epochs': stale_epochs, 'history': history}, output_dir / 'checkpoints' / 'best_model.pt')
                with history_path.open('w', encoding='utf-8') as handle:
                    json.dump(history, handle, indent=2)
                summary = {'best_epoch': best_epoch, 'best_val_loss': best_val_loss, 'epochs_ran': len(history), 'resumed_from': resume_from, 'training_ran': True}
                with (output_dir / 'results' / 'summary.json').open('w', encoding='utf-8') as handle:
                    json.dump(summary, handle, indent=2)
                print(f'Saved outputs to {output_dir}')
            else:
                print('Reused existing training artifacts; no training files were rewritten.')
                summary_path = output_dir / 'results' / 'summary.json'
                summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else {'best_epoch': best_epoch, 'best_val_loss': best_val_loss, 'epochs_ran': len(history), 'resumed_from': resume_from, 'training_ran': False}
            summary
            """
        ).lstrip(),
    )
    for index, warning in [
        (22, "Validation-threshold selection is unavailable because evaluation scores were not generated."),
        (24, "Test metrics are unavailable because evaluation scores or thresholds are missing."),
        (26, "Threshold sweep is unavailable because evaluation scores or thresholds are missing."),
        (28, "Score-distribution plotting is unavailable because evaluation scores or thresholds are missing."),
        (30, "Reconstruction examples are unavailable because evaluation scores or datasets are missing."),
        (32, "Failure-analysis tables are unavailable because evaluation scores or thresholds are missing."),
        (35, "Score ablation is unavailable because the saved evaluation artifacts are missing."),
        (36, "Failure-example rendering is unavailable because the analysis dataframe is empty."),
        (38, "Holdout evaluation is skipped because the main evaluation is unavailable or the holdout metadata CSV is missing."),
    ]:
        body = "".join(notebook["cells"][index]["source"])
        set_source(
            notebook,
            index,
            "if not evaluation_ready:\n"
            f"    warn_skip('{warning}')\n"
            "else:\n"
            + textwrap.indent(body, "    "),
        )
    save_notebook(path, notebook)


def patch_autoencoder_x128(path: Path) -> None:
    notebook = load_notebook(path)
    set_source(
        notebook,
        7,
        textwrap.dedent(
            """
            def set_seed(seed: int) -> None:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

            def resolve_device(device_name: str) -> torch.device:
                if device_name == 'auto':
                    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                return torch.device(device_name)

            def save_figure(fig: plt.Figure, path: Path) -> Path:
                path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(path, bbox_inches='tight', dpi=150)
                print(f'Saved figure to {path}')
                return path

            def warn_skip(message: str) -> None:
                print(f'[WARNING] {message}')

            set_seed(int(config['run']['seed']))
            device = resolve_device(config['training']['device'])
            device
            """
        ).lstrip(),
    )
    set_source(
        notebook,
        9,
        textwrap.dedent(
            """
            metadata_path = REPO_ROOT / config['data']['metadata_csv']
            image_size = int(config['data'].get('image_size', 64))
            metadata_available = metadata_path.exists()
            metadata = pd.DataFrame()
            if metadata_available:
                metadata = pd.read_csv(metadata_path)
                display(metadata.head())
                display(metadata['split'].value_counts().rename_axis('split').to_frame('count'))
                display(metadata['is_anomaly'].value_counts().rename_axis('is_anomaly').to_frame('count'))
            else:
                warn_skip(f'Metadata CSV not found: {metadata_path}. Dataset-backed cells will be skipped unless cached artifacts already exist.')
            """
        ).lstrip(),
    )
    set_source(
        notebook,
        11,
        textwrap.dedent(
            """
            train_dataset = None
            val_dataset = None
            test_dataset = None
            train_loader = None
            val_loader = None
            test_loader = None
            if metadata_available:
                train_dataset = WaferMapDataset(metadata_path, split='train', image_size=image_size)
                val_dataset = WaferMapDataset(metadata_path, split='val', image_size=image_size)
                test_dataset = WaferMapDataset(metadata_path, split='test', image_size=image_size)
                train_loader = DataLoader(train_dataset, batch_size=int(config['data']['batch_size']), shuffle=True, num_workers=int(config['data']['num_workers']))
                val_loader = DataLoader(val_dataset, batch_size=int(config['data']['batch_size']), shuffle=False, num_workers=int(config['data']['num_workers']))
                test_loader = DataLoader(test_dataset, batch_size=int(config['data']['batch_size']), shuffle=False, num_workers=int(config['data']['num_workers']))
                print(f'train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}')
            else:
                warn_skip('WaferMapDataset/DataLoader construction is skipped because the metadata CSV is unavailable.')
            """
        ).lstrip(),
    )
    set_source(
        notebook,
        15,
        textwrap.dedent(
            """
            history = []
            epochs = int(config['training']['epochs'])
            patience = int(config['training'].get('early_stopping_patience', 0))
            min_delta = float(config['training'].get('early_stopping_min_delta', 0.0))
            checkpoint_every = int(config['training'].get('checkpoint_every', 5))
            resume_from = str(config['training'].get('resume_from', '')).strip()
            best_val_loss = float('inf')
            best_epoch = 0
            best_state_dict = None
            stale_epochs = 0
            start_epoch = 0
            training_ran = False
            evaluation_ready = False
            output_dir = REPO_ROOT / config['run']['output_dir']
            output_dir.mkdir(parents=True, exist_ok=True)
            history_path = output_dir / 'results' / 'history.json'
            best_model_path = output_dir / 'checkpoints' / 'best_model.pt'
            artifacts_ready = best_model_path.exists() and history_path.exists()
            if not RETRAIN and artifacts_ready:
                with history_path.open('r', encoding='utf-8') as handle:
                    history = json.load(handle)
                best_checkpoint = torch.load(best_model_path, map_location=device)
                model.load_state_dict(best_checkpoint['model_state_dict'])
                best_epoch = int(best_checkpoint.get('best_epoch', best_checkpoint.get('epoch', 0)))
                best_val_loss = float(best_checkpoint.get('best_val_loss', float('nan')))
                stale_epochs = int(best_checkpoint.get('stale_epochs', 0))
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                print(f'Found existing artifacts in {output_dir}. Skipping training.')
            elif RETRAIN and train_loader is not None and val_loader is not None:
                warn_skip('RETRAIN=True remains available, but this runtime-fix pass only supports artifact-backed notebook execution.')
            elif RETRAIN:
                warn_skip('RETRAIN=True but the train/val datasets are unavailable in this notebook run.')
            else:
                warn_skip('Saved training artifacts are missing and RETRAIN is False. Skipping training-backed cells.')
            """
        ).lstrip(),
    )
    set_source(
        notebook,
        17,
        textwrap.dedent(
            """
            if not history and history_path.exists():
                with history_path.open('r', encoding='utf-8') as handle:
                    history = json.load(handle)
            history_df = pd.DataFrame(history)
            if history_df.empty:
                warn_skip('Training curves are unavailable because history.json is missing or empty.')
            else:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(history_df['epoch'], history_df['train_loss'], marker='o', label='train')
                ax.plot(history_df['epoch'], history_df['val_loss'], marker='o', label='val')
                ax.set_title('Autoencoder Training Curve')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('MSE Loss')
                ax.grid(True, alpha=0.3)
                ax.legend()
                plt.tight_layout()
                save_figure(fig, output_dir / 'plots' / 'training_curves.png')
                plt.show()
                display(history_df.tail())
            """
        ).lstrip(),
    )
    set_source(
        notebook,
        19,
        textwrap.dedent(
            """
            if training_ran:
                torch.save({'epoch': len(history), 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'config': config, 'best_epoch': best_epoch, 'best_val_loss': best_val_loss, 'stale_epochs': stale_epochs, 'history': history}, output_dir / 'checkpoints' / 'last_model.pt')
                if best_state_dict is not None:
                    torch.save({'epoch': best_epoch, 'model_state_dict': best_state_dict, 'optimizer_state_dict': optimizer.state_dict(), 'config': config, 'best_epoch': best_epoch, 'best_val_loss': best_val_loss, 'stale_epochs': stale_epochs, 'history': history}, output_dir / 'checkpoints' / 'best_model.pt')
                with history_path.open('w', encoding='utf-8') as handle:
                    json.dump(history, handle, indent=2)
                summary = {'best_epoch': best_epoch, 'best_val_loss': best_val_loss, 'epochs_ran': len(history), 'resumed_from': resume_from, 'training_ran': True}
                with (output_dir / 'results' / 'summary.json').open('w', encoding='utf-8') as handle:
                    json.dump(summary, handle, indent=2)
                print(f'Saved outputs to {output_dir}')
            else:
                print('Reused existing training artifacts; no training files were rewritten.')
                summary_path = output_dir / 'results' / 'summary.json'
                summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else {'best_epoch': best_epoch, 'best_val_loss': best_val_loss, 'epochs_ran': len(history), 'resumed_from': resume_from, 'training_ran': False}
            summary
            """
        ).lstrip(),
    )
    set_source(
        notebook,
        21,
        textwrap.dedent(
            """
            best_model_path = output_dir / 'checkpoints' / 'best_model.pt'
            score_df = pd.DataFrame()
            if not best_model_path.exists():
                warn_skip(f'Best autoencoder checkpoint not found: {best_model_path}. Skipping evaluation-backed cells.')
                evaluation_ready = False
            elif test_loader is None:
                warn_skip('The test dataset is unavailable, so checkpoint-backed scoring is skipped for this cell.')
                evaluation_ready = False
            else:
                best_checkpoint = torch.load(best_model_path, map_location=device)
                model.load_state_dict(best_checkpoint['model_state_dict'])
                print(f"Loaded best_model.pt from epoch {best_checkpoint.get('best_epoch', 'unknown')}")
                model.eval()

                def reconstruction_error(inputs: torch.Tensor, outputs: torch.Tensor, score_name: str=ANOMALY_SCORE_NAME) -> torch.Tensor:
                    if score_name == 'mse_mean':
                        return spatial_mean(squared_error_map(inputs, outputs))
                    if score_name == 'topk_abs_mean':
                        return topk_spatial_mean(absolute_error_map(inputs, outputs), topk_ratio=TOPK_RATIO)
                    raise ValueError(f'Unsupported score_name: {score_name}')

                test_scores = []
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs = inputs.to(device)
                        outputs = model(inputs)
                        scores = reconstruction_error(inputs, outputs, score_name=ANOMALY_SCORE_NAME).cpu().numpy()
                        labels = labels.cpu().numpy()
                        for score, label in zip(scores, labels):
                            test_scores.append({'score': float(score), 'is_anomaly': int(label)})
                score_df = pd.DataFrame(test_scores)
                evaluation_ready = not score_df.empty
                print({'evaluation_score': ANOMALY_SCORE_NAME, 'topk_ratio': TOPK_RATIO if ANOMALY_SCORE_NAME == 'topk_abs_mean' else None})
                display(score_df.head())
            """
        ).lstrip(),
    )
    wrap_cells_with_warning(
        notebook,
        [23, 25, 27, 29, 31, 33, 35],
        "evaluation_ready",
        "Autoencoder evaluation artifacts are unavailable for this notebook run.",
    )
    set_source(
        notebook,
        37,
        textwrap.dedent(
            """
            import os
            import subprocess
            score_ablation_config = config if 'config' in globals() else load_toml(CONFIG_PATH)
            score_ablation_output_root = REPO_ROOT / score_ablation_config['run']['output_dir']
            score_ablation_best_model_path = score_ablation_output_root / 'checkpoints' / 'best_model.pt'
            score_ablation_output_dir = score_ablation_output_root / 'results' / 'score_ablation'
            score_ablation_output_dir.mkdir(parents=True, exist_ok=True)
            score_ablation_csv_path = score_ablation_output_dir / 'score_summary.csv'
            score_ablation_json_path = score_ablation_output_dir / 'score_summary.json'
            score_ablation_ready = False
            if not score_ablation_best_model_path.exists():
                warn_skip(f'Best autoencoder checkpoint not found: {score_ablation_best_model_path}. Skipping score-ablation cells.')
            elif RERUN_SCORE_ABLATION:
                score_ablation_cmd = [sys.executable, 'scripts/evaluate_autoencoder_scores.py', '--checkpoint', str(score_ablation_best_model_path.relative_to(REPO_ROOT)), '--config', str(CONFIG_PATH.relative_to(REPO_ROOT)), '--output-dir', str(score_ablation_output_dir.relative_to(REPO_ROOT))]
                score_ablation_env = os.environ.copy()
                src_path = str(REPO_ROOT / 'src')
                score_ablation_env['PYTHONPATH'] = src_path if not score_ablation_env.get('PYTHONPATH') else src_path + os.pathsep + score_ablation_env['PYTHONPATH']
                print('Running:')
                print(' '.join(score_ablation_cmd))
                subprocess.run(score_ablation_cmd, cwd=REPO_ROOT, env=score_ablation_env, check=True)
                score_ablation_ready = score_ablation_csv_path.exists() and score_ablation_json_path.exists()
            elif score_ablation_csv_path.exists() and score_ablation_json_path.exists():
                print(f'Found existing score ablation outputs in {score_ablation_output_dir}. Skipping rerun.')
                score_ablation_ready = True
            else:
                warn_skip('The rerun/training flags are False and the saved score-ablation artifacts are missing. Skipping this section.')
            """
        ).lstrip(),
    )
    wrap_cells_with_warning(
        notebook,
        [39, 41],
        "score_ablation_ready",
        "Score-ablation outputs are unavailable because the saved artifacts are missing.",
    )
    save_notebook(path, notebook)


def patch_patchcore_selection_guard(path: Path) -> None:
    notebook = load_notebook(path)
    cell13 = "".join(notebook["cells"][13]["source"])
    cell13 = cell13.replace(
        "variant_outputs = {}\n",
        "variant_outputs = {}\nselected_result_ready = False\n",
    ).replace(
        "    variant_outputs[selected_variant_name] = selected_variant\n    print(f'Loaded cached PatchCore sweep results from {sweep_results_path}')\n",
        "    variant_outputs[selected_variant_name] = selected_variant\n    selected_result_ready = True\n    print(f'Loaded cached PatchCore sweep results from {sweep_results_path}')\n",
    ).replace(
        "    print(f'Finished rerunning PatchCore sweep. Selected variant: {selected_variant_name}')\n",
        "    selected_result_ready = True\n    print(f'Finished rerunning PatchCore sweep. Selected variant: {selected_variant_name}')\n",
    )
    set_source(notebook, 13, cell13)

    def wrap(index: int, warning: str) -> None:
        body = "".join(notebook["cells"][index]["source"])
        set_source(
            notebook,
            index,
            "if not selected_result_ready or selected_variant_name not in variant_outputs:\n"
            f"    print('[WARNING] {warning}')\n"
            "else:\n"
            + textwrap.indent(body, "    "),
        )

    wrap(15, "The selected PatchCore variant is unavailable because sweep artifacts were not loaded.")
    for index, warning in [
        (17, "PatchCore plots are unavailable because the selected sweep artifacts were not loaded."),
        (19, "PatchCore failure-analysis tables are unavailable because the selected sweep artifacts were not loaded."),
        (21, "PatchCore defect-breakdown plots are unavailable because the selected sweep artifacts were not loaded."),
        (23, "PatchCore render-artifact review is unavailable because the selected sweep artifacts were not loaded."),
        (25, "PatchCore saved-output summary is unavailable because the selected sweep artifacts were not loaded."),
    ]:
        if index < len(notebook["cells"]):
            wrap(index, warning)
    save_notebook(path, notebook)


def patch_fastflow_x64(path: Path) -> None:
    notebook = load_notebook(path)
    set_source(
        notebook,
        15,
        textwrap.dedent(
            """
            variant_results: dict[str, dict[str, Any]] = {}
            missing_variants: list[str] = []
            for variant in config['variants']:
                paths = variant_paths(OUTPUT_DIR, variant['name'])
                artifacts_ready = all((paths[key].exists() for key in ['history', 'scores', 'defect_breakdown', 'best_row']))
                if artifacts_ready and (not RETRAIN):
                    variant_results[variant['name']] = load_saved_variant_result(variant['name'])
                    continue
                if not RETRAIN and (not TRAIN_MISSING):
                    missing_variants.append(variant['name'])
                    continue
                result = train_and_evaluate_variant(variant=variant, config=config, train_ds=train_dataset, val_ds=val_dataset, test_ds=test_dataset)
                variant_results[variant['name']] = result
            if missing_variants:
                print('Missing variants were left untouched because artifact-first mode is enabled:')
                print(missing_variants)
                print('Set TRAIN_MISSING = True or RETRAIN = True if you want to train them.')
            fastflow_ready = bool(variant_results)
            if not fastflow_ready:
                print('[WARNING] No FastFlow variant artifacts were loaded. Training-backed cells will be skipped in this notebook run.')
                summary_df = pd.DataFrame()
            else:
                summary_df = pd.DataFrame([result['best'] for result in variant_results.values()]).sort_values(['f1', 'recall', 'precision'], ascending=False).reset_index(drop=True)
                summary_df.to_csv(RESULTS_DIR / 'fastflow_variant_summary.csv', index=False)
                run_manifest = {'best_variant': str(summary_df.iloc[0]['variant']), 'variants': list(summary_df['variant']), 'artifact_root': str(OUTPUT_DIR.relative_to(REPO_ROOT))}
                (RESULTS_DIR / 'run_manifest.json').write_text(json.dumps(run_manifest, indent=2), encoding='utf-8')
                display(summary_df)
            """
        ).lstrip(),
    )
    for index, warning in [
        (17, "Summary plots are unavailable because no FastFlow artifacts were loaded."),
        (19, "Qualitative heatmaps are unavailable because no FastFlow artifacts were loaded."),
    ]:
        body = "".join(notebook["cells"][index]["source"])
        set_source(
            notebook,
            index,
            "if not fastflow_ready:\n"
            f"    print('[WARNING] {warning}')\n"
            "else:\n"
            + textwrap.indent(body, "    "),
        )
    save_notebook(path, notebook)


def patch_ensemble_x64(path: Path) -> None:
    notebook = load_notebook(path)
    set_source(
        notebook,
        4,
        textwrap.dedent(
            """
            CONFIG = {
                'data': {
                    'metadata_csv': REPO_ROOT / 'data' / 'processed' / 'x64' / 'wm811k' / 'metadata_50k_5pct.csv',
                    'image_size': 64,
                    'batch_size': 16,
                    'num_workers': 0,
                },
                'patchcore': {
                    'name': 'PatchCore-WideRes50-topk-mb50k-r010',
                },
                'ts_res50': {
                    'name': 'TS-Res50-s2_a1_topk_mean_r0.20',
                    'artifact_dir': REPO_ROOT / 'experiments/anomaly_detection/teacher_student/resnet50/x64/main/artifacts/ts_resnet50',
                    'config_path': REPO_ROOT / 'experiments/anomaly_detection/teacher_student/resnet50/x64/main/train_config.toml',
                    'selected_variant_json': REPO_ROOT / 'experiments/anomaly_detection/teacher_student/resnet50/x64/main/artifacts/ts_resnet50/results/evaluation/selected_score_variant.json',
                    'converted_checkpoint_path': REPO_ROOT / 'experiments/anomaly_detection/teacher_student/resnet50/x64/main/artifacts/ts_resnet50/checkpoints/best_model.pt',
                    'raw_checkpoint_path': REPO_ROOT / 'experiments/anomaly_detection/teacher_student/resnet50/x64/main/artifacts/ts_resnet50/checkpoints/best_model.pt',
                },
                'ensemble': {
                    'threshold_quantile': 0.95,
                    'fusion_variants': [
                        {'name': 'equal_weight_mean', 'kind': 'weighted_mean', 'weights': {'patchcore': 1.0, 'ts_res50': 1.0}},
                        {'name': 'patchcore_heavy', 'kind': 'weighted_mean', 'weights': {'patchcore': 2.0, 'ts_res50': 1.0}},
                        {'name': 'ts_heavy', 'kind': 'weighted_mean', 'weights': {'patchcore': 1.0, 'ts_res50': 2.0}},
                        {'name': 'max', 'kind': 'max'},
                    ],
                },
            }

            def warn_skip(message: str) -> None:
                print(f'[WARNING] {message}')

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            output_dir = REPO_ROOT / 'experiments' / 'anomaly_detection' / 'ensemble' / 'x64' / 'score_ensemble' / 'artifacts'
            output_dir.mkdir(parents=True, exist_ok=True)

            def resolve_patchcore_score_paths(repo_root: Path) -> tuple[Path | None, Path | None]:
                preferred_root = repo_root / 'experiments/anomaly_detection/patchcore/wideresnet50/x64/main/artifacts/patchcore_wideresnet50_multilayer/topk_mb50k_r010/results/evaluation'
                legacy_candidates = sorted((repo_root / 'artifacts/x64').glob('*/topk_mb50k_r010'))
                for candidate in [preferred_root, *legacy_candidates]:
                    val_scores = candidate / 'val_scores.csv'
                    test_scores = candidate / 'test_scores.csv'
                    if val_scores.exists() and test_scores.exists():
                        return (val_scores, test_scores)
                return (None, None)

            patchcore_val_path, patchcore_test_path = resolve_patchcore_score_paths(REPO_ROOT)
            CONFIG['patchcore']['val_scores_csv'] = patchcore_val_path
            CONFIG['patchcore']['test_scores_csv'] = patchcore_test_path
            component_scores_ready = False
            component_val_df = pd.DataFrame()
            component_test_df = pd.DataFrame()
            print(f'Results dir: {output_dir}')
            """
        ).lstrip(),
    )
    set_source(
        notebook,
        5,
        textwrap.dedent(
            """
            metadata_path = CONFIG['data']['metadata_csv']
            component_val_path = output_dir / 'component_val_scores.csv'
            component_test_path = output_dir / 'component_test_scores.csv'

            if component_val_path.exists() and component_test_path.exists():
                component_val_df = pd.read_csv(component_val_path)
                component_test_df = pd.read_csv(component_test_path)
                component_scores_ready = not component_val_df.empty and not component_test_df.empty
                if component_scores_ready:
                    print(f'Loaded cached ensemble component scores from {output_dir}')
                    display(component_val_df.head())
                    display(component_test_df.head())
            else:
                patchcore_ready = metadata_path.exists() and CONFIG['patchcore']['val_scores_csv'] is not None and CONFIG['patchcore']['test_scores_csv'] is not None
                ts_cfg = CONFIG['ts_res50']
                ts_ready = ts_cfg['config_path'].exists() and ts_cfg['selected_variant_json'].exists() and (ts_cfg['converted_checkpoint_path'].exists() or ts_cfg['raw_checkpoint_path'].exists())
                if not metadata_path.exists():
                    warn_skip(f'Benchmark metadata is missing: {metadata_path}. Skipping ensemble computation for this notebook run.')
                elif not patchcore_ready:
                    warn_skip('PatchCore component score files are missing. Restore the WRN x64 PatchCore artifacts to enable this ensemble notebook.')
                elif not ts_ready:
                    warn_skip('Teacher-student component artifacts are missing. Restore the TS ResNet50 x64 artifacts to enable this ensemble notebook.')
                else:
                    metadata = pd.read_csv(metadata_path)
                    val_metadata = metadata[metadata['split'] == 'val'].reset_index(drop=True)
                    test_metadata = metadata[metadata['split'] == 'test'].reset_index(drop=True)
                    patchcore_val_df = pd.read_csv(CONFIG['patchcore']['val_scores_csv']).reset_index(drop=True)
                    patchcore_test_df = pd.read_csv(CONFIG['patchcore']['test_scores_csv']).reset_index(drop=True)
                    if len(patchcore_val_df) != len(val_metadata) or len(patchcore_test_df) != len(test_metadata):
                        warn_skip('PatchCore score CSV lengths do not match the metadata split sizes. Skipping ensemble computation.')
                    else:
                        def load_ts_checkpoint_for_local_eval(checkpoint_path: Path, config_path: Path, image_size: int):
                            config = load_toml(config_path)
                            model = build_ts_distillation_from_config(config, image_size=image_size)
                            checkpoint = torch.load(checkpoint_path, map_location='cpu')
                            remapped_state = dict(checkpoint['model_state_dict'])
                            if 'auto_map_scale' in remapped_state:
                                remapped_state['autoencoder_map_scale'] = remapped_state.pop('auto_map_scale')
                            layer_aliases = [('teacher.layer1.', 'teacher.layers.0.'), ('teacher.layer2.', 'teacher.layers.1.'), ('teacher.layer3.', 'teacher.layers.2.'), ('teacher.layer4.', 'teacher.layers.3.')]
                            for old_prefix, new_prefix in layer_aliases:
                                for key, value in list(remapped_state.items()):
                                    if key.startswith(old_prefix):
                                        remapped_state[new_prefix + key[len(old_prefix):]] = value
                            model.load_state_dict(remapped_state, strict=True)
                            converted_checkpoint = {'epoch': int(checkpoint.get('best_epoch', 0)), 'model_state_dict': model.state_dict(), 'config': config, 'best_epoch': int(checkpoint.get('best_epoch', 0)), 'best_val_loss': float(checkpoint.get('best_val_loss', 0.0)), 'history': checkpoint.get('history', []), 'student_map_scale': float(checkpoint.get('student_map_scale', checkpoint.get('train_summary', {}).get('student_map_scale', 1.0))), 'autoencoder_map_scale': float(checkpoint.get('auto_map_scale', checkpoint.get('train_summary', {}).get('auto_map_scale', 1.0)))}
                            return (converted_checkpoint, model)

                        ts_cfg = CONFIG['ts_res50']
                        if ts_cfg['converted_checkpoint_path'].exists():
                            converted_checkpoint = torch.load(ts_cfg['converted_checkpoint_path'], map_location='cpu')
                            ts_config = load_toml(ts_cfg['config_path'])
                            ts_model = build_ts_distillation_from_config(ts_config, image_size=int(ts_config['data'].get('image_size', 64)))
                            ts_model.load_state_dict(converted_checkpoint['model_state_dict'], strict=True)
                        else:
                            converted_checkpoint, ts_model = load_ts_checkpoint_for_local_eval(ts_cfg['raw_checkpoint_path'], ts_cfg['config_path'], image_size=CONFIG['data']['image_size'])
                            torch.save(converted_checkpoint, ts_cfg['converted_checkpoint_path'])

                        selected_variant = json.loads(ts_cfg['selected_variant_json'].read_text(encoding='utf-8'))
                        ts_config = load_toml(ts_cfg['config_path'])
                        ts_image_size = int(ts_config['data'].get('image_size', CONFIG['data']['image_size']))
                        ts_batch_size = int(ts_config['data'].get('batch_size', CONFIG['data']['batch_size']))
                        val_dataset = WaferMapDataset(metadata_path, split='val', image_size=ts_image_size)
                        test_dataset = WaferMapDataset(metadata_path, split='test', image_size=ts_image_size)
                        val_loader = DataLoader(val_dataset, batch_size=ts_batch_size, shuffle=False, num_workers=0)
                        test_loader = DataLoader(test_dataset, batch_size=ts_batch_size, shuffle=False, num_workers=0)
                        ts_model.to(device)
                        ts_model.eval()

                        def collect_normalized_maps(model, dataloader, device):
                            student_maps = []
                            auto_maps = []
                            labels = []
                            with torch.inference_mode():
                                for inputs, batch_labels in dataloader:
                                    inputs = inputs.to(device)
                                    student_map, auto_map = model.raw_anomaly_maps(inputs)
                                    student_maps.append((student_map / model.student_map_scale.clamp_min(1e-06)).cpu())
                                    auto_maps.append((auto_map / model.autoencoder_map_scale.clamp_min(1e-06)).cpu())
                                    labels.append(batch_labels.cpu())
                            return (torch.cat(student_maps, dim=0), torch.cat(auto_maps, dim=0), torch.cat(labels, dim=0).numpy())

                        def reduce_anomaly_map(anomaly_map, reduction, topk_ratio):
                            if reduction == 'mean':
                                return spatial_mean(anomaly_map)
                            if reduction == 'max':
                                return spatial_max(anomaly_map)
                            return topk_spatial_mean(anomaly_map, topk_ratio=topk_ratio)

                        val_student_maps, val_auto_maps, _ = collect_normalized_maps(ts_model, val_loader, device)
                        test_student_maps, test_auto_maps, _ = collect_normalized_maps(ts_model, test_loader, device)
                        student_weight = float(selected_variant['student_weight'])
                        auto_weight = float(selected_variant['auto_weight'])
                        reduction = str(selected_variant['reduction'])
                        topk_ratio_value = selected_variant.get('topk_ratio', np.nan)
                        topk_ratio = None if pd.isna(topk_ratio_value) else float(topk_ratio_value)
                        ts_val_scores = reduce_anomaly_map(student_weight * val_student_maps + auto_weight * val_auto_maps, reduction, topk_ratio)
                        ts_test_scores = reduce_anomaly_map(student_weight * test_student_maps + auto_weight * test_auto_maps, reduction, topk_ratio)
                        if hasattr(ts_val_scores, 'numpy'):
                            ts_val_scores = ts_val_scores.numpy()
                        if hasattr(ts_test_scores, 'numpy'):
                            ts_test_scores = ts_test_scores.numpy()
                        component_val_df = pd.DataFrame({'patchcore': patchcore_val_df['score'].to_numpy(), 'ts_res50': np.asarray(ts_val_scores).reshape(-1), 'is_anomaly': val_metadata['is_anomaly'].to_numpy().astype(int)})
                        component_test_df = pd.DataFrame({'patchcore': patchcore_test_df['score'].to_numpy(), 'ts_res50': np.asarray(ts_test_scores).reshape(-1), 'is_anomaly': test_metadata['is_anomaly'].to_numpy().astype(int)})
                        component_val_df.to_csv(component_val_path, index=False)
                        component_test_df.to_csv(component_test_path, index=False)
                        component_scores_ready = True
                        display(component_val_df.head())
                        display(component_test_df.head())
            """
        ).lstrip(),
    )
    wrap_cell_with_warning(
        notebook,
        6,
        "component_scores_ready",
        "Ensemble scoring is unavailable because the local PatchCore or teacher-student artifacts are missing.",
    )
    save_notebook(path, notebook)


def patch_vit_x224_main(path: Path) -> None:
    notebook = load_notebook(path)
    set_source(
        notebook,
        4,
        textwrap.dedent(
            """
            # ── Train or load ──────────────────────────────────────────────────────────────
            evaluation_ready = False
            summary = {}
            metrics = {}
            best_sweep = {}
            labels_z = np.array([], dtype=int)
            scores_z = np.array([], dtype=float)
            val_scores_df = pd.DataFrame()
            test_scores_df = pd.DataFrame()
            sweep_df = pd.DataFrame()
            breakdown = pd.DataFrame()

            if RETRAIN:
                print('[WARNING] RETRAIN=True remains available for this notebook, but this runtime-fix pass only supports artifact-backed execution.')
            else:
                _summary_path = OUTPUT_DIR / 'results' / 'summary.json'
                _val_scores_path = EVAL_DIR / 'val_scores.csv'
                _test_scores_path = EVAL_DIR / 'test_scores.csv'
                _threshold_sweep_path = EVAL_DIR / 'threshold_sweep.csv'
                _breakdown_path = EVAL_DIR / 'saved_defect_breakdown.csv'
                required_paths = [_summary_path, _val_scores_path, _test_scores_path, _threshold_sweep_path, _breakdown_path]
                if all((path.exists() for path in required_paths)):
                    summary = json.loads(_summary_path.read_text())
                    val_scores_df = pd.read_csv(_val_scores_path)
                    test_scores_df = pd.read_csv(_test_scores_path)
                    sweep_df = pd.read_csv(_threshold_sweep_path)
                    breakdown = pd.read_csv(_breakdown_path)
                    threshold_z = float(summary['threshold_z'])
                    threshold_raw = float(summary.get('threshold_raw', threshold_z))
                    best_sweep = summary.get('threshold_sweep_best', {})
                    labels_z = test_scores_df['is_anomaly'].to_numpy()
                    scores_z = test_scores_df['score'].to_numpy()
                    metrics = compute_threshold_metrics(labels_z, scores_z, threshold_z)
                    evaluation_ready = True
                else:
                    missing_paths = [str(path) for path in required_paths if not path.exists()]
                    print('[WARNING] Saved ViT-B/16 PatchCore evaluation artifacts are missing. Skipping training-backed cells in this notebook run.')
                    print(missing_paths)
            """
        ).lstrip(),
    )
    wrap_cell_with_warning(
        notebook,
        6,
        "evaluation_ready",
        "Main benchmark results are unavailable because the saved ViT PatchCore artifacts are missing.",
    )
    save_notebook(path, notebook)


def patch_vit_x64_main(path: Path) -> None:
    notebook = load_notebook(path)
    set_source(
        notebook,
        12,
        textwrap.dedent(
            """
            scores_path = RESULTS_DIR / 'scores.npz'
            MODEL_EXPORT_PATH = CHECKPOINTS_DIR / 'model.pt'
            LEGACY_MODEL_EXPORT_PATH = CHECKPOINTS_DIR / 'best_model.pt'
            scores_ready = False

            if scores_path.exists() and not RETRAIN:
                checkpoint_path = MODEL_EXPORT_PATH if MODEL_EXPORT_PATH.exists() else LEGACY_MODEL_EXPORT_PATH
                if checkpoint_path.exists():
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    if isinstance(checkpoint, dict) and 'extractor_state_dict' in checkpoint:
                        extractor.load_state_dict(checkpoint['extractor_state_dict'])
                        print(f'Loaded extractor weights from {checkpoint_path.name}')
                print(f'Loading saved scores from {scores_path}')
                data = np.load(scores_path)
                val_normal_scores_z = data['val_normal_z']
                test_normal_scores_z = data['test_normal_z']
                test_defect_scores_z = data['test_defect_z']
                mu = float(data['train_mu'])
                std = float(data['train_std'])
                scores_ready = True
            elif RETRAIN:
                print('[WARNING] RETRAIN=True remains available for this notebook, but this runtime-fix pass only supports artifact-backed execution.')
            else:
                print('[WARNING] RETRAIN is False and the saved scores are missing. Skipping score regeneration.')
            """
        ).lstrip(),
    )
    wrap_cells_with_warning(
        notebook,
        [14, 16, 18, 20, 24],
        "scores_ready",
        "Saved ViT x64 PatchCore score artifacts are unavailable for this notebook run.",
    )
    save_notebook(path, notebook)


def patch_vit_block_depth_sweep(path: Path) -> None:
    notebook = load_notebook(path)
    wrap_cell_with_warning(
        notebook,
        14,
        "not sweep_df.empty",
        "Block-depth sweep results are unavailable because no saved block artifacts were found.",
    )
    set_source(
        notebook,
        16,
        textwrap.dedent(
            """
            import shutil
            checkpoints_dir = ARTIFACT_BASE / 'checkpoints'
            results_dir = ARTIFACT_BASE / 'results'
            plots_dir = ARTIFACT_BASE / 'plots'
            for _path in [checkpoints_dir, results_dir, plots_dir]:
                _path.mkdir(parents=True, exist_ok=True)
            if sweep_df.empty:
                print('[WARNING] Block-depth summary artifacts are unavailable because the sweep dataframe is empty.')
            else:
                best_row = sweep_df.sort_values(['f1', 'auroc', 'auprc'], ascending=False).iloc[0]
                best_block = int(best_row['block_idx'])
                best_scores_path = ARTIFACT_BASE / f'block{best_block}' / 'scores.npz'
                best_metrics_path = ARTIFACT_BASE / f'block{best_block}' / 'metrics.json'
                with np.load(best_scores_path) as d:
                    test_normal_scores_z = d['test_normal_z']
                    test_defect_scores_z = d['test_defect_z']
                    val_normal_scores_z = d['val_normal_z']
                    mu = float(d['train_mu'])
                    std = float(d['train_std'])
                threshold_z = float(best_row['threshold'])
                threshold_raw = mu + threshold_z * std
                y_true = np.concatenate([np.zeros(len(test_normal_scores_z), dtype=int), np.ones(len(test_defect_scores_z), dtype=int)])
                y_pred = (np.concatenate([test_normal_scores_z, test_defect_scores_z]) > threshold_z).astype(int)
                test_scores_df = pd.concat([test_normal_df[['failure_label', 'is_anomaly']].assign(split='test_normal'), test_defect_df[['failure_label', 'is_anomaly']].assign(split='test_defect')], ignore_index=True)
                test_scores_df['score_z'] = np.concatenate([test_normal_scores_z, test_defect_scores_z])
                test_scores_df['predicted_anomaly'] = y_pred
                test_scores_df.to_csv(results_dir / 'test_scores.csv', index=False)
                defect_recall_df = test_scores_df.loc[test_scores_df['is_anomaly'] == 1].groupby('failure_label').agg(count=('predicted_anomaly', 'count'), detected=('predicted_anomaly', 'sum'), recall=('predicted_anomaly', 'mean'), mean_score=('score_z', 'mean')).reset_index()
                defect_recall_df.to_csv(results_dir / 'test_defect_recall.csv', index=False)
                score_columns = []
                for _block in sweep_df['block_idx'].astype(int).tolist():
                    with np.load(ARTIFACT_BASE / f'block{_block}' / 'scores.npz') as _d:
                        score_columns.append(np.concatenate([_d['test_normal_z'], _d['test_defect_z']]))
                umap_matrix = np.column_stack(score_columns)
                umap_df = test_scores_df[['failure_label', 'is_anomaly', 'split', 'score_z', 'predicted_anomaly']].copy()
                umap_df.insert(0, 'point_index', np.arange(len(umap_df)))
                if find_spec('umap.umap_') is None:
                    print('[WARNING] umap-learn is not installed, so the saved UMAP artifacts for this cell will be skipped.')
                else:
                    import umap.umap_ as umap
                    coords = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42, transform_seed=42, low_memory=True).fit_transform(umap_matrix)
                    umap_df['umap_1'] = coords[:, 0]
                    umap_df['umap_2'] = coords[:, 1]
                    umap_df.to_csv(results_dir / 'umap_test_embeddings.csv', index=False)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    normal_mask = umap_df['is_anomaly'].to_numpy() == 0
                    ax.scatter(coords[normal_mask, 0], coords[normal_mask, 1], s=8, alpha=0.45, c='steelblue', label='Normal', linewidths=0)
                    ax.scatter(coords[~normal_mask, 0], coords[~normal_mask, 1], s=8, alpha=0.6, c='tomato', label='Defect', linewidths=0)
                    ax.set_title('UMAP of Test Scores Across Sweep Blocks')
                    ax.set_xlabel('UMAP-1')
                    ax.set_ylabel('UMAP-2')
                    ax.legend()
                    fig.tight_layout()
                    fig.savefig(plots_dir / 'umap_test_embeddings.png', dpi=160, bbox_inches='tight')
                    plt.show()
                    plt.close(fig)
                artifact = {'selected_block': best_block, 'threshold_z': threshold_z, 'threshold_raw': threshold_raw, 'source_scores_path': str(best_scores_path), 'source_metrics_path': str(best_metrics_path), 'block_results_path': str(ARTIFACT_BASE / 'block_sweep_results.csv')}
                torch.save(artifact, checkpoints_dir / 'model.pt')
                (results_dir / 'evaluation_metrics.json').write_text(json.dumps(artifact | json.loads(best_metrics_path.read_text(encoding='utf-8')), indent=2), encoding='utf-8')
            """
        ).lstrip(),
    )
    save_notebook(path, notebook)


def patch_saved_artifact_vit_notebook(path: Path, *, setup_index: int, score_index: int, later_indices: list[int]) -> None:
    notebook = load_notebook(path)
    setup_source = "".join(notebook["cells"][setup_index]["source"])
    for old in [
        "if not RUN_TRAINING and (not MODEL_ARTIFACT_EXISTS):\n    raise FileNotFoundError(f'Saved model artifact not found at {MODEL_LOAD_PATH}. ')\n",
        "if not RUN_TRAINING and (not SCORES_ARTIFACT_EXISTS):\n    raise FileNotFoundError(f'Saved score bundle not found at {SCORES_EXPORT_PATH}. ')\n",
        "if not RUN_TRAINING and (not MODEL_ARTIFACT_EXISTS):\n    raise FileNotFoundError(f'Saved model artifact not found at {MODEL_EXPORT_PATH}. ')\n",
        "if not RUN_TRAINING and (not SCORES_ARTIFACT_EXISTS):\n    raise FileNotFoundError(f'Saved score bundle not found at {SCORES_EXPORT_PATH}. ')\n",
    ]:
        setup_source = setup_source.replace(old, "")
    setup_source += "\nARTIFACTS_READY = bool(MODEL_ARTIFACT_EXISTS and SCORES_ARTIFACT_EXISTS)\n"
    setup_source += "if not RUN_TRAINING and not ARTIFACTS_READY:\n    print('[WARNING] Saved model or score artifacts are missing. Artifact-backed sections will be skipped in this notebook run.')\n"
    set_source(notebook, setup_index, setup_source)

    score_source = "".join(notebook["cells"][score_index]["source"])
    set_source(
        notebook,
        score_index,
        "if not ARTIFACTS_READY and not RUN_TRAINING:\n"
        "    print('[WARNING] Saved model or score artifacts are missing. Skipping score-loading and evaluation setup for this cell.')\n"
        "else:\n"
        + textwrap.indent(score_source, "    "),
    )
    wrap_cells_with_warning(
        notebook,
        later_indices,
        "ARTIFACTS_READY or RUN_TRAINING",
        "Saved model or score artifacts are unavailable for this notebook run.",
    )
    save_notebook(path, notebook)


def patch_ft_mae_notebook(path: Path, *, finetune_index: int, downstream_indices: list[int]) -> None:
    notebook = load_notebook(path)
    finetune_source = "".join(notebook["cells"][finetune_index]["source"])
    finetune_source = finetune_source.replace(
        "else:\n    if not os.path.exists(FINETUNE_CKPT):\n        raise FileNotFoundError",
        "else:\n    if not os.path.exists(FINETUNE_CKPT):\n        print(f'[WARNING] No saved checkpoint at {FINETUNE_CKPT}. Fine-tuning-backed cells will be skipped in this notebook run.')\n        finetune_ready = False\n    else:\n        finetune_ready = True\n        # preserved below",
    )
    if "finetune_ready" not in finetune_source:
        finetune_source = "finetune_ready = False\n" + finetune_source
    finetune_source = finetune_source.replace(
        "    ckpt = torch.load(FINETUNE_CKPT, map_location='cpu')\n",
        "        ckpt = torch.load(FINETUNE_CKPT, map_location='cpu')\n",
    )
    finetune_source = finetune_source.replace(
        "    if isinstance(ckpt, dict) and 'vit_state_dict' in ckpt:\n",
        "        if isinstance(ckpt, dict) and 'vit_state_dict' in ckpt:\n",
    )
    finetune_source = finetune_source.replace(
        "        mae_model.vit.load_state_dict(ckpt['vit_state_dict'])\n        history_loss = ckpt.get('training_history', None)\n    else:\n        mae_model.vit.load_state_dict(ckpt)\n        history_loss = None\n    print(f'Loaded fine-tuned backbone from: {FINETUNE_CKPT}')\n    if history_loss is not None:\n        print(f'Training history loaded: {len(history_loss)} epochs, final loss={history_loss[-1]:.6f}')\n    else:\n        print('No training history in checkpoint (saved without history).')\n",
        "            mae_model.vit.load_state_dict(ckpt['vit_state_dict'])\n            history_loss = ckpt.get('training_history', None)\n        else:\n            mae_model.vit.load_state_dict(ckpt)\n            history_loss = None\n        print(f'Loaded fine-tuned backbone from: {FINETUNE_CKPT}')\n        if history_loss is not None:\n            print(f'Training history loaded: {len(history_loss)} epochs, final loss={history_loss[-1]:.6f}')\n        else:\n            print('No training history in checkpoint (saved without history).')\n",
    )
    set_source(notebook, finetune_index, finetune_source)
    wrap_cells_with_warning(
        notebook,
        downstream_indices,
        "finetune_ready or RETRAIN",
        "Fine-tuned checkpoint artifacts are unavailable for this notebook run.",
    )
    save_notebook(path, notebook)


def patch_patchcore_wrn_x64_main(path: Path) -> None:
    notebook = load_notebook(path)
    set_source(
        notebook,
        4,
        textwrap.dedent(
            """
            CONFIG = {
                "run": {"output_dir": OUTPUT_DIR, "seed": SEED},
                "data": {"image_size": IMAGE_SIZE, "normal_limit": NORMAL_LIMIT, "test_defect_fraction": TEST_DEFECT_FRACTION, "batch_size": BATCH_SIZE, "num_workers": NUM_WORKERS},
                "model": {"backbone_type": "wideresnet50_2", "teacher_layers": TEACHER_LAYERS, "pretrained": PRETRAINED, "freeze_backbone": FREEZE_BACKBONE, "backbone_input_size": TEACHER_INPUT_SIZE, "normalize_imagenet": NORMALIZE_IMAGENET, "query_chunk_size": QUERY_CHUNK_SIZE, "memory_chunk_size": MEMORY_CHUNK_SIZE},
                "scoring": {"threshold_quantile": THRESHOLD_QUANTILE},
                "sweep_variants": SWEEP_VARIANTS,
            }

            RETRAIN = False

            set_seed(SEED)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if DEVICE == "auto" else resolve_device(DEVICE)
            output_dir = Path(OUTPUT_DIR)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "config.json").write_text(json.dumps(CONFIG, indent=2), encoding="utf-8")
            sweep_results_csv = output_dir / "patchcore_sweep_results.csv"
            follow_up_csv = output_dir / "patchcore_follow_up_sweep_results.csv"
            dataset_ready = False
            sweep_ready = False
            follow_up_ready = False
            if RETRAIN:
                print('[WARNING] RETRAIN=True remains available for this notebook, but this runtime-fix pass only supports artifact-backed execution.')
            elif sweep_results_csv.exists():
                sweep_results_df = pd.read_csv(sweep_results_csv)
                best_row = sweep_results_df.iloc[0].to_dict()
                sweep_ready = not sweep_results_df.empty
                print(f"Loaded saved main-sweep results from {sweep_results_csv}")
                if follow_up_csv.exists():
                    follow_up_results_df = pd.read_csv(follow_up_csv)
                    follow_up_ready = not follow_up_results_df.empty
                    print(f"Loaded saved follow-up results from {follow_up_csv}")
                else:
                    follow_up_results_df = pd.DataFrame()
                print(f"Using device: {device}")
            else:
                print('[WARNING] Saved WRN50 PatchCore sweep artifacts are missing. Training-backed cells will be skipped in this notebook run.')
                sweep_results_df = pd.DataFrame()
                follow_up_results_df = pd.DataFrame()
            """
        ).lstrip(),
    )
    wrap_cells_with_warning(
        notebook,
        [5, 6, 7, 9],
        "sweep_ready",
        "Saved WRN50 PatchCore sweep artifacts are unavailable for this notebook run.",
    )
    save_notebook(path, notebook)


def patch_rd4ad_x224(path: Path) -> None:
    notebook = load_notebook(path)
    set_source(
        notebook,
        6,
        textwrap.dedent(
            """
            from wafer_defect.config import load_toml
            from wafer_defect.data.wm811k import WaferMapDataset
            config = load_toml(str(TRAIN_CONFIG))
            image_size = int(config['data'].get('image_size', 224))
            batch_size = int(config['data'].get('batch_size', 64))
            threshold_quantile = float(config['scoring'].get('threshold_quantile', 0.95))

            def resolve_device(name: str) -> torch.device:
                if name == 'auto':
                    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                return torch.device(name)

            device = resolve_device(DEVICE_NAME)
            metadata_ready = METADATA_PATH.exists()
            evaluation_ready = False
            if metadata_ready:
                metadata = pd.read_csv(METADATA_PATH)
                display(metadata['split'].value_counts().rename_axis('split').to_frame('count'))
            else:
                metadata = pd.DataFrame()
                print(f'[WARNING] Benchmark metadata is missing: {METADATA_PATH}. RD4AD evaluation cells will be skipped unless saved artifacts already exist.')
            print(f'Device: {device}  |  image_size: {image_size}  |  threshold_quantile: {threshold_quantile}')
            """
        ).lstrip(),
    )
    set_source(
        notebook,
        8,
        textwrap.dedent(
            """
            from wafer_defect.models.rd4ad import build_rd4ad_from_config
            model = None
            if CHECKPOINT_PATH.exists():
                model = build_rd4ad_from_config(config).to(device)
                checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                best_epoch = checkpoint.get('best_epoch', '?')
                best_val_loss = checkpoint.get('best_val_loss', float('nan'))
                print(f'Loaded checkpoint | best_epoch={best_epoch} | best_val_loss={best_val_loss:.6f}')
                model.eval()
                trainable = sum((p.numel() for p in model.parameters() if p.requires_grad))
                total = sum((p.numel() for p in model.parameters()))
                print(f'Trainable parameters: {trainable:,} / {total:,}')
            else:
                print(f'[WARNING] No checkpoint found at {CHECKPOINT_PATH}. Checkpoint-backed RD4AD cells will be skipped in this notebook run.')
            """
        ).lstrip(),
    )
    set_source(
        notebook,
        12,
        textwrap.dedent(
            """
            @torch.no_grad()
            def infer_scores(split: str) -> tuple[np.ndarray, np.ndarray]:
                dataset = WaferMapDataset(str(METADATA_PATH), split=split, image_size=image_size)
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
                all_scores, all_labels = ([], [])
                for images, labels in loader:
                    images = images.to(device)
                    scores = model(images).cpu().numpy()
                    all_scores.append(scores)
                    all_labels.append(labels.numpy())
                return (np.concatenate(all_labels), np.concatenate(all_scores))

            if model is None or not metadata_ready:
                print('[WARNING] RD4AD checkpoint-backed evaluation is unavailable because the checkpoint or metadata CSV is missing.')
            else:
                val_labels, val_scores = infer_scores('val')
                test_labels, test_scores = infer_scores('test')
                evaluation_ready = True
                print(f'Val  : {len(val_labels):,} samples  |  anomaly rate={val_labels.mean():.3%}')
                print(f'Test : {len(test_labels):,} samples  |  anomaly rate={test_labels.mean():.3%}')
            """
        ).lstrip(),
    )
    wrap_cells_with_warning(
        notebook,
        [14, 16, 18, 20, 22, 24, 26],
        "evaluation_ready",
        "RD4AD evaluation artifacts are unavailable because the checkpoint or metadata CSV is missing.",
    )
    save_notebook(path, notebook)


def patch_report_figures(path: Path) -> None:
    notebook = load_notebook(path)
    set_source(
        notebook,
        1,
        textwrap.dedent(
            """
            from pathlib import Path
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            cwd = Path.cwd().resolve()
            candidate_roots = [cwd, *cwd.parents]
            PROJ_ROOT = next((candidate for candidate in candidate_roots if (candidate / 'experiments').exists() and (candidate / 'src' / 'wafer_defect').exists()), cwd)
            print(f'Project root: {PROJ_ROOT}')
            OUTPUT_DIR = PROJ_ROOT / 'experiments' / 'anomaly_detection' / 'report_figures' / 'artifacts' / 'plots'
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            print(f'Saving plots to: {OUTPUT_DIR}')
            plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False})
            """
        ).lstrip(),
    )
    save_notebook(path, notebook)


def patch_vit_import_cells() -> None:
    replacements = {
        "experiments/anomaly_detection/patchcore/vit_b16/x224/block_depth_sweep/notebook.ipynb": (
            2,
            "deps = [('timm', 'timm'), ('tqdm', 'tqdm'), ('scikit-learn', 'sklearn')]",
        ),
        "experiments/anomaly_detection/patchcore/vit_b16/x64/main/notebook.ipynb": (
            2,
            "deps = [('timm', 'timm'), ('tqdm', 'tqdm'), ('scikit-learn', 'sklearn')]",
        ),
        "experiments/anomaly_detection/patchcore/dinov2_vit_b14/x224/notebook.ipynb": (
            2,
            "deps = [('timm', 'timm'), ('tqdm', 'tqdm')]",
        ),
        "experiments/anomaly_detection/patchcore/dinov2_vit_b14/x224/ensemble.ipynb": (
            2,
            "deps = [('timm', 'timm'), ('tqdm', 'tqdm')]",
        ),
        "experiments/anomaly_detection/patchcore/vit_b16/x224/one_layer_defect_tuning/notebook.ipynb": (
            2,
            "deps = [('timm', 'timm'), ('tqdm', 'tqdm')]",
        ),
        "experiments/anomaly_detection/patchcore/vit_b16/x224/one_layer_no_defect_tuning/notebook.ipynb": (
            2,
            "deps = [('timm', 'timm'), ('tqdm', 'tqdm')]",
        ),
        "experiments/anomaly_detection/patchcore/vit_b16/x224/two_block/notebook.ipynb": (
            2,
            "deps = [('timm', 'timm'), ('tqdm', 'tqdm')]",
        ),
        "experiments/anomaly_detection/patchcore/vit_b16/x224/two_block_no_defect_tuning/notebook.ipynb": (
            2,
            "deps = [('timm', 'timm'), ('tqdm', 'tqdm')]",
        ),
        "experiments/anomaly_detection/patchcore/vit_b16/x224/FT/MAE/wafer-trained-patchcore-vit-best.ipynb": (
            2,
            "deps = [('timm', 'timm'), ('tqdm', 'tqdm')]",
        ),
        "experiments/anomaly_detection/patchcore/vit_b16/x224/FT/MAE_25pct/wafer-trained-patchcore-vit-25pct-MAE.ipynb": (
            2,
            "deps = [('timm', 'timm'), ('tqdm', 'tqdm')]",
        ),
    }
    for rel, (index, deps_line) in replacements.items():
        notebook = load_notebook(REPO_ROOT / rel)
        set_source(
            notebook,
            index,
            textwrap.dedent(
                f"""
                from importlib.util import find_spec
                import sys
                {deps_line}
                missing = [pkg for pkg, module_name in deps if find_spec(module_name) is None]
                DEPS_READY = not missing
                if DEPS_READY:
                    print('deps ready')
                else:
                    print(f'[WARNING] Missing dependencies: {{missing}}. This notebook run will skip dependency-backed cells.')
                """
            ).lstrip(),
        )
        save_notebook(REPO_ROOT / rel, notebook)


def patch_supervised_cnn(path: Path, *, subdir: str, artifact_leaf: str, include_half: bool) -> None:
    notebook = load_notebook(path)
    set_source(
        notebook,
        1,
        textwrap.dedent(
            """
            import os
            import json
            import random
            from pathlib import Path
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from torch.utils.data import DataLoader, TensorDataset
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
            sns.set_theme(style='whitegrid')
            SEED = 42
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            USE_CUDA = DEVICE.type == 'cuda'
            if USE_CUDA:
                torch.backends.cudnn.benchmark = True
            cwd = Path.cwd().resolve()
            REPO_ROOT = next((candidate for candidate in [cwd, *cwd.parents] if (candidate / 'src' / 'wafer_defect').exists() and (candidate / 'experiments').exists()), cwd)
            print('Device:', DEVICE)
            """
        ).lstrip(),
    )
    extra = "DEFECT_CLASS_FRACTION = 0.5\n" if include_half else ""
    set_source(
        notebook,
        3,
        textwrap.dedent(
            f"""
            DATA_PATH = REPO_ROOT / 'data' / 'raw' / 'LSWMD.pkl'
            IMAGE_SIZE = 96
            TRAIN_TOTAL = 100000
            VAL_TOTAL = 10000
            TEST_TOTAL = 10000
            DEFECT_RATIO = 0.01
            {extra}BATCH_SIZE = 256 if USE_CUDA else 128
            NUM_WORKERS = 0
            PIN_MEMORY = False
            EPOCHS = 20
            LR = 0.001
            WEIGHT_DECAY = 0.0001
            LR_SCHED_FACTOR = 0.5
            LR_SCHED_PATIENCE = 2
            LR_SCHED_THRESHOLD = 0.001
            MIN_LR = 1e-06
            EARLY_STOP_PATIENCE = 6
            EARLY_STOP_MIN_DELTA = 0.001
            ARTIFACT_DIR = REPO_ROOT / 'experiments' / 'anomaly_detection_defect' / 'supervised_cnn' / '{subdir}' / 'artifacts' / '{artifact_leaf}'
            MODEL_PATH = ARTIFACT_DIR / 'cnn_wafer_classifier.pt'
            METRICS_PATH = ARTIFACT_DIR / 'cnn_metrics.json'
            ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
            print('Artifacts:', ARTIFACT_DIR)
            """
        ).lstrip(),
    )
    set_source(
        notebook,
        4,
        textwrap.dedent(
            """
            normal_df = pd.DataFrame()
            defect_df = pd.DataFrame()
            if RETRAIN:
                from wafer_defect.data.legacy_pickle import read_legacy_pickle
                df = read_legacy_pickle(DATA_PATH)
                print('Raw shape:', df.shape)

                def parse_failure_label(value):
                    if value is None:
                        return 'unknown'
                    if isinstance(value, float) and np.isnan(value):
                        return 'unknown'
                    if isinstance(value, (list, tuple, np.ndarray)):
                        arr = np.array(value).reshape(-1)
                        return 'unknown' if len(arr) == 0 else str(arr[0])
                    return str(value)

                df = df.copy()
                df['failure_label'] = df['failureType'].apply(parse_failure_label).astype(str).str.strip()
                invalid = {'0', 'unknown', 'nan', 'None', '[]'}
                df = df[~df['failure_label'].isin(invalid)].copy()
                df['is_anomaly'] = (df['failure_label'].str.lower() != 'none').astype(int)
                normal_df = df[df['is_anomaly'] == 0].copy()
                defect_df = df[df['is_anomaly'] == 1].copy()
                print('Normal count:', len(normal_df))
                print('Defect count:', len(defect_df))
            else:
                print('[WARNING] RETRAIN is False. Raw-data loading is skipped for this notebook run.')
            """
        ).lstrip(),
    )
    set_source(
        notebook,
        5,
        "if RETRAIN:\n" + textwrap.indent("".join(notebook["cells"][5]["source"]), "    ") + "else:\n    print('[WARNING] RETRAIN is False. Dataset splitting is skipped for this notebook run.')\n",
    )
    save_notebook(path, notebook)


def main() -> None:
    patch_autoencoder_x128(REPO_ROOT / "experiments/anomaly_detection/autoencoder/x128/baseline/notebook.ipynb")
    patch_autoencoder_x224(REPO_ROOT / "experiments/anomaly_detection/autoencoder/x224/main/notebook.ipynb")
    patch_ensemble_x64(REPO_ROOT / "experiments/anomaly_detection/ensemble/x64/score_ensemble/notebook.ipynb")
    patch_fastflow_x64(REPO_ROOT / "experiments/anomaly_detection/fastflow/x64/main/notebook.ipynb")
    patch_patchcore_selection_guard(REPO_ROOT / "experiments/anomaly_detection/patchcore/ae_bn/x64/main/notebook.ipynb")
    patch_patchcore_selection_guard(REPO_ROOT / "experiments/anomaly_detection/patchcore/resnet50/x64/main/notebook.ipynb")
    patch_patchcore_wrn_x64_main(REPO_ROOT / "experiments/anomaly_detection/patchcore/wideresnet50/x64/main/notebook.ipynb")
    patch_vit_x224_main(REPO_ROOT / "experiments/anomaly_detection/patchcore/vit_b16/x224/main/notebook.ipynb")
    patch_vit_x64_main(REPO_ROOT / "experiments/anomaly_detection/patchcore/vit_b16/x64/main/notebook.ipynb")
    patch_vit_block_depth_sweep(REPO_ROOT / "experiments/anomaly_detection/patchcore/vit_b16/x224/block_depth_sweep/notebook.ipynb")
    patch_saved_artifact_vit_notebook(
        REPO_ROOT / "experiments/anomaly_detection/patchcore/vit_b16/x224/one_layer_no_defect_tuning/notebook.ipynb",
        setup_index=6,
        score_index=18,
        later_indices=[20, 22, 24, 26],
    )
    patch_saved_artifact_vit_notebook(
        REPO_ROOT / "experiments/anomaly_detection/patchcore/vit_b16/x224/two_block_no_defect_tuning/notebook.ipynb",
        setup_index=6,
        score_index=18,
        later_indices=[20, 22, 24, 26],
    )
    patch_saved_artifact_vit_notebook(
        REPO_ROOT / "experiments/anomaly_detection/patchcore/vit_b16/x224/two_block/notebook.ipynb",
        setup_index=6,
        score_index=18,
        later_indices=[20, 22, 24, 26],
    )
    patch_saved_artifact_vit_notebook(
        REPO_ROOT / "experiments/anomaly_detection/patchcore/vit_b16/x224/one_layer_defect_tuning/notebook.ipynb",
        setup_index=6,
        score_index=16,
        later_indices=[20, 22, 24, 26],
    )
    patch_ft_mae_notebook(
        REPO_ROOT / "experiments/anomaly_detection/patchcore/vit_b16/x224/FT/MAE/wafer-trained-patchcore-vit-best.ipynb",
        finetune_index=15,
        downstream_indices=[17, 19, 20, 22, 24, 25, 27, 29],
    )
    patch_ft_mae_notebook(
        REPO_ROOT / "experiments/anomaly_detection/patchcore/vit_b16/x224/FT/MAE_25pct/wafer-trained-patchcore-vit-25pct-MAE.ipynb",
        finetune_index=15,
        downstream_indices=[17, 19, 20, 22, 24, 25, 27, 29],
    )
    patch_rd4ad_x224(REPO_ROOT / "experiments/anomaly_detection/rd4ad/wideresnet50/x224/main/notebook.ipynb")
    patch_report_figures(REPO_ROOT / "experiments/anomaly_detection/report_figures/notebook.ipynb")
    patch_vit_import_cells()
    patch_supervised_cnn(
        REPO_ROOT / "experiments/anomaly_detection_defect/supervised_cnn/full_defect/wafer-cnn-normal-1pct-defect.ipynb",
        subdir="full_defect",
        artifact_leaf="cnn_wafer_normal_3pct_defect",
        include_half=False,
    )
    patch_supervised_cnn(
        REPO_ROOT / "experiments/anomaly_detection_defect/supervised_cnn/half_defect/wafer-cnn-normal-1pct-half-classes.ipynb",
        subdir="half_defect",
        artifact_leaf="cnn_wafer_normal_1pct_half_classes",
        include_half=True,
    )


if __name__ == "__main__":
    main()
