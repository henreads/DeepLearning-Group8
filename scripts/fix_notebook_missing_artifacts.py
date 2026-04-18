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


def patch_autoencoder_x64(path: Path) -> None:
    notebook = load_notebook(path)
    is_baseline = 'baseline' in path.parts
    idx = {
        'curve': 17,
        'eval': 23 if is_baseline else 21,
        'threshold': 25 if is_baseline else 23,
        'metrics': 27 if is_baseline else 25,
        'sweep': 29 if is_baseline else 27,
        'distribution': 31 if is_baseline else 29,
        'recon': 33 if is_baseline else 31,
        'analysis': 35 if is_baseline else 33,
        'examples': 39 if is_baseline else 35,
        'ablation_run': 41 if is_baseline else 37,
        'ablation_results': 43 if is_baseline else 39,
        'ablation_plot': 45 if is_baseline else 41,
    }

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
            training_artifacts_ready = False
            evaluation_ready = False
            score_ablation_ready = False
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
                training_artifacts_ready = True
                print(f'Found existing artifacts in {output_dir}. Skipping training.')
            elif RETRAIN:
                if 'train_loader' not in globals() or train_loader is None or val_loader is None:
                    warn_skip('Training datasets are unavailable, so RETRAIN=True cannot run in this cell.')
                else:
                    if resume_from:
                        resume_path = Path(resume_from)
                        if not resume_path.is_absolute():
                            resume_path = REPO_ROOT / resume_path
                        checkpoint = torch.load(resume_path, map_location=device)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        start_epoch = int(checkpoint.get('epoch', 0))
                        best_val_loss = float(checkpoint.get('best_val_loss', best_val_loss))
                        best_epoch = int(checkpoint.get('best_epoch', best_epoch))
                        stale_epochs = int(checkpoint.get('stale_epochs', stale_epochs))
                        history = checkpoint.get('history', [])
                        best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                        print(f'Resumed from {resume_path} at epoch {start_epoch}')
                    print({'epochs': epochs, 'anomaly_score': ANOMALY_SCORE_NAME, 'topk_ratio': TOPK_RATIO})
                    for epoch in range(start_epoch, epochs):
                        train_metrics = run_autoencoder_epoch(model, train_loader, device, optimizer)
                        val_metrics = run_autoencoder_epoch(model, val_loader, device)
                        record = {'epoch': epoch + 1, 'train_loss': train_metrics.loss, 'val_loss': val_metrics.loss}
                        history.append(record)
                        print(record)
                        improved = best_val_loss - val_metrics.loss > min_delta
                        if improved:
                            best_val_loss = val_metrics.loss
                            best_epoch = epoch + 1
                            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                            stale_epochs = 0
                            torch.save({'epoch': epoch + 1, 'model_state_dict': best_state_dict, 'optimizer_state_dict': optimizer.state_dict(), 'config': config, 'best_epoch': best_epoch, 'best_val_loss': best_val_loss, 'stale_epochs': stale_epochs, 'history': history}, output_dir / 'checkpoints' / 'best_model.pt')
                        else:
                            stale_epochs += 1
                        latest_checkpoint = {'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'config': config, 'best_epoch': best_epoch, 'best_val_loss': best_val_loss, 'stale_epochs': stale_epochs, 'history': history}
                        torch.save(latest_checkpoint, output_dir / 'checkpoints' / 'latest_checkpoint.pt')
                        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
                            torch.save(latest_checkpoint, output_dir / 'checkpoints' / f'checkpoint_epoch_{epoch + 1}.pt')
                        if patience > 0 and stale_epochs >= patience:
                            print(f'Early stopping at epoch {epoch + 1}. Best epoch: {best_epoch}, best val loss: {best_val_loss:.6f}')
                            break
                    if best_state_dict is None:
                        best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    training_ran = True
                    training_artifacts_ready = True
            else:
                warn_skip('Saved training artifacts are missing and RETRAIN is False. Skipping this section.')
            """
        ).lstrip(),
    )

    set_source(
        notebook,
        idx['curve'],
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
        idx['eval'],
        textwrap.dedent(
            """
            best_model_path = output_dir / 'checkpoints' / 'best_model.pt'
            score_df = pd.DataFrame()
            if not best_model_path.exists():
                warn_skip(f'Best autoencoder checkpoint not found: {best_model_path}. Skipping evaluation-backed cells.')
                evaluation_ready = False
            elif 'test_loader' not in globals() or test_loader is None:
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

    set_source(
        notebook,
        idx['threshold'],
        textwrap.dedent(
            """
            val_score_series = pd.Series(dtype=float, name='val_score')
            threshold = float('nan')
            if not evaluation_ready:
                warn_skip('Validation-threshold selection is unavailable because evaluation scores were not generated.')
            elif 'val_loader' not in globals() or val_loader is None:
                warn_skip('The validation dataset is unavailable, so threshold selection is skipped for this cell.')
                evaluation_ready = False
            else:
                val_scores = []
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs = inputs.to(device)
                        outputs = model(inputs)
                        scores = reconstruction_error(inputs, outputs, score_name=ANOMALY_SCORE_NAME).cpu().numpy()
                        val_scores.extend(scores.tolist())
                val_score_series = pd.Series(val_scores, name='val_score')
                threshold = float(val_score_series.quantile(0.95))
                print(f'Chosen threshold from validation normals (95th percentile, {ANOMALY_SCORE_NAME}): {threshold:.6f}')
                display(val_score_series.describe())
            """
        ).lstrip(),
    )

    set_source(
        notebook,
        idx['metrics'],
        textwrap.dedent(
            """
            if not evaluation_ready or score_df.empty or pd.isna(threshold):
                warn_skip('Metric review is unavailable because the validation threshold or test scores were not generated.')
            else:
                score_df['predicted_anomaly'] = (score_df['score'] > threshold).astype(int)
                precision = precision_score(score_df['is_anomaly'], score_df['predicted_anomaly'], zero_division=0)
                recall = recall_score(score_df['is_anomaly'], score_df['predicted_anomaly'], zero_division=0)
                f1 = f1_score(score_df['is_anomaly'], score_df['predicted_anomaly'], zero_division=0)
                auroc = roc_auc_score(score_df['is_anomaly'], score_df['score'])
                auprc = average_precision_score(score_df['is_anomaly'], score_df['score'])
                cm = confusion_matrix(score_df['is_anomaly'], score_df['predicted_anomaly'])
                metrics_df = pd.DataFrame([{'metric': 'score_name', 'value': ANOMALY_SCORE_NAME}, {'metric': 'precision', 'value': precision}, {'metric': 'recall', 'value': recall}, {'metric': 'f1', 'value': f1}, {'metric': 'auroc', 'value': auroc}, {'metric': 'auprc', 'value': auprc}, {'metric': 'threshold', 'value': threshold}])
                display(metrics_df)
                cm_df = pd.DataFrame(cm, index=['true_normal', 'true_anomaly'], columns=['pred_normal', 'pred_anomaly'])
                display(cm_df)
                fig, ax = plt.subplots(figsize=(5, 4))
                heatmap = ax.imshow(cm_df.to_numpy(), cmap='Blues')
                ax.set_xticks(range(cm_df.shape[1]), cm_df.columns)
                ax.set_yticks(range(cm_df.shape[0]), cm_df.index)
                ax.set_title('Confusion Matrix At Validation Threshold')
                ax.set_xlabel('Predicted label')
                ax.set_ylabel('True label')
                for row_idx in range(cm_df.shape[0]):
                    for col_idx in range(cm_df.shape[1]):
                        value = int(cm_df.iat[row_idx, col_idx])
                        ax.text(col_idx, row_idx, str(value), ha='center', va='center', color='black')
                fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
                plt.tight_layout()
                save_figure(fig, output_dir / 'plots' / 'confusion_matrix.png')
                plt.show()
                score_df.to_csv(output_dir / 'results' / 'test_scores.csv', index=False)
                metrics_df.to_csv(output_dir / 'results' / 'metrics.csv', index=False)
            """
        ).lstrip(),
    )

    set_source(
        notebook,
        idx['sweep'],
        textwrap.dedent(
            """
            if not evaluation_ready or score_df.empty or pd.isna(threshold):
                warn_skip('Threshold-sweep review is unavailable because evaluation scores were not generated.')
            else:
                precision_curve, recall_curve, pr_thresholds = precision_recall_curve(score_df['is_anomaly'], score_df['score'])
                threshold_sweep_df = pd.DataFrame({'threshold': pr_thresholds, 'precision': precision_curve[:-1], 'recall': recall_curve[:-1]})
                threshold_sweep_df['f1'] = 2 * threshold_sweep_df['precision'] * threshold_sweep_df['recall'] / (threshold_sweep_df['precision'] + threshold_sweep_df['recall'] + 1e-12)
                threshold_sweep_df['predicted_anomalies'] = [int((score_df['score'] > t).sum()) for t in threshold_sweep_df['threshold']]
                best_f1_row = threshold_sweep_df.loc[threshold_sweep_df['f1'].idxmax()]
                threshold_sweep_df.to_csv(output_dir / 'results' / 'threshold_sweep.csv', index=False)
                display(threshold_sweep_df.sort_values('f1', ascending=False).head(10))
                print(f"Best F1 threshold: {best_f1_row['threshold']:.6f} | precision={best_f1_row['precision']:.4f}, recall={best_f1_row['recall']:.4f}, f1={best_f1_row['f1']:.4f}")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(threshold_sweep_df['threshold'], threshold_sweep_df['precision'], label='precision')
                ax.plot(threshold_sweep_df['threshold'], threshold_sweep_df['recall'], label='recall')
                ax.plot(threshold_sweep_df['threshold'], threshold_sweep_df['f1'], label='f1')
                ax.axvline(threshold, color='red', linestyle='--', label=f'validation threshold = {threshold:.4f}')
                ax.axvline(best_f1_row['threshold'], color='green', linestyle=':', label=f"best f1 threshold = {best_f1_row['threshold']:.4f}")
                ax.set_xlabel('Anomaly-score threshold')
                ax.set_ylabel('Metric value')
                ax.set_title(f'Threshold Sweep on Test Split ({ANOMALY_SCORE_NAME})')
                ax.legend()
                save_figure(fig, output_dir / 'plots' / 'threshold_sweep.png')
                plt.show()
            """
        ).lstrip(),
    )

    set_source(
        notebook,
        idx['distribution'],
        textwrap.dedent(
            """
            if not evaluation_ready or score_df.empty or pd.isna(threshold):
                warn_skip('Score-distribution plots are unavailable because evaluation scores were not generated.')
            else:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(score_df[score_df['is_anomaly'] == 0]['score'], bins=30, alpha=0.7, label='normal')
                ax.hist(score_df[score_df['is_anomaly'] == 1]['score'], bins=30, alpha=0.7, label='anomaly')
                ax.axvline(threshold, color='red', linestyle='--', label=f'threshold={threshold:.4f}')
                ax.set_title(f'Anomaly Score on Test Split ({ANOMALY_SCORE_NAME})')
                ax.set_xlabel(f'Per-sample score: {ANOMALY_SCORE_NAME}')
                ax.set_ylabel('Count')
                ax.legend()
                plt.tight_layout()
                save_figure(fig, output_dir / 'plots' / 'score_distribution.png')
                plt.show()
            """
        ).lstrip(),
    )

    set_source(
        notebook,
        idx['recon'],
        textwrap.dedent(
            """
            if not evaluation_ready or score_df.empty:
                warn_skip('Reconstruction examples are unavailable because evaluation scores were not generated.')
            else:
                normal_test_idx = score_df[score_df['is_anomaly'] == 0].index[:3].tolist()
                anomaly_test_idx = score_df[score_df['is_anomaly'] == 1].index[:3].tolist()
                selected_indices = normal_test_idx + anomaly_test_idx
                fig, axes = plt.subplots(2, len(selected_indices), figsize=(2.2 * len(selected_indices), 4.5))
                with torch.no_grad():
                    for col_idx, sample_idx in enumerate(selected_indices):
                        input_tensor, label = test_dataset[sample_idx]
                        output_tensor = model(input_tensor.unsqueeze(0).to(device)).squeeze(0).cpu()
                        title_prefix = 'anomaly' if int(label) == 1 else 'normal'
                        score = score_df.iloc[sample_idx]['score']
                        axes[0, col_idx].imshow(input_tensor.squeeze(0), cmap='viridis')
                        axes[0, col_idx].set_title(f'{title_prefix}')
                        axes[0, col_idx].axis('off')
                        axes[1, col_idx].imshow(output_tensor.squeeze(0), cmap='viridis')
                        axes[1, col_idx].set_title(f'score={score:.3f}')
                        axes[1, col_idx].axis('off')
                plt.suptitle('Input (top row) and Reconstruction (bottom row)', fontsize=10, y=0.98)
                plt.tight_layout()
                save_figure(fig, output_dir / 'plots' / 'reconstruction_examples.png')
                plt.show()
            """
        ).lstrip(),
    )

    set_source(
        notebook,
        idx['analysis'],
        textwrap.dedent(
            """
            if not evaluation_ready or score_df.empty or 'predicted_anomaly' not in score_df.columns:
                warn_skip('Failure analysis is unavailable because evaluation outputs were not generated.')
                analysis_df = pd.DataFrame()
            else:
                analysis_df = test_dataset.metadata.reset_index(drop=True).copy()
                analysis_df = pd.concat([analysis_df, score_df[['score', 'predicted_anomaly']].reset_index(drop=True)], axis=1)
                analysis_df['error_type'] = 'tn'
                analysis_df.loc[(analysis_df['is_anomaly'] == 0) & (analysis_df['predicted_anomaly'] == 1), 'error_type'] = 'fp'
                analysis_df.loc[(analysis_df['is_anomaly'] == 1) & (analysis_df['predicted_anomaly'] == 0), 'error_type'] = 'fn'
                analysis_df.loc[(analysis_df['is_anomaly'] == 1) & (analysis_df['predicted_anomaly'] == 1), 'error_type'] = 'tp'
                analysis_df['correct'] = analysis_df['is_anomaly'] == analysis_df['predicted_anomaly']
                error_summary_df = analysis_df.groupby('error_type').agg(count=('error_type', 'size'), mean_score=('score', 'mean')).reindex(['tp', 'fn', 'fp', 'tn'])
                defect_recall_df = analysis_df[analysis_df['is_anomaly'] == 1].groupby('defect_type').agg(count=('defect_type', 'size'), detected=('predicted_anomaly', 'sum'), mean_score=('score', 'mean')).sort_values(['detected', 'count'], ascending=[False, False])
                defect_recall_df['recall'] = defect_recall_df['detected'] / defect_recall_df['count']
                fp_defect_df = analysis_df[analysis_df['error_type'] == 'fp'].groupby('defect_type').agg(count=('defect_type', 'size'), mean_score=('score', 'mean')).sort_values(['count', 'mean_score'], ascending=[False, False])
                display(error_summary_df)
                display(defect_recall_df)
                display(fp_defect_df)
                display(analysis_df.head())
                analysis_df.to_csv(output_dir / 'results' / 'failure_analysis.csv', index=False)
                error_summary_df.to_csv(output_dir / 'results' / 'failure_error_summary.csv')
                defect_recall_df.to_csv(output_dir / 'results' / 'failure_defect_recall.csv')
                fp_defect_df.to_csv(output_dir / 'results' / 'failure_false_positive_breakdown.csv')
            """
        ).lstrip(),
    )

    set_source(
        notebook,
        idx['examples'],
        textwrap.dedent(
            """
            def show_error_examples(error_type: str, n_examples: int=6, score_order: str='desc') -> pd.DataFrame:
                if 'analysis_df' not in globals() or analysis_df.empty:
                    print(f'Failure analysis is unavailable for error_type={error_type!r}.')
                    return pd.DataFrame()
                subset = analysis_df[analysis_df['error_type'] == error_type].copy()
                if subset.empty:
                    print(f'No samples found for error_type={error_type!r}.')
                    return subset
                ascending = score_order == 'asc'
                subset = subset.sort_values('score', ascending=ascending).head(n_examples)
                n_rows = (len(subset) + 1) // 2
                fig, axes = plt.subplots(n_rows, 6, figsize=(6 * 2.2, n_rows * 2.5))
                if n_rows == 1:
                    axes = axes.reshape(1, -1)
                with torch.no_grad():
                    for plot_idx, (sample_idx, row) in enumerate(subset.iterrows()):
                        row_idx = plot_idx // 2
                        col_start = plot_idx % 2 * 3
                        input_tensor, label = test_dataset[sample_idx]
                        output_tensor = model(input_tensor.unsqueeze(0).to(device)).squeeze(0).cpu()
                        error_map = absolute_error_map(input_tensor.unsqueeze(0), output_tensor.unsqueeze(0)).squeeze(0).squeeze(0).cpu()
                        axes[row_idx, col_start].imshow(input_tensor.squeeze(0), cmap='viridis')
                        axes[row_idx, col_start].set_title(f"Input\\n{row.get('defect_type', '?')} | score={row['score']:.3f}", fontsize=8)
                        axes[row_idx, col_start].axis('off')
                        axes[row_idx, col_start + 1].imshow(output_tensor.squeeze(0), cmap='viridis')
                        axes[row_idx, col_start + 1].set_title(f"Reconstruction\\npred={row['predicted_anomaly']}", fontsize=8)
                        axes[row_idx, col_start + 1].axis('off')
                        axes[row_idx, col_start + 2].imshow(error_map, cmap='magma')
                        axes[row_idx, col_start + 2].set_title(f'Error Map\\n#{sample_idx}', fontsize=8)
                        axes[row_idx, col_start + 2].axis('off')
                for idx in range(len(subset) * 3, n_rows * 6):
                    row_idx = idx // 6
                    col_idx = idx % 6
                    axes[row_idx, col_idx].axis('off')
                plt.suptitle(f'{error_type.upper()} Examples', fontsize=11, fontweight='bold')
                plt.tight_layout()
                plt.subplots_adjust(hspace=0.2)
                save_figure(fig, output_dir / 'plots' / f'failure_examples_{error_type}.png')
                plt.show()
                return subset[['defect_type', 'score', 'predicted_anomaly', 'error_type']]

            if 'analysis_df' not in globals() or analysis_df.empty:
                warn_skip('Failure-example review is unavailable because failure analysis was not generated.')
            else:
                display(show_error_examples('fp', n_examples=6, score_order='desc'))
                display(show_error_examples('fn', n_examples=6, score_order='asc'))
                display(show_error_examples('tp', n_examples=4, score_order='desc'))
                display(show_error_examples('tn', n_examples=4, score_order='asc'))
            """
        ).lstrip(),
    )

    set_source(
        notebook,
        idx['ablation_run'],
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
            score_ablation_cmd = [sys.executable, 'scripts/evaluate_autoencoder_scores.py', '--checkpoint', str(score_ablation_best_model_path.relative_to(REPO_ROOT)), '--config', str(CONFIG_PATH.relative_to(REPO_ROOT)), '--output-dir', str(score_ablation_output_dir.relative_to(REPO_ROOT))]
            score_ablation_env = os.environ.copy()
            src_path = str(REPO_ROOT / 'src')
            score_ablation_env['PYTHONPATH'] = src_path if not score_ablation_env.get('PYTHONPATH') else src_path + os.pathsep + score_ablation_env['PYTHONPATH']
            score_ablation_ready = False
            if RERUN_SCORE_ABLATION:
                if not score_ablation_best_model_path.exists():
                    warn_skip(f'Best autoencoder checkpoint not found: {score_ablation_best_model_path}. Skipping score ablation review.')
                elif 'val_loader' not in globals() or val_loader is None or test_loader is None:
                    warn_skip('Score ablation needs the validation/test datasets, which are unavailable for this cell.')
                else:
                    print('Running:')
                    print(' '.join(score_ablation_cmd))
                    subprocess.run(score_ablation_cmd, cwd=REPO_ROOT, env=score_ablation_env, check=True)
                    score_ablation_ready = True
            elif score_ablation_csv_path.exists() and score_ablation_json_path.exists():
                print(f'Found existing score ablation outputs in {score_ablation_output_dir}. Skipping rerun.')
                score_ablation_ready = True
            else:
                warn_skip('Score ablation artifacts are missing and RERUN_SCORE_ABLATION is False. Skipping this section.')
            """
        ).lstrip(),
    )

    set_source(
        notebook,
        idx['ablation_results'],
        textwrap.dedent(
            """
            if not score_ablation_ready:
                warn_skip('Score ablation tables are unavailable because the score-ablation artifacts are missing.')
                score_ablation_df = pd.DataFrame()
                score_ablation_summary = {}
            else:
                score_ablation_df = pd.read_csv(score_ablation_output_dir / 'score_summary.csv')
                score_ablation_summary = json.loads((score_ablation_output_dir / 'score_summary.json').read_text(encoding='utf-8'))
                display(score_ablation_df)
                display(score_ablation_summary)
            """
        ).lstrip(),
    )

    set_source(
        notebook,
        idx['ablation_plot'],
        textwrap.dedent(
            """
            if not score_ablation_ready or score_ablation_df.empty:
                warn_skip('Score ablation plots are unavailable because the score-ablation artifacts are missing.')
            else:
                top_scores = score_ablation_df.sort_values('val_threshold_f1', ascending=False).reset_index(drop=True)
                fig, axes = plt.subplots(1, 3, figsize=(16, 4))
                axes[0].bar(top_scores['score_name'], top_scores['val_threshold_f1'])
                axes[0].set_title('Validation-Threshold F1')
                axes[0].tick_params(axis='x', rotation=35)
                axes[1].bar(top_scores['score_name'], top_scores['auroc'])
                axes[1].set_title('AUROC')
                axes[1].tick_params(axis='x', rotation=35)
                axes[2].bar(top_scores['score_name'], top_scores['auprc'])
                axes[2].set_title('AUPRC')
                axes[2].tick_params(axis='x', rotation=35)
                plt.tight_layout()
                save_figure(fig, score_ablation_output_root / 'plots' / 'score_ablation_summary.png')
                plt.show()
                display(top_scores[['score_name', 'val_threshold_f1', 'auroc', 'auprc', 'best_sweep_f1']])
            """
        ).lstrip(),
    )

    save_notebook(path, notebook)


def patch_vae_baseline(path: Path) -> None:
    notebook = load_notebook(path)

    set_source(
        notebook,
        3,
        textwrap.dedent(
            """
            from __future__ import annotations
            import json
            import os
            import subprocess
            import sys
            from pathlib import Path
            import matplotlib.pyplot as plt
            import pandas as pd
            import torch
            try:
                from IPython.display import display
            except ImportError:

                def display(obj: object) -> None:
                    print(obj)

            def warn_skip(message: str) -> None:
                print(f'[WARNING] {message}')

            REPO_ROOT = Path.cwd().resolve()
            if not (REPO_ROOT / 'src' / 'wafer_defect').exists():
                for candidate in [REPO_ROOT, *REPO_ROOT.parents]:
                    if (candidate / 'src' / 'wafer_defect').exists():
                        REPO_ROOT = candidate
                        break
            SRC_ROOT = REPO_ROOT / 'src'
            if str(SRC_ROOT) not in sys.path:
                sys.path.insert(0, str(SRC_ROOT))
            from wafer_defect.config import load_toml
            from wafer_defect.data.wm811k import WaferMapDataset
            from wafer_defect.models.vae import ConvVariationalAutoencoder, VAEOutput
            """
        ).lstrip(),
    )

    set_source(
        notebook,
        9,
        textwrap.dedent(
            """
            history_path = OUTPUT_DIR / 'results' / 'history.json'
            summary_path = OUTPUT_DIR / 'results' / 'summary.json'
            best_model_path = OUTPUT_DIR / 'checkpoints' / 'best_model.pt'
            artifacts_ready = history_path.exists() and summary_path.exists() and best_model_path.exists()
            training_ready = False
            history_df = pd.DataFrame()
            training_summary = {}
            env = os.environ.copy()
            env['PYTHONPATH'] = str(SRC_ROOT) if not env.get('PYTHONPATH') else str(SRC_ROOT) + os.pathsep + env['PYTHONPATH']
            if FORCE_RETRAIN:
                train_cmd = [sys.executable, 'scripts/train_vae.py', '--config', str(CONFIG_PATH.relative_to(REPO_ROOT))]
                print('Running:', ' '.join(train_cmd))
                subprocess.run(train_cmd, cwd=REPO_ROOT, env=env, check=True)
            elif artifacts_ready:
                print(f'Found existing training artifacts in {OUTPUT_DIR}. Skipping retraining.')
            else:
                warn_skip('Saved training artifacts are missing and FORCE_RETRAIN is False. Skipping this section.')
            if history_path.exists():
                history = json.loads(history_path.read_text(encoding='utf-8'))
                history_df = pd.DataFrame(history)
            if summary_path.exists():
                training_summary = json.loads(summary_path.read_text(encoding='utf-8'))
            training_ready = not history_df.empty
            if training_summary:
                display(pd.DataFrame([training_summary]))
            elif not training_ready:
                warn_skip('Training summary is unavailable because summary.json is missing.')
            """
        ).lstrip(),
    )

    set_source(
        notebook,
        11,
        textwrap.dedent(
            """
            if not training_ready:
                warn_skip('Training curves are unavailable because history.json is missing or empty.')
            else:
                fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), constrained_layout=True)
                axes[0].plot(history_df['epoch'], history_df['train_loss'], label='train')
                axes[0].plot(history_df['epoch'], history_df['val_loss'], label='val')
                axes[0].set_title('Total Loss')
                axes[0].set_xlabel('Epoch')
                axes[0].legend()
                axes[1].plot(history_df['epoch'], history_df['train_reconstruction_loss'], label='train')
                axes[1].plot(history_df['epoch'], history_df['val_reconstruction_loss'], label='val')
                axes[1].set_title('Reconstruction Loss')
                axes[1].set_xlabel('Epoch')
                axes[1].legend()
                axes[2].plot(history_df['epoch'], history_df['train_kl_loss'], label='train')
                axes[2].plot(history_df['epoch'], history_df['val_kl_loss'], label='val')
                axes[2].set_title('KL Loss')
                axes[2].set_xlabel('Epoch')
                axes[2].legend()
                fig.savefig(PLOTS_DIR / 'training_curves.png', dpi=160, bbox_inches='tight')
                display(fig)
                plt.close(fig)
            """
        ).lstrip(),
    )

    set_source(
        notebook,
        13,
        textwrap.dedent(
            """
            evaluation_summary_path = EVALUATION_DIR / 'summary.json'
            val_scores_path = EVALUATION_DIR / 'val_scores.csv'
            test_scores_path = EVALUATION_DIR / 'test_scores.csv'
            threshold_sweep_path = EVALUATION_DIR / 'threshold_sweep.csv'
            evaluation_ready = False
            evaluation_summary = {}
            val_scores_df = pd.DataFrame()
            test_scores_df = pd.DataFrame()
            threshold_sweep_df = pd.DataFrame()
            threshold = float('nan')
            best_sweep = {}
            if FORCE_EVALUATION_RERUN:
                if not best_model_path.exists():
                    warn_skip(f'Best checkpoint not found at {best_model_path}. Evaluation-backed cells will be skipped.')
                else:
                    eval_cmd = [sys.executable, 'scripts/evaluate_reconstruction_model.py', '--checkpoint', str(best_model_path.relative_to(REPO_ROOT)), '--config', str(CONFIG_PATH.relative_to(REPO_ROOT)), '--output-dir', str(EVALUATION_DIR.relative_to(REPO_ROOT))]
                    print('Running:', ' '.join(eval_cmd))
                    subprocess.run(eval_cmd, cwd=REPO_ROOT, env=env, check=True)
            elif not all((path.exists() for path in [evaluation_summary_path, val_scores_path, test_scores_path, threshold_sweep_path])):
                warn_skip('Saved evaluation artifacts are missing and FORCE_EVALUATION_RERUN is False. Skipping this section.')
            if all((path.exists() for path in [evaluation_summary_path, val_scores_path, test_scores_path, threshold_sweep_path])):
                evaluation_summary = json.loads(evaluation_summary_path.read_text(encoding='utf-8'))
                val_scores_df = pd.read_csv(val_scores_path)
                test_scores_df = pd.read_csv(test_scores_path)
                threshold_sweep_df = pd.read_csv(threshold_sweep_path)
                threshold = float(evaluation_summary['threshold'])
                best_sweep = dict(evaluation_summary.get('best_threshold_sweep', {}))
                evaluation_ready = True
                display(pd.DataFrame([evaluation_summary['metrics_at_validation_threshold']]))
                display(pd.DataFrame([best_sweep]))
            """
        ).lstrip(),
    )

    set_source(
        notebook,
        15,
        textwrap.dedent(
            """
            if not evaluation_ready:
                warn_skip('Evaluation plots are unavailable because the cached evaluation artifacts were not loaded.')
            else:
                cm = evaluation_summary['metrics_at_validation_threshold'].get('confusion_matrix', [[0, 0], [0, 0]])
                cm_df = pd.DataFrame(cm, index=['true_normal', 'true_anomaly'], columns=['pred_normal', 'pred_anomaly'])
                fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), constrained_layout=True)
                axes[0].hist(test_scores_df.loc[test_scores_df['is_anomaly'] == 0, 'score'], bins=30, alpha=0.7, label='normal')
                axes[0].hist(test_scores_df.loc[test_scores_df['is_anomaly'] == 1, 'score'], bins=30, alpha=0.7, label='anomaly')
                axes[0].axvline(threshold, color='red', linestyle='--', label=f'val threshold = {threshold:.4f}')
                axes[0].set_title('Test Score Distribution')
                axes[0].set_xlabel('Anomaly score')
                axes[0].legend()
                axes[1].plot(threshold_sweep_df['threshold'], threshold_sweep_df['precision'], label='precision')
                axes[1].plot(threshold_sweep_df['threshold'], threshold_sweep_df['recall'], label='recall')
                axes[1].plot(threshold_sweep_df['threshold'], threshold_sweep_df['f1'], label='f1')
                axes[1].axvline(threshold, color='red', linestyle='--', label='val threshold')
                axes[1].axvline(float(best_sweep['threshold']), color='green', linestyle=':', label='best sweep')
                axes[1].set_title('Threshold Sweep')
                axes[1].set_xlabel('Threshold')
                axes[1].legend()
                heatmap = axes[2].imshow(cm_df.to_numpy(), cmap='Blues')
                axes[2].set_xticks(range(cm_df.shape[1]), cm_df.columns)
                axes[2].set_yticks(range(cm_df.shape[0]), cm_df.index)
                axes[2].set_title('Confusion Matrix')
                axes[2].set_xlabel('Predicted label')
                axes[2].set_ylabel('True label')
                for row_idx in range(cm_df.shape[0]):
                    for col_idx in range(cm_df.shape[1]):
                        axes[2].text(col_idx, row_idx, str(int(cm_df.iat[row_idx, col_idx])), ha='center', va='center', color='black')
                fig.colorbar(heatmap, ax=axes[2], fraction=0.046, pad=0.04)
                fig.savefig(PLOTS_DIR / 'score_distribution_sweep_confusion.png', dpi=160, bbox_inches='tight')
                display(fig)
                plt.close(fig)
                display(cm_df)
                display(threshold_sweep_df.sort_values('f1', ascending=False).head(10))
            """
        ).lstrip(),
    )

    set_source(
        notebook,
        17,
        textwrap.dedent(
            """
            if 'test_dataset' not in globals() or test_dataset is None:
                warn_skip('Reconstruction examples are unavailable because the benchmark test dataset was not loaded.')
            elif not best_model_path.exists():
                warn_skip(f'Reconstruction examples are unavailable because the best checkpoint is missing: {best_model_path}')
            else:
                checkpoint = torch.load(best_model_path, map_location='cpu')
                eval_model = ConvVariationalAutoencoder(latent_dim=int(config['model']['latent_dim']), image_size=int(config['data']['image_size']))
                eval_model.load_state_dict(checkpoint['model_state_dict'])
                eval_model.eval()
                sample_count = min(4, len(test_dataset))
                sample_tensors = []
                sample_labels = []
                for index in range(sample_count):
                    tensor, label = test_dataset[index]
                    sample_tensors.append(tensor)
                    sample_labels.append(int(label))
                batch = torch.stack(sample_tensors, dim=0)
                with torch.inference_mode():
                    outputs = eval_model(batch)
                if not isinstance(outputs, VAEOutput):
                    raise TypeError('VAE model must return VAEOutput')
                recons = outputs.reconstruction.cpu()
                fig, axes = plt.subplots(sample_count, 4, figsize=(9, 2.5 * sample_count), constrained_layout=True)
                if sample_count == 1:
                    axes = [axes]
                for row in range(sample_count):
                    axes[row][0].imshow(batch[row, 0], cmap='gray')
                    axes[row][0].set_title(f'Input {row}')
                    axes[row][1].imshow(recons[row, 0], cmap='gray')
                    axes[row][1].set_title('Recon')
                    axes[row][2].imshow((batch[row, 0] - recons[row, 0]).abs(), cmap='magma')
                    axes[row][2].set_title('Abs Error')
                    axes[row][3].text(0.05, 0.6, f'label={sample_labels[row]}', fontsize=11)
                    axes[row][3].axis('off')
                    for col in range(3):
                        axes[row][col].set_xticks([])
                        axes[row][col].set_yticks([])
                fig.savefig(PLOTS_DIR / 'reconstruction_examples.png', dpi=160, bbox_inches='tight')
                display(fig)
                plt.close(fig)
            """
        ).lstrip(),
    )

    save_notebook(path, notebook)


def patch_vae_x224_metadata(path: Path) -> None:
    notebook = load_notebook(path)
    set_source(
        notebook,
        7,
        textwrap.dedent(
            """
            metadata_path = REPO_ROOT / config['data']['metadata_csv']
            metadata_ready = metadata_path.exists()
            metadata_df = pd.DataFrame()
            split_summary_df = pd.DataFrame()
            test_dataset = None
            if metadata_ready:
                metadata_df = pd.read_csv(metadata_path)
                split_summary_df = metadata_df.groupby(['split', 'is_anomaly']).size().reset_index(name='count').sort_values(['split', 'is_anomaly']).reset_index(drop=True)
                test_dataset = WaferMapDataset(metadata_path, split='test', image_size=int(config['data']['image_size']))
                print(f'Metadata CSV: {metadata_path}')
                display(split_summary_df)
                display(metadata_df.head(5))
            else:
                warn_skip(f'Metadata CSV is missing: {metadata_path}. Dataset-backed review cells will be skipped.')
            """
        ).lstrip(),
    )
    save_notebook(path, notebook)


def patch_patchcore_wrn_x224_review(path: Path, metadata_cell: int, selected_cell: int, analysis_cell: int) -> None:
    notebook = load_notebook(path)

    set_source(
        notebook,
        metadata_cell,
        textwrap.dedent(
            """
            metadata_ready = METADATA_PATH.exists()
            if metadata_ready:
                metadata = pd.read_csv(METADATA_PATH)
                test_metadata = metadata[metadata['split'] == 'test'].reset_index(drop=True)
            else:
                metadata = pd.DataFrame()
                test_metadata = pd.DataFrame()
                print(f'[WARNING] Benchmark metadata is missing: {METADATA_PATH}. Defect-analysis cells will be skipped.')
            sweep_results_path = RESULTS_DIR / 'patchcore_sweep_results.csv'
            sweep_summary_path = RESULTS_DIR / 'patchcore_sweep_summary.json'
            review_ready = sweep_results_path.exists() and sweep_summary_path.exists()
            if review_ready:
                sweep_results_df = pd.read_csv(sweep_results_path)
                sweep_summary = json.loads(sweep_summary_path.read_text(encoding='utf-8'))
                selected_variant_name = str(SELECTED_VARIANT_NAME or sweep_summary['best_variant']['name'])
            else:
                sweep_results_df = pd.DataFrame()
                sweep_summary = {}
                selected_variant_name = None
                print('[WARNING] Cached sweep artifacts are missing. Variant-review cells will be skipped until the artifacts folder is restored.')
            if metadata_ready:
                display(metadata['split'].value_counts().rename_axis('split').to_frame('count'))
            if not sweep_results_df.empty:
                display(sweep_results_df)
            print(f'Selected variant: {selected_variant_name}')
            """
        ).lstrip(),
    )

    set_source(
        notebook,
        selected_cell,
        textwrap.dedent(
            """
            if not review_ready or not selected_variant_name:
                selected_variant = {}
                summary = {}
                val_scores_df = pd.DataFrame()
                test_scores_df = pd.DataFrame()
                threshold_sweep_df = pd.DataFrame()
                metrics = {}
                best_sweep = {}
                threshold = float('nan')
                print('[WARNING] Selected-variant review is unavailable because the cached sweep artifacts are missing.')
            else:
                selected_variant = load_variant_outputs(selected_variant_name)
                summary = selected_variant['summary']
                val_scores_df = selected_variant['val_scores_df']
                test_scores_df = selected_variant['test_scores_df']
                threshold_sweep_df = selected_variant['threshold_sweep_df']
                metrics = selected_variant['metrics']
                best_sweep = selected_variant['best_sweep']
                threshold = float(summary['threshold'])
                metrics_df = pd.DataFrame([{'metric': 'precision', 'value': metrics['precision']}, {'metric': 'recall', 'value': metrics['recall']}, {'metric': 'f1', 'value': metrics['f1']}, {'metric': 'auroc', 'value': metrics['auroc']}, {'metric': 'auprc', 'value': metrics['auprc']}, {'metric': 'threshold', 'value': threshold}])
                confusion_df = pd.DataFrame(metrics['confusion_matrix'], index=['true_normal', 'true_anomaly'], columns=['pred_normal', 'pred_anomaly'])
                display(metrics_df)
                display(confusion_df)
                plot_df = sweep_results_df.copy().sort_values(['f1', 'auroc'], ascending=False).reset_index(drop=True)
                plot_df['label'] = plot_df['name']
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                axes[0].barh(plot_df['label'], plot_df['f1'], color='#355070')
                axes[0].set_title('PatchCore Variant F1')
                axes[0].invert_yaxis()
                axes[1].barh(plot_df['label'], plot_df['auroc'], color='#6d597a')
                axes[1].set_title('PatchCore Variant AUROC')
                axes[1].invert_yaxis()
                plt.tight_layout()
                fig.savefig(PLOTS_DIR / 'variant_comparison_metrics.png', dpi=200, bbox_inches='tight')
                plt.show()
                plt.close(fig)
            """
        ).lstrip(),
    )

    set_source(
        notebook,
        analysis_cell,
        textwrap.dedent(
            """
            if review_ready and metadata_ready and not test_scores_df.empty and len(test_metadata) == len(test_scores_df):
                analysis_df, error_summary_df, defect_recall_df = compute_failure_tables(test_metadata, test_scores_df, threshold)
                analysis_df.to_csv(RESULTS_DIR / 'selected_analysis_with_predictions.csv', index=False)
                error_summary_df.reset_index().to_csv(RESULTS_DIR / 'selected_error_summary.csv', index=False)
                defect_recall_df.reset_index().to_csv(RESULTS_DIR / 'selected_defect_recall.csv', index=False)
                display(error_summary_df)
                display(defect_recall_df)
                fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                axes[0].bar(error_summary_df.index.astype(str), error_summary_df['count'], color=VARIANT_COLOR_ANOMALY if 'VARIANT_COLOR_ANOMALY' in globals() else '#e76f51')
                axes[0].set_title(f'Prediction Outcome Counts\\n{selected_variant_name}')
                axes[0].set_ylabel('count')
                top_defects_df = defect_recall_df.head(10).reset_index()
                axes[1].barh(top_defects_df['defect_type'], top_defects_df['recall'], color=VARIANT_COLOR_DEFECT if 'VARIANT_COLOR_DEFECT' in globals() else '#8ab17d')
                axes[1].set_xlim(0.0, 1.0)
                axes[1].set_title('Top Defect-Type Recall')
                axes[1].set_xlabel('recall')
                axes[1].invert_yaxis()
                plt.tight_layout()
                fig.savefig(PLOTS_DIR / 'defect_breakdown.png', dpi=200, bbox_inches='tight')
                plt.show()
                plt.close(fig)
            else:
                analysis_df = pd.DataFrame()
                error_summary_df = pd.DataFrame()
                defect_recall_df = pd.DataFrame()
                print('[WARNING] Test metadata is unavailable or misaligned with cached scores. Skipping defect-analysis plots for this cell.')

            if sweep_results_df.empty and not selected_variant_name:
                rendered_variants_df = pd.DataFrame(columns=['variant_name', 'plots_dir', 'evaluation_dir'])
                print('[WARNING] Cached variant rendering is unavailable because no sweep artifacts were found.')
            else:
                variant_names = sweep_results_df['name'].astype(str).tolist() if ('RENDER_ALL_CACHED_VARIANTS' in globals() and RENDER_ALL_CACHED_VARIANTS) or ('RENDER_ALL_SAVED_VARIANTS' in globals() and RENDER_ALL_SAVED_VARIANTS) else []
                if 'VARIANTS_TO_RENDER' in globals():
                    variant_names.extend([str(name) for name in VARIANTS_TO_RENDER])
                if selected_variant_name:
                    variant_names.append(selected_variant_name)
                ordered_variant_names = []
                seen = set()
                for name in variant_names:
                    if name not in seen:
                        ordered_variant_names.append(name)
                        seen.add(name)
                rendered_rows = []
                for variant_name in ordered_variant_names:
                    payload = load_variant_outputs(variant_name)
                    render_info = render_variant_artifacts(variant_name, payload)
                    rendered_rows.append({'variant_name': variant_name, 'plots_dir': render_info['plots_dir'], 'evaluation_dir': render_info['evaluation_dir']})
                rendered_variants_df = pd.DataFrame(rendered_rows)
            display(rendered_variants_df)
            """
        ).lstrip(),
    )

    save_notebook(path, notebook)


def patch_patchcore_wrn_x224_multilayer_umap_followup(path: Path) -> None:
    notebook = load_notebook(path)
    set_source(
        notebook,
        15,
        textwrap.dedent(
            """
            REGENERATE_UMAP = False
            if not selected_variant or 'variant_umap_dir' not in selected_variant or 'variant_plots_dir' not in selected_variant:
                selected_umap_dir = None
                selected_plots_dir = None
                umap_png_path = None
                print('[WARNING] Selected-variant UMAP assets are unavailable because no cached variant could be restored.')
            else:
                selected_umap_dir = selected_variant['variant_umap_dir']
                selected_plots_dir = selected_variant['variant_plots_dir']
                umap_png_path = selected_plots_dir / 'umap_by_split.png'
                if REGENERATE_UMAP:
                    try:
                        import umap as umap_module
                        from wafer_defect.evaluation.umap_reference import export_reference_umap_bundle
                        required_npy = {'train_embeddings': selected_umap_dir / 'train_embeddings.npy', 'val_embeddings': selected_umap_dir / 'val_embeddings.npy', 'val_labels': selected_umap_dir / 'val_labels.npy', 'test_embeddings': selected_umap_dir / 'test_embeddings.npy', 'test_labels': selected_umap_dir / 'test_labels.npy'}
                        missing = [k for k, p in required_npy.items() if not p.exists()]
                        if missing:
                            raise FileNotFoundError(f'Missing embedding files (re-run training to regenerate): {missing}')
                        train_emb = np.load(required_npy['train_embeddings'])
                        val_emb = np.load(required_npy['val_embeddings'])
                        val_lbl = np.load(required_npy['val_labels'])
                        test_emb = np.load(required_npy['test_embeddings'])
                        test_lbl = np.load(required_npy['test_labels'])
                        umap_out = export_reference_umap_bundle(output_dir=selected_umap_dir, umap_module=umap_module, train_normal_embeddings=train_emb, val_embeddings=val_emb, val_labels=val_lbl, test_embeddings=test_emb, test_labels=test_lbl, pca_components=50, n_neighbors=15, min_dist=0.1, knn_k=15, metric='euclidean', random_state=42)
                        generated_png = selected_umap_dir / 'plots' / 'embedding_umap.png'
                        if generated_png.exists():
                            shutil.copy2(generated_png, umap_png_path)
                        print(f'UMAP regenerated and saved to {umap_png_path}')
                    except ImportError:
                        print('umap-learn not available. Install with: pip install umap-learn')
                    except FileNotFoundError as e:
                        print(f'Cannot regenerate UMAP: {e}')
                if umap_png_path.exists():
                    shutil.copy2(umap_png_path, PLOTS_DIR / 'selected_variant_umap_by_split.png')
                    display(Image(filename=str(umap_png_path)))
                else:
                    print(f'UMAP image not found at {umap_png_path}')
                    print('Set REGENERATE_UMAP = True to compute it (requires embedding .npy files).')
            """
        ).lstrip(),
    )

    set_source(
        notebook,
        17,
        textwrap.dedent(
            """
            if sweep_results_df.empty and not selected_variant_name:
                rendered_variants_df = pd.DataFrame(columns=['variant_name', 'plots_dir', 'evaluation_dir', 'checkpoint_present'])
                print('[WARNING] Cached variant rendering is unavailable because no sweep artifacts were found.')
            else:
                variant_names = sweep_results_df['name'].astype(str).tolist() if RENDER_ALL_SAVED_VARIANTS else []
                variant_names.extend([str(name) for name in VARIANTS_TO_RENDER])
                if selected_variant_name:
                    variant_names.append(selected_variant_name)
                ordered_variant_names = []
                seen = set()
                for name in variant_names:
                    if name not in seen:
                        ordered_variant_names.append(name)
                        seen.add(name)
                rendered_rows = []
                for variant_name in ordered_variant_names:
                    payload = load_variant_outputs(variant_name)
                    render_info = render_variant_artifacts(variant_name, payload)
                    rendered_rows.append({'variant_name': variant_name, 'plots_dir': render_info['plots_dir'], 'evaluation_dir': render_info['evaluation_dir'], 'checkpoint_present': (payload['variant_checkpoints_dir'] / 'best_model.pt').exists()})
                rendered_variants_df = pd.DataFrame(rendered_rows)
            display(rendered_variants_df)
            """
        ).lstrip(),
    )

    set_source(
        notebook,
        19,
        textwrap.dedent(
            """
            selected_variant_checkpoint = ''
            if selected_variant and 'variant_checkpoints_dir' in selected_variant:
                selected_variant_checkpoint = str(selected_variant['variant_checkpoints_dir'] / 'best_model.pt')
            saved_outputs = {'artifact_root': str(ARTIFACT_ROOT), 'results_dir': str(RESULTS_DIR), 'plots_dir': str(PLOTS_DIR), 'selected_variant_name': selected_variant_name, 'selected_variant_checkpoint': selected_variant_checkpoint, 'rendered_variants': rendered_variants_df['variant_name'].tolist() if 'variant_name' in rendered_variants_df.columns else []}
            saved_outputs
            """
        ).lstrip(),
    )
    save_notebook(path, notebook)


def patch_patchcore_weighted_x224(path: Path) -> None:
    notebook = load_notebook(path)
    set_source(
        notebook,
        3,
        textwrap.dedent(
            """
            from pathlib import Path
            import json
            import sys
            import matplotlib.pyplot as plt
            import pandas as pd
            from IPython.display import display

            cwd = Path.cwd().resolve()
            candidate_roots = [cwd, *cwd.parents]
            REPO_ROOT = None
            for candidate in candidate_roots:
                if (candidate / 'src' / 'wafer_defect').exists() and (candidate / 'configs').exists():
                    REPO_ROOT = candidate
                    break
            if REPO_ROOT is None:
                raise RuntimeError('Could not locate repo root containing src/wafer_defect and configs/')
            ARTIFACT_ROOT = REPO_ROOT / 'experiments/anomaly_detection/patchcore/wideresnet50/x224/weighted/artifacts/patchcore-wideresnet50-weighted'
            RESULTS_DIR = ARTIFACT_ROOT / 'results'
            PLOTS_DIR = ARTIFACT_ROOT / 'plots'
            sweep_results_path = RESULTS_DIR / 'patchcore_sweep_results.csv'
            sweep_summary_path = RESULTS_DIR / 'patchcore_sweep_summary.json'
            sweep_results_df = pd.DataFrame()
            sweep_summary = {}
            best_variant = {}
            if sweep_results_path.exists() and sweep_summary_path.exists():
                sweep_results_df = pd.read_csv(sweep_results_path)
                sweep_summary = json.loads(sweep_summary_path.read_text(encoding='utf-8'))
                best_variant = sweep_summary.get('best_variant', {})
                display(sweep_results_df.head())
                display(pd.Series(best_variant))
            else:
                print('[WARNING] Weighted sweep artifacts are missing. Review plots will be skipped until the artifacts folder is restored.')
            """
        ).lstrip(),
    )
    set_source(
        notebook,
        6,
        textwrap.dedent(
            """
            if sweep_results_df.empty:
                print('[WARNING] Weighted sweep comparison is unavailable because the cached sweep CSV is missing.')
            else:
                top_plot_df = sweep_results_df.sort_values(['f1', 'auroc'], ascending=False).head(15).copy()
                top_plot_df['label'] = top_plot_df['name']
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                axes[0].barh(top_plot_df['label'], top_plot_df['f1'], color='#264653')
                axes[0].set_title('Top Weighted Variants by F1')
                axes[0].invert_yaxis()
                axes[1].barh(top_plot_df['label'], top_plot_df['auroc'], color='#2a9d8f')
                axes[1].set_title('Top Weighted Variants by AUROC')
                axes[1].invert_yaxis()
                plt.tight_layout()
                fig.savefig(PLOTS_DIR / 'weighted_top_variants.png', dpi=200, bbox_inches='tight')
                plt.show()
                plt.close(fig)
                best_by_weight_df = sweep_results_df.sort_values(['weight_name', 'f1', 'auroc'], ascending=[True, False, False]).groupby('weight_name', as_index=False).first().sort_values('f1', ascending=False)
                best_by_weight_df.to_csv(RESULTS_DIR / 'best_by_weight_name.csv', index=False)
                display(best_by_weight_df)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.scatter(sweep_results_df['recall'], sweep_results_df['precision'], c=sweep_results_df['f1'], cmap='viridis', s=70)
                for _, row in best_by_weight_df.iterrows():
                    ax.text(row['recall'], row['precision'], row['weight_name'], fontsize=8)
                ax.set_xlabel('recall')
                ax.set_ylabel('precision')
                ax.set_title('Weighted Sweep Precision-Recall Tradeoff')
                plt.tight_layout()
                fig.savefig(PLOTS_DIR / 'weighted_precision_recall_scatter.png', dpi=200, bbox_inches='tight')
                plt.show()
                plt.close(fig)
            """
        ).lstrip(),
    )
    set_source(
        notebook,
        8,
        textwrap.dedent(
            """
            saved_outputs = {'artifact_root': str(ARTIFACT_ROOT), 'results_dir': str(RESULTS_DIR), 'plots_dir': str(PLOTS_DIR), 'best_variant_name': best_variant.get('name') if isinstance(best_variant, dict) else None}
            saved_outputs
            """
        ).lstrip(),
    )
    save_notebook(path, notebook)


def patch_svdd_baseline(path: Path) -> None:
    notebook = load_notebook(path)

    set_source(notebook, 3, notebook["cells"][3]["source"] if False else textwrap.dedent("""\
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import torch
import tomllib
try:
    from IPython.display import display
except ImportError:

    def display(obj: object) -> None:
        print(obj)

def warn_skip(message: str) -> None:
    print(f'[WARNING] {message}')

REPO_ROOT = Path.cwd().resolve()
if not (REPO_ROOT / 'src' / 'wafer_defect').exists():
    for candidate in [REPO_ROOT, *REPO_ROOT.parents]:
        if (candidate / 'src' / 'wafer_defect').exists():
            REPO_ROOT = candidate
            break
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from wafer_defect.config import load_toml
from wafer_defect.data.wm811k import WaferMapDataset
"""))

    set_source(notebook, 9, textwrap.dedent("""\
history_path = OUTPUT_DIR / 'results' / 'history.json'
summary_path = OUTPUT_DIR / 'results' / 'summary.json'
best_model_path = OUTPUT_DIR / 'checkpoints' / 'best_model.pt'
artifacts_ready = history_path.exists() and summary_path.exists() and best_model_path.exists()
training_ready = False
history_df = pd.DataFrame()
training_summary = {}
env = os.environ.copy()
env['PYTHONPATH'] = str(SRC_ROOT) if not env.get('PYTHONPATH') else str(SRC_ROOT) + os.pathsep + env['PYTHONPATH']
if RETRAIN:
    train_cmd = [sys.executable, 'scripts/train_svdd.py', '--config', str(CONFIG_PATH.relative_to(REPO_ROOT))]
    print('Running:', ' '.join(train_cmd))
    subprocess.run(train_cmd, cwd=REPO_ROOT, env=env, check=True)
elif artifacts_ready:
    print(f'Found existing training artifacts in {OUTPUT_DIR}. Skipping retraining.')
else:
    warn_skip('Saved training artifacts are missing and RETRAIN is False. Skipping this section.')
if history_path.exists():
    history = json.loads(history_path.read_text(encoding='utf-8'))
    history_df = pd.DataFrame(history)
if summary_path.exists():
    training_summary = json.loads(summary_path.read_text(encoding='utf-8'))
training_ready = not history_df.empty
if training_summary:
    display(pd.DataFrame([training_summary]))
elif not training_ready:
    warn_skip('Training summary is unavailable because summary.json is missing.')
"""))

    set_source(notebook, 11, textwrap.dedent("""\
if not training_ready:
    warn_skip('Training curves are unavailable because history.json is missing or empty.')
else:
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.plot(history_df['epoch'], history_df['train_loss'], label='train')
    ax.plot(history_df['epoch'], history_df['val_loss'], label='val')
    ax.set_title('SVDD Distance Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    fig.savefig(PLOTS_DIR / 'training_curves.png', dpi=160, bbox_inches='tight')
    display(fig)
    plt.close(fig)
"""))

    set_source(notebook, 13, textwrap.dedent("""\
evaluation_summary_path = EVALUATION_DIR / 'summary.json'
val_scores_path = EVALUATION_DIR / 'val_scores.csv'
test_scores_path = EVALUATION_DIR / 'test_scores.csv'
threshold_sweep_path = EVALUATION_DIR / 'threshold_sweep.csv'
evaluation_ready = False
evaluation_summary = {}
val_scores_df = pd.DataFrame()
test_scores_df = pd.DataFrame()
threshold_sweep_df = pd.DataFrame()
threshold = float('nan')
best_sweep = {}
if RERUN_EVALUATION:
    if not best_model_path.exists():
        warn_skip(f'Best checkpoint not found at {best_model_path}. Evaluation-backed cells will be skipped.')
    else:
        eval_cmd = [sys.executable, 'scripts/evaluate_reconstruction_model.py', '--checkpoint', str(best_model_path.relative_to(REPO_ROOT)), '--config', str(CONFIG_PATH.relative_to(REPO_ROOT)), '--output-dir', str(EVALUATION_DIR.relative_to(REPO_ROOT))]
        print('Running:', ' '.join(eval_cmd))
        subprocess.run(eval_cmd, cwd=REPO_ROOT, env=env, check=True)
elif not all((path.exists() for path in [evaluation_summary_path, val_scores_path, test_scores_path, threshold_sweep_path])):
    warn_skip('Saved evaluation artifacts are missing and RERUN_EVALUATION is False. Skipping this section.')
if all((path.exists() for path in [evaluation_summary_path, val_scores_path, test_scores_path, threshold_sweep_path])):
    evaluation_summary = json.loads(evaluation_summary_path.read_text(encoding='utf-8'))
    val_scores_df = pd.read_csv(val_scores_path)
    test_scores_df = pd.read_csv(test_scores_path)
    threshold_sweep_df = pd.read_csv(threshold_sweep_path)
    threshold = float(evaluation_summary['threshold'])
    best_sweep = dict(evaluation_summary.get('best_threshold_sweep', {}))
    evaluation_ready = True
    display(pd.DataFrame([evaluation_summary['metrics_at_validation_threshold']]))
    display(pd.DataFrame([best_sweep]))
"""))

    set_source(notebook, 15, textwrap.dedent("""\
if not evaluation_ready:
    warn_skip('Evaluation plots are unavailable because the cached evaluation artifacts were not loaded.')
else:
    focus_low = float(threshold_sweep_df['threshold'].quantile(0.01))
    focus_high = float(threshold_sweep_df['threshold'].quantile(0.99))
    focus_low = min(focus_low, threshold, float(best_sweep['threshold']))
    focus_high = max(focus_high, threshold, float(best_sweep['threshold']))
    focus_pad = max((focus_high - focus_low) * 0.1, 1e-06)
    focus_low = max(0.0, focus_low - focus_pad)
    focus_high = focus_high + focus_pad
    cm = evaluation_summary['metrics_at_validation_threshold'].get('confusion_matrix', [[0, 0], [0, 0]])
    cm_df = pd.DataFrame(cm, index=['true_normal', 'true_anomaly'], columns=['pred_normal', 'pred_anomaly'])
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.hist(test_scores_df.loc[test_scores_df['is_anomaly'] == 0, 'score'], bins=30, alpha=0.7, label='normal')
    ax.hist(test_scores_df.loc[test_scores_df['is_anomaly'] == 1, 'score'], bins=30, alpha=0.7, label='anomaly')
    ax.axvline(threshold, color='red', linestyle='--', label=f'val threshold = {threshold:.2e}')
    ax.set_title('SVDD Test Score Distribution')
    ax.set_xlabel('L2 distance from SVDD center')
    ax.set_ylabel('Count')
    ax.legend()
    fig.savefig(PLOTS_DIR / 'score_distribution.png', dpi=160, bbox_inches='tight')
    display(fig)
    plt.close(fig)
    display(cm_df)
    display(threshold_sweep_df.sort_values('f1', ascending=False).head(10))
"""))

    set_source(notebook, 17, textwrap.dedent("""\
if not evaluation_ready:
    warn_skip('Top-scored example review is unavailable because the cached evaluation artifacts were not loaded.')
else:
    ranked_test_scores_df = test_scores_df.sort_values('score', ascending=False).reset_index(drop=True)
    top_k = min(TOP_K, len(ranked_test_scores_df))
    rows = 2
    cols = max(1, top_k // 2)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), constrained_layout=True)
    axes = axes.ravel() if hasattr(axes, 'ravel') else [axes]
    for idx in range(top_k):
        row = ranked_test_scores_df.iloc[idx]
        wafer_map, label = test_dataset[int(row['sample_index'])]
        axes[idx].imshow(wafer_map[0], cmap='gray')
        axes[idx].set_title(f"score={row['score']:.4f}\\nlabel={int(label)}")
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
    for idx in range(top_k, len(axes)):
        axes[idx].axis('off')
    fig.savefig(PLOTS_DIR / 'top_scored_examples.png', dpi=160, bbox_inches='tight')
    display(fig)
    plt.close(fig)
"""))

    save_notebook(path, notebook)


def patch_vae_sweep(path: Path, value_col: str, tag_expr: str, best_label: str) -> None:
    notebook = load_notebook(path)
    set_source(notebook, 3, textwrap.dedent("""\
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
try:
    from IPython.display import display
except ImportError:

    def display(obj: object) -> None:
        print(obj)

def warn_skip(message: str) -> None:
    print(f'[WARNING] {message}')

REPO_ROOT = Path.cwd().resolve()
if not (REPO_ROOT / 'src' / 'wafer_defect').exists():
    for candidate in [REPO_ROOT, *REPO_ROOT.parents]:
        if (candidate / 'src' / 'wafer_defect').exists():
            REPO_ROOT = candidate
            break
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from wafer_defect.config import load_toml
"""))

    set_source(notebook, 9, textwrap.dedent(f"""\
sweep_ready = False
summary_payload = {{}}
{value_col}_sweep_df = pd.DataFrame()
if summary_path.exists():
    summary_payload = json.loads(summary_path.read_text(encoding='utf-8'))
    {value_col}_sweep_df = pd.DataFrame(summary_payload['results']).sort_values('val_threshold_f1', ascending=False).reset_index(drop=True)
    sweep_ready = not {value_col}_sweep_df.empty
    display({value_col}_sweep_df)
else:
    warn_skip('Saved sweep artifacts are missing, so the cached sweep review cells will be skipped.')
"""))

    set_source(notebook, 11, textwrap.dedent(f"""\
if not sweep_ready:
    warn_skip('Sweep metric plots are unavailable because the saved sweep summary is missing.')
else:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    sorted_df = {value_col}_sweep_df.sort_values('{value_col}').reset_index(drop=True)
    axes[0].plot(sorted_df['{value_col}'], sorted_df['val_threshold_f1'], marker='o', label='val-threshold F1')
    axes[0].plot(sorted_df['{value_col}'], sorted_df['best_sweep_f1'], marker='s', label='best-sweep F1')
    axes[0].set_title('F1 vs {best_label}')
    axes[0].set_xlabel('{value_col}')
    axes[0].legend()
    axes[1].plot(sorted_df['{value_col}'], sorted_df['auroc'], marker='o', color='#1f77b4')
    axes[1].set_title('AUROC vs {best_label}')
    axes[1].set_xlabel('{value_col}')
    axes[2].plot(sorted_df['{value_col}'], sorted_df['auprc'], marker='o', color='#2ca02c')
    axes[2].set_title('AUPRC vs {best_label}')
    axes[2].set_xlabel('{value_col}')
    fig.savefig(PLOTS_DIR / '{value_col}_sweep_metrics.png', dpi=160, bbox_inches='tight')
    display(fig)
    plt.close(fig)
"""))

    set_source(notebook, 13, textwrap.dedent(f"""\
if not sweep_ready:
    warn_skip('Training-curve comparison is unavailable because the saved sweep summary is missing.')
else:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)
    plotted_any = False
    for sweep_value in sorted({value_col}_sweep_df['{value_col}'].tolist()):
        tag = {tag_expr}
        history_path = SWEEP_ROOT / tag / 'results' / 'history.json'
        if not history_path.exists():
            warn_skip(f'Missing training history for {{tag}}: {{history_path}}. Skipping this variant in the plot.')
            continue
        history_df = pd.DataFrame(json.loads(history_path.read_text(encoding='utf-8')))
        axes[0].plot(history_df['epoch'], history_df['val_loss'], label=f'{value_col}={{sweep_value}}')
        axes[1].plot(history_df['epoch'], history_df['val_reconstruction_loss'], label=f'{value_col}={{sweep_value}}')
        plotted_any = True
    if not plotted_any:
        plt.close(fig)
        warn_skip('No reusable training-history artifacts were found for the sweep comparison cell.')
    else:
        axes[0].set_title('Validation Total Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].legend(fontsize=8)
        axes[1].set_title('Validation Reconstruction Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].legend(fontsize=8)
        fig.savefig(PLOTS_DIR / '{value_col}_sweep_training_curves.png', dpi=160, bbox_inches='tight')
        display(fig)
        plt.close(fig)
"""))

    set_source(notebook, 15, textwrap.dedent(f"""\
if not sweep_ready:
    warn_skip('Best-run inspection is unavailable because the saved sweep summary is missing.')
else:
    best_row = {value_col}_sweep_df.iloc[0].to_dict()
    best_value = best_row['{value_col}']
    best_tag = {tag_expr.replace('sweep_value', 'best_value')}
    best_run_dir = SWEEP_ROOT / best_tag
    best_eval_summary_path = best_run_dir / 'results' / 'evaluation' / 'summary.json'
    best_test_scores_path = best_run_dir / 'results' / 'evaluation' / 'test_scores.csv'
    best_threshold_sweep_path = best_run_dir / 'results' / 'evaluation' / 'threshold_sweep.csv'
    if not all((path.exists() for path in [best_eval_summary_path, best_test_scores_path, best_threshold_sweep_path])):
        warn_skip(f'Best-run evaluation artifacts are missing for {{best_tag}}. Skipping this cell.')
    else:
        best_eval_summary = json.loads(best_eval_summary_path.read_text(encoding='utf-8'))
        best_test_scores_df = pd.read_csv(best_test_scores_path)
        best_threshold_sweep_df = pd.read_csv(best_threshold_sweep_path)
        cm = best_eval_summary['metrics_at_validation_threshold'].get('confusion_matrix', [[0, 0], [0, 0]])
        cm_df = pd.DataFrame(cm, index=['true_normal', 'true_anomaly'], columns=['pred_normal', 'pred_anomaly'])
        print(f'Best {best_label} by validation-threshold F1: {{best_value}}')
        print(f'Best run directory: {{best_run_dir}}')
        display(pd.DataFrame([best_row]))
        display(pd.DataFrame([best_eval_summary['metrics_at_validation_threshold']]))
        display(pd.DataFrame([best_eval_summary['best_threshold_sweep']]))
        display(cm_df)
"""))

    save_notebook(path, notebook)


def main() -> None:
    for rel in [
        'experiments/anomaly_detection/autoencoder/x64/baseline/notebook.ipynb',
        'experiments/anomaly_detection/autoencoder/x64/batchnorm/notebook.ipynb',
        'experiments/anomaly_detection/autoencoder/x64/residual/notebook.ipynb',
    ]:
        patch_autoencoder_x64(REPO_ROOT / rel)

    patch_vae_baseline(REPO_ROOT / 'experiments/anomaly_detection/vae/x64/baseline/notebook.ipynb')
    patch_vae_baseline(REPO_ROOT / 'experiments/anomaly_detection/vae/x224/main/notebook.ipynb')
    patch_vae_x224_metadata(REPO_ROOT / 'experiments/anomaly_detection/vae/x224/main/notebook.ipynb')
    patch_svdd_baseline(REPO_ROOT / 'experiments/anomaly_detection/svdd/x64/baseline/notebook.ipynb')
    patch_vae_sweep(
        REPO_ROOT / 'experiments/anomaly_detection/vae/x64/beta_sweep/notebook.ipynb',
        value_col='beta',
        tag_expr="f\"beta_{str(sweep_value).replace('.', 'p')}\"",
        best_label='beta',
    )
    patch_vae_sweep(
        REPO_ROOT / 'experiments/anomaly_detection/vae/x64/latent_dim_sweep/notebook.ipynb',
        value_col='latent_dim',
        tag_expr='latent_dim_tag(sweep_value)',
        best_label='latent-dim',
    )
    patch_patchcore_wrn_x224_review(
        REPO_ROOT / 'experiments/anomaly_detection/patchcore/wideresnet50/x224/layer2/notebook.ipynb',
        metadata_cell=6,
        selected_cell=10,
        analysis_cell=12,
    )
    patch_patchcore_wrn_x224_review(
        REPO_ROOT / 'experiments/anomaly_detection/patchcore/wideresnet50/x224/layer3/notebook.ipynb',
        metadata_cell=6,
        selected_cell=10,
        analysis_cell=12,
    )
    patch_patchcore_wrn_x224_review(
        REPO_ROOT / 'experiments/anomaly_detection/patchcore/wideresnet50/x224/multilayer_umap/notebook.ipynb',
        metadata_cell=7,
        selected_cell=11,
        analysis_cell=13,
    )
    patch_patchcore_wrn_x224_multilayer_umap_followup(
        REPO_ROOT / 'experiments/anomaly_detection/patchcore/wideresnet50/x224/multilayer_umap/notebook.ipynb'
    )
    patch_patchcore_weighted_x224(
        REPO_ROOT / 'experiments/anomaly_detection/patchcore/wideresnet50/x224/weighted/notebook.ipynb'
    )


if __name__ == '__main__':
    main()
