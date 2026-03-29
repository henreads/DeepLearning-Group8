## 120k Labeled Anomaly Scripts

These are repository-friendly CLI wrappers for the larger labeled WRN50 PatchCore workflow.

They do not replace the notebooks under `notebooks/anomaly_120k_labeled/`; they mirror the main experiment entry points in plain Python so the run logic is easier to diff and log in Git.

- `run_patchcore_wrn50_120k.py`
  Runs the baseline `120k / 10k / 20k` WRN50 PatchCore sweep and writes the same bundle-style outputs as notebook `1`.
- `run_patchcore_wrn50_120k_memorybank_sweep.py`
  Runs the notebook `4` memory-bank and feature sweep from the command line and writes `notebook4_results.csv` plus per-experiment folders.
- `review_patchcore_wrn50_120k_thresholds.py`
  Reproduces the threshold-policy table from notebook `3` for a saved baseline bundle.
- `evaluate_best_patchcore_large_eval.py`
  Rebuilds the best saved notebook-4 PatchCore recipe without retraining, scores a larger labeled pool or the full labeled pool, retunes the threshold on a separate calibration slice, and reports both holdout and full-pool threshold comparisons.
- `build_kaggle_large_eval_bundle.py`
  Packages the large-eval script, helper modules, saved notebook-4 results, cached WRN50 torchvision weights, and a Kaggle notebook template into one uploadable bundle.

Example usage:

```powershell
python scripts/anomaly_120k_labeled/run_patchcore_wrn50_120k.py
python scripts/anomaly_120k_labeled/run_patchcore_wrn50_120k_memorybank_sweep.py --group coverage_sampling
python scripts/anomaly_120k_labeled/review_patchcore_wrn50_120k_thresholds.py
python scripts/anomaly_120k_labeled/evaluate_best_patchcore_large_eval.py --evaluation-total 0 --calibration-size 10000
python scripts/anomaly_120k_labeled/build_kaggle_large_eval_bundle.py
```

Notes:

- `--evaluation-total 0` means "score the entire chosen pool".
- `--evaluation-pool full_labeled` scores all labeled rows, including rows that may overlap the source train split used to build the memory bank.
- `--evaluation-pool unseen_labeled` excludes source-train rows from the scored pool.
- The script now writes both holdout-report metrics and full scored-pool metrics so threshold retuning can be compared on a larger population without losing the cleaner holdout view.
