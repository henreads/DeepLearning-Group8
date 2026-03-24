## 120k Labeled Anomaly Scripts

These are repository-friendly CLI wrappers for the larger labeled WRN50 PatchCore workflow.

They do not replace the notebooks under `notebooks/anomaly_120k_labeled/`; they mirror the main experiment entry points in plain Python so the run logic is easier to diff and log in Git.

- `run_patchcore_wrn50_120k.py`
  Runs the baseline `120k / 10k / 20k` WRN50 PatchCore sweep and writes the same bundle-style outputs as notebook `1`.
- `run_patchcore_wrn50_120k_memorybank_sweep.py`
  Runs the notebook `4` memory-bank and feature sweep from the command line and writes `notebook4_results.csv` plus per-experiment folders.
- `review_patchcore_wrn50_120k_thresholds.py`
  Reproduces the threshold-policy table from notebook `3` for a saved baseline bundle.

Example usage:

```powershell
python scripts/anomaly_120k_labeled/run_patchcore_wrn50_120k.py
python scripts/anomaly_120k_labeled/run_patchcore_wrn50_120k_memorybank_sweep.py --group coverage_sampling
python scripts/anomaly_120k_labeled/review_patchcore_wrn50_120k_thresholds.py
```
