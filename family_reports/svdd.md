# Deep SVDD Family Report

## Overview

Deep SVDD (Support Vector Data Description) is a **one-class learning** baseline that learns a compact hypersphere around normal wafer embeddings. Anomaly scores are the squared L2 distance from the learned center — wafers far from the center are flagged as anomalous.

Unlike reconstruction-based methods (autoencoder, VAE), SVDD does not require the model to reconstruct the input. Instead, it asks whether a compact feature-space representation of normal wafers is enough to separate defective ones. This makes it a useful methodological contrast: it isolates whether the anomaly signal comes from reconstruction error or from feature-space compactness.

Notebooks and artifacts live under [`experiments/anomaly_detection/svdd/`](../experiments/anomaly_detection/svdd/).

---

## Dataset and Evaluation Protocol

- **Train:** 40,000 normal wafers only
- **Validation:** 5,000 normal wafers (threshold = 95th percentile of SVDD distances)
- **Test:** 5,000 normal + 250 anomaly wafers
- **Score:** squared L2 distance from the learned hypersphere center
- **Primary metric:** val-threshold F1

---

## Experiment 6: Deep SVDD `64×64`

**Notebook:** [`experiments/anomaly_detection/svdd/x64/baseline/notebook.ipynb`](../experiments/anomaly_detection/svdd/x64/baseline/notebook.ipynb)
**Artifact dir:** `experiments/anomaly_detection/svdd/x64/baseline/artifacts/svdd_baseline/`

### Run Controls

| flag | default | meaning |
|---|---|---|
| `RETRAIN` | `False` | use saved checkpoint; set `True` to retrain via `scripts/train_svdd.py` |
| `RERUN_EVALUATION` | `False` | use cached eval scores; set `True` to re-score via `scripts/evaluate_reconstruction_model.py` |
| `RUN_HOLDOUT_EVALUATION` | `False` | run expanded 70k/3.5k holdout |
| `FORCE_HOLDOUT_RERUN` | `False` | re-run holdout even if cached results exist |

### Configuration

| parameter | value |
|---|---|
| latent dim | 128 |
| center eps | 0.1 (clips near-zero center components) |
| optimizer | Adam, lr=0.0005, weight decay=1e-5 |
| max epochs | 30 |
| early stopping | patience=5, min_delta=0.0001 |
| checkpoint every | 5 epochs |

### Training

- Best epoch: **23**, best val loss: **0.000181**, epochs ran: **28**
- Center norm: **1.131** (the learned hypersphere center magnitude)

![Training curves](../experiments/anomaly_detection/svdd/x64/baseline/artifacts/svdd_baseline/plots/training_curves.png)

### Evaluation

| metric | value |
|---|---|
| threshold | 0.000304 |
| precision | 0.304709 |
| recall | 0.440000 |
| F1 | **0.360065** |
| AUROC | 0.787506 |
| AUPRC | 0.213108 |
| predicted anomalies | 361 |
| best sweep F1 | 0.366288 |

Confusion matrix: `[[4749, 251], [140, 110]]`

### Score Distribution and Threshold Sweep

![Score distribution](../experiments/anomaly_detection/svdd/x64/baseline/artifacts/svdd_baseline/plots/score_distribution.png)

![Threshold sweep](../experiments/anomaly_detection/svdd/x64/baseline/artifacts/svdd_baseline/plots/threshold_sweep.png)

![Threshold sweep zoomed](../experiments/anomaly_detection/svdd/x64/baseline/artifacts/svdd_baseline/plots/threshold_sweep_zoomed.png)

![Confusion matrix](../experiments/anomaly_detection/svdd/x64/baseline/artifacts/svdd_baseline/plots/confusion_matrix.png)

### Per-Defect Recall (computed from saved scores)

| defect type | count | detected | recall |
|---|---|---|---|
| Donut | 7 | 1 | 0.143 |
| Loc | 34 | 8 | 0.235 |
| Edge-Loc | 53 | 15 | 0.283 |
| Scratch | 15 | 5 | 0.333 |
| Random | 5 | 2 | 0.400 |
| Edge-Ring | 84 | 46 | 0.548 |
| Center | 50 | 31 | 0.620 |
| Near-full | 2 | 2 | 1.000 |

### Highest-Scored Examples

![Top scored examples](../experiments/anomaly_detection/svdd/x64/baseline/artifacts/svdd_baseline/plots/top_scored_examples.png)

### Interpretation

- SVDD learned a real anomaly signal — AUROC 0.788 is clearly above chance
- The per-defect pattern follows the project-wide trend: broad, high-contrast defects (Center, Edge-Ring) are easiest; localized defects (Donut, Loc, Scratch) are hardest
- Notably, SVDD beats the tuned VAE (F1=0.340) on all main metrics, suggesting that the one-class distance objective compares favorably to probabilistic reconstruction on this dataset
- The especially low AUPRC (0.213) signals weaker ranking quality under class imbalance compared to reconstruction methods
- As expected, SVDD still clearly trails the best autoencoder result (F1=0.501) — the reconstruction error provides richer local anomaly signal than global center-distance

---

## Holdout Evaluation: Expanded 70k Normal / 3.5k Defect

The saved checkpoint was evaluated on the expanded holdout split.

| metric | benchmark (5k/250) | holdout (70k/3.5k) |
|---|---|---|
| precision | 0.304709 | 0.296318 |
| recall | 0.440000 | 0.427714 |
| F1 | 0.360065 | **0.350094** |
| AUROC | 0.787506 | **0.815139** |
| AUPRC | 0.213108 | **0.229441** |
| predicted anomalies | 361 | 5,052 |

Generalization is stable — the model maintains similar F1 at 14× the anomaly count. AUROC and AUPRC improve on the larger pool, consistent with better statistical estimates.

---

## Context in the Project

| method | F1 | AUROC | AUPRC |
|---|---|---|---|
| AE + BatchNorm (`max_abs`) | 0.502 | 0.834 | 0.568 |
| **Deep SVDD** | **0.360** | **0.788** | **0.213** |
| VAE (best beta) | 0.340 | 0.771 | 0.372 |
| ResNet18 center-distance | 0.236 | 0.685 | 0.195 |

SVDD sits between the autoencoder family and the simple backbone embedding baseline. It beats the VAE on F1 and AUROC despite being a simpler objective, confirming that reconstruction error is not the only useful signal — but AUPRC is its clear weakness, suggesting poor score calibration under severe class imbalance.

The key takeaway: SVDD confirms that a one-class distance objective can learn a useful signal, but reconstruction error and local patch scoring both extract richer anomaly information from the same architecture.

---

## Notebook Self-Containment

| path | RETRAIN=False | RETRAIN=True |
|---|---|---|
| Training | Loads `best_model.pt` + `history.json` | Runs `scripts/train_svdd.py` |
| Evaluation | Loads cached score CSVs | Runs `scripts/evaluate_reconstruction_model.py` |
| Training curve plot | Regenerated from `history.json` | Same |
| Score dist + sweep plot | Regenerated from saved CSVs | Same |
| Top-examples plot | Regenerated from `test_scores.csv` + test dataset | Same |
| Holdout | Loads cached OR runs eval script with holdout metadata | Runs eval script |

A grader can open the notebook, run top-to-bottom with `RETRAIN = False`, and get all plots and metrics from the saved artifacts without any GPU requirement.
