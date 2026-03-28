# Overleaf / ChatGPT Handoff For Final Wafer Report

Use this note as the briefing to ChatGPT when converting the repo's experiment record into a polished Overleaf LaTeX report.

## Goal

Produce a streamlined, complete technical report in LaTeX for Overleaf that:

- introduces the wafer anomaly-detection problem clearly
- explains the wafer-manufacturing meaning of false positives and false negatives
- explains the dataset and split design
- explains why training uses only normal wafers instead of mixing defects into training
- presents the experiment sequence as a logical progression rather than as an unstructured lab notebook
- uses the saved figures and tables from this repo
- ends with a grounded interpretation of metrics, UMAP geometry, and threshold-selection limitations

The tone should feel like a proper project report or conference-style methods/results paper, not a changelog.

## Main Source File

Primary narrative source:

- `REPORT.md`

This is the main experiment record and should be treated as the backbone source for the final writeup.

## Additional Files To Pass To ChatGPT

These files should be provided together with `REPORT.md` so ChatGPT has enough context to produce a conclusive LaTeX report.

### Core methods / dataset files

- `scripts/prepare_wm811k.py`
- `configs/data/data.toml`

These establish:

- dataset origin from `LSWMD.pkl`
- normal/defect label handling
- split generation logic
- the `50k_5pct` split configuration

### Training-config files for methods section accuracy

- `configs/training/train_autoencoder.toml`
- `configs/training/train_autoencoder_batchnorm.toml`
- `configs/training/train_autoencoder_batchnorm_dropout.toml`
- `configs/training/train_autoencoder_residual.toml`
- `configs/training/train_resnet18_backbone.toml`
- `configs/training/train_patchcore_resnet18.toml`
- `configs/training/train_patchcore_resnet50.toml`
- `configs/training/train_ts_resnet18.toml`
- `configs/training/train_ts_resnet50.toml`
- `configs/training/train_wideresnet50_backbone.toml`

These are useful for:

- model naming consistency
- input-size / backbone details
- score reduction names
- making the methods section less hand-wavy

### Holdout evaluation files

- `artifacts/x64/holdout70k_3p5k_evaluations/leaderboard.csv`
- `artifacts/x64/holdout70k_3p5k_evaluations/compiled_full.csv`
- `artifacts/x64/holdout70k_3p5k_evaluations/_histograms_all/histogram_run_summary.csv`

These are needed for:

- the expanded test-set discussion
- holdout leaderboard tables
- choosing only the histogram outputs that really belong to the `70k / 3.5k` evaluation

### Figure folders

- `artifacts/report_plots/`
- `artifacts/umaps/`
- `artifacts/x64/holdout70k_3p5k_evaluations/_histograms_all/`

These contain:

- report summary plots
- UMAP figures
- holdout histogram figures

## Important Cautions

### 1. `REPORT.md` is the main source of truth for narrative and metrics

If there is a conflict between casual assumptions and the report wording, prioritize `REPORT.md` unless the code or CSV files clearly contradict it.

### 2. `collated_scores.csv` should not be used

The file:

- `artifacts/x64/holdout70k_3p5k_evaluations/collated_scores.csv`

is empty and should be ignored.

### 3. Not every histogram file in `_histograms_all` is actually from the `70k / 3.5k` holdout

Before selecting histogram figures, filter by:

- `n_test_normal = 70000`
- `n_test_anomaly = 3500`

using:

- `artifacts/x64/holdout70k_3p5k_evaluations/_histograms_all/histogram_run_summary.csv`

Some entries in that folder are older small-test exports and should not be mislabeled as expanded-holdout evidence.

### 4. Not every UMAP corresponds to the best row of a sweep

Example:

- the ResNet18 PatchCore UMAP in `artifacts/umaps/patchcore_resnet18_10A/max_mb50k/...` is a qualitative export from the saved `max_mb50k` artifact
- it should not be described as the selected best ResNet18 PatchCore variant, because the best reported ResNet18 PatchCore row is `mean_mb50k`

The report already notes this distinction. Preserve that caution.

### 5. The expanded-holdout bundle does not yet include every newest winner

In particular, the later WRN `x224` PatchCore winner from the main report is not part of that holdout leaderboard bundle. Do not claim the holdout leaderboard is a full replacement for the full main-report ranking.

## Explicit Context ChatGPT Should Use

These points should be treated as author intent.

### Problem framing

This project is about anomaly detection on wafer maps, not supervised defect classification.

The system is intended to learn what a normal wafer looks like, then flag wafers whose spatial patterns deviate from that learned normal manifold.

### Why train on normal wafers only

Use this rationale directly:

> We train only on normal wafers because the project is framed as anomaly detection, labeled defect wafers are heterogeneous and incomplete, and mixing defects into training would bias the model toward known defect patterns instead of learning a robust model of normal behavior.

You may rephrase it, but preserve the meaning.

### Manufacturing meaning of false positives and false negatives

Use this framing explicitly:

- False positive: a normal wafer flagged as anomalous. This causes unnecessary manual review, reinspection, possible scrap, and throughput loss.
- False negative: a defective wafer missed by the model. This allows defect escape downstream and is usually the more serious manufacturing failure.

### Threshold policy

The main deployed threshold in the report is:

- the `95th` percentile of validation-normal scores

This is the fair deployment-style threshold.

The best threshold sweep is:

- analysis only
- optimistic because it uses test labels
- useful to show ceiling / threshold sensitivity, but not the main operating point

### UMAP motivation

The report's UMAP discussion should support this idea:

- metrics alone show performance, but UMAP helps reveal overlap structure, local clustering, and why the fixed `95th`-percentile threshold may be reasonable yet still naive
- the UMAPs suggest future thresholding work could use richer structure than a single percentile cutoff

Do not overclaim that UMAP itself is a decision rule. It is an interpretive diagnostic.

## Suggested Report Structure

ChatGPT should use a structure close to this:

1. Introduction
2. Problem Context and Manufacturing Motivation
3. Dataset and Split Construction
4. Why Normal-Only Training Fits the Problem
5. Experimental Methodology
6. Logical Experiment Progression
7. Main Results
8. Expanded Holdout Validation
9. UMAP-Based Embedding Analysis
10. Discussion of Thresholding and Limitations
11. Conclusion and Next Steps

## Desired Experiment Narrative

The report should present the experiments as a logical flow:

1. Start with the plain autoencoder baseline to establish the anomaly-detection pipeline.
2. Explore AE variants such as BatchNorm, dropout, residual changes, and resolution to test whether better in-domain training is enough.
3. Conclude that own-training reconstruction models help, but local defect sensitivity remains limited.
4. Move to teacher-student and pretrained-backbone methods once it becomes clear that stronger frozen feature extractors are needed.
5. Test simple pretrained embedding baselines first, showing that backbone strength alone is not enough if scoring remains global.
6. Move to PatchCore and other local-anomaly methods to exploit pretrained spatial features.
7. Explore backbone choice and image-size effects, especially the jump to direct `224x224`.
8. Use UMAP as a later-stage diagnostic once ranking metrics show that thresholding and manifold overlap are now a bottleneck.

## Figure Usage Guidance

ChatGPT should preferentially use:

### Main summary plots

From `artifacts/report_plots/`:

- `overall_experiment_comparison.png`
- `autoencoder_family_comparison.png`
- `compact_baseline_comparison.png`
- `patchcore_family_comparison.png`
- `ts_family_comparison.png`
- `wrn_family_comparison.png`

### Holdout plots

From `artifacts/x64/holdout70k_3p5k_evaluations/_histograms_all/`, but only after confirming they are true `70k / 3.5k` rows in `histogram_run_summary.csv`.

Good representative choices already discussed in the report:

- `ts_resnet50.png`
- `autoencoder_batchnorm__max_abs.png`
- `patchcore_resnet50__mean_mb50k.png`

### UMAP plots

From `artifacts/umaps/`:

- `wideresnet50A_embedding_baseline/evaluation/plots/embedding_umap.png`
- `patchcore_resnet18_10A/max_mb50k/evaluation/plots/umap_by_split.png`
- `patchcore_resnet18_10A/max_mb50k/evaluation/plots/umap_by_score.png`
- `18A2-patchcore-wideresnet50-multilayer-umap/topk_mb50k_r005_x224/plots/umap_by_split.png`

## What ChatGPT Should Not Do

- Do not rewrite the work as if all experiments were equally important.
- Do not flatten the progression into a generic model zoo comparison.
- Do not describe the best-threshold sweep as the deployed result.
- Do not imply the expanded holdout contains every latest main-report winner.
- Do not claim clean anomaly/normal separation where the UMAPs or histograms show substantial overlap.

## Output Request To ChatGPT

Ask ChatGPT to produce:

1. A polished Overleaf-ready LaTeX report body.
2. Suggested figure placements with `\\includegraphics` paths.
3. Suggested table placements for:
   - main experiment leaderboard
   - expanded holdout leaderboard
4. A short abstract.
5. A concise conclusion section.

If needed, ask for:

- a main-paper version
- and a shorter appendix version for the extra sweep details

## Copy-Paste Prompt For ChatGPT

Use the following prompt together with the files listed above:

> I want you to convert the attached `REPORT.md` and supporting repo files into a polished Overleaf LaTeX technical report. Please keep the experiment story logical rather than chronological-noise-heavy: start from the problem, manufacturing context, dataset and split design, explain why anomaly detection is trained on pure normal wafers, then present the experiment progression from autoencoder baselines to AE variants, then teacher-student / pretrained backbones, then PatchCore and image-size experiments, and finally UMAP-based interpretation and thresholding limitations. Use the attached CSVs and figures as evidence. Treat the 95th percentile of validation-normal scores as the main deployment threshold, and treat best-threshold sweeps as analysis only. Be careful that the expanded holdout histogram folder contains mixed exports, so only use rows confirmed by `histogram_run_summary.csv` with `70000` normal and `3500` anomaly test samples. Also be careful that not every UMAP corresponds to the best sweep row; preserve the distinctions already stated in the report. Please produce an Overleaf-ready LaTeX report body with section structure, figures, tables, and captions.

