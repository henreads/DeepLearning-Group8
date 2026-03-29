# `scripts/dev/` Guide

This folder contains maintenance utilities used to clean, reorganize, and standardize the repo during the submission pass.

These scripts are useful for contributors, but they are not part of the main grader workflow.

## What They Do

- notebook curation
  Rebuild imported or exploratory notebooks into the cleaned submission format.
- artifact normalization
  Move outputs into `checkpoints/`, `plots/`, and `results/` layouts.
- branch renaming and migration
  Align experiment folder names with what the notebooks actually do.
- report support
  Generate tables or plots from saved result bundles.
- inspection utilities
  Small helpers for checking dataset integrity or result consistency.

## How To Treat This Folder

- Safe to keep in the repo as provenance for the cleanup work.
- Not required for graders to run the final submission.
- Best used when we need to repeat a repo-organization step instead of hand-editing notebooks again.
