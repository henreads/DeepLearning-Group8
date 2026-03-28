# PatchCore WRN50 x224 on Modal

This folder contains a standalone Modal app for the `18A`-style direct `224x224` WideResNet50-2 multilayer PatchCore run.

It is intentionally separate from the notebooks so you can:

- run from your terminal with `modal run`
- keep Modal-specific code in one place
- persist raw data and artifacts in Modal Volumes

The standalone Modal defaults are slightly lighter than the original notebook so the run is less painful remotely:

- batch size: `128`
- memory bank: `500,000` patches
- minimum memory images: `640`

The original notebook-style benchmark setting is still the heavier `600,000`-patch variant, so treat this Modal app as a close validation run rather than a byte-for-byte reproduction.

## What it runs

- Backbone: `WideResNet50-2`
- Feature layers: `layer2 + layer3`
- Default best variant: `topk_mb50k_r005_x224`
- Optional full `18A` sweep:
  - `topk_mb50k_r010_x224`
  - `topk_mb50k_r005_x224`
- Threshold rule: `95th` percentile of validation-normal scores
- Optional secondary holdout:
  - `70,000` normals
  - `3,500` defects

## Volumes

The app uses two Modal Volumes:

- `wm811k-data`
- `wm811k-artifacts`

Create them once if you want:

```powershell
modal volume create wm811k-data
modal volume create wm811k-artifacts
```

The code can also lazily create them if missing.

## One-time raw pickle upload

Upload your local `LSWMD.pkl` into the data volume:

```powershell
modal run modal_apps/patchcore_wrn50_x224/app.py --mode upload-only --raw-pickle data/raw/LSWMD.pkl
```

That stores the file at `/raw/LSWMD.pkl` inside the `wm811k-data` volume.

## Run the best x224 variant on the secondary holdout

```powershell
modal run modal_apps/patchcore_wrn50_x224/app.py --mode run --split-mode holdout70k_3p5k --variants topk_mb50k_r005_x224
```

## Run the full 18A x224 sweep

```powershell
modal run modal_apps/patchcore_wrn50_x224/app.py --mode run --split-mode holdout70k_3p5k --variants topk_mb50k_r010_x224,topk_mb50k_r005_x224
```

## Upload and run in one command

```powershell
modal run modal_apps/patchcore_wrn50_x224/app.py --mode upload-and-run --raw-pickle data/raw/LSWMD.pkl --split-mode holdout70k_3p5k --variants topk_mb50k_r005_x224
```

## Download artifacts back locally

Example commands:

```powershell
modal volume get wm811k-artifacts patchcore_wrn50_x224/holdout70k_3p5k/patchcore_sweep_summary.json patchcore_sweep_summary.json
modal volume get wm811k-artifacts patchcore_wrn50_x224/holdout70k_3p5k/topk_mb50k_r005_x224/best_model.pt best_model.pt
```

## Notes

- The selected `best_model.pt` is large because it includes the fitted PatchCore memory bank.
- With the current Modal-friendly default `500,000`-patch memory bank, expect the memory-bank payload to be roughly `2.9 GiB` before extra checkpoint overhead.
- Remote Modal artifacts do not automatically sync back into your git repo. Use `modal volume get` when you want them locally.
