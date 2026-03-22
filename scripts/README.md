## Scripts

This project is notebook-first.

Older standalone experiment-runner scripts were removed so the notebooks remain the main place to launch experiments.

Keep these top-level scripts when you need notebook support or reproducible CLI utilities:

- `prepare_wm811k.py` for dataset preparation
  It now supports both the default `50k` benchmark split and the separate larger labeled WRN50 PatchCore split via `--config`.
- `evaluate_reconstruction_model.py` for shared checkpoint evaluation
- `evaluate_autoencoder_scores.py` for autoencoder score ablations
- `train_vae.py` because the VAE notebook calls it directly
- `train_ts_distillation.py` because notebook `12` calls it directly

Other helpers are grouped to reduce top-level clutter:

- `dev/` contains one-off inspection tools
- `classifier/` contains the supervised multiclass classifier workflow scripts
