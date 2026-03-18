## Scripts

This project is notebook-first.

Older standalone experiment-runner scripts were removed so the notebooks remain the main place to launch experiments.

Keep these top-level scripts when you need notebook support or reproducible CLI utilities:

- `prepare_wm811k.py` for dataset preparation
- `evaluate_reconstruction_model.py` for shared checkpoint evaluation
- `evaluate_autoencoder_scores.py` for autoencoder score ablations
- `ensemble_multiclass_classifier.py` for averaged or stacked multiclass ensemble evaluation
- `predict_unlabeled_multiclass_ensemble.py` for unlabeled inference with an averaged or stacked classifier ensemble
- `train_vae.py` because the VAE notebook calls it directly
- `train_ts_distillation.py` because notebook `12` calls it directly

Other helpers are grouped to reduce top-level clutter:

- `dev/` contains one-off inspection tools
