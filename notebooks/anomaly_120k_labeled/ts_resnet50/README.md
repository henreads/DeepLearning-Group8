TS-ResNet50 120k Labeled Notebooks

This folder brings the teacher-student ResNet50 idea onto the labeled `120k / 10k / 20k` anomaly split with one controlled first test instead of a large sweep.

Contents

- `1_ts_resnet50_training_120k.ipynb`
- `configs/training/train_ts_resnet50_120k.toml`

Single-test plan

- reuse the same labeled `120k / 10k / 20k` metadata family already used by the WRN50 PatchCore workflow
- train one teacher-student model with the same core recipe that worked on the `CT` branch: `ResNet50` teacher, `layer2`, feature autoencoder hidden dim `128`
- keep the default deployment score close to the friend-branch setup: student-only `topk_mean`
- add a post-training score sweep over student and autoencoder branch weights plus reduction choices to see whether rescoring unlocks a stronger operating point
- compare the teacher-student result against the saved PatchCore controls if the local artifact bundles are present

Out of scope for this first notebook

- teacher-layer ablations
- image-size sweeps
- ensembles
- Modal packaging

Expected outputs

- training artifacts under `artifacts/x64/ts_resnet50_120k`
- score-sweep summaries under `artifacts/x64/ts_resnet50_120k/evaluation`
