# Teacher-Student Family

This family contains distillation-based anomaly detectors.

The teacher-student approach trains a student to reproduce teacher features on normal wafers only. Anomaly scores are then derived from the mismatch between teacher and student features at test time.

## Why This Family Exists

Teacher-student distillation is one of the stronger feature-based anomaly ideas in the project.

It provides a middle ground between plain reconstruction and nearest-neighbor memory methods:

- feature-aware rather than pixel-only
- normal-only training
- suitable for localized defects

## Branches

- `resnet18/x64/main/`
  Main teacher-student baseline on the benchmark split.
- `resnet50/x64/main/`
  Higher-capacity ResNet50 branch with local checkpoints and evaluation outputs.
- `resnet50/x64/layer_ablation/`
  Follow-up study comparing different teacher-layer choices.
- `wideresnet50_2/x64/layer2_self_contained/`
  Single-layer Wide ResNet50-2 branch.
- `wideresnet50_2/x64/multilayer_self_contained/`
  Multilayer Wide ResNet50-2 branch.

## Common Files In A Branch

- `README.md`
- `notebook.ipynb`
- `train_config.toml` when the branch uses a local training config
- `artifacts/checkpoints/`
- `artifacts/plots/`
- `artifacts/results/`
