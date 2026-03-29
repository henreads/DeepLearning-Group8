# Backbone Embedding Family

This family contains anomaly baselines built from frozen pretrained visual backbones.

Rather than learning a reconstruction model, these branches extract backbone features and evaluate whether normal wafers form a sufficiently compact feature space for anomaly scoring.

## Why This Family Exists

These branches isolate the contribution of pretrained features from the rest of the anomaly-detection stack.

They are useful for understanding whether strong generic visual representations already provide a workable anomaly signal before introducing more specialized training objectives.

## Branches

- `resnet18/x64/baseline/`
  Frozen ResNet18 embedding baseline.
- `wide_resnet50_2/x64/baseline/`
  Frozen Wide ResNet50-2 embedding baseline.

## Common Files In A Branch

- `README.md`
- `notebook.ipynb`
- `train_config.toml`
- `artifacts/`
