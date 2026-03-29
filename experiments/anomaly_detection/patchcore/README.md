# PatchCore Family

This family contains nearest-neighbor memory-bank anomaly detectors.

PatchCore stores reference features from normal training data and scores a test wafer by measuring how far its local features drift from that memory bank.

## Why This Family Exists

PatchCore is the main local-anomaly family in the project.

It is especially relevant for wafer maps because small localized defects can be difficult for reconstruction models to preserve, while feature-memory methods can stay sensitive to local deviations.

## Major Subfamilies

- `ae_bn/x64/main/`
  PatchCore on top of the BatchNorm autoencoder backbone.
- `resnet18/x64/` and `resnet50/x64/`
  Main benchmark PatchCore branches on CNN backbones.
- `wideresnet50/`
  Wide ResNet50 PatchCore experiments across the local `x64` benchmark, the labeled `120k` follow-up, and the higher-resolution review branches.
- `efficientnet_b0/` and `efficientnet_b1/`
  EfficientNet-based PatchCore branches.
- `vit_b16/`
  Transformer-based PatchCore branches.

## Resolution Use In This Family

- `x64` is the main benchmark comparison setting.
- `x224` and `x240` are used for higher-resolution follow-ups and for backbones whose pretrained scale is naturally larger than `64x64`.

## Common Files In A Branch

- `README.md`
- `notebook.ipynb`
- local config files when the branch uses them
- `artifacts/checkpoints/`
- `artifacts/plots/`
- `artifacts/results/`

Some high-resolution follow-up branches are review-oriented because only saved score artifacts were retained, while the main benchmark branches are set up as fuller local experiment notebooks.
