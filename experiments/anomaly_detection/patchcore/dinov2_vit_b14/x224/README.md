# PatchCore DINOv2 ViT-B/14 (`x224`, frozen)

PatchCore memory bank built on frozen DINOv2 ViT-B/14 features at 224×224 input resolution.

## Method

No training. One forward pass over 40k training normals builds a 600k-patch coreset memory bank. Test wafers are scored by nearest-neighbour distance from their patch embeddings to the bank.

DINOv2 differs from the supervised ViT-B/16 baseline in two ways:

- **Patch size 14** instead of 16 → 256 tokens per image instead of 196
- **DINO self-distillation pretraining** on 142M images → patch tokens retain strong local spatial structure at all depths, unlike supervised ViT which collapses to class-discriminative representations at deeper blocks

## Settings

| Setting | Value | Baseline (frozen ViT-B/16) |
|---|---|---|
| Backbone | `vit_base_patch14_dinov2.lvd142m` | `vit_base_patch16_224.augreg_in21k_ft_in1k` |
| Patch size | 14 | 16 |
| Tokens/image | 256 | 196 |
| Feature block | 9 | 6 |
| Projection dim | 128 | 128 |
| Memory bank | 600k | 600k |
| NN k | 3 | 3 |
| Top-k patch ratio | 0.10 | 0.10 |
| Threshold | q=0.95 on tune-normal | q=0.95 |

## Block Sweep Results

Block sweep across layers 6, 9, 11 to find the best feature depth given positional embedding interpolation from 37×37 → 16×16.

| Metric | Block 6 | Block 9 | Block 11 | Frozen ViT-B/16 |
|---|---|---|---|---|
| AUROC | **0.926** | 0.915 | 0.885 | 0.956 |
| AUPRC | 0.549 | **0.561** | 0.504 | 0.671 |
| F1 | **0.521** | 0.492 | 0.443 | 0.595 |
| Center | **0.760** | 0.560 | 0.680 | 0.618 |
| Edge-Loc | **0.604** | 0.547 | 0.340 | 0.705 |
| Edge-Ring | **0.774** | 0.738 | 0.524 | **0.941** |
| Loc | 0.676 | **0.706** | 0.647 | 0.829 |
| Scratch | 0.800 | **0.933** | 0.867 | 0.727 |

**Block 6** is the best overall DINOv2 block. **Block 9** uniquely maximises Scratch recall (0.933 — highest in the project) at the cost of global defect coverage. Deeper layers (block 11) degrade further, consistent with positional interpolation distortion accumulating through attention.

DINOv2 block 6 is still ~3 AUROC points below frozen ViT-B/16, confirming the positional embedding mismatch limits global spatial reasoning on 224×224 inputs. A score-ensemble (max-fusion of normalised scores from frozen ViT-B/16 + DINOv2 block 9) is the recommended path to combine Edge-Ring and Scratch strengths.

## Files

Per-block structure:

```
block_6/   — block 6 sweep (best overall: AUROC 0.926, F1 0.521)
block9/    — block 9 sweep (best Scratch: 0.933)
block_11/  — block 11 sweep
main/      — primary experiment (block 9, full workflow)
```

Each block folder contains `artifacts/results/evaluation_metrics.json` and `artifacts/results/scores.npz`.
