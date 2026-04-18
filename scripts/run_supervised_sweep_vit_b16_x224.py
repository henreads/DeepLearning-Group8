#!/usr/bin/env python3
"""Runner: Supervised ViT-B/16 linear probe defect-data sweep with holdout classes.

Trains a linear classification head on top of a frozen ViT-B/16 backbone,
varying the number of labeled defect training samples across a sweep of
fractions of the non-holdout labeled defect pool.

Key design:
- One or more defect classes are withheld entirely from training and validation.
  All their labeled wafers appear only in the test set.
- The sweep varies N defects from the remaining (seen) classes for training.
- Evaluation reports overall metrics + per-class recall, showing the model
  fails on unseen defect types regardless of how much seen-class data is used.
- The zero-defect PatchCore baseline is referenced externally for comparison.

Usage:
    python scripts/run_supervised_sweep_vit_b16_x224.py \\
        --config experiments/anomaly_detection_defect/supervised_sweep/vit_b16/x224/main/train_config.toml \\
        --raw-pickle data/raw/LSWMD.pkl
"""

from __future__ import annotations

import argparse
import json
from pathlib import PureWindowsPath
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------

def _resolve_project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "src" / "wafer_defect").exists() and (candidate / "experiments").exists():
            return candidate
    return Path.cwd().resolve()


PROJECT_ROOT = _resolve_project_root()
_src = str(PROJECT_ROOT / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from wafer_defect.config import load_toml  # noqa: E402
from wafer_defect.data.legacy_pickle import read_legacy_pickle, unwrap_legacy_value  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGENET_MEAN = 0.4490
IMAGENET_STD = 0.2260


# ---------------------------------------------------------------------------
# ViT encoder (frozen, 1-channel adapted)
# ---------------------------------------------------------------------------

def build_vit_encoder(pretrained: bool = True) -> nn.Module:
    """Frozen ViT-B/16 with patch embedding adapted from 3-channel to 1-channel."""
    import timm
    vit = timm.create_model(
        "vit_base_patch16_224.augreg_in21k_ft_in1k",
        pretrained=pretrained,
        num_classes=0,  # returns [CLS] token embedding, dim=768
    )
    orig = vit.patch_embed.proj
    adapted = nn.Conv2d(
        1, orig.out_channels,
        kernel_size=orig.kernel_size,
        stride=orig.stride,
        padding=orig.padding,
        bias=orig.bias is not None,
    )
    with torch.no_grad():
        adapted.weight.copy_(orig.weight.mean(dim=1, keepdim=True))
        if orig.bias is not None and adapted.bias is not None:
            adapted.bias.copy_(orig.bias)
    vit.patch_embed.proj = adapted
    for p in vit.parameters():
        p.requires_grad = False
    vit.eval()
    return vit


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

def normalize_map(wafer_map: Any, image_size: int) -> np.ndarray:
    arr = np.asarray(wafer_map, dtype=np.float32) / 2.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=(image_size, image_size), mode="nearest")
    return resized.squeeze().numpy()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_defects_from_pickle(raw_pickle: Path, image_size: int) -> pd.DataFrame:
    """Extract all labeled defect wafers from LSWMD raw pickle.

    Returns DataFrame with columns: image (ndarray), defect_type (str).
    """
    print(f"[supervised-sweep] reading {raw_pickle} ...", flush=True)
    df = read_legacy_pickle(raw_pickle)
    records: list[dict] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting defects"):
        train_label = unwrap_legacy_value(row.get("trianTestLabel", ""))
        if not train_label:
            continue  # unlabeled

        failure = unwrap_legacy_value(row.get("failureType", "")).strip()
        if not failure or failure.lower() in {"none", ""}:
            continue  # normal

        wafer_map = row.get("waferMap", None)
        if wafer_map is None:
            continue

        try:
            image = normalize_map(wafer_map, image_size)
        except Exception:
            continue

        records.append({"image": image, "defect_type": failure})

    result = pd.DataFrame(records)
    print(f"[supervised-sweep] loaded {len(result)} labeled defects", flush=True)
    print(f"[supervised-sweep] class counts:\n{result['defect_type'].value_counts().to_string()}", flush=True)
    return result


def load_normal_images(metadata_csv: Path, split: str, count: int, seed: int) -> np.ndarray:
    """Load pre-processed normal wafer arrays from existing metadata CSV."""
    df = pd.read_csv(metadata_csv)
    normals = df[(df["split"] == split) & (df["is_anomaly"] == 0)]
    if len(normals) < count:
        raise ValueError(f"Need {count} {split} normals, only {len(normals)} available.")
    sampled = normals.sample(n=count, random_state=seed).reset_index(drop=True)

    def resolve_array_path(raw_path: str) -> Path:
        path = Path(str(raw_path))
        if path.exists():
            return path
        repo_relative = PROJECT_ROOT / path
        if repo_relative.exists():
            return repo_relative
        # Compatibility with stale metadata created on Windows before Modal runs.
        windows_name = PureWindowsPath(str(raw_path)).name
        fallback = metadata_csv.parent / "arrays_50k_5pct" / windows_name
        if fallback.exists():
            return fallback
        raise FileNotFoundError(f"Array file not found for metadata path: {raw_path}")

    images = np.stack([
        np.load(resolve_array_path(p)) for p in tqdm(sampled["array_path"], desc=f"Loading {split} normals")
    ])
    return images  # (N, H, W) float32


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(
    encoder: nn.Module,
    images: np.ndarray,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
    desc: str = "Embedding",
) -> np.ndarray:
    """Forward pass images through frozen ViT, return [CLS] embeddings (N, 768)."""
    encoder.eval()
    mean = torch.tensor(IMAGENET_MEAN, device=device)
    std = torch.tensor(IMAGENET_STD, device=device)
    parts: list[np.ndarray] = []
    for start in tqdm(range(0, len(images), batch_size), desc=desc, leave=False):
        batch = torch.from_numpy(images[start:start + batch_size]).to(device).unsqueeze(1)
        batch = (batch - mean) / std
        with torch.autocast(device_type=device.type, enabled=use_amp):
            emb = encoder(batch)
        parts.append(emb.float().cpu().numpy())
    return np.concatenate(parts, axis=0)


def load_or_extract_embeddings(
    *,
    encoder: nn.Module,
    images: np.ndarray,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
    desc: str,
    cache_path: Path,
) -> np.ndarray:
    """Load cached embeddings if present; otherwise extract and save atomically."""
    if cache_path.exists():
        print(f"[supervised-sweep] loading cached embeddings: {cache_path}", flush=True)
        return np.load(cache_path)

    emb = extract_embeddings(encoder, images, batch_size, device, use_amp, desc)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp_path.open("wb") as fh:
        np.save(fh, emb)
    tmp_path.replace(cache_path)
    print(f"[supervised-sweep] cached embeddings: {cache_path}", flush=True)
    return emb


# ---------------------------------------------------------------------------
# Defect splitting with holdout classes
# ---------------------------------------------------------------------------

def build_defect_splits(
    defect_df: pd.DataFrame,
    holdout_types: list[str],
    test_defect_count: int,
    val_count: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Split defects into test / val / train_pool respecting holdout classes.

    Test set (total = test_defect_count, e.g. 250):
        Stratified proportionally across ALL defect types including holdout types.
        This keeps the benchmark identical to the zero-defect PatchCore evaluation
        so AUROC/AUPRC/F1 are directly comparable.

    Val set:
        Defects from non-holdout (seen) classes only, used for early stopping.

    Train pool:
        Only non-holdout (seen) class defects remain here.
        Holdout classes never appear in train or val.

    Per-class recall on the test set reveals zero recall on holdout types
    regardless of how many seen-class defects were used in training.

    Returns dict with keys: test, val, train_pool.
    Each row has an emb_idx column set by the caller.
    """
    total = len(defect_df)
    test_parts, val_parts, train_parts = [], [], []

    for dtype in defect_df["defect_type"].unique():
        group = defect_df[defect_df["defect_type"] == dtype].sample(
            frac=1, random_state=seed
        ).reset_index(drop=True)

        # Proportional share of the 250-defect test set (all types including holdout)
        n_test = max(1, round(test_defect_count * len(group) / total))
        n_test = min(n_test, len(group))
        test_parts.append(group.iloc[:n_test])

        remaining = group.iloc[n_test:]

        if dtype in holdout_types:
            # Holdout classes: nothing goes to val or train
            continue

        # Seen classes: split remainder into val and train pool
        n_val = max(1, round(val_count * len(group) / total))
        n_val = min(n_val, len(remaining))
        val_parts.append(remaining.iloc[:n_val])
        train_parts.append(remaining.iloc[n_val:])

    test_df = pd.concat(test_parts).reset_index(drop=True)
    val_df = pd.concat(val_parts).reset_index(drop=True) if val_parts else pd.DataFrame()
    train_pool_df = pd.concat(train_parts).reset_index(drop=True) if train_parts else pd.DataFrame()

    holdout_in_test = test_df[test_df["defect_type"].isin(holdout_types)]
    seen_in_test = test_df[~test_df["defect_type"].isin(holdout_types)]

    print(
        f"[supervised-sweep] test defects: {len(test_df)} total "
        f"({len(seen_in_test)} seen, {len(holdout_in_test)} holdout {holdout_types})",
        flush=True,
    )
    print(
        f"[supervised-sweep] val defects: {len(val_df)} (seen classes only)",
        flush=True,
    )
    print(
        f"[supervised-sweep] train defect pool: {len(train_pool_df)} (seen classes only)",
        flush=True,
    )

    return {"test": test_df, "val": val_df, "train_pool": train_pool_df}


# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    def __init__(self, in_dim: int = 768) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def train_linear_probe(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    val_emb: np.ndarray,
    val_labels: np.ndarray,
    epochs: int,
    lr: float,
    wd: float,
    device: torch.device,
    use_amp: bool,
    patience: int,
    seed: int,
) -> LinearProbe:
    torch.manual_seed(seed)

    val_X = torch.from_numpy(val_emb).float().to(device)
    val_y = torch.from_numpy(val_labels).long()

    n_normal = int((train_labels == 0).sum())
    n_defect = int((train_labels == 1).sum())
    total = len(train_labels)
    class_weights = torch.tensor(
        [total / (2.0 * n_normal), total / (2.0 * n_defect)],
        dtype=torch.float32,
        device=device,
    )

    model = LinearProbe(train_emb.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_X = torch.from_numpy(train_emb).float()
    train_y = torch.from_numpy(train_labels).long()
    loader = DataLoader(TensorDataset(train_X, train_y), batch_size=512, shuffle=True)

    best_val_f1 = -1.0
    best_state: dict | None = None
    no_improve = 0

    epoch_bar = tqdm(range(epochs), desc="Training probe epochs", leave=False)
    for epoch in epoch_bar:
        model.train()
        batch_bar = tqdm(
            loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            leave=False,
            total=len(loader),
        )
        for xb, yb in batch_bar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            batch_bar.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")

        model.eval()
        with torch.no_grad():
            val_pred = model(val_X).argmax(dim=1).cpu().numpy()
        val_f1 = float(f1_score(val_y.numpy(), val_pred, zero_division=0))
        epoch_bar.set_postfix(val_f1=f"{val_f1:.4f}", best=f"{max(best_val_f1, 0.0):.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _defect_probs(
    model: LinearProbe, emb: np.ndarray, device: torch.device
) -> np.ndarray:
    model.eval()
    logits = model(torch.from_numpy(emb).float().to(device)).cpu()
    return torch.softmax(logits, dim=1)[:, 1].numpy()


def evaluate(
    model: LinearProbe,
    test_normal_emb: np.ndarray,
    test_defect_emb: np.ndarray,
    test_defect_types: list[str],
    holdout_types: list[str],
    train_normal_emb: np.ndarray,
    threshold_quantile: float,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate on the benchmark test set (5k normals + 250 defects).

    The test set contains both seen and holdout defect classes in their natural
    proportions, matching the zero-defect PatchCore benchmark exactly.
    AUROC/AUPRC/F1 are therefore directly comparable to PatchCore ViT-B/16.
    Per-class recall exposes near-zero detection of holdout classes.
    """
    # Threshold from training normals — same protocol as zero-defect PatchCore
    normal_probs = _defect_probs(model, train_normal_emb, device)
    threshold = float(np.quantile(normal_probs, threshold_quantile))

    test_emb = np.concatenate([test_normal_emb, test_defect_emb])
    test_labels = np.array(
        [0] * len(test_normal_emb) + [1] * len(test_defect_emb), dtype=np.int64
    )
    all_probs = _defect_probs(model, test_emb, device)
    pred = (all_probs >= threshold).astype(int)

    auroc = float(roc_auc_score(test_labels, all_probs))
    auprc = float(average_precision_score(test_labels, all_probs))
    f1 = float(f1_score(test_labels, pred, zero_division=0))
    precision = float(precision_score(test_labels, pred, zero_division=0))
    recall = float(recall_score(test_labels, pred, zero_division=0))

    # Best-sweep threshold
    best_f1, best_thresh = 0.0, threshold
    for t in np.linspace(float(all_probs.min()), float(all_probs.max()), 200):
        p = (all_probs >= t).astype(int)
        ft = float(f1_score(test_labels, p, zero_division=0))
        if ft > best_f1:
            best_f1, best_thresh = ft, t
    best_pred = (all_probs >= best_thresh).astype(int)

    # Per-class recall on defect portion only
    defect_probs = all_probs[len(test_normal_emb):]
    defect_pred = (defect_probs >= threshold).astype(int)
    per_class: dict[str, dict] = {}
    for dtype in sorted(set(test_defect_types)):
        mask = np.array([t == dtype for t in test_defect_types])
        n = int(mask.sum())
        detected = int(defect_pred[mask].sum())
        per_class[dtype] = {
            "seen_in_training": dtype not in holdout_types,
            "n": n,
            "detected": detected,
            "recall": detected / n if n > 0 else 0.0,
        }

    return {
        "auroc": auroc,
        "auprc": auprc,
        "threshold": threshold,
        "threshold_quantile": threshold_quantile,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "best_sweep_f1": best_f1,
        "best_sweep_threshold": best_thresh,
        "best_sweep_precision": float(precision_score(test_labels, best_pred, zero_division=0)),
        "best_sweep_recall": float(recall_score(test_labels, best_pred, zero_division=0)),
        "n_test_normal": len(test_normal_emb),
        "n_test_defect": len(test_defect_emb),
        "per_class_recall": per_class,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--raw-pickle", default=None)
    parser.add_argument("--normal-metadata-csv", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_toml(args.config)
    run_cfg = cfg["run"]
    data_cfg = cfg["data"]
    split_cfg = cfg["split"]
    sweep_cfg = cfg["sweep"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    score_cfg = cfg["scoring"]

    output_dir = Path(args.output_dir or run_cfg["output_dir"])
    raw_pickle = Path(args.raw_pickle or data_cfg["raw_pickle"])
    normal_metadata_csv = Path(args.normal_metadata_csv or data_cfg["normal_metadata_csv"])
    seed = int(run_cfg.get("seed", 42))
    image_size = int(data_cfg.get("image_size", 224))
    batch_size = int(data_cfg.get("batch_size", 256))

    holdout_types: list[str] = list(split_cfg["holdout_defect_types"])
    test_defect_count = int(split_cfg["test_defect_count"])
    val_count = int(split_cfg["val_defect_count"])
    n_train_normal = int(split_cfg["normal_train_count"])
    n_val_normal = int(split_cfg["normal_val_count"])
    n_test_normal = int(split_cfg["normal_test_count"])
    fractions: list[float] = [float(f) for f in sweep_cfg["fractions"]]
    threshold_quantile = float(score_cfg.get("threshold_quantile", 0.95))

    device_str = str(train_cfg.get("device", "auto"))
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        if device_str == "auto" else device_str
    )
    use_amp = bool(train_cfg.get("use_amp", True)) and device.type == "cuda"

    results_dir = output_dir / "results"
    checkpoint_dir = output_dir / "checkpoints"
    embedding_dir = output_dir / "embeddings"
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    embedding_dir.mkdir(parents=True, exist_ok=True)

    print(f"[supervised-sweep] device={device}, use_amp={use_amp}", flush=True)
    print(f"[supervised-sweep] holdout classes: {holdout_types}", flush=True)

    # ---- Build encoder ----
    print("[supervised-sweep] loading ViT-B/16 ...", flush=True)
    encoder = build_vit_encoder(pretrained=bool(model_cfg.get("pretrained", True))).to(device)

    # ---- Normal embeddings ----
    print("[supervised-sweep] embedding train normals ...", flush=True)
    train_normal_images = load_normal_images(normal_metadata_csv, "train", n_train_normal, seed)
    train_normal_emb = load_or_extract_embeddings(
        encoder=encoder,
        images=train_normal_images,
        batch_size=batch_size,
        device=device,
        use_amp=use_amp,
        desc="Train normals",
        cache_path=embedding_dir / "train_normal_emb.npy",
    )
    del train_normal_images

    print("[supervised-sweep] embedding val normals ...", flush=True)
    val_normal_images = load_normal_images(normal_metadata_csv, "val", n_val_normal, seed)
    val_normal_emb = load_or_extract_embeddings(
        encoder=encoder,
        images=val_normal_images,
        batch_size=batch_size,
        device=device,
        use_amp=use_amp,
        desc="Val normals",
        cache_path=embedding_dir / "val_normal_emb.npy",
    )
    del val_normal_images

    print("[supervised-sweep] embedding test normals ...", flush=True)
    test_normal_images = load_normal_images(normal_metadata_csv, "test", n_test_normal, seed)
    test_normal_emb = load_or_extract_embeddings(
        encoder=encoder,
        images=test_normal_images,
        batch_size=batch_size,
        device=device,
        use_amp=use_amp,
        desc="Test normals",
        cache_path=embedding_dir / "test_normal_emb.npy",
    )
    del test_normal_images

    # ---- Defect embeddings ----
    defect_df = load_defects_from_pickle(raw_pickle, image_size)
    defect_images = np.stack(defect_df["image"].tolist())
    defect_df = defect_df.drop(columns=["image"]).reset_index(drop=True)
    defect_df["emb_idx"] = range(len(defect_df))

    print("[supervised-sweep] embedding all defects ...", flush=True)
    all_defect_emb = load_or_extract_embeddings(
        encoder=encoder,
        images=defect_images,
        batch_size=batch_size,
        device=device,
        use_amp=use_amp,
        desc="Defects",
        cache_path=embedding_dir / "all_defect_emb.npy",
    )
    del defect_images

    # ---- Split defects ----
    splits = build_defect_splits(defect_df, holdout_types, test_defect_count, val_count, seed)
    test_def_df = splits["test"]
    val_df = splits["val"]
    train_pool_df = splits["train_pool"]

    # Pre-index embeddings for each split
    test_def_emb = all_defect_emb[test_def_df["emb_idx"].values]
    val_def_emb = all_defect_emb[val_df["emb_idx"].values]

    # Val set: normals + seen-class defects (holdout types never appear here)
    full_val_emb = np.concatenate([val_normal_emb, val_def_emb])
    full_val_labels = np.array(
        [0] * len(val_normal_emb) + [1] * len(val_df), dtype=np.int64
    )

    pool_size = len(train_pool_df)
    print(f"[supervised-sweep] train defect pool size: {pool_size}", flush=True)

    # ---- Sweep ----
    sweep_results: list[dict] = []

    for frac in fractions:
        slug = f"frac_{frac:.4f}".replace(".", "p")
        summary_path = results_dir / f"summary_{slug}.json"
        checkpoint_path = checkpoint_dir / f"linear_probe_{slug}.pt"
        if summary_path.exists() and checkpoint_path.exists():
            print(
                f"\n[supervised-sweep] fraction={frac:.4f} already complete; "
                f"reusing {summary_path.name}",
                flush=True,
            )
            sweep_results.append(json.loads(summary_path.read_text(encoding="utf-8")))
            continue

        n_defects = max(1, int(round(frac * pool_size)))
        n_defects = min(n_defects, pool_size)
        print(f"\n[supervised-sweep] fraction={frac:.4f} → {n_defects} training defects", flush=True)

        sampled_def = train_pool_df.sample(n=n_defects, random_state=seed)
        sampled_def_emb = all_defect_emb[sampled_def["emb_idx"].values]

        train_emb = np.concatenate([train_normal_emb, sampled_def_emb])
        train_labels = np.array(
            [0] * len(train_normal_emb) + [1] * n_defects, dtype=np.int64
        )

        model = train_linear_probe(
            train_emb, train_labels,
            full_val_emb, full_val_labels,
            epochs=int(train_cfg.get("epochs", 50)),
            lr=float(train_cfg.get("learning_rate", 0.001)),
            wd=float(train_cfg.get("weight_decay", 1e-5)),
            device=device,
            use_amp=use_amp,
            patience=int(train_cfg.get("early_stopping_patience", 7)),
            seed=seed,
        )

        metrics = evaluate(
            model=model,
            test_normal_emb=test_normal_emb,
            test_defect_emb=test_def_emb,
            test_defect_types=test_def_df["defect_type"].tolist(),
            holdout_types=holdout_types,
            train_normal_emb=train_normal_emb,
            threshold_quantile=threshold_quantile,
            device=device,
        )

        result: dict = {
            "fraction": frac,
            "n_defects_train": n_defects,
            "n_normals_train": len(train_normal_emb),
            **metrics,
        }
        sweep_results.append(result)
        print(
            f"  F1={result['f1']:.3f}  AUROC={result['auroc']:.3f}  "
            f"best_F1={result['best_sweep_f1']:.3f}",
            flush=True,
        )
        seen_recalls = {
            k: f"{v['recall']:.2f}" for k, v in result["per_class_recall"].items()
            if v["seen_in_training"]
        }
        holdout_recalls = {
            k: f"{v['recall']:.2f}" for k, v in result["per_class_recall"].items()
            if not v["seen_in_training"]
        }
        print(f"  Seen class recall:    {seen_recalls}", flush=True)
        print(f"  Holdout class recall (unseen types): {holdout_recalls}", flush=True)

        summary_path.write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        torch.save(model.state_dict(), checkpoint_path)

    # ---- Sweep summary ----
    sweep_summary: dict = {
        "sweep_results": sweep_results,
        "config": {
            "backbone": model_cfg["backbone"],
            "holdout_defect_types": holdout_types,
            "test_defect_count": test_defect_count,
            "val_defect_count": val_count,
            "normal_train_count": n_train_normal,
            "total_labeled_defects": len(defect_df),
            "train_defect_pool_size": pool_size,
            "fractions": fractions,
        },
    }
    (results_dir / "sweep_summary.json").write_text(
        json.dumps(sweep_summary, indent=2), encoding="utf-8"
    )
    print(f"\n[supervised-sweep] done — results in {results_dir}", flush=True)


if __name__ == "__main__":
    main()
