from __future__ import annotations

import json
import math
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandas.core.indexes as core_indexes
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

LABEL_NORMAL = "none"
LABEL_DEFECT = "pattern"

DEFAULT_SPLIT_CONFIG: dict[str, int] = {
    "train_total": 120_000,
    "train_anomalies": 6_000,
    "val_total": 10_000,
    "val_anomalies": 500,
    "test_total": 20_000,
    "test_anomalies": 1_000,
}

DEFAULT_VARIANTS: list[dict[str, Any]] = [
    {"name": "topk_mb50k_r010", "memory_bank_size": 50_000, "reduction": "topk_mean", "topk_ratio": 0.10},
    {"name": "topk_mb50k_r015", "memory_bank_size": 50_000, "reduction": "topk_mean", "topk_ratio": 0.15},
    {"name": "topk_mb50k_r005", "memory_bank_size": 50_000, "reduction": "topk_mean", "topk_ratio": 0.05},
    {"name": "mean_mb50k", "memory_bank_size": 50_000, "reduction": "mean", "topk_ratio": 0.10},
]


def build_wideresnet50_2(pretrained: bool) -> nn.Module:
    try:
        from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
    except ImportError as exc:
        raise ImportError(
            "torchvision is required to build the WideResNet50-2 PatchCore backbone."
        ) from exc

    weights = Wide_ResNet50_2_Weights.DEFAULT if pretrained else None
    return wide_resnet50_2(weights=weights)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def resolve_bundle_root(start: str | Path | None = None) -> Path:
    start_path = Path(start or Path.cwd()).resolve()
    for candidate in [start_path, *start_path.parents]:
        if (candidate / "helpers" / "patchcore_wrn50_kaggle.py").exists() and (candidate / "README.md").exists():
            return candidate
        if (
            (
                candidate
                / "notebooks"
                / "anomaly_120k_labeled"
                / "patchcore_wrn50"
                / "helpers"
                / "patchcore_wrn50_kaggle.py"
            ).exists()
            and (candidate / "configs").exists()
        ):
            return candidate
    raise FileNotFoundError("Could not locate the Kaggle bundle root.")


def auto_find_raw_pickle(explicit_path: str | Path | None = None) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if path.exists():
            return path.resolve()
        raise FileNotFoundError(f"Raw pickle not found: {path}")

    search_roots = [
        Path.cwd(),
        Path("/kaggle/input"),
        Path("/kaggle/working"),
        Path("/root"),
        Path("/mnt/data"),
        Path("/workspace"),
    ]
    for root in search_roots:
        if root.exists():
            matches = sorted(root.rglob("LSWMD.pkl"))
            if matches:
                return matches[0].resolve()

    raise FileNotFoundError("Could not find LSWMD.pkl. Set RAW_PICKLE explicitly.")


def read_legacy_pickle(path: Path) -> pd.DataFrame:
    sys.modules["pandas.indexes"] = core_indexes
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def unwrap_legacy_value(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "size") and getattr(value, "size") == 0:
        return ""
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return str(value).strip()


def normalize_map(wafer_map: np.ndarray, image_size: int) -> np.ndarray:
    wafer_map = np.asarray(wafer_map, dtype=np.float32) / 2.0
    tensor = torch.from_numpy(wafer_map).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=(image_size, image_size), mode="nearest")
    return resized.squeeze(0).squeeze(0).numpy()


def infer_label_from_row(row: pd.Series) -> str | None:
    failure = unwrap_legacy_value(row.get("failureType", "")).lower()
    if failure == "none":
        return LABEL_NORMAL
    if failure:
        return LABEL_DEFECT
    return None


def split_slug(split_config: dict[str, int]) -> str:
    return (
        f"train{split_config['train_total']}_a{split_config['train_anomalies']}"
        f"_val{split_config['val_total']}_a{split_config['val_anomalies']}"
        f"_test{split_config['test_total']}_a{split_config['test_anomalies']}"
    )


def metadata_paths(bundle_root: Path, image_size: int, split_config: dict[str, int]) -> tuple[Path, Path]:
    slug = split_slug(split_config)
    processed_dir = bundle_root / "data" / "processed" / f"x{image_size}" / "wm811k_patchcore_custom"
    metadata_path = processed_dir / f"metadata_{slug}.csv"
    arrays_dir = processed_dir / f"arrays_{slug}"
    return metadata_path, arrays_dir


def _validate_split_counts(split_config: dict[str, int], normal_count: int, defect_count: int) -> None:
    requested_normals = (
        split_config["train_total"]
        - split_config["train_anomalies"]
        + split_config["val_total"]
        - split_config["val_anomalies"]
        + split_config["test_total"]
        - split_config["test_anomalies"]
    )
    requested_defects = (
        split_config["train_anomalies"] + split_config["val_anomalies"] + split_config["test_anomalies"]
    )
    if requested_normals > normal_count:
        raise ValueError(f"Requested {requested_normals} normals but only {normal_count} are available.")
    if requested_defects > defect_count:
        raise ValueError(f"Requested {requested_defects} defects but only {defect_count} are available.")


def build_labeled_split_dataframe(raw_df: pd.DataFrame, split_config: dict[str, int], seed: int) -> pd.DataFrame:
    df = raw_df.copy()
    df["failureTypeText"] = df["failureType"].map(unwrap_legacy_value)
    df["trianTestLabelText"] = df["trianTestLabel"].map(unwrap_legacy_value)
    df["label"] = df.apply(infer_label_from_row, axis=1)
    df = df[df["label"].notna()].reset_index(drop=True)

    normal_df = df[df["label"] == LABEL_NORMAL].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    defect_df = df[df["label"] == LABEL_DEFECT].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    _validate_split_counts(split_config, len(normal_df), len(defect_df))

    train_normals = split_config["train_total"] - split_config["train_anomalies"]
    val_normals = split_config["val_total"] - split_config["val_anomalies"]
    test_normals = split_config["test_total"] - split_config["test_anomalies"]

    normal_offsets = [0, train_normals, train_normals + val_normals, train_normals + val_normals + test_normals]
    defect_offsets = [
        0,
        split_config["train_anomalies"],
        split_config["train_anomalies"] + split_config["val_anomalies"],
        split_config["train_anomalies"] + split_config["val_anomalies"] + split_config["test_anomalies"],
    ]

    train_normal_df = normal_df.iloc[normal_offsets[0] : normal_offsets[1]].copy()
    val_normal_df = normal_df.iloc[normal_offsets[1] : normal_offsets[2]].copy()
    test_normal_df = normal_df.iloc[normal_offsets[2] : normal_offsets[3]].copy()

    train_defect_df = defect_df.iloc[defect_offsets[0] : defect_offsets[1]].copy()
    val_defect_df = defect_df.iloc[defect_offsets[1] : defect_offsets[2]].copy()
    test_defect_df = defect_df.iloc[defect_offsets[2] : defect_offsets[3]].copy()

    train_normal_df["split"] = "train"
    val_normal_df["split"] = "val"
    test_normal_df["split"] = "test"
    train_defect_df["split"] = "train"
    val_defect_df["split"] = "val"
    test_defect_df["split"] = "test"

    export_df = pd.concat(
        [
            train_normal_df,
            train_defect_df,
            val_normal_df,
            val_defect_df,
            test_normal_df,
            test_defect_df,
        ],
        ignore_index=True,
    )
    return export_df


def prepare_dataset(
    raw_pickle: Path,
    bundle_root: Path,
    image_size: int,
    split_config: dict[str, int],
    seed: int = 42,
    overwrite: bool = False,
) -> Path:
    metadata_path, arrays_dir = metadata_paths(bundle_root, image_size, split_config)
    if metadata_path.exists() and not overwrite:
        return metadata_path

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    arrays_dir.mkdir(parents=True, exist_ok=True)

    raw_df = read_legacy_pickle(raw_pickle)
    export_df = build_labeled_split_dataframe(raw_df, split_config=split_config, seed=seed)

    records: list[dict[str, Any]] = []
    for row_index, row in export_df.iterrows():
        array_name = f"wafer_{row_index:07d}.npy"
        array_path = arrays_dir / array_name
        raw_map = np.asarray(row["waferMap"])
        np.save(array_path, normalize_map(raw_map, image_size=image_size))
        records.append(
            {
                "array_path": array_path.relative_to(bundle_root).as_posix(),
                "label": row["label"],
                "defect_type": row["failureTypeText"] or "unlabeled",
                "is_anomaly": int(row["label"] == LABEL_DEFECT),
                "split": row["split"],
                "source_split": row["trianTestLabelText"] or "unlabeled",
                "original_height": int(raw_map.shape[0]),
                "original_width": int(raw_map.shape[1]),
            }
        )

    pd.DataFrame(records).to_csv(metadata_path, index=False)
    return metadata_path


def split_summary(metadata: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        metadata.groupby(["split", "is_anomaly"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["split", "is_anomaly"])
        .reset_index(drop=True)
    )
    grouped["label_name"] = grouped["is_anomaly"].map({0: "normal", 1: "anomaly"})
    return grouped


def split_summary_wide(metadata: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_name in ["train", "val", "test"]:
        split_df = metadata[metadata["split"] == split_name]
        anomalies = int(split_df["is_anomaly"].sum())
        total = len(split_df)
        rows.append(
            {
                "split": split_name,
                "count": total,
                "anomalies": anomalies,
                "normals": total - anomalies,
                "anomaly_pct": 0.0 if total == 0 else anomalies / total,
            }
        )
    return pd.DataFrame(rows)


def defect_type_summary(metadata: pd.DataFrame) -> pd.DataFrame:
    defects = metadata[metadata["is_anomaly"] == 1]
    if defects.empty:
        return pd.DataFrame(columns=["split", "defect_type", "count"])
    return (
        defects.groupby(["split", "defect_type"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["split", "count"], ascending=[True, False])
        .reset_index(drop=True)
    )


class WaferArrayDataset(Dataset):
    def __init__(self, metadata_csv: str | Path, split: str, bundle_root: str | Path | None = None) -> None:
        self.metadata_path = Path(metadata_csv).resolve()
        self.bundle_root = Path(bundle_root).resolve() if bundle_root else resolve_bundle_root(self.metadata_path)
        self.metadata = pd.read_csv(self.metadata_path)
        self.metadata = self.metadata[self.metadata["split"] == split].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.metadata.iloc[index]
        wafer_map = np.load(self.bundle_root / row["array_path"]).astype(np.float32)
        tensor = torch.from_numpy(wafer_map).unsqueeze(0)
        label = torch.tensor(int(row["is_anomaly"]), dtype=torch.long)
        return tensor, label


def summarize_threshold_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    precision = float(precision_score(labels, predicted, zero_division=0))
    recall = float(recall_score(labels, predicted, zero_division=0))
    f1 = float(0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall))
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "false_positive_rate": float(0.0 if fp + tn == 0 else fp / (fp + tn)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "predicted_anomalies": int(predicted.sum()),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def sweep_threshold_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    max_false_positive_rate: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold in np.unique(scores):
        metrics = summarize_threshold_metrics(labels, scores, float(threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "false_positive_rate": float(metrics["false_positive_rate"]),
                "tp": int(metrics["tp"]),
                "fp": int(metrics["fp"]),
                "tn": int(metrics["tn"]),
                "fn": int(metrics["fn"]),
                "predicted_anomalies": int(metrics["predicted_anomalies"]),
            }
        )
    sweep_df = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    eligible_df = sweep_df
    if max_false_positive_rate is not None:
        eligible_df = sweep_df[sweep_df["false_positive_rate"] <= float(max_false_positive_rate)].reset_index(drop=True)
        if eligible_df.empty:
            raise ValueError(
                "No threshold satisfied the requested max_false_positive_rate="
                f"{float(max_false_positive_rate):.6f}."
            )
    best_row = eligible_df.sort_values(["f1", "precision", "recall"], ascending=False).iloc[0]
    return sweep_df, best_row.to_dict()


def select_validation_threshold(
    val_scores_df: pd.DataFrame,
    *,
    threshold_quantile: float,
    threshold_strategy: str = "normal_quantile",
    max_false_positive_rate: float | None = None,
) -> dict[str, Any]:
    strategy = str(threshold_strategy).lower()
    val_labels = val_scores_df["is_anomaly"].to_numpy()
    val_scores = val_scores_df["score"].to_numpy()
    val_threshold_sweep_df, best_validation_sweep = sweep_threshold_metrics(
        val_labels,
        val_scores,
        max_false_positive_rate=max_false_positive_rate if strategy == "validation_f1" else None,
    )

    if strategy == "normal_quantile":
        val_normal_scores = val_scores_df.loc[val_scores_df["is_anomaly"] == 0, "score"]
        threshold = float(val_normal_scores.quantile(threshold_quantile))
    elif strategy == "validation_f1":
        threshold = float(best_validation_sweep["threshold"])
    else:
        raise ValueError(
            "threshold_strategy must be one of {'normal_quantile', 'validation_f1'}, "
            f"but received {threshold_strategy!r}."
        )

    selected_metrics = summarize_threshold_metrics(val_labels, val_scores, threshold)
    return {
        "threshold": float(threshold),
        "threshold_strategy": strategy,
        "threshold_quantile": float(threshold_quantile),
        "max_false_positive_rate": None
        if max_false_positive_rate is None
        else float(max_false_positive_rate),
        "selected_metrics": selected_metrics,
        "val_threshold_sweep_df": val_threshold_sweep_df,
        "best_validation_sweep": best_validation_sweep,
    }


class WideResNet50_2MultiLayerExtractor(nn.Module):
    def __init__(
        self,
        teacher_layers: list[str],
        pretrained: bool = True,
        input_size: int = 224,
        freeze_backbone: bool = True,
        normalize_imagenet: bool = True,
    ) -> None:
        super().__init__()
        self.teacher_layers = [str(layer).lower() for layer in teacher_layers]
        backbone = build_wideresnet50_2(pretrained=pretrained)

        original_conv = backbone.conv1
        adapted_conv = nn.Conv2d(
            1,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            adapted_conv.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))

        backbone.conv1 = adapted_conv
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.input_size = int(input_size)
        self.normalize_imagenet = bool(normalize_imagenet)
        self.register_buffer("image_mean", torch.tensor([0.4490], dtype=torch.float32).view(1, 1, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.2260], dtype=torch.float32).view(1, 1, 1, 1))
        self.feature_dims = {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}
        downsample_map = {"layer1": 4, "layer2": 8, "layer3": 16, "layer4": 32}
        self.output_spatial = max(1, self.input_size // min(downsample_map[layer] for layer in self.teacher_layers))

        if freeze_backbone:
            for parameter in self.parameters():
                parameter.requires_grad = False

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        if self.normalize_imagenet:
            x = (x - self.image_mean) / self.image_std
        return x

    def forward_feature_maps(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        x = self.preprocess(x)
        x = self.stem(x)
        x = self.layer1(x)
        if "layer1" in self.teacher_layers:
            outputs["layer1"] = x
        x = self.layer2(x)
        if "layer2" in self.teacher_layers:
            outputs["layer2"] = x
        x = self.layer3(x)
        if "layer3" in self.teacher_layers:
            outputs["layer3"] = x
        x = self.layer4(x)
        if "layer4" in self.teacher_layers:
            outputs["layer4"] = x
        return outputs


class MultiLayerPatchCoreModel(nn.Module):
    def __init__(
        self,
        teacher_layers: list[str],
        reduction: str = "topk_mean",
        topk_ratio: float = 0.10,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        backbone_input_size: int = 224,
        normalize_imagenet: bool = True,
        query_chunk_size: int = 1024,
        memory_chunk_size: int = 4096,
    ) -> None:
        super().__init__()
        self.teacher_layers = [str(layer).lower() for layer in teacher_layers]
        self.reduction = str(reduction)
        self.topk_ratio = float(topk_ratio)
        self.query_chunk_size = int(query_chunk_size)
        self.memory_chunk_size = int(memory_chunk_size)

        self.teacher = WideResNet50_2MultiLayerExtractor(
            teacher_layers=self.teacher_layers,
            pretrained=pretrained,
            input_size=backbone_input_size,
            freeze_backbone=freeze_backbone,
            normalize_imagenet=normalize_imagenet,
        )
        self.feature_dim = int(sum(self.teacher.feature_dims[layer] for layer in self.teacher_layers))
        self.reduced_spatial = int(self.teacher.output_spatial)
        self.register_buffer("memory_bank", torch.empty(0, self.feature_dim))

    @property
    def patches_per_image(self) -> int:
        return self.reduced_spatial * self.reduced_spatial

    def patch_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        feature_maps = self.teacher.forward_feature_maps(x)
        target_size = max(feature_map.shape[-1] for feature_map in feature_maps.values())
        embeddings: list[torch.Tensor] = []
        for layer_name in self.teacher_layers:
            feature_map = feature_maps[layer_name]
            if feature_map.shape[-1] != target_size or feature_map.shape[-2] != target_size:
                feature_map = F.interpolate(
                    feature_map,
                    size=(target_size, target_size),
                    mode="bilinear",
                    align_corners=False,
                )
            layer_embeddings = feature_map.permute(0, 2, 3, 1).reshape(x.shape[0], -1, feature_map.shape[1])
            embeddings.append(F.normalize(layer_embeddings, p=2, dim=-1))
        return F.normalize(torch.cat(embeddings, dim=-1), p=2, dim=-1)

    def set_memory_bank(self, memory_bank: torch.Tensor) -> None:
        normalized = F.normalize(memory_bank.to(dtype=torch.float32), p=2, dim=1)
        self.memory_bank = normalized.to(device=self.memory_bank.device, dtype=self.memory_bank.dtype)

    def nearest_patch_distances(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        batch_size, patch_count, _ = patch_embeddings.shape
        flat_queries = patch_embeddings.reshape(-1, self.feature_dim)
        mins: list[torch.Tensor] = []
        for query_start in range(0, flat_queries.shape[0], self.query_chunk_size):
            query_chunk = flat_queries[query_start : query_start + self.query_chunk_size]
            chunk_best = None
            for memory_start in range(0, self.memory_bank.shape[0], self.memory_chunk_size):
                memory_chunk = self.memory_bank[memory_start : memory_start + self.memory_chunk_size]
                distances = torch.cdist(query_chunk, memory_chunk)
                current_best = distances.min(dim=1).values
                chunk_best = current_best if chunk_best is None else torch.minimum(chunk_best, current_best)
            mins.append(chunk_best)
        return torch.cat(mins, dim=0).reshape(batch_size, patch_count)

    def reduce_patch_distances(self, patch_distances: torch.Tensor) -> torch.Tensor:
        if self.reduction == "max":
            return patch_distances.max(dim=1).values
        if self.reduction == "mean":
            return patch_distances.mean(dim=1)
        topk = max(1, int(math.ceil(patch_distances.shape[1] * self.topk_ratio)))
        return torch.topk(patch_distances, k=topk, dim=1).values.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_embeddings = self.patch_embeddings(x)
        patch_distances = self.nearest_patch_distances(patch_embeddings)
        return self.reduce_patch_distances(patch_distances)


def sample_memory_indices(dataset_size: int, memory_bank_size: int, patches_per_image: int, seed: int) -> np.ndarray:
    image_count = min(dataset_size, max(1, math.ceil(memory_bank_size / patches_per_image)))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(dataset_size, size=image_count, replace=False))


def build_memory_subset(dataset: Dataset, memory_bank_size: int, patches_per_image: int, seed: int) -> Subset:
    indices = sample_memory_indices(len(dataset), memory_bank_size, patches_per_image, seed)
    return Subset(dataset, indices.tolist())


def collect_memory_bank(
    model: MultiLayerPatchCoreModel,
    dataloader: DataLoader,
    device: torch.device,
    target_size: int,
    seed: int,
) -> torch.Tensor:
    model.eval()
    patch_batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            normal_mask = labels == 0
            if not torch.any(normal_mask):
                continue
            embeddings = model.patch_embeddings(inputs[normal_mask]).reshape(-1, model.feature_dim)
            patch_batches.append(embeddings.cpu())
    memory_bank = torch.cat(patch_batches, dim=0)
    if memory_bank.shape[0] > target_size:
        generator = torch.Generator().manual_seed(seed)
        keep = torch.randperm(memory_bank.shape[0], generator=generator)[:target_size]
        memory_bank = memory_bank[keep]
    return memory_bank


def collect_scores(model: MultiLayerPatchCoreModel, dataloader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for inputs, labels in dataloader:
            scores = model(inputs.to(device)).cpu().numpy()
            for score, label in zip(scores.tolist(), labels.tolist()):
                rows.append({"score": float(score), "is_anomaly": int(label)})
    return pd.DataFrame(rows)


def run_patchcore_variant(
    variant: dict[str, Any],
    *,
    train_dataset: Dataset,
    val_loader: DataLoader,
    test_loader: DataLoader,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    output_dir: Path,
    seed: int,
    teacher_layers: list[str],
    pretrained: bool,
    freeze_backbone: bool,
    backbone_input_size: int,
    normalize_imagenet: bool,
    threshold_quantile: float,
    query_chunk_size: int,
    memory_chunk_size: int,
    threshold_strategy: str = "normal_quantile",
    max_validation_false_positive_rate: float | None = None,
) -> dict[str, Any]:
    model = MultiLayerPatchCoreModel(
        teacher_layers=teacher_layers,
        reduction=str(variant["reduction"]),
        topk_ratio=float(variant["topk_ratio"]),
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        backbone_input_size=backbone_input_size,
        normalize_imagenet=normalize_imagenet,
        query_chunk_size=query_chunk_size,
        memory_chunk_size=memory_chunk_size,
    ).to(device)

    memory_subset = build_memory_subset(
        train_dataset,
        memory_bank_size=int(variant["memory_bank_size"]),
        patches_per_image=model.patches_per_image,
        seed=seed,
    )
    memory_loader = DataLoader(memory_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    memory_bank = collect_memory_bank(
        model,
        dataloader=memory_loader,
        device=device,
        target_size=int(variant["memory_bank_size"]),
        seed=seed,
    )
    model.set_memory_bank(memory_bank)

    val_scores_df = collect_scores(model, val_loader, device)
    test_scores_df = collect_scores(model, test_loader, device)
    threshold_selection = select_validation_threshold(
        val_scores_df,
        threshold_quantile=threshold_quantile,
        threshold_strategy=threshold_strategy,
        max_false_positive_rate=max_validation_false_positive_rate,
    )
    threshold = float(threshold_selection["threshold"])
    val_metrics = threshold_selection["selected_metrics"]
    val_threshold_sweep_df = threshold_selection["val_threshold_sweep_df"]
    best_validation_sweep = threshold_selection["best_validation_sweep"]

    labels = test_scores_df["is_anomaly"].to_numpy()
    scores = test_scores_df["score"].to_numpy()
    metrics = summarize_threshold_metrics(labels, scores, threshold)
    threshold_sweep_df, best_sweep = sweep_threshold_metrics(labels, scores)

    variant_output_dir = output_dir / str(variant["name"])
    variant_output_dir.mkdir(parents=True, exist_ok=True)
    val_scores_df.to_csv(variant_output_dir / "val_scores.csv", index=False)
    test_scores_df.to_csv(variant_output_dir / "test_scores.csv", index=False)
    val_threshold_sweep_df.to_csv(variant_output_dir / "val_threshold_sweep.csv", index=False)
    threshold_sweep_df.to_csv(variant_output_dir / "threshold_sweep.csv", index=False)

    row = {
        "name": str(variant["name"]),
        "memory_bank_size": int(variant["memory_bank_size"]),
        "memory_subset_images": int(len(memory_subset)),
        "patches_per_image": int(model.patches_per_image),
        "feature_dim": int(model.feature_dim),
        "reduction": str(variant["reduction"]),
        "topk_ratio": float(variant["topk_ratio"]),
        "threshold_strategy": str(threshold_strategy),
        "max_validation_false_positive_rate": None
        if max_validation_false_positive_rate is None
        else float(max_validation_false_positive_rate),
        "threshold": float(threshold),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "auroc": float(metrics["auroc"]),
        "auprc": float(metrics["auprc"]),
        "false_positive_rate": float(metrics["false_positive_rate"]),
        "validation_precision": float(val_metrics["precision"]),
        "validation_recall": float(val_metrics["recall"]),
        "validation_f1": float(val_metrics["f1"]),
        "validation_false_positive_rate": float(val_metrics["false_positive_rate"]),
        "best_validation_sweep_threshold": float(best_validation_sweep["threshold"]),
        "best_validation_sweep_precision": float(best_validation_sweep["precision"]),
        "best_validation_sweep_recall": float(best_validation_sweep["recall"]),
        "best_validation_sweep_f1": float(best_validation_sweep["f1"]),
        "best_sweep_threshold": float(best_sweep["threshold"]),
        "best_sweep_precision": float(best_sweep["precision"]),
        "best_sweep_recall": float(best_sweep["recall"]),
        "best_sweep_f1": float(best_sweep["f1"]),
        "predicted_anomalies": int(metrics["predicted_anomalies"]),
        "output_dir": str(variant_output_dir),
    }
    summary = {
        **row,
        "teacher_backbone": "wideresnet50_2",
        "teacher_layers": teacher_layers,
        "threshold_quantile": float(threshold_quantile),
        "selected_threshold": {
            "strategy": str(threshold_strategy),
            "threshold": float(threshold),
            "max_validation_false_positive_rate": None
            if max_validation_false_positive_rate is None
            else float(max_validation_false_positive_rate),
        },
        "validation_metrics_at_selected_threshold": val_metrics,
        "test_metrics_at_selected_threshold": metrics,
        "metrics_at_validation_threshold": metrics,
        "best_validation_threshold_sweep": {
            "threshold": float(best_validation_sweep["threshold"]),
            "precision": float(best_validation_sweep["precision"]),
            "recall": float(best_validation_sweep["recall"]),
            "f1": float(best_validation_sweep["f1"]),
            "false_positive_rate": float(best_validation_sweep["false_positive_rate"]),
            "predicted_anomalies": int(best_validation_sweep["predicted_anomalies"]),
        },
        "best_threshold_sweep": {
            "threshold": float(best_sweep["threshold"]),
            "precision": float(best_sweep["precision"]),
            "recall": float(best_sweep["recall"]),
            "f1": float(best_sweep["f1"]),
            "false_positive_rate": float(best_sweep["false_positive_rate"]),
            "predicted_anomalies": int(best_sweep["predicted_anomalies"]),
        },
    }
    (variant_output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "row": row,
        "summary": summary,
        "metrics": metrics,
        "best_sweep": best_sweep,
        "best_validation_sweep": best_validation_sweep,
        "threshold": threshold,
        "val_metrics": val_metrics,
        "val_scores_df": val_scores_df,
        "test_scores_df": test_scores_df,
        "val_threshold_sweep_df": val_threshold_sweep_df,
        "threshold_sweep_df": threshold_sweep_df,
    }


def attach_scores_to_metadata(metadata: pd.DataFrame, scores_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    scored = metadata.reset_index(drop=True).copy()
    scored["score"] = scores_df["score"].to_numpy()
    scored["predicted_is_anomaly"] = (scored["score"] >= threshold).astype(int)
    scored["error_type"] = "correct"
    scored.loc[(scored["is_anomaly"] == 0) & (scored["predicted_is_anomaly"] == 1), "error_type"] = "false_positive"
    scored.loc[(scored["is_anomaly"] == 1) & (scored["predicted_is_anomaly"] == 0), "error_type"] = "false_negative"
    return scored


def load_wafer_array(bundle_root: Path, relative_array_path: str) -> np.ndarray:
    return np.load((bundle_root / relative_array_path).resolve())
