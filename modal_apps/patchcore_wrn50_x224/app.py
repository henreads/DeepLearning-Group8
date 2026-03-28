from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import modal


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[2] if len(CURRENT_FILE.parents) >= 3 else CURRENT_FILE.parent
SRC_DIR = (REPO_ROOT / "src").resolve()

APP_NAME = "patchcore-wrn50-x224"
DATA_VOLUME_NAME = "wm811k-data"
ARTIFACT_VOLUME_NAME = "wm811k-artifacts"

REMOTE_DATA_ROOT = "/vol/data"
REMOTE_ARTIFACT_ROOT = "/vol/output"
REMOTE_RAW_PICKLE_PATH = f"{REMOTE_DATA_ROOT}/raw/LSWMD.pkl"

IMAGE_SIZE = 224
NORMAL_LIMIT = 50_000
TEST_DEFECT_FRACTION = 0.05
SECONDARY_TEST_NORMALS = 70_000
SECONDARY_TEST_DEFECTS = 3_500
SEED = 42
BATCH_SIZE = 128
NUM_WORKERS = 0
TEACHER_INPUT_SIZE = 224
TEACHER_LAYERS = ["layer2", "layer3"]
PRETRAINED = True
FREEZE_BACKBONE = True
NORMALIZE_IMAGENET = True
THRESHOLD_QUANTILE = 0.95
QUERY_CHUNK_SIZE = 1024
MEMORY_CHUNK_SIZE = 4096
MIN_MEMORY_IMAGES = 640
SELECTED_CHECKPOINT_NAME = "best_model.pt"

DEFAULT_VARIANTS: list[dict[str, Any]] = [
    {"name": "topk_mb50k_r010_x224", "memory_bank_size": 500_000, "reduction": "topk_mean", "topk_ratio": 0.10},
    {"name": "topk_mb50k_r005_x224", "memory_bank_size": 500_000, "reduction": "topk_mean", "topk_ratio": 0.05},
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "torchvision",
        "tqdm",
    )
    .add_local_dir(str(SRC_DIR), remote_path="/root/src", copy=True)
    .env({"PYTHONPATH": "/root/src"})
)

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)

app = modal.App(APP_NAME)


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> "torch.device":
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_count_slug(count: int) -> str:
    if count >= 1000 and count % 1000 == 0:
        return f"{count // 1000}k"
    if count >= 1000 and count % 100 == 0:
        return f"{count / 1000:.1f}".replace(".", "p") + "k"
    return str(count)


def build_holdout_suffix(test_normals: int, test_defects: int) -> str:
    return f"holdout{format_count_slug(test_normals)}_{format_count_slug(test_defects)}"


def build_metadata_filename(split_mode: str) -> str:
    if split_mode == "report_50k_5pct":
        return "metadata_50k_5pct.csv"
    if split_mode == "holdout70k_3p5k":
        return f"metadata_50k_5pct_{build_holdout_suffix(SECONDARY_TEST_NORMALS, SECONDARY_TEST_DEFECTS)}.csv"
    raise ValueError(f"Unsupported split_mode: {split_mode}")


def build_arrays_dir_name(split_mode: str) -> str:
    if split_mode == "report_50k_5pct":
        return "arrays"
    if split_mode == "holdout70k_3p5k":
        return f"arrays_{build_holdout_suffix(SECONDARY_TEST_NORMALS, SECONDARY_TEST_DEFECTS)}"
    raise ValueError(f"Unsupported split_mode: {split_mode}")


def normalize_map(wafer_map: "np.ndarray", image_size: int) -> "np.ndarray":
    import numpy as np
    import torch
    import torch.nn.functional as F

    wafer_map = np.asarray(wafer_map, dtype=np.float32) / 2.0
    tensor = torch.from_numpy(wafer_map).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return resized.squeeze(0).squeeze(0).numpy()


def infer_label_from_failure_type(failure_type_text: str) -> str | None:
    failure = failure_type_text.lower()
    if failure == "none":
        return "none"
    if failure:
        return "pattern"
    return None


def split_normals(normal_df: "pd.DataFrame", seed: int) -> "pd.DataFrame":
    shuffled = normal_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    shuffled.loc[: train_end - 1, "split"] = "train"
    shuffled.loc[train_end: val_end - 1, "split"] = "val"
    shuffled.loc[val_end:, "split"] = "test"
    return shuffled


def sample_test_defects(
    defect_df: "pd.DataFrame",
    normal_df: "pd.DataFrame",
    fraction: float,
    seed: int,
) -> "pd.DataFrame":
    requested = max(1, int(round(int((normal_df["split"] == "test").sum()) * fraction)))
    sampled = defect_df.sample(n=min(requested, len(defect_df)), random_state=seed).copy()
    sampled["split"] = "test"
    return sampled


def build_secondary_holdout_export_df(
    df: "pd.DataFrame",
    normal_limit: int,
    test_defect_fraction: float,
    secondary_test_normals: int,
    secondary_test_defects: int,
    seed: int,
) -> "pd.DataFrame":
    import pandas as pd

    normal_df = df[df["label"] == "none"].copy()
    defect_df = df[df["label"] == "pattern"].copy()

    base_normals = normal_df.sample(n=min(normal_limit, len(normal_df)), random_state=seed).copy()
    base_normals = split_normals(base_normals, seed)
    base_defects = sample_test_defects(defect_df, base_normals, test_defect_fraction, seed)

    used_raw_indices = set(base_normals["raw_index"].astype(int).tolist())
    used_raw_indices.update(base_defects["raw_index"].astype(int).tolist())

    normal_pool = normal_df[~normal_df["raw_index"].isin(used_raw_indices)].copy()
    defect_pool = defect_df[~defect_df["raw_index"].isin(used_raw_indices)].copy()

    if secondary_test_normals > len(normal_pool):
        raise ValueError(
            f"Requested {secondary_test_normals} secondary-holdout normals, but only {len(normal_pool)} are available."
        )
    if secondary_test_defects > len(defect_pool):
        raise ValueError(
            f"Requested {secondary_test_defects} secondary-holdout defects, but only {len(defect_pool)} are available."
        )

    holdout_normals = normal_pool.sample(n=secondary_test_normals, random_state=seed).copy()
    holdout_normals["split"] = "test"
    holdout_defects = defect_pool.sample(n=secondary_test_defects, random_state=seed).copy()
    holdout_defects["split"] = "test"

    base_train_val = base_normals[base_normals["split"].isin(["train", "val"])].copy()
    return pd.concat([base_train_val, holdout_normals, holdout_defects], ignore_index=True)


def parse_variants(variants_text: str) -> list[dict[str, Any]]:
    if not variants_text.strip():
        return [dict(variant) for variant in DEFAULT_VARIANTS]
    names = [name.strip() for name in variants_text.split(",") if name.strip()]
    variant_lookup = {str(variant["name"]): dict(variant) for variant in DEFAULT_VARIANTS}
    missing = [name for name in names if name not in variant_lookup]
    if missing:
        raise ValueError(f"Unknown variant(s): {missing}")
    return [variant_lookup[name] for name in names]


def prepare_dataset(
    raw_pickle_path: Path,
    processed_root: Path,
    split_mode: str,
    force_rebuild_dataset: bool,
) -> Path:
    import numpy as np
    import pandas as pd
    from tqdm.auto import tqdm

    from wafer_defect.data.legacy_pickle import read_legacy_pickle, unwrap_legacy_value

    metadata_path = processed_root / build_metadata_filename(split_mode)
    arrays_dir = processed_root / build_arrays_dir_name(split_mode)

    if metadata_path.exists() and not force_rebuild_dataset:
        print(f"Reusing processed metadata: {metadata_path}")
        return metadata_path

    processed_root.mkdir(parents=True, exist_ok=True)
    arrays_dir.mkdir(parents=True, exist_ok=True)

    raw_df = read_legacy_pickle(raw_pickle_path).copy()
    raw_df["raw_index"] = np.arange(len(raw_df), dtype=np.int64)
    raw_df["failureTypeText"] = raw_df["failureType"].map(unwrap_legacy_value)
    raw_df["trianTestLabelText"] = raw_df["trianTestLabel"].map(unwrap_legacy_value)
    raw_df["label"] = raw_df["failureTypeText"].map(infer_label_from_failure_type)
    raw_df = raw_df[raw_df["label"].notna()].reset_index(drop=True)

    if split_mode == "holdout70k_3p5k":
        export_df = build_secondary_holdout_export_df(
            raw_df,
            normal_limit=NORMAL_LIMIT,
            test_defect_fraction=TEST_DEFECT_FRACTION,
            secondary_test_normals=SECONDARY_TEST_NORMALS,
            secondary_test_defects=SECONDARY_TEST_DEFECTS,
            seed=SEED,
        )
    elif split_mode == "report_50k_5pct":
        normal_df = raw_df[raw_df["label"] == "none"].copy()
        defect_df = raw_df[raw_df["label"] == "pattern"].copy()
        normal_df = normal_df.sample(n=min(NORMAL_LIMIT, len(normal_df)), random_state=SEED).copy()
        normal_df = split_normals(normal_df, SEED)
        defect_df = sample_test_defects(defect_df, normal_df, TEST_DEFECT_FRACTION, SEED)
        export_df = pd.concat([normal_df, defect_df], ignore_index=True)
    else:
        raise ValueError(f"Unsupported split_mode: {split_mode}")

    split_summary = (
        export_df.groupby(["split", "label"]).size().rename("count").reset_index().sort_values(["split", "label"])
    )
    print(split_summary.to_string(index=False))

    records: list[dict[str, Any]] = []
    for row_index, row in tqdm(export_df.iterrows(), total=len(export_df), desc="Preparing arrays"):
        array_path = arrays_dir / f"wafer_{row_index:07d}.npy"
        raw_map = row["waferMap"]
        np.save(array_path, normalize_map(raw_map, image_size=IMAGE_SIZE))
        records.append(
            {
                "array_path": str(array_path),
                "label": row["label"],
                "defect_type": row["failureTypeText"] or "unlabeled",
                "is_anomaly": int(row["label"] == "pattern"),
                "split": row["split"],
                "source_split": row["trianTestLabelText"] or "unlabeled",
                "original_height": int(raw_map.shape[0]),
                "original_width": int(raw_map.shape[1]),
            }
        )

    pd.DataFrame(records).to_csv(metadata_path, index=False)
    print(f"Saved processed metadata: {metadata_path}")
    return metadata_path


def load_existing_variant_row(summary_path: Path) -> dict[str, Any]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    keys = [
        "name",
        "memory_bank_size",
        "memory_subset_images",
        "patches_per_image",
        "feature_dim",
        "reduction",
        "topk_ratio",
        "threshold",
        "precision",
        "recall",
        "f1",
        "auroc",
        "auprc",
        "best_sweep_threshold",
        "best_sweep_precision",
        "best_sweep_recall",
        "best_sweep_f1",
        "predicted_anomalies",
        "output_dir",
    ]
    return {key: payload[key] for key in keys}


def summarize_threshold_metrics(labels: "np.ndarray", scores: "np.ndarray", threshold: float) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score, roc_auc_score

    predicted = (scores >= threshold).astype(int)
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
        "predicted_anomalies": int(predicted.sum()),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
    }


def sweep_threshold_metrics(labels: "np.ndarray", scores: "np.ndarray") -> tuple["pd.DataFrame", dict[str, Any]]:
    import numpy as np
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for threshold in np.unique(scores):
        metrics = summarize_threshold_metrics(labels, scores, float(threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "predicted_anomalies": metrics["predicted_anomalies"],
            }
        )
    sweep_df = pd.DataFrame(rows).sort_values(["f1", "precision", "recall"], ascending=False).reset_index(drop=True)
    return pd.DataFrame(rows), sweep_df.iloc[0].to_dict()


@app.function(
    image=image,
    gpu="L40S",
    memory=65536,
    timeout=24 * 60 * 60,
    volumes={
        REMOTE_DATA_ROOT: data_volume,
        REMOTE_ARTIFACT_ROOT: artifact_volume,
    },
)
def run_patchcore_x224(
    split_mode: str = "holdout70k_3p5k",
    variants_text: str = "topk_mb50k_r005_x224",
    force_rebuild_dataset: int = 0,
    force_rerun: int = 0,
    save_selected_checkpoint: int = 1,
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, Subset
    from tqdm.auto import tqdm

    from wafer_defect.models.resnet import ResNetFeatureExtractor

    set_seed(SEED)
    device = resolve_device()

    raw_pickle_path = Path(REMOTE_RAW_PICKLE_PATH)
    if not raw_pickle_path.exists():
        raise FileNotFoundError(
            f"Raw pickle not found at {raw_pickle_path}. Upload it first with --mode upload-only or --mode upload-and-run."
        )

    processed_root = Path(REMOTE_DATA_ROOT) / "processed" / f"x{IMAGE_SIZE}" / "wm811k_patchcore_wrn50_x224"
    metadata_path = prepare_dataset(
        raw_pickle_path=raw_pickle_path,
        processed_root=processed_root,
        split_mode=split_mode,
        force_rebuild_dataset=bool(force_rebuild_dataset),
    )
    data_volume.commit()

    output_dir = Path(REMOTE_ARTIFACT_ROOT) / "patchcore_wrn50_x224" / split_mode
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = parse_variants(variants_text)
    config_payload = {
        "run": {
            "app_name": APP_NAME,
            "split_mode": split_mode,
            "seed": SEED,
            "save_selected_checkpoint": bool(save_selected_checkpoint),
        },
        "data": {
            "metadata_csv": str(metadata_path),
            "image_size": IMAGE_SIZE,
            "normal_limit": NORMAL_LIMIT,
            "test_defect_fraction": TEST_DEFECT_FRACTION,
            "secondary_test_normals": SECONDARY_TEST_NORMALS,
            "secondary_test_defects": SECONDARY_TEST_DEFECTS,
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
        },
        "model": {
            "teacher_backbone": "wideresnet50_2",
            "teacher_layers": TEACHER_LAYERS,
            "pretrained": PRETRAINED,
            "freeze_backbone": FREEZE_BACKBONE,
            "teacher_input_size": TEACHER_INPUT_SIZE,
            "normalize_imagenet": NORMALIZE_IMAGENET,
            "query_chunk_size": QUERY_CHUNK_SIZE,
            "memory_chunk_size": MEMORY_CHUNK_SIZE,
            "min_memory_images": MIN_MEMORY_IMAGES,
        },
        "scoring": {"threshold_quantile": THRESHOLD_QUANTILE},
        "variants": variants,
    }
    (output_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    class VolumeWaferDataset(Dataset):
        def __init__(self, metadata_csv: Path, split: str) -> None:
            self.metadata = pd.read_csv(metadata_csv)
            self.metadata = self.metadata[self.metadata["split"] == split].reset_index(drop=True)

        def __len__(self) -> int:
            return len(self.metadata)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
            row = self.metadata.iloc[index]
            wafer_map = np.load(row["array_path"]).astype(np.float32)
            return torch.from_numpy(wafer_map).unsqueeze(0), torch.tensor(int(row["is_anomaly"]), dtype=torch.long)

    train_dataset = VolumeWaferDataset(metadata_path, split="train")
    val_dataset = VolumeWaferDataset(metadata_path, split="val")
    test_dataset = VolumeWaferDataset(metadata_path, split="test")

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    def _teacher_spatial_size(input_size: int, layer_names: list[str]) -> int:
        downsample = {"layer1": 4, "layer2": 8, "layer3": 16, "layer4": 32}
        return max(1, input_size // min(downsample[name] for name in layer_names))

    class MultiLayerPatchCoreModel(nn.Module):
        def __init__(self, variant: dict[str, Any]) -> None:
            super().__init__()
            self.teacher_layers = [str(x).lower() for x in TEACHER_LAYERS]
            self.reduction = str(variant["reduction"])
            self.topk_ratio = float(variant["topk_ratio"])
            self.query_chunk_size = int(QUERY_CHUNK_SIZE)
            self.memory_chunk_size = int(MEMORY_CHUNK_SIZE)
            self.teacher = ResNetFeatureExtractor(
                backbone_name="wideresnet50_2",
                pretrained=PRETRAINED,
                input_size=TEACHER_INPUT_SIZE,
                freeze_backbone=FREEZE_BACKBONE,
                normalize_imagenet=NORMALIZE_IMAGENET,
            )
            self.feature_dims = {"layer2": 512, "layer3": 1024}
            self.feature_dim = sum(self.feature_dims.values())
            self.reduced_spatial = _teacher_spatial_size(TEACHER_INPUT_SIZE, self.teacher_layers)
            self.register_buffer("memory_bank", torch.empty(0, self.feature_dim))

        @property
        def patches_per_image(self) -> int:
            return self.reduced_spatial * self.reduced_spatial

        def forward_feature_maps(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            outputs: dict[str, torch.Tensor] = {}
            x = self.teacher.preprocess(x)
            x = self.teacher.stem(x)
            x = self.teacher.layer1(x)
            x = self.teacher.layer2(x)
            if "layer2" in self.teacher_layers:
                outputs["layer2"] = x
            x = self.teacher.layer3(x)
            if "layer3" in self.teacher_layers:
                outputs["layer3"] = x
            x = self.teacher.layer4(x)
            if "layer4" in self.teacher_layers:
                outputs["layer4"] = x
            return outputs

        def patch_embeddings(self, x: torch.Tensor) -> torch.Tensor:
            feature_maps = self.forward_feature_maps(x)
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
            self.memory_bank = F.normalize(memory_bank.to(dtype=torch.float32), p=2, dim=1).to(
                device=self.memory_bank.device,
                dtype=self.memory_bank.dtype,
            )

        def nearest_patch_distances(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
            flat_queries = patch_embeddings.reshape(-1, self.feature_dim)
            all_mins: list[torch.Tensor] = []
            for query_start in range(0, flat_queries.shape[0], self.query_chunk_size):
                query_chunk = flat_queries[query_start : query_start + self.query_chunk_size]
                chunk_best = None
                for memory_start in range(0, self.memory_bank.shape[0], self.memory_chunk_size):
                    memory_chunk = self.memory_bank[memory_start : memory_start + self.memory_chunk_size]
                    distances = torch.cdist(query_chunk, memory_chunk)
                    current_best = distances.min(dim=1).values
                    chunk_best = current_best if chunk_best is None else torch.minimum(chunk_best, current_best)
                all_mins.append(chunk_best)
            return torch.cat(all_mins, dim=0).reshape(patch_embeddings.shape[0], patch_embeddings.shape[1])

        def reduce_patch_distances(self, patch_distances: torch.Tensor) -> torch.Tensor:
            if self.reduction == "max":
                return patch_distances.max(dim=1).values
            if self.reduction == "mean":
                return patch_distances.mean(dim=1)
            topk = max(1, int(math.ceil(patch_distances.shape[1] * self.topk_ratio)))
            return torch.topk(patch_distances, k=topk, dim=1).values.mean(dim=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.reduce_patch_distances(self.nearest_patch_distances(self.patch_embeddings(x)))

    def sample_memory_indices(
        dataset_size: int,
        memory_bank_size: int,
        patches_per_image: int,
        seed: int,
        min_images: int = 0,
    ) -> np.ndarray:
        image_count = min(dataset_size, max(min_images, math.ceil(memory_bank_size / patches_per_image)))
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(dataset_size, size=image_count, replace=False))

    def build_memory_subset(
        dataset: Dataset,
        memory_bank_size: int,
        patches_per_image: int,
        seed: int,
        min_images: int = 0,
    ) -> Subset:
        indices = sample_memory_indices(len(dataset), memory_bank_size, patches_per_image, seed, min_images=min_images)
        return Subset(dataset, indices.tolist())

    def collect_memory_bank(
        model: MultiLayerPatchCoreModel,
        dataloader: DataLoader,
        target_size: int,
        seed: int,
    ) -> torch.Tensor:
        patch_batches: list[torch.Tensor] = []
        model.eval()
        with torch.inference_mode():
            for inputs, labels in tqdm(dataloader, desc="Collect memory bank"):
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

    def collect_scores(model: MultiLayerPatchCoreModel, dataloader: DataLoader) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        model.eval()
        with torch.inference_mode():
            for inputs, labels in tqdm(dataloader, desc="Score split"):
                scores = model(inputs.to(device)).cpu().numpy()
                for score, label in zip(scores.tolist(), labels.tolist()):
                    rows.append({"score": float(score), "is_anomaly": int(label)})
        return pd.DataFrame(rows)

    def build_checkpoint_payload(
        model: MultiLayerPatchCoreModel,
        variant: dict[str, Any],
        threshold: float,
    ) -> dict[str, Any]:
        return {
            "model_type": "patchcore",
            "checkpoint_format": "patchcore_selected_variant_v1",
            "model_state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
            "config": {
                "training": {"device": "auto"},
                "data": {
                    "metadata_csv": str(metadata_path),
                    "image_size": int(IMAGE_SIZE),
                    "batch_size": int(BATCH_SIZE),
                    "num_workers": int(NUM_WORKERS),
                },
                "model": {
                    "type": "patchcore",
                    "backbone_type": "wideresnet50_2",
                    "teacher_layers": list(TEACHER_LAYERS),
                    "pretrained": bool(PRETRAINED),
                    "freeze_backbone": bool(FREEZE_BACKBONE),
                    "backbone_input_size": int(TEACHER_INPUT_SIZE),
                    "normalize_imagenet": bool(NORMALIZE_IMAGENET),
                    "query_chunk_size": int(QUERY_CHUNK_SIZE),
                    "memory_chunk_size": int(MEMORY_CHUNK_SIZE),
                    "memory_bank_size": int(model.memory_bank.shape[0]),
                    "reduction": str(variant["reduction"]),
                    "topk_ratio": float(variant["topk_ratio"]),
                },
                "scoring": {"threshold_quantile": float(THRESHOLD_QUANTILE), "threshold": float(threshold)},
            },
            "variant": {
                "name": str(variant["name"]),
                "memory_bank_size": int(variant["memory_bank_size"]),
                "reduction": str(variant["reduction"]),
                "topk_ratio": float(variant["topk_ratio"]),
            },
            "threshold": float(threshold),
        }

    memory_bank_cache: dict[int, dict[str, Any]] = {}
    sweep_rows: list[dict[str, Any]] = []

    for variant in variants:
        variant_name = str(variant["name"])
        variant_output_dir = output_dir / variant_name
        variant_output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = variant_output_dir / "summary.json"

        if summary_path.exists() and not bool(force_rerun):
            print(f"Reusing existing results for {variant_name}")
            row = load_existing_variant_row(summary_path)
            sweep_rows.append(row)
            continue

        cache_key = int(variant["memory_bank_size"])
        if cache_key not in memory_bank_cache:
            cache_model = MultiLayerPatchCoreModel(variant).to(device)
            memory_subset = build_memory_subset(
                train_dataset,
                memory_bank_size=cache_key,
                patches_per_image=cache_model.patches_per_image,
                seed=SEED,
                min_images=MIN_MEMORY_IMAGES,
            )
            memory_loader = DataLoader(memory_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
            memory_bank = collect_memory_bank(cache_model, memory_loader, target_size=cache_key, seed=SEED)
            memory_bank_cache[cache_key] = {
                "memory_bank": memory_bank,
                "memory_subset_images": int(len(memory_subset)),
                "patches_per_image": int(cache_model.patches_per_image),
                "feature_dim": int(cache_model.feature_dim),
            }

        model = MultiLayerPatchCoreModel(variant).to(device)
        cache_entry = memory_bank_cache[cache_key]
        model.set_memory_bank(cache_entry["memory_bank"])

        val_scores_df = collect_scores(model, val_loader)
        test_scores_df = collect_scores(model, test_loader)
        threshold = float(val_scores_df.loc[val_scores_df["is_anomaly"] == 0, "score"].quantile(THRESHOLD_QUANTILE))
        labels = test_scores_df["is_anomaly"].to_numpy()
        scores = test_scores_df["score"].to_numpy()
        metrics = summarize_threshold_metrics(labels, scores, threshold)
        threshold_sweep_df, best_sweep = sweep_threshold_metrics(labels, scores)

        val_scores_df.to_csv(variant_output_dir / "val_scores.csv", index=False)
        test_scores_df.to_csv(variant_output_dir / "test_scores.csv", index=False)
        threshold_sweep_df.to_csv(variant_output_dir / "threshold_sweep.csv", index=False)

        row = {
            "name": variant_name,
            "memory_bank_size": int(variant["memory_bank_size"]),
            "memory_subset_images": int(cache_entry["memory_subset_images"]),
            "patches_per_image": int(cache_entry["patches_per_image"]),
            "feature_dim": int(cache_entry["feature_dim"]),
            "reduction": str(variant["reduction"]),
            "topk_ratio": float(variant["topk_ratio"]),
            "threshold": float(threshold),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
            "auroc": float(metrics["auroc"]),
            "auprc": float(metrics["auprc"]),
            "best_sweep_threshold": float(best_sweep["threshold"]),
            "best_sweep_precision": float(best_sweep["precision"]),
            "best_sweep_recall": float(best_sweep["recall"]),
            "best_sweep_f1": float(best_sweep["f1"]),
            "predicted_anomalies": int(metrics["predicted_anomalies"]),
            "output_dir": str(variant_output_dir),
        }
        summary_payload = {
            **row,
            "teacher_backbone": "wideresnet50_2",
            "teacher_layers": TEACHER_LAYERS,
            "threshold_quantile": float(THRESHOLD_QUANTILE),
            "metrics_at_validation_threshold": metrics,
            "best_threshold_sweep": {
                "threshold": float(best_sweep["threshold"]),
                "precision": float(best_sweep["precision"]),
                "recall": float(best_sweep["recall"]),
                "f1": float(best_sweep["f1"]),
                "predicted_anomalies": int(best_sweep["predicted_anomalies"]),
            },
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        sweep_rows.append(row)

    sweep_results_df = pd.DataFrame(sweep_rows).sort_values(["f1", "auroc"], ascending=False).reset_index(drop=True)
    sweep_results_df.to_csv(output_dir / "patchcore_sweep_results.csv", index=False)
    best_row = dict(sweep_results_df.iloc[0].to_dict())
    (output_dir / "patchcore_sweep_summary.json").write_text(
        json.dumps(
            {
                "sweep_variants": [variant["name"] for variant in variants],
                "base_output_dir": str(output_dir),
                "teacher_backbone": "wideresnet50_2",
                "teacher_layers": TEACHER_LAYERS,
                "best_variant": best_row,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    checkpoint_path = output_dir / str(best_row["name"]) / SELECTED_CHECKPOINT_NAME
    checkpoint_manifest_path = output_dir / "selected_checkpoint.json"
    if bool(save_selected_checkpoint):
        best_variant = next(variant for variant in variants if str(variant["name"]) == str(best_row["name"]))
        cache_key = int(best_variant["memory_bank_size"])
        if cache_key not in memory_bank_cache:
            checkpoint_model = MultiLayerPatchCoreModel(best_variant).to(device)
            memory_subset = build_memory_subset(
                train_dataset,
                memory_bank_size=cache_key,
                patches_per_image=checkpoint_model.patches_per_image,
                seed=SEED,
                min_images=MIN_MEMORY_IMAGES,
            )
            memory_loader = DataLoader(memory_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
            memory_bank_cache[cache_key] = {
                "memory_bank": collect_memory_bank(checkpoint_model, memory_loader, target_size=cache_key, seed=SEED),
                "memory_subset_images": int(len(memory_subset)),
                "patches_per_image": int(checkpoint_model.patches_per_image),
                "feature_dim": int(checkpoint_model.feature_dim),
            }

        if not checkpoint_path.exists() or bool(force_rerun):
            checkpoint_model = MultiLayerPatchCoreModel(best_variant).to(device)
            checkpoint_model.set_memory_bank(memory_bank_cache[cache_key]["memory_bank"])
            checkpoint_payload = build_checkpoint_payload(
                checkpoint_model,
                best_variant,
                threshold=float(best_row["threshold"]),
            )
            torch.save(checkpoint_payload, checkpoint_path)

        checkpoint_manifest = {
            "best_variant": str(best_row["name"]),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_name": SELECTED_CHECKPOINT_NAME,
            "approx_size_gb": round(checkpoint_path.stat().st_size / (1000 ** 3), 3),
            "memory_bank_shape": [int(best_row["memory_bank_size"]), int(best_row["feature_dim"])],
            "note": "Checkpoint includes the fitted PatchCore memory bank, so the file is expected to be large.",
        }
        checkpoint_manifest_path.write_text(json.dumps(checkpoint_manifest, indent=2), encoding="utf-8")

    artifact_volume.commit()

    metadata_df = pd.read_csv(metadata_path)
    split_counts = (
        metadata_df.groupby(["split", "label"]).size().rename("count").reset_index().to_dict(orient="records")
    )

    result = {
        "split_mode": split_mode,
        "metadata_csv": str(metadata_path),
        "output_dir": str(output_dir),
        "variants": [variant["name"] for variant in variants],
        "best_variant": best_row,
        "split_counts": split_counts,
        "selected_checkpoint": str(checkpoint_path) if bool(save_selected_checkpoint) else "",
    }
    print(json.dumps(result, indent=2))
    return result


def upload_raw_pickle(raw_pickle: Path) -> str:
    if not raw_pickle.exists():
        raise FileNotFoundError(f"Raw pickle not found: {raw_pickle}")
    vol = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
    remote_volume_path = "/raw/LSWMD.pkl"
    with vol.batch_upload(force=True) as batch:
        batch.put_file(str(raw_pickle), remote_volume_path)
    return remote_volume_path


@app.local_entrypoint()
def main(
    mode: str = "run",
    raw_pickle: str = "data/raw/LSWMD.pkl",
    split_mode: str = "holdout70k_3p5k",
    variants: str = "topk_mb50k_r005_x224",
    force_rebuild_dataset: int = 0,
    force_rerun: int = 0,
    save_selected_checkpoint: int = 1,
) -> None:
    raw_pickle_path = Path(raw_pickle).resolve()

    if mode not in {"upload-only", "run", "upload-and-run"}:
        raise ValueError(f"Unsupported mode: {mode}")

    if mode in {"upload-only", "upload-and-run"}:
        remote_path = upload_raw_pickle(raw_pickle_path)
        print(f"Uploaded raw pickle to volume path: {remote_path}")

    if mode in {"run", "upload-and-run"}:
        result = run_patchcore_x224.remote(
            split_mode=split_mode,
            variants_text=variants,
            force_rebuild_dataset=force_rebuild_dataset,
            force_rerun=force_rerun,
            save_selected_checkpoint=save_selected_checkpoint,
        )
        print(json.dumps(result, indent=2))
