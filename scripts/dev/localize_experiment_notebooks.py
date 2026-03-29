from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_notebook(path: Path, notebook: dict) -> None:
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def clear_outputs(notebook: dict) -> None:
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []


def set_cell_source(notebook: dict, cell_index: int, source: str) -> None:
    notebook["cells"][cell_index]["source"] = source.splitlines(keepends=True)


def replace_in_cells(notebook: dict, old: str, new: str) -> None:
    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        if old in source:
            cell["source"] = source.replace(old, new).splitlines(keepends=True)


def patch_wrn_x64_main() -> None:
    path = REPO_ROOT / "experiments/anomaly_detection/patchcore/wideresnet50/x64/main/notebook.ipynb"
    nb = load_notebook(path)

    set_cell_source(
        nb,
        0,
        """# PatchCore + WideResNet50-2 Multilayer (All-in-One)

This notebook is designed to run locally from the repository on any machine with the project dependencies installed.

Place `LSWMD.pkl` somewhere accessible locally, then run all cells. The notebook will prepare the shared `64x64` split, fit a multilayer `WideResNet50-2` PatchCore model using `layer2 + layer3`, run a small critical sweep, and save repo-compatible summaries.
""",
    )

    set_cell_source(
        nb,
        2,
        """from pathlib import Path

def resolve_repo_root(start: Path | None = None) -> Path:
    start_path = (start or Path.cwd()).resolve()
    for candidate in [start_path, *start_path.parents]:
        if (candidate / "experiments").exists() and (candidate / "data").exists():
            return candidate
    raise RuntimeError("Could not locate the repository root.")

REPO_ROOT = resolve_repo_root()

RAW_PICKLE = "LSWMD.pkl"  # leave blank to auto-find LSWMD.pkl
OUTPUT_DIR = str(
    (
        REPO_ROOT
        / "experiments/anomaly_detection/patchcore/wideresnet50/x64/main/artifacts/patchcore_wideresnet50_multilayer"
    ).resolve()
)
IMAGE_SIZE = 64
NORMAL_LIMIT = 50_000
TEST_DEFECT_FRACTION = 0.05
BATCH_SIZE = 64
NUM_WORKERS = 0
DEVICE = "auto"
SEED = 42
TEACHER_INPUT_SIZE = 224
TEACHER_LAYERS = ["layer2", "layer3"]
PRETRAINED = True
FREEZE_BACKBONE = True
NORMALIZE_IMAGENET = True
THRESHOLD_QUANTILE = 0.95
QUERY_CHUNK_SIZE = 1024
MEMORY_CHUNK_SIZE = 4096
SWEEP_VARIANTS = [
    {"name": "mean_mb20k", "memory_bank_size": 20_000, "reduction": "mean", "topk_ratio": 0.10},
    {"name": "mean_mb50k", "memory_bank_size": 50_000, "reduction": "mean", "topk_ratio": 0.10},
    {"name": "topk_mb50k_r010", "memory_bank_size": 50_000, "reduction": "topk_mean", "topk_ratio": 0.10},
    {"name": "topk_mb50k_r015", "memory_bank_size": 50_000, "reduction": "topk_mean", "topk_ratio": 0.15},
]
""",
    )

    clear_outputs(nb)
    save_notebook(path, nb)


def patch_wrn_labeled_120k() -> None:
    notebook_paths = [
        REPO_ROOT / "experiments/anomaly_detection/patchcore/wideresnet50/x64/labeled_120k/notebook.ipynb",
        REPO_ROOT / "experiments/anomaly_detection/patchcore/wideresnet50/x64/labeled_120k/dataset_helper.ipynb",
        REPO_ROOT / "experiments/anomaly_detection/patchcore/wideresnet50/x64/labeled_120k/results_review.ipynb",
        REPO_ROOT / "experiments/anomaly_detection/patchcore/wideresnet50/x64/labeled_120k/threshold_policies.ipynb",
    ]

    # Main training notebook.
    nb = load_notebook(notebook_paths[0])
    set_cell_source(
        nb,
        0,
        """# PatchCore + WideResNet50-2 Multilayer Training

This notebook runs the current best anomaly detector from the report on a larger labeled local split.

Model setup:
- PatchCore with a frozen pretrained `WideResNet50-2`
- multilayer feature bank from `layer2 + layer3`
- validation-threshold evaluation at the `0.95` normal quantile
- small comparison sweep around the report winner `topk_mb50k_r010`
""",
    )
    set_cell_source(
        nb,
        1,
        """import os
from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch
from IPython.display import display
from torch.utils.data import DataLoader

cwd = Path.cwd().resolve()
BUNDLE_ROOT = None
for candidate in [cwd, *cwd.parents]:
    if (candidate / "helpers" / "patchcore_wrn50_local.py").exists():
        BUNDLE_ROOT = candidate
        break

if BUNDLE_ROOT is None:
    raise RuntimeError("Could not locate the local WRN PatchCore experiment root.")

HELPERS_ROOT = BUNDLE_ROOT / "helpers"
if str(HELPERS_ROOT) not in sys.path:
    sys.path.insert(0, str(HELPERS_ROOT))

from patchcore_wrn50_local import (
    DEFAULT_SPLIT_CONFIG,
    DEFAULT_VARIANTS,
    WaferArrayDataset,
    attach_scores_to_metadata,
    auto_find_raw_pickle,
    defect_type_summary,
    load_wafer_array,
    prepare_dataset,
    resolve_data_root,
    resolve_device,
    resolve_output_root,
    run_patchcore_variant,
    set_seed,
    split_summary_wide,
)
""",
    )
    clear_outputs(nb)
    save_notebook(notebook_paths[0], nb)

    # Dataset helper notebook.
    nb = load_notebook(notebook_paths[1])
    set_cell_source(
        nb,
        0,
        """# WideResNet50-2 PatchCore Dataset Helper

This helper notebook prepares the labeled local split used by the current best anomaly detector.

Requested split:
- train: `120,000` total with `6,000` anomalies
- validation: `10,000` total with `500` anomalies
- test: `20,000` total with exactly `1,000` anomalies
""",
    )
    set_cell_source(
        nb,
        1,
        """import os
from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

cwd = Path.cwd().resolve()
BUNDLE_ROOT = None
for candidate in [cwd, *cwd.parents]:
    if (candidate / "helpers" / "patchcore_wrn50_local.py").exists():
        BUNDLE_ROOT = candidate
        break

if BUNDLE_ROOT is None:
    raise RuntimeError("Could not locate the local WRN PatchCore experiment root.")

HELPERS_ROOT = BUNDLE_ROOT / "helpers"
if str(HELPERS_ROOT) not in sys.path:
    sys.path.insert(0, str(HELPERS_ROOT))

from patchcore_wrn50_local import (
    DEFAULT_SPLIT_CONFIG,
    auto_find_raw_pickle,
    defect_type_summary,
    load_wafer_array,
    metadata_paths,
    prepare_dataset,
    resolve_data_root,
    resolve_output_root,
    set_seed,
    split_summary,
    split_summary_wide,
)
""",
    )
    clear_outputs(nb)
    save_notebook(notebook_paths[1], nb)

    # Results review notebook.
    nb = load_notebook(notebook_paths[2])
    set_cell_source(
        nb,
        0,
        """# PatchCore WRN50 120k Local Results Review

This notebook loads the local artifact bundle for the labeled `120k / 10k / 20k` PatchCore WideResNet50-2 run and summarizes the final evaluation.

It is meant to be a lightweight review notebook for:
- the selected variant and threshold
- the comparison sweep across WRN50 PatchCore settings
- held-out test metrics on the `20k` evaluation split
- the main false positive and false negative tables
""",
    )
    set_cell_source(
        nb,
        1,
        """from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

cwd = Path.cwd().resolve()
EXPERIMENT_ROOT = None
for candidate in [cwd, *cwd.parents]:
    if (candidate / "helpers" / "patchcore_wrn50_local.py").exists() and (candidate / "README.md").exists():
        EXPERIMENT_ROOT = candidate
        break

if EXPERIMENT_ROOT is None:
    raise RuntimeError("Could not locate the local WRN PatchCore experiment root.")

ARTIFACT_DIR = EXPERIMENT_ROOT / "artifacts" / "patchcore_wrn50_multilayer_120k_5pct"

if not ARTIFACT_DIR.exists():
    raise FileNotFoundError(
        f"No local artifact bundle was found at {ARTIFACT_DIR}. Run notebook.ipynb first to generate it."
    )
""",
    )
    replace_in_cells(nb, 'RUN_ROOT', 'EXPERIMENT_ROOT')
    replace_in_cells(nb, 'run_root', 'experiment_root')
    clear_outputs(nb)
    save_notebook(notebook_paths[2], nb)

    # Threshold policies notebook.
    nb = load_notebook(notebook_paths[3])
    set_cell_source(
        nb,
        1,
        """from pathlib import Path
import sys

NOTEBOOK_ROOT = Path.cwd().resolve()
if str(NOTEBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Markdown, display

from helpers.patchcore_threshold_tools import (
    build_review_policy_summary,
    build_single_threshold_policy_table,
    build_threshold_sweep,
    load_variant_artifacts,
)


def resolve_experiment_root(start: Path | None = None) -> Path:
    start_path = (start or Path.cwd()).resolve()
    for candidate in [start_path, *start_path.parents]:
        if (candidate / "helpers" / "patchcore_wrn50_local.py").exists() and (candidate / "README.md").exists():
            return candidate
    raise FileNotFoundError("Could not locate the local WRN PatchCore experiment root.")


EXPERIMENT_ROOT = resolve_experiment_root()
BUNDLE_DIR = EXPERIMENT_ROOT / "artifacts" / "patchcore_wrn50_multilayer_120k_5pct"
VARIANT_NAME = 'topk_mb50k_r005'

MIN_RECALL = 0.70
MAX_FALSE_POSITIVE_RATE = 0.03
MAX_AUTO_NORMAL_ANOMALY_RATE = 0.01
MIN_AUTO_ANOMALY_PRECISION = 0.60
""",
    )
    clear_outputs(nb)
    save_notebook(notebook_paths[3], nb)


def patch_ensemble() -> None:
    path = REPO_ROOT / "experiments/anomaly_detection/ensemble/x64/score_ensemble/notebook.ipynb"
    nb = load_notebook(path)

    set_cell_source(
        nb,
        3,
        """CONFIG = {
    "data": {
        "metadata_csv": REPO_ROOT / "data" / "processed" / "x64" / "wm811k" / "metadata_50k_5pct.csv",
        "image_size": 64,
        "batch_size": 16,
        "num_workers": 0,
    },
    "patchcore": {
        "name": "PatchCore-WideRes50-topk-mb50k-r010",
    },
    "ts_res50": {
        "name": "TS-Res50-s2_a1_topk_mean_r0.20",
        "artifact_dir": REPO_ROOT / "experiments/anomaly_detection/teacher_student/resnet50/x64/main/artifacts/ts_resnet50",
        "config_path": REPO_ROOT / "experiments/anomaly_detection/teacher_student/resnet50/x64/main/train_config.toml",
        "selected_variant_json": REPO_ROOT
        / "experiments/anomaly_detection/teacher_student/resnet50/x64/main/artifacts/ts_resnet50/results/evaluation/selected_score_variant.json",
        "converted_checkpoint_path": REPO_ROOT
        / "experiments/anomaly_detection/teacher_student/resnet50/x64/main/artifacts/ts_resnet50/checkpoints/best_model.pt",
        "raw_checkpoint_path": REPO_ROOT
        / "experiments/anomaly_detection/teacher_student/resnet50/x64/main/artifacts/ts_resnet50/checkpoints/best_model.pt",
    },
}


def resolve_patchcore_score_paths(repo_root: Path) -> tuple[Path, Path]:
    preferred_root = (
        repo_root
        / "experiments/anomaly_detection/patchcore/wideresnet50/x64/main/artifacts/patchcore_wideresnet50_multilayer/topk_mb50k_r010/results/evaluation"
    )
    legacy_candidates = sorted((repo_root / "artifacts/x64").glob("*/topk_mb50k_r010"))

    for candidate in [preferred_root, *legacy_candidates]:
        val_scores = candidate / "val_scores.csv"
        test_scores = candidate / "test_scores.csv"
        if val_scores.exists() and test_scores.exists():
            return val_scores, test_scores

    raise FileNotFoundError(
        "Could not locate local PatchCore score files. Run the WRN x64 PatchCore notebook or restore the saved local export."
    )


CONFIG["patchcore"]["val_scores_csv"], CONFIG["patchcore"]["test_scores_csv"] = resolve_patchcore_score_paths(REPO_ROOT)
""",
    )

    replace_in_cells(nb, "def remap_kaggle_ts_checkpoint(kaggle_checkpoint_path: Path, config_path: Path, image_size: int):", "def load_ts_checkpoint_for_local_eval(checkpoint_path: Path, config_path: Path, image_size: int):")
    replace_in_cells(nb, "checkpoint = torch.load(kaggle_checkpoint_path, map_location=\"cpu\")", "checkpoint = torch.load(checkpoint_path, map_location=\"cpu\")")
    replace_in_cells(nb, "converted_checkpoint, ts_model = remap_kaggle_ts_checkpoint(", "converted_checkpoint, ts_model = load_ts_checkpoint_for_local_eval(")
    clear_outputs(nb)
    save_notebook(path, nb)


def patch_teacher_student_note() -> None:
    path = REPO_ROOT / "experiments/anomaly_detection/teacher_student/wideresnet50_2/x64/layer2_self_contained/notebook.ipynb"
    nb = load_notebook(path)
    replace_in_cells(nb, "# same protocol as Kaggle notebook", "# same protocol as the local training notebook")
    clear_outputs(nb)
    save_notebook(path, nb)


def main() -> None:
    patch_wrn_x64_main()
    patch_wrn_labeled_120k()
    patch_ensemble()
    patch_teacher_student_note()


if __name__ == "__main__":
    main()
