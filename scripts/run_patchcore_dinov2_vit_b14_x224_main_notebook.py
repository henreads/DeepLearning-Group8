from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any


NOTEBOOK_PATH = Path(
    "experiments/anomaly_detection/patchcore/dinov2_vit_b14/x224/notebook.ipynb"
)


def _patch_cell_source(
    cell_id: str,
    source: str,
    *,
    metadata_csv: Path | None,
    output_dir: Path,
    force_rerun: bool,
    num_workers: int,
    batch_size: int,
    vit_feature_block: int,
) -> str:
    if cell_id == "cell-06-config" and metadata_csv is not None:
        source = source.replace(
            "cwd = Path.cwd().resolve()\n"
            "PROJECT_ROOT = None\n"
            "for candidate in [cwd, *cwd.parents]:\n"
            "    if (candidate / 'src' / 'wafer_defect').exists() and (candidate / 'configs').exists():\n"
            "        PROJECT_ROOT = candidate\n"
            "        break\n"
            "if PROJECT_ROOT is None:\n"
            "    raise RuntimeError('Could not locate repo root containing src/wafer_defect and configs/')\n",
            (
                "# Modal runner executes from the repo root already.\n"
                "PROJECT_ROOT = Path.cwd().resolve()\n"
            ),
        )
        source = source.replace(
            "DATA_PATH  = str(PROJECT_ROOT / 'data' / 'raw' / 'LSWMD.pkl')\n",
            (
                "DATA_PATH  = str(PROJECT_ROOT / 'data' / 'raw' / 'LSWMD.pkl')\n"
                f"METADATA_CSV = r'{metadata_csv.as_posix()}'\n"
            ),
        )
        source = source.replace(
            "VIT_FEATURE_BLOCK = 9      # sweep candidate: try 6, 9, 11\n",
            f"VIT_FEATURE_BLOCK = {vit_feature_block}      # overridden by runner\n",
        )
        source = source.replace(
            "ARTIFACT_DIR    = str(PROJECT_ROOT / 'experiments/anomaly_detection/patchcore/dinov2_vit_b14/x224/main/artifacts')\n",
            f"ARTIFACT_DIR    = r'{output_dir.as_posix()}'\n",
        )
        source = source.replace(
            "ARTIFACT_DIR    = str(PROJECT_ROOT / 'experiments/anomaly_detection/patchcore/dinov2_vit_b14/x224/block9/artifacts')\n",
            f"ARTIFACT_DIR    = r'{output_dir.as_posix()}'\n",
        )
    if cell_id == "cell-08-load" and metadata_csv is not None:
        return f"""# -- 3. Load processed metadata ------------------------------------------------
df = pd.read_csv(r"{metadata_csv.as_posix()}").copy()
df["failure_label"] = df.get("defect_type", "unlabeled").astype(str).str.strip()
print("Metadata shape:", df.shape)

normal_df = df[df["is_anomaly"] == 0].copy()
defect_df = df[df["is_anomaly"] == 1].copy()

print(f"Labeled: {{len(df):,}}   Normal: {{len(normal_df):,}}   Defect: {{len(defect_df):,}}")
print("\\nDefect breakdown:")
print(defect_df["failure_label"].value_counts())
"""
    if cell_id == "cell-10-split" and metadata_csv is not None:
        return """# -- 4. Split from cached processed metadata -----------------------------------
train_normal_df = (
    normal_df[normal_df["split"] == "train"]
    .head(TRAIN_NORMAL_N)
    .reset_index(drop=True)
    .copy()
)
tune_normal_df = (
    normal_df[normal_df["split"] == "val"]
    .head(TUNE_NORMAL_N)
    .reset_index(drop=True)
    .copy()
)
test_normal_df = (
    normal_df[normal_df["split"] == "test"]
    .head(TEST_NORMAL_N)
    .reset_index(drop=True)
    .copy()
)
test_defect_df = (
    defect_df[defect_df["split"] == "test"]
    .head(TEST_DEFECT_N)
    .reset_index(drop=True)
    .copy()
)

del df, normal_df, defect_df
gc.collect()

print(f"Train normal : {len(train_normal_df):>7,}  (memory bank)")
print(f"Tune  normal : {len(tune_normal_df):>7,}  (threshold calibration)")
print(f"Test  normal : {len(test_normal_df):>7,}")
print(f"Test  defect : {len(test_defect_df):>7,}")
print("\\nDefect classes in test set:")
print(test_defect_df["failure_label"].value_counts())
"""
    if cell_id == "cell-12-dataset" and metadata_csv is not None:
        return """# -- 5. Cached-array dataset ---------------------------------------------------
class WaferDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True).copy()

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        array_path = (PROJECT_ROOT / str(row["array_path"])).resolve()
        wafer_map = np.load(array_path).astype(np.float32)
        raw = np.clip(np.rint(wafer_map * 2.0), 0, 2).astype(np.int64)
        x = torch.tensor(raw, dtype=torch.long)
        x = F.one_hot(x, num_classes=3).permute(2, 0, 1).float()
        return x, int(row["is_anomaly"])


loader_kw = dict(
    batch_size  = BATCH_SIZE,
    shuffle     = False,
    num_workers = NUM_WORKERS,
    pin_memory  = USE_CUDA,
    persistent_workers = (NUM_WORKERS > 0),
)

train_loader       = DataLoader(WaferDataset(train_normal_df), **loader_kw)
tune_normal_loader = DataLoader(WaferDataset(tune_normal_df),  **loader_kw)
test_normal_loader = DataLoader(WaferDataset(test_normal_df),  **loader_kw)
test_defect_loader = DataLoader(WaferDataset(test_defect_df),  **loader_kw)

# Smoke-test
xb, yb = next(iter(train_loader))
print(f"Batch shape: {tuple(xb.shape)}  dtype={xb.dtype}")
"""
    if cell_id == "cell-14-model":
        source = source.replace(
            "        self.vit = timm.create_model(\n"
            "            BACKBONE_NAME,\n"
            "            pretrained=True,\n"
            "            num_classes=0,\n"
            "        )\n",
            (
                "        self.vit = timm.create_model(\n"
                "            BACKBONE_NAME,\n"
                "            pretrained=True,\n"
                "            num_classes=0,\n"
                "            img_size=IMAGE_SIZE,\n"
                "        )\n"
            ),
        )
    replacements = {
        "FORCE_REBUILD_SCORES = False": f"FORCE_REBUILD_SCORES = {str(force_rerun)}",
        "FORCE_RERUN_UMAP     = False": f"FORCE_RERUN_UMAP     = {str(force_rerun)}",
        "BATCH_SIZE       = 128": f"BATCH_SIZE       = {batch_size}",
        "NUM_WORKERS      = 0": f"NUM_WORKERS      = {num_workers}",
    }
    for old, new in replacements.items():
        source = source.replace(old, new)
    return source


def execute_notebook(
    *,
    notebook_path: Path,
    metadata_csv: Path | None,
    output_dir: Path,
    force_rerun: bool,
    num_workers: int,
    batch_size: int,
    vit_feature_block: int,
) -> dict[str, Any]:
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    globals_dict: dict[str, Any] = {"__name__": "__main__"}

    code_cells = [cell for cell in nb["cells"] if cell.get("cell_type") == "code"]
    total_cells = len(code_cells)
    for index, cell in enumerate(code_cells, start=1):
        cell_id = cell.get("id", f"cell_{index}")
        code = "".join(cell.get("source", []))
        code = _patch_cell_source(
            cell_id,
            code,
            metadata_csv=metadata_csv,
            output_dir=output_dir,
            force_rerun=force_rerun,
            num_workers=num_workers,
            batch_size=batch_size,
            vit_feature_block=vit_feature_block,
        )
        print(
            f"[patchcore-dinov2-vit-b14-x224-main] starting code cell {index}/{total_cells} ({cell_id})",
            flush=True,
        )
        exec(compile(code, f"{notebook_path.name}::{cell_id}", "exec"), globals_dict)
        print(
            f"[patchcore-dinov2-vit-b14-x224-main] finished code cell {index}/{total_cells}",
            flush=True,
        )

    summary_export_path = Path(str(globals_dict["SUMMARY_EXPORT_PATH"]))
    metrics_export_path = Path(str(globals_dict["METRICS_EXPORT_PATH"]))
    model_export_path = Path(str(globals_dict["MODEL_EXPORT_PATH"]))
    umap_png_path = Path(str(globals_dict["UMAP_PNG_PATH"]))

    summary: dict[str, Any] = {}
    if summary_export_path.exists():
        summary = json.loads(summary_export_path.read_text(encoding="utf-8"))

    manifest = {
        "status": "complete",
        "output_dir": str(Path(str(globals_dict["ARTIFACT_DIR"]))),
        "checkpoint_path": str(model_export_path),
        "summary_path": str(summary_export_path),
        "metrics_path": str(metrics_export_path),
        "umap_png_path": str(umap_png_path),
        "metrics": summary.get("metrics", {}),
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-pickle", required=True)
    parser.add_argument("--metadata-csv", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--vit-feature-block", type=int, default=9)
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    raw_pickle_path = Path(args.raw_pickle).resolve()
    if not raw_pickle_path.exists():
        raise FileNotFoundError(f"Raw pickle not found: {raw_pickle_path}")

    metadata_csv_path: Path | None = None
    if args.metadata_csv:
        metadata_csv_path = Path(args.metadata_csv).resolve()
        if not metadata_csv_path.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv_path}")

    expected_raw_pickle = Path("data/raw/LSWMD.pkl").resolve()
    expected_raw_pickle.parent.mkdir(parents=True, exist_ok=True)
    if raw_pickle_path != expected_raw_pickle:
        shutil.copy2(raw_pickle_path, expected_raw_pickle)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = execute_notebook(
        notebook_path=NOTEBOOK_PATH,
        metadata_csv=metadata_csv_path,
        output_dir=output_dir,
        force_rerun=args.force_rerun,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        vit_feature_block=args.vit_feature_block,
    )
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
