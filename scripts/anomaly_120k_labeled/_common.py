"""Shared helpers for the 120k labeled anomaly scripts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_HELPERS_ROOT = PROJECT_ROOT / "notebooks" / "anomaly_120k_labeled" / "helpers"


def load_helper_module(filename: str, module_name: str) -> ModuleType:
    module_path = NOTEBOOK_HELPERS_ROOT / filename
    if not module_path.exists():
        raise FileNotFoundError(f"Could not find helper module: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
