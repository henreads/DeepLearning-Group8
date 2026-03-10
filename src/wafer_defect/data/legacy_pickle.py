from __future__ import annotations

import pickle
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import pandas.core.indexes as core_indexes


def read_legacy_pickle(path: str | Path) -> pd.DataFrame:
    sys.modules["pandas.indexes"] = core_indexes
    with Path(path).open("rb") as handle:
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

