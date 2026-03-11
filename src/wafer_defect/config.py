"""Small helpers for loading TOML-based experiment configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import tomllib


def load_toml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)
