"""YAML configuration loading. Every method value lives in `configs/` (CLAUDE.md §7)."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: config must be a YAML mapping, got {type(data).__name__}")
    return data
