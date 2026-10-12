from pathlib import Path

import pytest

from tcc.utils.config import load_config


def test_load_config_returns_mapping(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("seed: 0\nwindow:\n  T: 24\n", encoding="utf-8")
    assert load_config(path) == {"seed": 0, "window": {"T": 24}}


def test_load_config_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)
