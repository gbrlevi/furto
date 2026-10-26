from pathlib import Path

import pytest

from tcc.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_load_config_returns_mapping(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("seed: 0\nwindow:\n  T: 24\n", encoding="utf-8")
    assert load_config(path) == {"seed": 0, "window": {"T": 24}}


def test_load_config_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)


def test_gate_off_variant_differs_from_c1_base_only_in_the_quality_gate() -> None:
    base = load_config(ROOT / "configs" / "c1_base.yaml")
    off = load_config(ROOT / "configs" / "c1_gate_off.yaml")
    assert base["treatment"]["quality_gate"] == "on" and off["treatment"]["quality_gate"] == "off"
    off["treatment"]["quality_gate"] = "on"
    assert off == base
