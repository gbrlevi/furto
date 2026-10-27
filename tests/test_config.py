from pathlib import Path
from typing import Any

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


def _base() -> dict[str, Any]:
    return load_config(ROOT / "configs" / "c1_base.yaml")


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_seed_configs_differ_from_c1_base_only_in_the_training_seed(seed: int) -> None:
    cfg = load_config(ROOT / "configs" / f"c1_seed{seed}.yaml")
    assert cfg["train"]["seed"] == seed and cfg["split"]["seed"] == 0
    cfg["train"]["seed"] = 0
    assert cfg == _base()


@pytest.mark.parametrize("t", [12, 48])
def test_window_sensitivities_differ_from_c1_base_only_in_t(t: int) -> None:
    cfg = load_config(ROOT / "configs" / f"c1_t{t}.yaml")
    assert cfg["treatment"]["T"] == t
    cfg["treatment"]["T"] = 24
    assert cfg == _base()


def test_nan_zero_sensitivity_differs_only_in_the_nan_confidence_of_the_dataset() -> None:
    cfg = load_config(ROOT / "configs" / "c1_nan_zero.yaml")
    dataset = load_config(ROOT / cfg["dataset_config"])
    assert dataset["nan_confidence"] == "zero"
    dataset["nan_confidence"] = "interpolate"
    assert dataset == load_config(ROOT / "configs" / "poselift.yaml")
    cfg["dataset_config"] = "configs/poselift.yaml"
    assert cfg == _base()
