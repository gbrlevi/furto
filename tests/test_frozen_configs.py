"""The frozen configurations of session 3 (D.9) are consistent with PREDECLARED.md."""

import hashlib
import json
from pathlib import Path

import pytest

from tcc.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "configs" / "frozen"
NAMES = ["main", "nan_zero", "t12", "t48", "gate_off"]
SHARED = ["operating_points", "alerts", "threshold_candidates", "offline", "events",
          "bootstrap", "device", "batch_size"]  # fmt: skip


def _load(name: str) -> dict:  # type: ignore[type-arg]
    return load_config(FROZEN / f"{name}.yaml")


def test_predeclared_rules_and_the_five_configurations_exist() -> None:
    assert (FROZEN / "PREDECLARED.md").exists()
    assert sorted(p.stem for p in FROZEN.glob("*.yaml")) == sorted(NAMES)


def test_every_configuration_shares_the_predeclared_evaluation_rules() -> None:
    main = _load("main")
    assert main["operating_points"] == {"main": 5, "secondary": [2, 10]}
    assert main["offline"] == {"frame_score": "center", "sigmas": [1, 2, 3, 4, 5, 6]}
    assert main["bootstrap"] == {"n_resamples": 2000, "seed": 0, "level": 0.95}
    assert main["occlusion"]["fractions"] == [0.1, 0.2, 0.3, 0.4, 0.5]
    for name in NAMES:
        cfg = _load(name)
        assert cfg["name"] == name
        assert {k: cfg[k] for k in SHARED} == {k: main[k] for k in SHARED}
        assert ("occlusion" in cfg) == (name == "main")


def _runs_available() -> bool:
    return all((ROOT / m["run"] / "model.pt").exists() for n in NAMES for m in _load(n)["models"])


@pytest.mark.skipif(not _runs_available(), reason="training runs not available")
def test_frozen_checkpoints_match_their_hashes_and_training_configurations() -> None:
    expected = {
        "main": ("c1_base", [0, 1, 2, 3, 4]),
        "nan_zero": ("c1_nan_zero", [0]),
        "t12": ("c1_t12", [0]),
        "t48": ("c1_t48", [0]),
        "gate_off": ("c1_gate_off", [0]),
    }
    for name, (base, seeds) in expected.items():
        models = _load(name)["models"]
        found = []
        for m in models:
            run = ROOT / m["run"]
            assert hashlib.sha256((run / "model.pt").read_bytes()).hexdigest() == m["sha256"]
            meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
            assert meta["git"]["dirty"] is False
            found.append(meta["config"]["train"]["seed"])
            stem = meta["argv"][meta["argv"].index("--config") + 1]
            assert Path(stem).stem in ({base} | {f"c1_seed{s}" for s in seeds if s})
        assert found == seeds
