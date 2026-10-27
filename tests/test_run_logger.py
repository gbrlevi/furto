"""One test per field required by rule 4 of CLAUDE.md, plus `dirty` (rule 5)."""

import hashlib
import json
import platform
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tcc.utils.run_logger import record_artifact, start_run, write_meta


def _git(repo: Path, *args: str) -> str:
    cmd = ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args]
    return subprocess.run(cmd, cwd=repo, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    path = tmp_path / "repo"
    path.mkdir()
    _git(path, "init", "-q")
    (path / "a.txt").write_text("a\n", encoding="utf-8")
    _git(path, "add", "a.txt")
    _git(path, "commit", "-q", "-m", "init")
    return path


def _meta(run_dir: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    return data


def _run(repo: Path, tmp_path: Path, **kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("config", {"T": 24})
    kwargs.setdefault("seeds", {"global": 0})
    run_dir = start_run("probe", runs_root=tmp_path / "runs", repo=repo, **kwargs)
    return _meta(run_dir)


def test_run_dir_is_named_by_timestamp_and_name(repo: Path, tmp_path: Path) -> None:
    run_dir = start_run("probe", config={}, seeds={}, runs_root=tmp_path / "runs", repo=repo)
    assert run_dir.parent == tmp_path / "runs"
    assert re.fullmatch(r"\d{8}-\d{6}_probe", run_dir.name)
    assert (run_dir / "meta.json").is_file()


def test_records_seeds(repo: Path, tmp_path: Path) -> None:
    meta = _run(repo, tmp_path, seeds={"global": 3, "split": 7})
    assert meta["seeds"] == {"global": 3, "split": 7}


def test_records_python_torch_cuda_and_library_versions(repo: Path, tmp_path: Path) -> None:
    import numpy
    import torch

    versions = _run(repo, tmp_path)["versions"]
    assert versions["python"] == platform.python_version()
    assert versions["torch"] == torch.__version__
    assert versions["cuda"] == torch.version.cuda
    assert "cudnn" in versions
    assert versions["libraries"]["numpy"] == numpy.__version__


def test_records_model_versions_and_weight_hashes(repo: Path, tmp_path: Path) -> None:
    weights = tmp_path / "pose.pth"
    weights.write_bytes(b"fake weights")
    models = {"pose": {"name": "hrnet_w32", "version": "mmpose 1.3", "weights": str(weights)}}
    recorded = _run(repo, tmp_path, models=models)["models"]
    assert recorded["pose"]["name"] == "hrnet_w32"
    assert recorded["pose"]["version"] == "mmpose 1.3"
    assert recorded["pose"]["weights"] == str(weights)
    assert recorded["pose"]["weights_sha256"] == hashlib.sha256(b"fake weights").hexdigest()


def test_models_field_is_present_when_no_model_is_used(repo: Path, tmp_path: Path) -> None:
    assert _run(repo, tmp_path)["models"] == {}


def test_missing_weights_file_fails(repo: Path, tmp_path: Path) -> None:
    models = {"pose": {"name": "x", "version": "1", "weights": str(tmp_path / "missing.pth")}}
    with pytest.raises(FileNotFoundError):
        _run(repo, tmp_path, models=models)


def test_records_hardware(repo: Path, tmp_path: Path) -> None:
    hardware = _run(repo, tmp_path)["hardware"]
    assert hardware["platform"] == platform.platform()
    assert hardware["cpu"]
    assert hardware["cpu_count"] >= 1
    assert hardware["ram_bytes"] > 0
    assert isinstance(hardware["gpus"], list)


def test_records_git_commit(repo: Path, tmp_path: Path) -> None:
    head = _git(repo, "rev-parse", "HEAD").strip()
    assert _run(repo, tmp_path)["git"]["commit"] == head


def test_clean_tree_is_not_dirty(repo: Path, tmp_path: Path) -> None:
    assert _run(repo, tmp_path)["git"]["dirty"] is False


def test_modified_file_makes_tree_dirty(repo: Path, tmp_path: Path) -> None:
    (repo / "a.txt").write_text("changed\n", encoding="utf-8")
    git = _run(repo, tmp_path)["git"]
    assert git["dirty"] is True
    assert any("a.txt" in line for line in git["status"])


def test_untracked_file_makes_tree_dirty(repo: Path, tmp_path: Path) -> None:
    (repo / "new.py").write_text("x = 1\n", encoding="utf-8")
    assert _run(repo, tmp_path)["git"]["dirty"] is True


def test_outside_git_repo_counts_as_dirty(tmp_path: Path) -> None:
    not_repo = tmp_path / "plain"
    not_repo.mkdir()
    meta = _run(not_repo, tmp_path)
    assert meta["git"]["commit"] is None
    assert meta["git"]["dirty"] is True


def test_write_meta_uses_the_given_directory(repo: Path, tmp_path: Path) -> None:
    target = tmp_path / "reference"
    target.mkdir()
    write_meta(
        target, "ref", {"cmd": ["python", "x.py"]}, {"global": 0}, repo=repo, filename="m.json"
    )
    meta = json.loads((target / "m.json").read_text(encoding="utf-8"))
    assert meta["name"] == "ref"
    assert meta["config"] == {"cmd": ["python", "x.py"]}
    assert meta["git"]["dirty"] is False


def test_records_full_config(repo: Path, tmp_path: Path) -> None:
    config = {"T": 24, "S": 6, "paths": {"data": "data/raw/poselift"}, "k": [1, 2]}
    assert _run(repo, tmp_path, config=config)["config"] == config


def test_record_artifact_adds_the_file_sha256_to_meta(repo: Path, tmp_path: Path) -> None:
    """Session 3, P17 item 18: the checkpoint hash goes to the run's meta.json."""
    run_dir = start_run("probe", config={}, seeds={}, runs_root=tmp_path / "runs", repo=repo)
    ckpt = run_dir / "model.pt"
    ckpt.write_bytes(b"weights")
    record_artifact(run_dir, "model.pt", ckpt)
    meta = _meta(run_dir)
    assert meta["artifacts"]["model.pt"] == hashlib.sha256(b"weights").hexdigest()
    assert meta["config"] == {}
