"""Test gate of ADR-0009 (session 3, task D.8)."""

import csv
import hashlib
import subprocess
from pathlib import Path

import pytest

from tcc.utils.frozen import FrozenConfigError, check_frozen, log_test_opening


def _git(repo: Path, *args: str) -> None:
    cmd = ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args]
    subprocess.run(cmd, cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    path = tmp_path / "repo"
    (path / "configs" / "frozen").mkdir(parents=True)
    (path / "configs" / "frozen" / "main.yaml").write_bytes(b"name: main\n")
    (path / "configs" / "frozen" / "PREDECLARED.md").write_bytes(b"# rules\n")
    (path / "configs" / "free.yaml").write_text("name: free\n", encoding="utf-8")
    _git(path, "init", "-q")
    _git(path, "add", ".")
    _git(path, "commit", "-q", "-m", "init")
    return path


def test_committed_frozen_config_on_a_clean_tree_opens_the_test(repo: Path) -> None:
    out = check_frozen(repo / "configs" / "frozen" / "main.yaml", repo)
    assert out["config"] == "configs/frozen/main.yaml"
    assert out["config_sha256"] == hashlib.sha256(b"name: main\n").hexdigest()
    assert out["predeclared_sha256"] == hashlib.sha256(b"# rules\n").hexdigest()
    assert len(out["commit"]) == 40 and len(out["config_blob"]) == 40


def test_config_outside_the_frozen_folder_is_refused(repo: Path) -> None:
    with pytest.raises(FrozenConfigError, match="frozen"):
        check_frozen(repo / "configs" / "free.yaml", repo)


def test_uncommitted_frozen_config_is_refused(repo: Path) -> None:
    (repo / "configs" / "frozen" / "new.yaml").write_text("name: new\n", encoding="utf-8")
    with pytest.raises(FrozenConfigError, match="not committed"):
        check_frozen(repo / "configs" / "frozen" / "new.yaml", repo)


def test_edited_frozen_config_is_refused(repo: Path) -> None:
    (repo / "configs" / "frozen" / "main.yaml").write_text("name: changed\n", encoding="utf-8")
    with pytest.raises(FrozenConfigError, match="uncommitted"):
        check_frozen(repo / "configs" / "frozen" / "main.yaml", repo)


def test_missing_predeclared_rules_are_refused(repo: Path) -> None:
    _git(repo, "rm", "-q", "configs/frozen/PREDECLARED.md")
    _git(repo, "commit", "-q", "-m", "drop")
    with pytest.raises(FrozenConfigError, match="PREDECLARED"):
        check_frozen(repo / "configs" / "frozen" / "main.yaml", repo)


def test_any_dirty_file_refuses_the_test(repo: Path) -> None:
    (repo / "notes.txt").write_text("x\n", encoding="utf-8")
    with pytest.raises(FrozenConfigError, match="uncommitted"):
        check_frozen(repo / "configs" / "frozen" / "main.yaml", repo)


def test_every_opening_is_logged_and_numbered(repo: Path, tmp_path: Path) -> None:
    log = tmp_path / "runs" / "test_evaluations.csv"
    frozen = check_frozen(repo / "configs" / "frozen" / "main.yaml", repo)
    first = log_test_opening(log, Path("runs/a"), frozen)
    other = log_test_opening(log, Path("runs/b"), frozen | {"config": "configs/frozen/t12.yaml"})
    again = log_test_opening(log, Path("runs/c"), frozen)
    assert (first["opening_of_this_config"], first["opening_overall"]) == (1, 1)
    assert (other["opening_of_this_config"], other["opening_overall"]) == (1, 2)
    assert (again["opening_of_this_config"], again["opening_overall"]) == (2, 3)
    with log.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["run_dir"] for r in rows] == ["runs/a", "runs/b", "runs/c"]
    assert rows[0]["config_sha256"] == frozen["config_sha256"]
