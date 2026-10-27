"""Test gate (ADR-0009): the test set is opened only with a committed frozen configuration.

`check_frozen` refuses unless the configuration lives in `configs/frozen/`, it and the
`PREDECLARED.md` next to it are tracked by git, and the whole tree is clean (rule 5).
`log_test_opening` appends one row per opening to `runs/test_evaluations.csv`; it is
called before the test is read, so an opening that crashes is still counted.
"""

import csv
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

FROZEN_DIR = Path("configs") / "frozen"
PREDECLARED = "PREDECLARED.md"
LOG_FIELDS = (
    "opened_at", "run_dir", "config", "config_sha256", "config_blob", "predeclared_sha256",
    "commit", "opening_of_this_config", "opening_overall",
)  # fmt: skip


class FrozenConfigError(RuntimeError):
    pass


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_frozen(config: Path, repo: Path = Path(".")) -> dict[str, str]:
    """Raise FrozenConfigError unless `config` may open the test; else return its hashes
    (file bytes and git blob, which does not depend on line endings) and the commit."""
    rel = config.resolve().relative_to(repo.resolve())
    if rel.parent != FROZEN_DIR:
        raise FrozenConfigError(f"{rel}: --final needs a configuration in {FROZEN_DIR}/")
    predeclared = FROZEN_DIR / PREDECLARED
    for path in (rel, predeclared):
        if _git(repo, "ls-files", "--error-unmatch", str(path)).returncode != 0:
            raise FrozenConfigError(f"{path} is not committed")
    status = _git(repo, "status", "--porcelain")
    if status.returncode != 0 or status.stdout.strip():
        raise FrozenConfigError("the tree has uncommitted changes (rule 5)")
    return {
        "config": rel.as_posix(),
        "config_sha256": sha256(repo / rel),  # bytes on disk
        "config_blob": _git(repo, "rev-parse", f"HEAD:{rel.as_posix()}").stdout.strip(),
        "predeclared_sha256": sha256(repo / predeclared),
        "commit": _git(repo, "rev-parse", "HEAD").stdout.strip(),
    }


def log_test_opening(log: Path, run_dir: Path, frozen: dict[str, str]) -> dict[str, Any]:
    """Append the opening to the CSV log and return the row written."""
    rows: list[dict[str, str]] = []
    if log.exists():
        with log.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    row: dict[str, Any] = {
        "opened_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_dir": run_dir.as_posix(),
        **frozen,
        "opening_of_this_config": 1 + sum(r["config"] == frozen["config"] for r in rows),
        "opening_overall": 1 + len(rows),
    }
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if not rows:
            writer.writeheader()
        writer.writerow(row)
    return row
