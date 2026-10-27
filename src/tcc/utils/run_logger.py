"""Automatic run record: `runs/<YYYYMMDD-HHMMSS>_<name>/meta.json` (CLAUDE.md, rules 4 and 5)."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil
import torch


def start_run(
    name: str,
    config: Mapping[str, Any],
    seeds: Mapping[str, int],
    models: Mapping[str, Mapping[str, str]] | None = None,
    runs_root: Path = Path("runs"),
    repo: Path = Path("."),
) -> Path:
    """Create the run directory, write `meta.json` and return the directory.

    `models` maps a role (detector, tracker, pose) to `name`, `version` and,
    optionally, `weights`, a file path whose SHA-256 is recorded.
    """
    now = datetime.now().astimezone()
    run_dir = runs_root / f"{now:%Y%m%d-%H%M%S}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    write_meta(run_dir, name, config, seeds, models, repo)
    return run_dir


def write_meta(
    run_dir: Path,
    name: str,
    config: Mapping[str, Any],
    seeds: Mapping[str, int],
    models: Mapping[str, Mapping[str, str]] | None = None,
    repo: Path = Path("."),
    filename: str = "meta.json",
) -> None:
    """Write the run record into an existing directory chosen by the caller."""
    now = datetime.now().astimezone()
    meta = {
        "name": name,
        "created_at": now.isoformat(timespec="seconds"),
        "argv": sys.argv,
        "seeds": dict(seeds),
        "versions": _versions(),
        "models": {role: _model(info) for role, info in (models or {}).items()},
        "hardware": _hardware(),
        "git": _git_state(repo),
        "config": config,
    }
    (run_dir / filename).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def record_artifact(run_dir: Path, name: str, path: Path) -> None:
    """Add the SHA-256 of a file the run produced (e.g. its checkpoint) to its meta.json."""
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.setdefault("artifacts", {})[name] = _sha256(path)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def _versions() -> dict[str, Any]:
    libraries = {
        dist.metadata["Name"].lower(): dist.version for dist in importlib.metadata.distributions()
    }
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": _cudnn_version(),
        "libraries": dict(sorted(libraries.items())),
    }


def _cudnn_version() -> int | None:
    # torch.backends.cudnn has no type annotations.
    if not torch.backends.cudnn.is_available():  # type: ignore[no-untyped-call]
        return None
    version: int | None = torch.backends.cudnn.version()  # type: ignore[no-untyped-call]
    return version


def _model(info: Mapping[str, str]) -> dict[str, str]:
    record = dict(info)
    if "weights" in info:
        record["weights_sha256"] = _sha256(Path(info["weights"]))
    return record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hardware() -> dict[str, Any]:
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append(
                {
                    "name": props.name,
                    "total_memory_bytes": props.total_memory,
                    "capability": f"{props.major}.{props.minor}",
                }
            )
    return {
        "platform": platform.platform(),
        "cpu": _cpu_name(),
        "cpu_count": os.cpu_count(),
        "ram_bytes": psutil.virtual_memory().total,
        "gpus": gpus,
        "nvidia_driver": _nvidia_driver(),
    }


def _cpu_name() -> str:
    if sys.platform == "win32":
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        )
        return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _nvidia_driver() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.strip().splitlines()[0] if out.strip() else None


def _git_state(repo: Path) -> dict[str, Any]:
    """Commit and dirty flag. Untracked files count as dirty; no repository counts as dirty."""
    try:
        commit = _run_git(repo, "rev-parse", "HEAD").strip()
        status = _run_git(repo, "status", "--porcelain").splitlines()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": True, "status": []}
    return {"commit": commit, "dirty": bool(status), "status": status}


def _run_git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout
