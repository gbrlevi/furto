"""Layout from CLAUDE.md §3 and the ban on importing legacy code (ADR-0006)."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ["data", "preprocess", "model", "eval", "pipeline", "utils"]


def test_layout_directories_exist() -> None:
    for name in ["configs", "scripts", "tests", "src/tcc"]:
        assert (ROOT / name).is_dir(), name
    for pkg in PACKAGES:
        assert (ROOT / "src" / "tcc" / pkg / "__init__.py").is_file(), pkg


def test_no_code_imports_legacy() -> None:
    pattern = re.compile(r"^\s*(from|import)\s+legacy\b", re.MULTILINE)
    offenders = [
        path
        for folder in ["src", "scripts", "tests"]
        for path in (ROOT / folder).rglob("*.py")
        if pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_preprocess_and_eval_do_not_import_pipeline_or_model() -> None:
    pattern = re.compile(r"^\s*(from|import)\s+tcc\.(pipeline|model)\b", re.MULTILINE)
    offenders = [
        path
        for pkg in ["preprocess", "eval"]
        for path in (ROOT / "src" / "tcc" / pkg).rglob("*.py")
        if pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
