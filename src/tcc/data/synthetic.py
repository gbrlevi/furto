"""Pre-registered synthetic probe (session 4): prompts, manifest and annotations.

The rules are in `configs/synthetic_probe/PROTOCOLO.md`. This module only checks the group's
data and never corrects it: anything out of protocol raises SyntheticDataError.
"""

import csv
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

Row = dict[str, str]

VARIANTS = ("concealment", "normal")
# Technical failures of section 2 of the protocol, the only reasons for a new attempt
RETRY_REASONS = ("corrompido", "duracao_menor_8s", "mais_de_uma_pessoa", "recusa")
MAX_ATTEMPTS = 2
GENERATED, NOT_GENERATED = "gerado", "nao_gerado"
MANIFEST_COLUMNS = (
    "id", "status", "generated_at", "generator", "model_version", "settings", "attempts",
    "retry_reasons",
)  # fmt: skip
ANNOTATION_COLUMNS = (
    "id", "action_appeared", "concealment_start_s", "concealment_end_s", "artifacts",
)  # fmt: skip
ACTION_VALUES = ("sim", "nao", "ambiguo")
ARTIFACT_VALUES = ("nenhum", "leves", "graves")
NO_INTERVAL = "nenhum"


class SyntheticDataError(RuntimeError):
    pass


def final_prompts(cfg: Mapping[str, Any]) -> dict[str, str]:  # Any: YAML config
    """{id: prompt} in the declared order; prompt = style + person + variant, joined by
    single spaces."""
    by_id = {}
    for pair in cfg["pairs"]:
        for variant in VARIANTS:
            text = f"{cfg['style']} The person is {pair['person']}. {pair[variant]}"
            by_id[f"{pair['id']}_{variant}"] = text
    return {i: by_id[i] for i in cfg["order"]}


def _read(path: Path, columns: Sequence[str]) -> list[Row]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != tuple(columns):
            raise SyntheticDataError(f"{path.name}: columns must be {','.join(columns)}")
        return [{k: (v or "").strip() for k, v in row.items()} for row in reader]


def read_manifest(path: Path) -> list[Row]:
    return _read(path, MANIFEST_COLUMNS)


def read_annotations(path: Path) -> list[Row]:
    return _read(path, ANNOTATION_COLUMNS)


def _check_ids(rows: Sequence[Row], ids: Sequence[str], name: str) -> None:
    found = [r["id"] for r in rows]
    if sorted(found) != sorted(ids):
        raise SyntheticDataError(f"{name}: ids must be exactly {list(ids)}, got {found}")


def validate_manifest(rows: Sequence[Row], ids: Sequence[str], frozen_at: datetime) -> None:
    """Exactly the declared ids; generated after the freeze; new attempts only for a listed
    technical failure; a video not generated stays listed with its failure."""
    _check_ids(rows, ids, "manifest")
    for r in rows:
        vid = r["id"]
        if r["status"] not in (GENERATED, NOT_GENERATED):
            raise SyntheticDataError(f"{vid}: status {r['status']!r}")
        if not r["attempts"].isdigit() or not 1 <= int(r["attempts"]) <= MAX_ATTEMPTS:
            raise SyntheticDataError(f"{vid}: attempts must be 1 to {MAX_ATTEMPTS}")
        attempts = int(r["attempts"])
        reasons = [x for x in r["retry_reasons"].split(";") if x]
        if any(x not in RETRY_REASONS for x in reasons):
            raise SyntheticDataError(f"{vid}: reasons must be among {RETRY_REASONS}")
        if r["status"] == NOT_GENERATED:
            if not 1 <= len(reasons) <= attempts:
                raise SyntheticDataError(f"{vid}: not generated needs its failure registered")
            continue
        if len(reasons) != attempts - 1:
            raise SyntheticDataError(f"{vid}: each new attempt needs one reason")
        when = datetime.fromisoformat(r["generated_at"]).astimezone()
        if when <= frozen_at:
            raise SyntheticDataError(f"{vid}: generated at {when}, not after the freeze")


def annotation_files(folder: Path) -> list[Path]:
    """The two independent annotation files."""
    paths = sorted(folder.glob("annotations_*.csv"))
    if len(paths) != 2:
        raise SyntheticDataError(f"{folder}: need two annotation files, found {len(paths)}")
    return paths


def check_annotations_before(paths: Sequence[Path], when: datetime) -> dict[str, str]:
    """Raise if a file changed after `when` (the first probe run); return the modification
    times, to be recorded in the run's meta.json."""
    mtimes = {}
    for path in paths:
        mtime = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
        if mtime >= when:
            raise SyntheticDataError(f"{path.name}: modified at {mtime}, after {when}")
        mtimes[path.name] = mtime.isoformat(timespec="seconds")
    return mtimes


def _interval(row: Row) -> tuple[float, float] | None:
    start, end = row["concealment_start_s"], row["concealment_end_s"]
    if start == NO_INTERVAL and end == NO_INTERVAL:
        return None
    try:
        a, b = float(start), float(end)
    except ValueError:
        raise SyntheticDataError(
            f"{row['id']}: interval must be two numbers or {NO_INTERVAL}"
        ) from None
    if not 0 <= a < b:
        raise SyntheticDataError(f"{row['id']}: interval {a} to {b}")
    return a, b


def validate_annotations(rows: Sequence[Row], ids: Sequence[str]) -> None:
    _check_ids(rows, ids, "annotations")
    for r in rows:
        if r["action_appeared"] not in ACTION_VALUES:
            raise SyntheticDataError(f"{r['id']}: action_appeared {r['action_appeared']!r}")
        if r["artifacts"] not in ARTIFACT_VALUES:
            raise SyntheticDataError(f"{r['id']}: artifacts {r['artifacts']!r}")
        _interval(r)


def reference_intervals(a: Sequence[Row], b: Sequence[Row]) -> dict[str, list[tuple[float, float]]]:
    """{id: merged intervals}: the union of the two annotators' intervals, in seconds."""
    others = {r["id"]: r for r in b}
    out = {}
    for row in a:
        found = [_interval(row), _interval(others[row["id"]])]
        merged: list[tuple[float, float]] = []
        for start, end in sorted(x for x in found if x is not None):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        out[row["id"]] = merged
    return out


def reference_labels(
    intervals: Sequence[tuple[float, float]], fps: float, n_frames: int
) -> npt.NDArray[np.int64]:
    """(n_frames,) 1 on the frames of the intervals, ready for `tcc.eval.events.event_hits`."""
    y = np.zeros(n_frames, np.int64)
    for start, end in intervals:
        y[math.floor(start * fps) : min(math.ceil(end * fps), n_frames)] = 1
    return y
