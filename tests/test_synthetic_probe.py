"""Pre-registered synthetic probe (session 4): final prompts, manifest and annotations."""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from tcc.data.synthetic import (
    SyntheticDataError,
    annotation_files,
    check_annotations_before,
    final_prompts,
    read_annotations,
    read_manifest,
    reference_intervals,
    reference_labels,
    validate_annotations,
    validate_manifest,
)
from tcc.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "synthetic_probe" / "prompts.yaml"
FROZEN_AT = datetime.fromisoformat("2026-11-01T20:00:00-03:00")
IDS = [f"p{p}_{v}" for p in range(1, 6) for v in ("concealment", "normal")]
MANIFEST = "id,status,generated_at,generator,model_version,settings,attempts,retry_reasons\n"
ANNOTATIONS = "id,action_appeared,concealment_start_s,concealment_end_s,artifacts\n"


def _manifest(path: Path, overrides: dict[str, str] | None = None) -> Path:
    rows = [f"{i},gerado,2026-11-02T10:00:00-03:00,G,1,16:9,1," for i in IDS]
    for key, line in (overrides or {}).items():
        rows[IDS.index(key)] = line
    path.write_text(MANIFEST + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def _annotations(path: Path, overrides: dict[str, str] | None = None) -> Path:
    rows = [
        f"{i},sim,3,7,nenhum" if "concealment" in i else f"{i},sim,nenhum,nenhum,leves" for i in IDS
    ]
    for key, line in (overrides or {}).items():
        rows[IDS.index(key)] = line
    path.write_text(ANNOTATIONS + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_the_ten_final_prompts_start_with_the_same_style_block_in_the_declared_order() -> None:
    cfg = load_config(CONFIG)
    prompts = final_prompts(cfg)
    assert list(prompts) == cfg["order"] and len(prompts) == 10
    assert all(p.startswith(cfg["style"]) for p in prompts.values())


def test_the_two_prompts_of_each_pair_differ_only_in_the_variant() -> None:
    cfg = load_config(CONFIG)
    prompts = final_prompts(cfg)
    for pair in cfg["pairs"]:
        a, b = prompts[f"{pair['id']}_concealment"], prompts[f"{pair['id']}_normal"]
        assert a.endswith(pair["concealment"]) and b.endswith(pair["normal"])
        assert a[: -len(pair["concealment"])] == b[: -len(pair["normal"])]
        assert f"The person is {pair['person']}. " in a


def test_valid_manifest_passes(tmp_path: Path) -> None:
    validate_manifest(read_manifest(_manifest(tmp_path / "m.csv")), IDS, FROZEN_AT)


def test_manifest_needs_exactly_the_ten_ids(tmp_path: Path) -> None:
    rows = read_manifest(_manifest(tmp_path / "m.csv"))
    with pytest.raises(SyntheticDataError, match="ids"):
        validate_manifest(rows[:-1], IDS, FROZEN_AT)


def test_generation_before_the_freeze_is_refused(tmp_path: Path) -> None:
    rows = read_manifest(_manifest(tmp_path / "m.csv"))
    with pytest.raises(SyntheticDataError, match="freeze"):
        validate_manifest(rows, IDS, FROZEN_AT + timedelta(days=30))


def test_a_new_attempt_needs_a_listed_technical_failure(tmp_path: Path) -> None:
    good = {"p1_normal": "p1_normal,gerado,2026-11-02T10:00:00-03:00,G,1,16:9,2,corrompido"}
    validate_manifest(read_manifest(_manifest(tmp_path / "a.csv", good)), IDS, FROZEN_AT)
    for reasons, attempts in [("nao_ficou_bom", 2), ("", 2), ("corrompido;recusa", 3)]:
        line = f"p1_normal,gerado,2026-11-02T10:00:00-03:00,G,1,16:9,{attempts},{reasons}"
        rows = read_manifest(_manifest(tmp_path / "b.csv", {"p1_normal": line}))
        with pytest.raises(SyntheticDataError, match="p1_normal"):
            validate_manifest(rows, IDS, FROZEN_AT)


def test_a_video_not_generated_stays_in_the_manifest_with_its_failure(tmp_path: Path) -> None:
    line = "p2_normal,nao_gerado,,G,1,16:9,1,recusa"
    rows = read_manifest(_manifest(tmp_path / "m.csv", {"p2_normal": line}))
    validate_manifest(rows, IDS, FROZEN_AT)
    bad = "p2_normal,nao_gerado,,G,1,16:9,1,"
    rows = read_manifest(_manifest(tmp_path / "n.csv", {"p2_normal": bad}))
    with pytest.raises(SyntheticDataError, match="p2_normal"):
        validate_manifest(rows, IDS, FROZEN_AT)


def test_exactly_two_annotation_files_are_required(tmp_path: Path) -> None:
    _annotations(tmp_path / "annotations_a.csv")
    with pytest.raises(SyntheticDataError, match="two"):
        annotation_files(tmp_path)
    _annotations(tmp_path / "annotations_b.csv")
    assert [p.name for p in annotation_files(tmp_path)] == [
        "annotations_a.csv",
        "annotations_b.csv",
    ]


def test_annotation_values_are_validated_and_a_video_without_the_action_stays(
    tmp_path: Path,
) -> None:
    kept = {"p3_concealment": "p3_concealment,ambiguo,nenhum,nenhum,graves"}
    rows = read_annotations(_annotations(tmp_path / "a.csv", kept))
    validate_annotations(rows, IDS)
    assert len(rows) == 10
    for line in ["p1_normal,talvez,nenhum,nenhum,leves", "p1_normal,sim,5,3,leves",
                 "p1_normal,sim,3,nenhum,leves", "p1_normal,sim,nenhum,nenhum,muitos"]:  # fmt: skip
        rows = read_annotations(_annotations(tmp_path / "b.csv", {"p1_normal": line}))
        with pytest.raises(SyntheticDataError, match="p1_normal"):
            validate_annotations(rows, IDS)


def test_annotations_must_be_filled_before_any_run(tmp_path: Path) -> None:
    path = _annotations(tmp_path / "annotations_a.csv")
    mtimes = check_annotations_before([path], datetime.now().astimezone() + timedelta(hours=1))
    assert set(mtimes) == {"annotations_a.csv"}
    with pytest.raises(SyntheticDataError, match="annotations_a"):
        check_annotations_before([path], datetime.now().astimezone() - timedelta(hours=1))


def test_reference_interval_is_the_union_of_the_two_annotated_intervals(tmp_path: Path) -> None:
    a = read_annotations(_annotations(tmp_path / "a.csv"))
    over = {"p1_concealment": "p1_concealment,sim,5,8,leves",
            "p2_concealment": "p2_concealment,sim,8,9,leves",
            "p1_normal": "p1_normal,sim,1,2,leves"}  # fmt: skip
    b = read_annotations(_annotations(tmp_path / "b.csv", over))
    union = reference_intervals(a, b)
    assert union["p1_concealment"] == [(3.0, 8.0)]  # overlapping
    assert union["p2_concealment"] == [(3.0, 7.0), (8.0, 9.0)]  # disjoint: both stay
    assert union["p1_normal"] == [(1.0, 2.0)]  # only one annotator saw it
    assert union["p2_normal"] == []


def test_reference_labels_mark_the_frames_of_the_intervals() -> None:
    y = reference_labels([(1.0, 2.0)], fps=15, n_frames=60)
    np.testing.assert_array_equal(np.flatnonzero(y), np.arange(15, 30))
    assert reference_labels([(3.0, 99.0)], 15, 60).sum() == 15
    assert reference_labels([], 15, 60).sum() == 0
