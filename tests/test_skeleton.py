"""Skeleton tests from the porting map (docs/tcc/PORTE_POC_LUIZ.md)."""

import re
from pathlib import Path

import numpy as np
import pytest

from tcc.data import skeleton

# Edge list of the data contract, copied from docs/tcc/CONTEXTO_TCC.md (ADR-0004).
CONTRACT_EDGES = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
]  # fmt: skip
CONTEXTO = Path(__file__).resolve().parents[1] / "docs" / "tcc" / "CONTEXTO_TCC.md"


def _undirected(edges: list[tuple[int, int]]) -> set[frozenset[int]]:
    return {frozenset(e) for e in edges}


def test_skeleton_has_19_edges_equal_to_contract() -> None:
    assert len(skeleton.EDGES) == 19
    assert len(_undirected(skeleton.EDGES)) == 19
    assert _undirected(skeleton.EDGES) == _undirected(CONTRACT_EDGES)


@pytest.mark.skipif(not CONTEXTO.exists(), reason="CONTEXTO_TCC.md is not versioned")
def test_skeleton_edges_equal_edges_listed_in_contexto() -> None:
    text = CONTEXTO.read_text(encoding="utf-8")
    block = text.split("**Arestas (base 0, 19 no total):**", 1)[1].split("```")[1]
    listed = [(int(a), int(b)) for a, b in re.findall(r"\((\d+),(\d+)\)", block)]
    assert len(listed) == 19
    assert _undirected(listed) == _undirected(skeleton.EDGES)


def test_joint_names_follow_coco17_order() -> None:
    assert len(skeleton.JOINT_NAMES) == skeleton.V == 17
    assert skeleton.JOINT_NAMES[0] == "nose"
    assert skeleton.JOINT_NAMES[9] == "left_wrist"
    assert skeleton.JOINT_NAMES[12] == "right_hip"
    assert skeleton.JOINT_NAMES[16] == "right_ankle"
    assert [skeleton.JOINT_NAMES[i] for i in skeleton.HIPS] == ["left_hip", "right_hip"]
    assert [skeleton.JOINT_NAMES[i] for i in skeleton.SHOULDERS] == [
        "left_shoulder",
        "right_shoulder",
    ]


def test_adjacency_is_symmetric() -> None:
    a = skeleton.normalized_adjacency()
    assert a.shape == (17, 17)
    assert a.dtype == np.float32
    np.testing.assert_array_equal(a, a.T)


def test_adjacency_is_d_inv_sqrt_a_plus_i_d_inv_sqrt() -> None:
    a = np.zeros((17, 17))
    for i, j in CONTRACT_EDGES:
        a[i, j] = a[j, i] = 1.0
    a_hat = a + np.eye(17)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(a_hat.sum(axis=1)))
    expected = d_inv_sqrt @ a_hat @ d_inv_sqrt
    np.testing.assert_allclose(skeleton.normalized_adjacency(), expected, rtol=1e-6)
    assert np.linalg.eigvalsh(expected).max() == pytest.approx(1.0)


def test_mirror_pairs_swap_left_and_right() -> None:
    assert len(skeleton.MIRROR_PAIRS) == 8
    for left, right in skeleton.MIRROR_PAIRS:
        assert skeleton.JOINT_NAMES[left].startswith("left_")
        assert skeleton.JOINT_NAMES[right] == "right_" + skeleton.JOINT_NAMES[left][5:]
    mirrored = [skeleton.JOINT_NAMES[i] for i in skeleton.MIRROR_INDEX]
    assert mirrored[0] == "nose"
    assert mirrored[5] == "right_shoulder"
    assert mirrored[6] == "left_shoulder"


def test_mirror_index_is_an_involution() -> None:
    idx = np.array(skeleton.MIRROR_INDEX)
    np.testing.assert_array_equal(idx[idx], np.arange(17))


def test_mirroring_preserves_the_edge_set() -> None:
    idx = skeleton.MIRROR_INDEX
    mirrored = [(idx[i], idx[j]) for i, j in skeleton.EDGES]
    assert _undirected(mirrored) == _undirected(skeleton.EDGES)
