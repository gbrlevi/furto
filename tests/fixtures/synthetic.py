"""Deterministic synthetic pose data, only to exercise the code.

Performance numbers obtained on it mean nothing about PoseLift.

Adaptado de luizvitord/poc_furto@9bc4a91 (Luiz Vitor Dantas Freitas).
"""

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]

# standing pose in pixels, origin at the feet (y grows downwards)
BASE = np.array([
    [0, -170], [5, -175], [-5, -175], [10, -172], [-10, -172],
    [25, -140], [-25, -140], [30, -108], [-30, -108], [32, -78], [-32, -78],
    [15, -88], [-15, -88], [16, -45], [-16, -45], [17, 0], [-17, 0],
], dtype=np.float32)  # fmt: skip


@dataclass
class SyntheticVideo:
    name: str
    split: str  # "train" or "test"
    people: dict[int, tuple[IntArray, FloatArray]]  # person_id -> (frame indices, (n, 17, 3))
    labels: IntArray  # one 0/1 label per frame


def person(n: int, rng: np.random.Generator, theft: tuple[int, int] | None = None) -> FloatArray:
    """(n, 17, 3) sequence of a person walking, looking and picking products.

    theft=(start, end): brings the right wrist to the hip and keeps it there.
    """
    scale = rng.uniform(0.8, 1.3)
    x0, y0 = rng.uniform(300, 1500), rng.uniform(700, 1000)
    speed = rng.choice([-1, 1]) * rng.uniform(1.0, 3.0)
    phase = rng.uniform(0, 6.28)
    picks = set(rng.choice(n, size=max(1, n // 120), replace=False).tolist())
    out = np.zeros((n, 17, 3), np.float32)
    reach = 0.0
    for t in range(n):
        p = BASE.copy()
        walking = 1.0 if (t // 60) % 3 != 2 else 0.1  # alternates walking and stopping
        w = np.sin(0.35 * t + phase) * walking
        p[[13, 15], 0] += 10 * w
        p[[14, 16], 0] -= 10 * w
        p[[7, 9], 0] -= 8 * w
        p[[8, 10], 0] += 8 * w
        if t in picks:
            reach = 1.0
        reach *= 0.93  # arm goes up to the shelf and back
        p[9] += reach * np.array([35, -90])
        p[7] += reach * np.array([18, -45])
        if theft and theft[0] <= t < theft[1]:
            k = min(1.0, (t - theft[0]) / 8, (theft[1] - t) / 8)
            p[10] = (1 - k) * p[10] + k * (p[12] + np.array([-4, 6]))
            p[8] = (1 - k) * p[8] + k * (p[6] + np.array([-14, 30]))
            p[:5, 0] += 6 * k  # head tilts
        x = x0 + speed * t * walking
        out[t, :, :2] = p * scale + [x, y0]
        out[t, :, :2] += rng.normal(0, 1.5, (17, 2))
        out[t, :, 2] = rng.uniform(0.55, 0.98, 17)
        out[t, rng.random(17) < 0.03, 2] = 0.05  # lost points
    return out


def video(
    n: int, n_people: int, rng: np.random.Generator, with_theft: bool
) -> tuple[dict[int, tuple[IntArray, FloatArray]], IntArray]:
    people: dict[int, tuple[IntArray, FloatArray]] = {}
    labels = np.zeros(n, np.int64)
    for i in range(n_people):
        start, end = sorted(rng.choice(n, 2, replace=False))
        start, end = int(max(0, start - 40)), int(min(n, max(end, start + 80)))
        theft = None
        if with_theft and i == 0:
            a = rng.integers(start + 20, end - 40)
            theft = (int(a), int(a + rng.integers(20, 40)))
            labels[theft[0] : theft[1]] = 1
        local = None if theft is None else (theft[0] - start, theft[1] - start)
        people[i + 1] = (np.arange(start, end), person(end - start, rng, local))
    return people, labels


def generate(seed: int = 1, n_train: int = 24, n_test: int = 8) -> list[SyntheticVideo]:
    """Same random stream as the original script: identical output for the same seed."""
    rng = np.random.default_rng(seed)
    videos = []
    for k in range(n_train + n_test):
        name = f"C{k % 6 + 1}_{k:03d}"
        is_test = k >= n_train
        people, labels = video(int(rng.integers(250, 450)), int(rng.integers(1, 3)), rng, is_test)
        videos.append(SyntheticVideo(name, "test" if is_test else "train", people, labels))
    return videos


def write_pkl(root: Path, videos: list[SyntheticVideo]) -> None:
    """Write `<name>.pkl` + `<name>.npy` in the layout of the original script (bbox in XYWH)."""
    root.mkdir(parents=True, exist_ok=True)
    for v in videos:
        frames: dict[int, list[dict[str, object]]] = {}
        for pid, (idx, seq) in v.people.items():
            for q, kp in zip(idx, seq, strict=True):
                x, y = kp[:, 0], kp[:, 1]
                frames.setdefault(int(q), []).append(
                    {
                        "person_id": pid,
                        "bbox": [
                            float(x.min()),
                            float(y.min()),
                            float(x.max() - x.min()),
                            float(y.max() - y.min()),
                        ],
                        "keypoints": kp,
                    }
                )
        with open(root / f"{v.name}.pkl", "wb") as f:
            pickle.dump(frames, f)
        np.save(root / f"{v.name}.npy", v.labels)
