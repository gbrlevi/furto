"""Official PoseLift train/test split.

The split is the one distributed with the dataset: the `Pickle_files/Train` and
`Pickle_files/Test` folders (docs/tcc/FORMATO_POSELIFT.md). Without them this module
fails; it never infers the split from the labels (rule 1).
"""

from pathlib import Path

import numpy as np

from tcc.data.poselift import LABEL_DIR, SPLIT_DIRS


class MissingOfficialSplitError(RuntimeError):
    pass


def official_split(root: Path) -> dict[str, list[str]]:
    """Return {"train": [video ids], "test": [video ids]}, sorted."""
    base = root / "Pickle_files"
    split: dict[str, list[str]] = {}
    for name, folder in SPLIT_DIRS.items():
        path = base / folder
        if not path.is_dir():
            raise MissingOfficialSplitError(f"official split folder not found: {path}")
        split[name] = sorted(p.stem for p in path.glob("*.pkl"))
        if not split[name]:
            raise MissingOfficialSplitError(f"official split folder is empty: {path}")

    overlap = set(split["train"]) & set(split["test"])
    if overlap:
        raise MissingOfficialSplitError(f"videos in both train and test: {sorted(overlap)}")
    labelled = {p.stem for p in (base / LABEL_DIR).glob("*.npy")}
    missing = sorted(set(split["test"]) - labelled)
    if missing:
        raise MissingOfficialSplitError(f"test videos without labels: {missing}")
    extra = sorted(labelled - set(split["test"]))
    if extra:
        raise MissingOfficialSplitError(f"labels for videos outside the test split: {extra}")
    return split


def split_train_val(
    video_ids: list[str], fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Split training videos into train and validation by video, never by window.

    Same permutation as `dividir_por_video` in the original PoC, so the same ids in the
    same order give the same validation videos.
    Adaptado de luizvitord/poc_furto@9bc4a91 (Luiz Vitor Dantas Freitas).
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(video_ids))
    n_val = max(1, int(round(fraction * len(video_ids))))
    val = {video_ids[i] for i in idx[:n_val]}
    return [v for v in video_ids if v not in val], [v for v in video_ids if v in val]
