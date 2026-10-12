"""Official PoseLift train/test split.

The split is the one distributed with the dataset: the `Pickle_files/Train` and
`Pickle_files/Test` folders (docs/tcc/FORMATO_POSELIFT.md). Without them this module
fails; it never infers the split from the labels (rule 1).
"""

from pathlib import Path

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
