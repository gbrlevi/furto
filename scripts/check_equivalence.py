"""Equivalence of the port with the original PoC reference, on validation videos only.

- split: the ported train/validation split gives the reference validation videos;
- treatment: with the original reading (as_is, one), the ported treatment gives exactly
  the windows stored in val_ref.npz.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tcc.data.poselift import load_video
from tcc.data.split import official_split, split_train_val
from tcc.preprocess.windows import video_windows
from tcc.utils.config import load_config
from tcc.utils.run_logger import start_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/equivalence_poc.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = {**load_config(Path(config["dataset_config"])), **config["dataset_overrides"]}
    config["dataset"] = dataset
    run_dir = start_run(
        "check_equivalence", config=config, seeds={"split": config["split"]["seed"]}
    )

    ref = np.load(Path(config["reference_dir"]) / "val_ref.npz")
    ref_val = sorted(str(v) for v in ref["videos"])
    root = Path(dataset["root"])
    train_ids = official_split(root)["train"]
    _, val = split_train_val(train_ids, config["split"]["val_fraction"], config["split"]["seed"])

    t = config["treatment"]
    per_video: dict[str, Any] = {}
    for vid in ref_val:
        x, meta = video_windows(
            load_video(root, "train", vid, dataset),
            t["T"],
            t["stride_eval"],
            t["max_gap"],
            t["conf_min"],
            t["center"],
        )
        rx, rmeta = ref[f"{vid}__windows"], ref[f"{vid}__window_meta"]
        same_shape = x.shape == rx.shape
        per_video[vid] = {
            "windows": int(len(x)),
            "reference_windows": int(len(rx)),
            "identical_windows": bool(same_shape and np.array_equal(x, rx)),
            "identical_meta": bool(same_shape and np.array_equal(meta[:, 1:], rmeta[:, 1:])),
            "max_abs_diff": float(np.abs(x - rx).max()) if same_shape and len(x) else None,
        }
    results = {
        "split": {"ported": sorted(val), "reference": ref_val, "identical": sorted(val) == ref_val},
        "treatment": {
            "videos": len(per_video),
            "windows": sum(v["windows"] for v in per_video.values()),
            "reference_windows": sum(v["reference_windows"] for v in per_video.values()),
            "all_identical": all(
                v["identical_windows"] and v["identical_meta"] for v in per_video.values()
            ),
            "per_video": per_video,
        },
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(run_dir)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_video"}
                      for k, v in results.items()}, indent=2))  # fmt: skip


if __name__ == "__main__":
    main()
