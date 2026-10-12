"""Share of training data in trajectories shorter than T, production configuration.

Decision of the group (12/10/2026): acceptable below 10% of the training frames.
Measured on the official training partition only, never on test.
"""

import argparse
import json
from pathlib import Path

from tcc.data.poselift import load_video
from tcc.data.split import official_split
from tcc.data.stats import short_trajectory_share
from tcc.utils.config import load_config
from tcc.utils.run_logger import start_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/c1_base.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = load_config(Path(config["dataset_config"]))
    config["dataset"] = dataset
    run_dir = start_run("short_tracks", config=config, seeds={})

    root = Path(dataset["root"])
    videos = [load_video(root, "train", v, dataset) for v in official_split(root)["train"]]
    t = config["treatment"]
    share = short_trajectory_share(videos, t["max_gap"], t["T"])
    results = {
        **share,
        "short_rows_fraction": share["short_rows"] / share["rows"],
        "frames_lost_fraction": share["frames_lost"] / share["frames"],
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(run_dir)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
