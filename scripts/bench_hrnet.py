"""Split the HRNet pose cost into network forward time and `inference_topdown` overhead.

Run with the pose environment. Times, on the GPU, the network forward on a batch of
crops (256 x 192) and one `inference_topdown` call with the same number of boxes on a
real video frame.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from tcc.utils.config import load_config
from tcc.utils.run_logger import start_run


def _median_ms(fn, repeats: int) -> float:  # type: ignore[no-untyped-def]
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def main() -> None:
    from mmpose.apis import inference_topdown, init_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/bench_hrnet.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = start_run("bench_hrnet", config=config, seeds={"global": config["seed"]})
    torch.manual_seed(config["seed"])

    cap = cv2.VideoCapture(config["frame_source"])
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("cannot read the frame source")
    h, w = frame.shape[:2]
    results = {}
    for name, pose in config["models"].items():
        model = init_model(pose["config"], pose["weights"], device=config["device"])
        model.eval()
        flip = bool(model.cfg.model.test_cfg.get("flip_test", False))
        per_model = {"flip_test": flip}
        for n in config["boxes"]:
            x = torch.randn(n, 3, 256, 192, device=config["device"])
            with torch.no_grad():
                for _ in range(config["warmup"]):
                    model(x, mode="tensor")
                forward = _median_ms(lambda x=x: model(x, mode="tensor"), config["repeats"])
            boxes = np.array(
                [[w * (0.1 + 0.15 * i), h * 0.2, w * (0.2 + 0.15 * i), h * 0.8] for i in range(n)],
                np.float32,
            )
            for _ in range(config["warmup"]):
                inference_topdown(model, frame, boxes, bbox_format="xyxy")
            topdown = _median_ms(
                lambda boxes=boxes: inference_topdown(model, frame, boxes, bbox_format="xyxy"),
                config["repeats"],
            )
            per_model[f"{n}_boxes"] = {
                "forward_ms": forward,
                "inference_topdown_ms": topdown,
                "forward_with_flip_ms_estimate": forward * (2 if flip else 1),
            }
        results[name] = per_model
        del model
        torch.cuda.empty_cache()
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(run_dir)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
