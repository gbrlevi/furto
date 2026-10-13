"""Split the HRNet pose cost into network forward time and `inference_topdown` overhead.

Run with the pose environment. Times, on the GPU, the network forward on a batch of
crops (256 x 192) and one `inference_topdown` call with the same number of boxes on a
real video frame, with test-time flip on and off and with cuDNN autotuning on and off.
Only measures: the extraction config itself is not changed here.
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
        for flip in config["flip_test"]:
            for benchmark in config["cudnn_benchmark"]:
                torch.backends.cudnn.benchmark = benchmark
                model = init_model(pose["config"], pose["weights"], device=config["device"])
                model.eval()
                model.test_cfg["flip_test"] = flip
                key = f"{name}|flip_test={flip}|cudnn_benchmark={benchmark}"
                results[key] = {}
                for n in config["boxes"]:
                    x = torch.randn(n, 3, 256, 192, device=config["device"])
                    with torch.no_grad():
                        for _ in range(config["warmup"]):
                            model(x, None, mode="tensor")
                        forward = _median_ms(
                            lambda x=x, m=model: m(x, None, mode="tensor"), config["repeats"]
                        )
                    boxes = np.array(
                        [[w * (0.1 + 0.15 * i), h * 0.2, w * (0.2 + 0.15 * i), h * 0.8]
                         for i in range(n)],
                        np.float32,
                    )  # fmt: skip
                    for _ in range(config["warmup"]):
                        inference_topdown(model, frame, boxes, bbox_format="xyxy")
                    topdown = _median_ms(
                        lambda b=boxes, m=model: inference_topdown(m, frame, b, bbox_format="xyxy"),
                        config["repeats"],
                    )
                    results[key][f"{n}_boxes"] = {
                        "forward_ms": forward,
                        "inference_topdown_ms": topdown,
                    }
                del model
                torch.cuda.empty_cache()
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(run_dir)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
