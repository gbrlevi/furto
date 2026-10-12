"""Record a run that exercises the detected GPU."""

import argparse
import json
import time
from pathlib import Path

import torch

from tcc.utils.config import load_config
from tcc.utils.run_logger import start_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/smoke_gpu.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = start_run("smoke_gpu", config=config, seeds={"global": config["seed"]})

    if not torch.cuda.is_available():
        raise SystemExit(f"CUDA not available; run recorded at {run_dir}")
    torch.manual_seed(config["seed"])
    n = config["matrix_size"]
    a = torch.randn(n, n, device="cuda")
    torch.cuda.synchronize()
    start = time.perf_counter()
    checksum = (a @ a).sum().item()
    torch.cuda.synchronize()
    result = {
        "device": torch.cuda.get_device_name(0),
        "matmul_seconds": time.perf_counter() - start,
        "checksum": checksum,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(run_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
