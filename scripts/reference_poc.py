"""Run one step of the reference with Luiz's original PoC (docs/tcc/PORTE_POC_LUIZ.md).

Run it with the legacy interpreter so that the meta file records that environment:

    .venv-legacy/Scripts/python.exe scripts/reference_poc.py --config configs/reference_poc.yaml

The original script runs as a subprocess; nothing from legacy/ is imported here.
Each step writes meta_<step>.json and stdout_<step>.log and is never repeated.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from tcc.utils.config import load_config
from tcc.utils.run_logger import write_meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/reference_poc.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    out = Path(config["output_dir"])
    step = config["step"]
    if (out / f"meta_{step}.json").exists():
        raise SystemExit(f"step {step!r} already ran in {out}; the reference is never redone")
    out.mkdir(parents=True, exist_ok=True)

    command = [sys.executable, config["script"], *(str(a) for a in config["args"])]
    config["command"] = command
    write_meta(
        out,
        f"poc_luiz_reference_{step}",
        config,
        seeds={"semente": config["seed"]},
        filename=f"meta_{step}.json",
    )
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    with open(out / f"stdout_{step}.log", "w", encoding="utf-8") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
    print(out)


if __name__ == "__main__":
    main()
