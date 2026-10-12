"""Reference training with Luiz's original PoC, unmodified (docs/tcc/PORTE_POC_LUIZ.md).

Run it with the legacy interpreter so that meta.json records that environment:

    .venv-legacy/Scripts/python.exe scripts/reference_poc.py

The original script runs as a subprocess; nothing from legacy/ is imported here.
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
    out = Path(config["output_root"]) / config["name"]
    if out.exists():
        raise SystemExit(f"{out} already exists; the reference is never overwritten")
    out.mkdir(parents=True)

    command = [sys.executable, config["script"], "--dados", config["data"]]
    command += ["--saida", config["output_root"], "--nome", config["name"]]
    command += [str(a) for a in config["extra_args"]]
    config["command"] = command
    write_meta(out, "poc_luiz_reference_train", config, seeds={"semente": config["seed"]})

    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    with open(out / "stdout.log", "w", encoding="utf-8") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
    print(out)


if __name__ == "__main__":
    main()
