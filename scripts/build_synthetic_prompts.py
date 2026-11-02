"""Session 4, item 2: write the 10 final prompts of the synthetic probe, in the declared order.

The group copies them from `prompts_final.txt` into the generator, without editing. The run
records the SHA-256 of the pre-registration files at this moment.
"""

import argparse
import json
from pathlib import Path

from tcc.data.synthetic import final_prompts
from tcc.utils.config import load_config
from tcc.utils.frozen import sha256
from tcc.utils.run_logger import start_run

PROTOCOL = Path("configs/synthetic_probe/PROTOCOLO.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic_probe/prompts.yaml"))
    parser.add_argument("--out", type=Path, default=Path("data/raw/synthetic/prompts_final.txt"))
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = start_run("build_synthetic_prompts", config=config, seeds={})
    prompts = final_prompts(config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"NEGATIVE PROMPT (if the generator accepts it): {config['generator']['negative_prompt']}"
    ]
    for vid, text in prompts.items():
        lines += ["", f"### {vid}", text]
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    hashes = {str(p.as_posix()): sha256(p) for p in (PROTOCOL, args.config)}
    (run_dir / "results.json").write_text(
        json.dumps({"sha256": hashes}, indent=2), encoding="utf-8"
    )
    print(json.dumps(hashes, indent=1))
    print(args.out)
    print(run_dir)


if __name__ == "__main__":
    main()
