"""Zero-shot clip scores of a video-language model, one .npz per video and prompt.

Usage: vlm_scores.py CONFIG SET PROMPT [PROMPT ...], SET in dev, holdout, normal.
`normal` is the threshold and measure parts of the normal split of the base config.
"""

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

from tcc.utils.config import load_config
from tcc.utils.run_logger import start_run
from tcc.video.clips import parse_annotations, people_crop
from tcc.video.vlm import clip_layout, yes_no_margin


def video_paths(config: dict[str, Any], base: dict[str, Any], which: str) -> list[Path]:
    root = Path(config["videos"])
    if which == "dev":
        names = parse_annotations(Path(base["annotations"]).read_text(), base["category"])
        return [root / f"{k}.mp4" for k in sorted(names)]
    if which == "holdout":
        labels = yaml.safe_load(Path(config["holdout_labels"]).read_text(encoding="utf-8"))
        return [root / f"{k}.mp4" for k in sorted(labels["videos"])]
    normal = sorted(Path(config["normal_videos"]).glob("*.mp4"))
    order = np.random.default_rng(base["seed"]).permutation(len(normal))
    n_train = int(base["normal_split"][0] * len(normal))
    return [normal[i] for i in order[n_train:]]  # threshold and measure parts


def frames_at(path: Path, wanted: set[int]) -> tuple[dict[int, np.ndarray], int, float]:
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    out, n = {}, 0
    while capture.grab():
        if n in wanted:
            out[n] = cv2.cvtColor(capture.retrieve()[1], cv2.COLOR_BGR2RGB)
        n += 1
    capture.release()
    return out, n, fps


def main() -> None:
    config = load_config(Path(sys.argv[1]))
    which, prompts = sys.argv[2], sys.argv[3:]
    base = load_config(Path(config["base"]))
    run = start_run(
        f"vlm_scores_{which}",
        config={**config, "set": which, "prompts_run": prompts},
        seeds={},
        models={"vlm": {"hub_id": config["model"]}},
    )
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config["model"], quantization_config=quant, device_map="cuda:0"
    ).eval()
    processor = AutoProcessor.from_pretrained(config["model"])
    tok = processor.tokenizer
    yes, no = (tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "No"))
    head = model.lm_head.weight.detach().float().cpu().numpy()
    texts = {
        p: processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [{"type": "video"}, {"type": "text", "text": config["prompts"][p]}],
                }
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        for p in prompts
    }
    paths = video_paths(config, base, which)
    for i, path in enumerate(paths):
        targets = {p: Path(config["out"]) / p / f"{path.stem}.npz" for p in prompts}
        todo = [p for p in prompts if not targets[p].exists()]
        if not todo:
            continue
        capture = cv2.VideoCapture(str(path))
        n_est, fps0 = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), capture.get(cv2.CAP_PROP_FPS)
        capture.release()
        step, span, hop, _ = clip_layout(
            n_est, fps0, config["sample_fps"], config["frames_per_clip"], config["hop_seconds"]
        )
        frames, n, fps = frames_at(path, set(range(0, n_est + span, step)))
        starts = clip_layout(
            n, fps, config["sample_fps"], config["frames_per_clip"], config["hop_seconds"]
        )[3]
        scores = {p: np.zeros(len(starts)) for p in todo}
        for c, s in enumerate(starts):
            clip = np.stack([frames[s + k * step] for k in range(config["frames_per_clip"])])
            crop = config.get("crop")
            if crop:
                boxes = np.load(Path(crop["boxes"]) / f"{path.stem}.npz")["boxes"]
                h, w = clip.shape[1:3]
                x1, y1, x2, y2 = people_crop(boxes, s, s + span - 1, w, h, crop["margin"])
                clip = np.stack(
                    [cv2.resize(f[y1:y2, x1:x2], (crop["size"], crop["size"])) for f in clip]
                )
            for p in todo:
                inputs = processor(
                    text=[texts[p]],
                    videos=[clip],
                    return_tensors="pt",
                    videos_kwargs={"fps": config["sample_fps"], "do_sample_frames": False},
                ).to("cuda:0")
                with torch.no_grad():
                    out = model(**inputs, output_hidden_states=True, logits_to_keep=1)
                hidden = out.hidden_states[-1][0, -1].float().cpu().numpy()
                scores[p][c] = yes_no_margin(hidden, head[yes], head[no])
        for p in todo:
            targets[p].parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                targets[p], scores=scores[p], starts=starts, span=span, n_frames=n, fps=fps
            )
        print(i + 1, "of", len(paths), path.stem, len(starts), flush=True)
    print("run", run)


if __name__ == "__main__":
    main()
