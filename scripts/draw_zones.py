"""Click tool to draw the shelf and checkout zones of a camera, once per installation.

Usage: draw_zones.py VIDEO ZONES_YAML [--id ID] [--propose AUTO_ZONES_YAML]

Opens the median background of the video (people who move are removed). With --propose, the
zones found by scripts/auto_zones.py for that video are loaded first, so the installer only
confirms or fixes them. Writes the zones of ID (default: the video file name) into ZONES_YAML,
keeping the other cameras already there, in the format of configs/zones/.

  left click   add a corner          right click  remove the last corner
  s / c        next zone is a shelf / a checkout
  enter or n   close the zone (3 corners or more)
  u            remove the last closed zone of the current kind
  q            save and quit         esc          quit without saving
"""

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from auto_zones import COLOURS, background

WINDOW = "zones"
HELP = "click corners | enter: close | s/c: shelf/checkout | u: undo | q: save | esc: quit"


def draw(image: np.ndarray, zones: dict[str, list[Any]], points: list[Any], kind: str,
         scale: float) -> np.ndarray:  # fmt: skip
    out = image.copy()
    for k, polygons in zones.items():
        for poly in polygons:
            cv2.polylines(out, [np.int32(np.asarray(poly) * scale)], True, COLOURS[k], 2)
    corners = [tuple(np.int32(np.asarray(p) * scale).tolist()) for p in points]
    for p, q in zip(corners, corners[1:], strict=False):
        cv2.line(out, p, q, COLOURS[kind], 1)
    for p in corners:
        cv2.circle(out, p, 4, COLOURS[kind], -1)
    cv2.putText(out, f"drawing: {kind}", (10, 25), 0, 0.7, COLOURS[kind], 2)
    cv2.putText(out, HELP, (10, out.shape[0] - 10), 0, 0.45, (255, 255, 255), 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video")
    parser.add_argument("zones")
    parser.add_argument("--id")
    parser.add_argument("--propose")
    args = parser.parse_args()
    vid = args.id or Path(args.video).stem
    image = background(args.video, 15)
    scale = min(1.0, 1280 / image.shape[1])
    shown = cv2.resize(image, None, fx=scale, fy=scale)
    zones: dict[str, list[Any]] = {"shelf": [], "checkout": []}
    if args.propose:
        proposed = yaml.safe_load(Path(args.propose).read_text(encoding="utf-8")).get(vid, {})
        zones = {k: list(proposed.get(k, [])) for k in zones}
    points: list[list[float]] = []
    state = {"kind": "shelf"}

    def on_mouse(event: int, x: int, y: int, *_: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([round(x / scale, 1), round(y / scale, 1)])
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    while True:
        cv2.imshow(WINDOW, draw(shown, zones, points, state["kind"], scale))
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("s"), ord("c")):
            state["kind"] = "shelf" if key == ord("s") else "checkout"
        elif key in (13, ord("n")) and len(points) >= 3:
            zones[state["kind"]].append(list(points))
            points.clear()
        elif key == ord("u") and zones[state["kind"]]:
            zones[state["kind"]].pop()
        elif key == ord("q"):
            path = Path(args.zones)
            saved = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
            saved = saved or {}
            saved[vid] = zones
            path.write_text(yaml.safe_dump(saved, sort_keys=True, default_flow_style=None),
                            encoding="utf-8")  # fmt: skip
            print(vid, {k: len(v) for k, v in zones.items()}, "->", path)
            break
        elif key == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
