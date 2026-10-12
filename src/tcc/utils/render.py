"""Draw skeletons on a neutral background. No raw video is ever read or written here."""

import cv2
import numpy as np
import numpy.typing as npt

from tcc.data.skeleton import EDGES

BACKGROUND = (236, 239, 240)  # BGR, light neutral gray
BONE = (214, 120, 42)  # BGR of #2a78d6
JOINT = (61, 61, 61)
TEXT = (61, 61, 61)


def draw_skeletons(
    keypoints: npt.NDArray[np.float32],
    track_ids: npt.NDArray[np.int64],
    size: tuple[int, int],
    scale: float,
    conf_min: float,
    caption: str = "",
) -> npt.NDArray[np.uint8]:
    """Render (k, 17, 3) contract keypoints (x, y in pixels) on a (h, w) canvas.

    `size` is (width, height) of the source image; the canvas is `size * scale`.
    Joints with confidence below `conf_min` are not drawn.
    """
    width, height = int(size[0] * scale), int(size[1] * scale)
    img = np.empty((height, width, 3), np.uint8)
    img[:] = BACKGROUND
    for kp, tid in zip(keypoints, track_ids, strict=True):
        xy = np.round(kp[:, :2] * scale).astype(np.int32)
        ok = kp[:, 2] >= conf_min
        for a, b in EDGES:
            if ok[a] and ok[b]:
                cv2.line(img, tuple(xy[a]), tuple(xy[b]), BONE, 2, cv2.LINE_AA)
        for j in np.flatnonzero(ok):
            cv2.circle(img, tuple(xy[j]), 3, JOINT, -1, cv2.LINE_AA)
        if ok.any():
            top = xy[ok][np.argmin(xy[ok][:, 1])]
            cv2.putText(img, str(int(tid)), (int(top[0]), int(top[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT, 1, cv2.LINE_AA)  # fmt: skip
    if caption:
        cv2.putText(img, caption, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT, 1, cv2.LINE_AA)
    return img
