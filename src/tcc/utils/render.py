"""Draw skeletons and score marks. Raw frames are only drawn on when a caller passes one."""

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


GREEN = (0, 131, 0)  # BGR of #008300
RED = (72, 73, 227)  # BGR of #e34948
NO_SCORE = (150, 150, 150)
WHITE = (255, 255, 255)
INTERPOLATED_ALPHA = 0.45  # opacity of poses filled by the treatment (drawing only)


def score_color(score: float, threshold: float) -> tuple[int, int, int]:
    """Green at 0, red at or above the threshold, linear in between; gray without score."""
    if not np.isfinite(score):
        return NO_SCORE
    t = float(np.clip(score / threshold, 0.0, 1.0))
    g, r = np.array(GREEN, float), np.array(RED, float)
    b, gg, rr = np.round((1 - t) * g + t * r).astype(int).tolist()
    return (b, gg, rr)


def annotate_frame(
    canvas: npt.NDArray[np.uint8],
    keypoints: npt.NDArray[np.float32],
    track_ids: npt.NDArray[np.int64],
    boxes: npt.NDArray[np.float32],
    scores: npt.NDArray[np.float64],
    threshold: float,
    scale: float,
    conf_min: float,
    alert: bool,
    caption: str = "",
    interpolated: npt.NDArray[np.bool_] | None = None,
    reasons: list[str] | None = None,
) -> npt.NDArray[np.uint8]:
    """Draw skeleton, score-coloured box, id and score of each person on a copy of
    `canvas` (already at `scale`), plus an ALERTA banner when `alert` is true.

    People marked in `interpolated` (pose filled by the treatment, no detection in this
    frame) are drawn translucent and labelled "interpolada". A non-empty entry of
    `reasons` (quality gate, ADR-0014) gives a gray box labelled with the reason.
    """
    img = canvas.copy()
    n = len(track_ids)
    interpolated = np.zeros(n, np.bool_) if interpolated is None else interpolated
    reasons = [""] * n if reasons is None else reasons
    rows = zip(keypoints, track_ids, boxes, scores, interpolated, reasons, strict=True)
    for kp, tid, box, s, interp, reason in rows:
        layer = img.copy() if interp else img
        xy = np.round(kp[:, :2] * scale).astype(np.int32)
        ok = kp[:, 2] >= conf_min
        for a, b in EDGES:
            if ok[a] and ok[b]:
                cv2.line(layer, tuple(xy[a]), tuple(xy[b]), BONE, 2, cv2.LINE_AA)
        color = NO_SCORE if reason else score_color(float(s), threshold)
        x1, y1, x2, y2 = np.round(box * scale).astype(int).tolist()
        cv2.rectangle(layer, (x1, y1), (x2, y2), color, 2)
        label = f"{int(tid)} " + (reason or (f"{s:.3f}" if np.isfinite(s) else "--"))
        if interp:
            label += " interpolada"
        cv2.putText(
            layer, label, (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
            cv2.LINE_AA,
        )  # fmt: skip
        if interp:
            blend = cv2.addWeighted(layer, INTERPOLATED_ALPHA, img, 1 - INTERPOLATED_ALPHA, 0)
            img = np.asarray(blend, np.uint8)
    if alert:
        cv2.rectangle(img, (0, 0), (img.shape[1], 28), RED, -1)
        cv2.putText(img, "ALERTA", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
    if caption:
        y = img.shape[0] - 10
        cv2.putText(img, caption, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT, 1, cv2.LINE_AA)
    return img


def blank_canvas(size: tuple[int, int], scale: float) -> npt.NDArray[np.uint8]:
    """Neutral background of the source `size` (width, height) times `scale`."""
    img = np.empty((int(size[1] * scale), int(size[0] * scale), 3), np.uint8)
    img[:] = BACKGROUND
    return img
