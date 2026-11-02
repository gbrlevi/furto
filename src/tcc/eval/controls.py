"""Controls of what the autoencoder score reacts to: perturbed copies of real windows and a
few simple descriptors of each window. Windows are (N, 3, T, V) after the treatment, with
channels x, y, confidence; joints follow COCO-17."""

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]

# (wrist, elbow, hip) of each side, COCO-17
ARMS = {"left": (9, 7, 11), "right": (10, 8, 12)}


def hand_to_hip(x: FloatArray, side: str, frames: int, ramp: int) -> FloatArray:
    """The wrist moves to the hip of its side over `ramp` frames, starting `frames` frames
    before the end, and stays there; the elbow goes half the way."""
    out = x.copy()
    wrist, elbow, hip = ARMS[side]
    t = x.shape[2]
    w = np.clip((np.arange(t) - (t - frames)) / ramp, 0, 1)[None, None, :]
    out[:, :2, :, wrist] = x[:, :2, :, wrist] * (1 - w) + x[:, :2, :, hip] * w
    out[:, :2, :, elbow] = x[:, :2, :, elbow] * (1 - w / 2) + x[:, :2, :, hip] * w / 2
    return out


def raise_wrist(x: FloatArray, side: str, frames: int, height: float) -> FloatArray:
    """The wrist jumps to `height` torsos above the hip centre in the last `frames` frames."""
    out = x.copy()
    wrist = ARMS[side][0]
    out[:, 0, -frames:, wrist] = 0.0
    out[:, 1, -frames:, wrist] = -height  # image y grows downwards
    return out


def upside_down(x: FloatArray) -> FloatArray:
    out = x.copy()
    out[:, 1] *= -1
    return out


def add_noise(x: FloatArray, sd: float, rng: np.random.Generator) -> FloatArray:
    out = x.copy()
    out[:, :2] += rng.normal(0, sd, out[:, :2].shape).astype(np.float32)
    return out


def shuffle_joints(x: FloatArray, rng: np.random.Generator) -> FloatArray:
    return x[:, :, :, rng.permutation(x.shape[3])].copy()


def descriptors(x: FloatArray) -> dict[str, npt.NDArray[np.float64]]:
    """Per window: mean joint speed, mean and maximum wrist speed of each side, and mean
    confidence. Speeds in torsos per frame."""
    xy = x[:, :2].transpose(0, 2, 3, 1).astype(np.float64)  # N, T, V, 2
    speed = np.linalg.norm(np.diff(xy, axis=1), axis=-1)  # N, T-1, V
    out = {"pose_speed": speed.mean((1, 2)), "confidence": x[:, 2].mean((1, 2)).astype(np.float64)}
    for side, (wrist, _, _) in ARMS.items():
        out[f"{side}_wrist_speed_mean"] = speed[:, :, wrist].mean(1)
        out[f"{side}_wrist_speed_max"] = speed[:, :, wrist].max(1)
    return out
