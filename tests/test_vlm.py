import numpy as np

from tcc.video.vlm import clip_layout, yes_no_margin


def test_clip_layout_at_two_frames_per_second_of_a_30_fps_video() -> None:
    step, span, hop, starts = clip_layout(300, 30.0, 2.0, 8, 2.0)
    assert (step, span, hop) == (15, 106, 60)
    assert starts.tolist() == [0, 60, 120, 180]  # the last clip ends at frame 285 < 300
    assert (starts + span <= 300).all()


def test_clip_layout_of_a_video_shorter_than_a_clip_has_no_clip() -> None:
    assert clip_layout(50, 30.0, 2.0, 8, 2.0)[3].tolist() == []


def test_yes_no_margin_is_the_difference_of_the_two_logits() -> None:
    h = np.array([1.0, 2.0], np.float32)
    assert (
        yes_no_margin(h, np.array([3.0, 0.0], np.float32), np.array([0.0, 1.0], np.float32)) == 1.0
    )
