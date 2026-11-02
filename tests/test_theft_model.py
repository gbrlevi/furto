import numpy as np

from tcc.video.theft_model import clip_labels, fit_linear, predict_linear, within_video_auc


def test_clip_is_theft_only_with_enough_annotated_frames() -> None:
    y = np.array([0, 0, 1, 1, 1, 1, 0, 0])
    assert clip_labels(np.array([0, 2, 5]), y, 3, 0.5).tolist() == [0, 1, 0]


def test_within_video_auc_of_an_oracle_is_one_and_above_chance() -> None:
    labels = [np.array([0] * 30 + [1] * 10 + [0] * 30) for _ in range(10)]
    result = within_video_auc(labels, [y.astype(float) for y in labels], 200, 0)
    assert result["auc"] == 1.0
    assert result["null_high"] < 1.0
    assert result["p"] < 0.01


def test_within_video_auc_of_random_scores_stays_inside_chance() -> None:
    rng = np.random.default_rng(0)
    labels = [np.array([0] * 30 + [1] * 10 + [0] * 30) for _ in range(30)]
    result = within_video_auc(labels, [rng.random(70) for _ in labels], 200, 0)
    assert result["null_low"] <= result["auc"] <= result["null_high"]


def test_linear_model_separates_clips_it_can_separate() -> None:
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 1, (50, 4)), rng.normal(3, 1, (50, 4))])
    y = np.array([0] * 50 + [1] * 50)
    score = predict_linear(fit_linear(x, y, 1.0), x)
    assert score[y == 1].mean() > score[y == 0].mean()
