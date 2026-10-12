import numpy as np
import pytest
import torch

from tcc.data.skeleton import JOINT_NAMES, MIRROR_PAIRS
from tcc.model.dataset import WindowDataset, mirror
from tcc.model.loss import weighted_error
from tcc.model.stgcn_ae import STGCNAutoencoder, count_parameters, state_dict_from_poc
from tcc.model.train import train
from tcc.preprocess.trajectories import Trajectory
from tcc.preprocess.windows import make_windows
from tcc.utils.seed import set_seed
from tests.fixtures.synthetic import generate

POC_NAMES = {
    "data_bn.": "bn_dados.",
    "encoder.": "enc.",
    "decoder.": "dec.",
    "to_latent.": "para_latente.",
    "from_latent.": "de_latente.",
    "head.": "saida.",
}


@pytest.mark.parametrize("t", [12, 24, 48])
def test_autoencoder_maps_3xTx17_to_2xTx17(t: int) -> None:
    model = STGCNAutoencoder().eval()
    out = model(torch.randn(5, 3, t, 17))
    assert out.shape == (5, 2, t, 17)


def test_default_model_has_the_poc_parameter_count() -> None:
    # 166,078: PoC default model (REGISTRO, 12/10/2026; reference run stdout)
    assert count_parameters(STGCNAutoencoder()) == 166_078


def test_poc_state_dict_names_map_onto_the_ported_model() -> None:
    model = STGCNAutoencoder()
    poc_state = {}
    for key, value in model.state_dict().items():
        for ours, theirs in POC_NAMES.items():
            if key.startswith(ours):
                poc_state[theirs + key[len(ours) :].replace("importance", "importancia")] = value
    assert len(poc_state) == len(model.state_dict())
    STGCNAutoencoder().load_state_dict(state_dict_from_poc(poc_state), strict=True)


def test_loss_normalized_by_confidence_sum() -> None:
    target = torch.zeros(1, 3, 2, 17)
    target[0, 2, 0, 0] = 0.5  # one confident point
    target[0, 2, 1, 1] = 0.5
    rec = torch.zeros(1, 2, 2, 17)
    rec[0, 0, 0, 0] = 2.0  # error 4 on the first point
    assert weighted_error(rec, target, "conf").item() == pytest.approx(0.5 * 4 / 1.0)
    assert weighted_error(rec, target, "tv").item() == pytest.approx(0.5 * 4 / (2 * 17))


def test_loss_is_nan_without_confident_points() -> None:
    target = torch.zeros(2, 3, 4, 17)
    target[1, 2] = 1.0
    out = weighted_error(torch.zeros(2, 2, 4, 17), target, "conf")
    assert torch.isnan(out[0]) and out[1].item() == 0.0


def test_mirror_swaps_left_and_right_and_negates_x() -> None:
    x = np.random.default_rng(0).normal(size=(3, 24, 17)).astype(np.float32)
    m = mirror(x)
    for left, right in MIRROR_PAIRS:
        assert JOINT_NAMES[left].startswith("left_")
        np.testing.assert_array_equal(m[0, :, left], -x[0, :, right])
        np.testing.assert_array_equal(m[1, :, left], x[1, :, right])
    np.testing.assert_array_equal(m[0, :, 0], -x[0, :, 0])
    np.testing.assert_array_equal(mirror(m), x)


def _windows() -> np.ndarray:
    x = np.random.default_rng(1).normal(size=(4, 3, 24, 17)).astype(np.float32)
    x[:, 2] = 0.8
    return x


def test_dataset_without_augmentation_returns_the_window_twice() -> None:
    ds = WindowDataset(_windows(), np.random.default_rng(0), augment=False)
    inp, tgt = ds[2]
    np.testing.assert_array_equal(inp.numpy(), _windows()[2])
    np.testing.assert_array_equal(tgt.numpy(), _windows()[2])


def test_joint_removal_only_on_input_by_default() -> None:
    ds = WindowDataset(_windows(), np.random.default_rng(0), augment=True, p_drop_joint=0.5)
    inp, tgt = ds[0]
    dropped = (inp[2] == 0).all(0)
    assert dropped.any()
    assert (tgt[2] > 0).all()


def test_mask_target_removes_joints_from_the_target_too() -> None:
    ds = WindowDataset(
        _windows(), np.random.default_rng(0), augment=True, p_drop_joint=0.5, mask_target=True
    )
    inp, tgt = ds[0]
    assert torch.equal(inp, tgt)


def test_overfits_a_single_batch() -> None:
    """Acceptance (ROADMAP stage 5): on one small batch the loss falls close to zero."""
    rng = set_seed(0)
    v = generate(seed=3, n_train=1, n_test=0)[0]
    trajs = [Trajectory(p, idx, seq) for p, (idx, seq) in v.people.items()]
    x, _ = make_windows(trajs, 24, 24, 0.1, "frame")
    ds = WindowDataset(x[:8], rng)
    cfg = {
        "batch_size": 8,
        "lr": 1e-3,
        "epochs": 150,
        "patience": 10**6,
        "lr_plateau": {"enabled": False},
    }
    model = STGCNAutoencoder(dropout=0.0)
    result = train(
        model, ds, ds, cfg, "conf", torch.device("cpu"), torch.Generator().manual_seed(0)
    )
    first, last = result.history[0]["train_loss"], result.history[-1]["train_loss"]
    assert last < 0.005
    assert last < 1e-3 * first
