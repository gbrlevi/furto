"""Equivalence with the original PoC reference run (docs/tcc/PORTE_POC_LUIZ.md).

Needs data/reference/poc_luiz_seed0 (not versioned); skipped without it.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from tcc.model.stgcn_ae import STGCNAutoencoder, state_dict_from_poc
from tcc.model.train import window_scores

REF = Path(__file__).resolve().parents[1] / "data" / "reference" / "poc_luiz_seed0"
# Declared tolerance (float32; cuDNN may choose other kernels; CPU vs GPU differs slightly)
RTOL, ATOL = 1e-5, 1e-7

pytestmark = pytest.mark.skipif(
    not (REF / "val_ref.npz").exists(), reason="reference run not available"
)


def test_reference_checkpoint_in_ported_model_reproduces_validation_window_scores() -> None:
    ref = np.load(REF / "val_ref.npz")
    ck = torch.load(REF / "modelo.pt", map_location="cpu", weights_only=True)
    m = ck["modelo"]
    model = STGCNAutoencoder(m["canais"], m["latente"], m["k_t"], m["dropout"])
    model.load_state_dict(state_dict_from_poc(ck["estado"]), strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    for vid in ref["videos"]:
        x = ref[f"{vid}__windows"]
        ours = window_scores(model, x, device, "conf", 1024)
        np.testing.assert_allclose(
            ours, ref[f"{vid}__window_scores"], rtol=RTOL, atol=ATOL, equal_nan=True
        )
