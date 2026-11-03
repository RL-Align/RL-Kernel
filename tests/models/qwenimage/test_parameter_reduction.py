# SPDX-License-Identifier: Apache-2.0
"""Cancellation regressions for the shared parameter-gradient reduction."""
import pytest
import torch

from rl_engine.backends.cuda.model_specific.qwenimage.adaln_mod_projection import require_extension
from rl_engine.backends.extension import _C

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rows", [255, 256, 257, 2048])
def test_parameter_sum_retains_small_contributions(dtype, rows):
    require_extension()
    # Ascending ordinary FP32 summation loses all the unit contributions.
    grad = torch.ones(rows, 2, device="cuda", dtype=torch.float32)
    grad[0] = 2**26
    grad[-1] = -(2**26)
    act = torch.ones(rows, 7, device="cuda", dtype=dtype)
    act[:, 1::2] = 1.5
    dw, db = _C.qwenimage_det_linear_backward_weight(grad, act, dtype, True)
    golden = grad.double().cpu().T @ act.double().cpu()
    golden_bias = grad.double().cpu().sum(0)
    assert torch.equal(dw.cpu(), golden.to(dtype))
    assert torch.equal(db.cpu(), golden_bias.to(dtype))
    again = _C.qwenimage_det_linear_backward_weight(grad, act, dtype, True)
    for left, right in zip((dw, db), again):
        assert torch.equal(left.view(torch.uint8), right.view(torch.uint8))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_bias_only_entrypoint_matches_joint_reduction(dtype):
    grad = torch.randn(257, 17, device="cuda", dtype=torch.float32)
    act = torch.randn(257, 3, device="cuda", dtype=dtype)
    _, expected = _C.qwenimage_det_linear_backward_weight(grad, act, dtype, True)
    actual = _C.qwenimage_det_linear_backward_bias(grad, dtype)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def test_stale_v1_extension_is_rejected(monkeypatch):
    monkeypatch.delattr(_C, "qwenimage_det_linear_backward_bias")
    with pytest.raises(RuntimeError, match="rebuild"):
        require_extension()
