# SPDX-License-Identifier: Apache-2.0
"""Exercise the opt-in A3 Qwen gate/up forward through its autograd entry point."""
from unittest.mock import patch

import pytest
import torch


def same(label, a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert a.isfinite().all() and b.isfinite().all()
    assert torch.equal(a.view(torch.int16), b.view(torch.int16)), label


def test_a3_gate_up_forward_and_backward(monkeypatch):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available() or torch.npu.get_device_name(0) != "Ascend910_9362":
        pytest.skip("requires A3 9362")
    pytest.importorskip("triton")
    pytest.importorskip("rl_engine._C_npu")
    from rl_engine.backends.ascend.gemm import a3_cube_trial
    from rl_engine.backends.ascend.gemm.det_gemm import DetGemmAscendOp

    torch.manual_seed(42)
    op = DetGemmAscendOp()
    opts = dict(device="npu:0", dtype=torch.bfloat16)
    x = torch.randn(129, 4096, **opts)
    w = torch.randn(12288, 4096, **opts, requires_grad=True)
    dy = torch.randn(2, 12288, **opts)
    grads = []
    for flag in ("0", "1"):
        monkeypatch.setenv("RLK_A3_FFN_CUBE", flag)
        a = x[:2].detach().requires_grad_()
        with patch.object(
            a3_cube_trial, "gate_up_linear", wraps=a3_cube_trial.gate_up_linear
        ) as kernel:
            y = op.linear(a, w)
            assert kernel.call_count == int(flag), "Unexpected backend selection"
        grads.append(torch.autograd.grad(y, (a, w), dy))
    same("input gradient, identical upstream", grads[0][0], grads[1][0])
    same("weight gradient, identical upstream", grads[0][1], grads[1][1])
    with torch.no_grad():
        y = op.linear(x, w)
        same("129 rows vs 17-row chunks", y, torch.cat([op.linear(v, w) for v in x.split(17)]))
        with torch.inference_mode():
            same("inference vs no_grad", y, op.linear(x, w))
