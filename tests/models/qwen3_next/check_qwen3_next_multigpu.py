# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Required allocation-only checks. Missing GPUs fail instead of skipping."""

import pytest
import torch


@pytest.mark.parametrize("device", [1, 2, 3])
@pytest.mark.parametrize("gated", [False, True])
def test_norm_guard_restores_current_device_and_uses_input_stream(device, gated):
    from rl_engine import _C

    assert torch.cuda.device_count() == 4, "Four allocated GPUs are required"
    torch.cuda.set_device(device)
    x = torch.randn(32, 128, device=f"cuda:{device}", dtype=torch.bfloat16)
    weight = torch.randn(128, device=x.device, dtype=x.dtype)
    gate = torch.randn_like(x)
    dy = torch.randn_like(x)

    def invoke(w=weight):
        if gated:
            out, rstd = _C.rmsnorm_gated_forward(x, w, gate, 1e-6)
            dx = _C.rmsnorm_gated_backward_dx(dy, x, w, gate, rstd)
        else:
            out, rstd = _C.rmsnorm_forward(x, w, 1e-6, 1.0)
            dx = _C.rmsnorm_backward_dx(dy, x, w, rstd, 1.0)
        return out, rstd, dx

    expected = invoke()
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    # set_stream changes the current device: deliberately switch back before C++ entry.
    torch.cuda.set_stream(stream)
    torch.cuda.set_device(0)
    actual = invoke()
    assert torch.cuda.current_device() == 0
    stream.synchronize()
    for got, want in zip(actual, expected):
        assert got.device == x.device
        assert torch.isfinite(got).all()
        assert torch.equal(got.contiguous().view(torch.uint8), want.contiguous().view(torch.uint8))
    with pytest.raises(RuntimeError, match="device"):
        invoke(weight.to("cuda:0"))
    assert torch.cuda.current_device() == 0
