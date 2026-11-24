# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from rl_engine.models.inkling2.log_scaling import (
    apply_full_attention_log_scaling,
    full_attention_tau,
)

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
    ),
]


@pytest.mark.parametrize("device", DEVICES)
def test_tau_boundaries_and_packed_reset(device):
    p = torch.tensor([0, 8190, 8191, 8192, 16383, 0], device=device)
    tau = full_attention_tau(p)
    assert tau.dtype == torch.float32
    assert torch.equal(tau[[0, 1, 2, 5]], torch.ones(4, device=device))
    assert tau[3] > 1
    assert tau[4].item() == pytest.approx(1.0 + 0.1 * math.log(2.0), abs=1e-7)


@pytest.mark.parametrize("device", DEVICES)
def test_scaling_cast_order_backward_and_local_bypass(device):
    q = torch.full((3, 24, 128), 1.5, device=device, dtype=torch.bfloat16, requires_grad=True)
    r = torch.full((3, 24, 16), -0.75, device=device, dtype=torch.bfloat16, requires_grad=True)
    p = torch.tensor([8191, 16383, 0], device=device)
    scaled_q, scaled_r = apply_full_attention_log_scaling(q, r, p, is_local=False)
    tau = full_attention_tau(p)[:, None, None]
    assert torch.equal(scaled_q, (q.float() * tau).bfloat16())
    assert torch.equal(scaled_r, (r.float() * tau).bfloat16())
    assert torch.equal(scaled_q[1:2], apply_full_attention_log_scaling(
        q[1:2], r[1:2], p[1:2], is_local=False
    )[0])
    (scaled_q.float().sum() + scaled_r.float().sum()).backward()
    assert torch.equal(q.grad, tau.bfloat16().expand_as(q))
    assert torch.equal(r.grad, tau.bfloat16().expand_as(r))
    local_q, local_r = apply_full_attention_log_scaling(q, r, p, is_local=True)
    assert local_q is q and local_r is r


def test_invalid_positions():
    with pytest.raises(ValueError, match="nonnegative"):
        full_attention_tau(torch.tensor([-1, 0]))
    with pytest.raises(ValueError, match="integer"):
        full_attention_tau(torch.ones(2))
