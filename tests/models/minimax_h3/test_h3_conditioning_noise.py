# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The conditioning noise mix on its row: ``bind`` against the provider's ``scale_noise``."""

from __future__ import annotations

import pytest
import torch

from rl_engine.reference.conditioning.conditioning_noise_mix import NativeConditioningNoiseMixOp
from rl_engine.validation.models import h3_conditioning_noise as hcn

scheduling = pytest.importorskip(
    "diffusers.schedulers.scheduling_minimax_h3",
    reason="the Diffusers MiniMax-H3 scheduler is the reference",
)


def _scheduler():
    return scheduling.MiniMaxH3Scheduler()


def _case(dtype, noise_dtype=torch.float32, shape=(1, 24, 3, 8, 12)):
    gen = torch.Generator().manual_seed(11)
    return torch.randn(shape, generator=gen).to(dtype), torch.randn(shape, generator=gen).to(
        noise_dtype
    )


@pytest.mark.parametrize("t", [0.999, 0.5, 1.0])
def test_same_dtype_runs_the_row_bitwise(t):
    scheduler = _scheduler()
    sample, noise = _case(torch.float32)
    want = scheduler.scale_noise(sample, t, noise)
    with hcn.bind(scheduler, NativeConditioningNoiseMixOp()) as calls:
        got = scheduler.scale_noise(sample, t, noise)
    assert calls == {"row": 1, "provider": 0}
    assert got.dtype == want.dtype and torch.equal(got.view(-1), want.view(-1))
    assert "scale_noise" not in scheduler.__dict__


def test_mixed_dtypes_stay_on_the_provider():
    scheduler = _scheduler()
    sample, noise = _case(torch.bfloat16)
    want = scheduler.scale_noise(sample, 0.999, noise)
    with hcn.bind(scheduler, NativeConditioningNoiseMixOp()) as calls:
        got = scheduler.scale_noise(sample, 0.999, noise)
    assert calls == {"row": 0, "provider": 1}
    assert torch.equal(got, want)


def test_a_wrapped_scale_noise_is_refused():
    scheduler = _scheduler()
    scheduler.scale_noise = lambda *a: None
    with pytest.raises(RuntimeError, match="already wrapped"):
        with hcn.bind(scheduler, NativeConditioningNoiseMixOp()):
            pass
