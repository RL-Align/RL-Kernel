# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CPU tests for the mHC entry: dispatch through kernel_registry to the reference.

The aiter path on ROCm is tested in tests/backends/rocm/test_rocm_mhc_*.py.
"""

import pytest
import torch

from rl_engine.ops.mhc import mhc_head, mhc_post, mhc_pre
from rl_engine.reference.mhc import NativeMHCHeadOp, NativeMHCPostOp, NativeMHCPreOp
from rl_engine.runtime.registry import kernel_registry

_N, _C, _M = 4, 16, 3


def _inputs(seed=0):
    rng = torch.Generator().manual_seed(seed)
    rows = 2 * _N + _N * _N
    residual = torch.randn(_M, _N, _C, generator=rng).to(torch.bfloat16)
    fn = torch.randn(rows, _N * _C, generator=rng) * 0.1
    scale = torch.rand(3, generator=rng) + 0.5
    base = torch.randn(rows, generator=rng) * 0.1
    head_fn = torch.randn(_N, _N * _C, generator=rng) * 0.1
    head_scale = torch.rand(1, generator=rng) + 0.5
    head_base = torch.randn(_N, generator=rng) * 0.1
    y = torch.randn(_M, _C, generator=rng).to(torch.bfloat16)
    return residual, fn, scale, base, head_fn, head_scale, head_base, y


@pytest.fixture
def get_op_calls(monkeypatch):
    calls = []
    real_get_op = kernel_registry.get_op

    def recording_get_op(op_type, device=None):
        calls.append((op_type, torch.device(device).type))
        return real_get_op(op_type, device=device)

    monkeypatch.setattr(kernel_registry, "get_op", recording_get_op)
    return calls


def test_entry_dispatches_to_the_reference_on_cpu(get_op_calls):
    residual, fn, scale, base, head_fn, head_scale, head_base, y = _inputs()

    got_pre = mhc_pre(residual, fn, scale, base, sinkhorn_repeat=5)
    want_pre = NativeMHCPreOp()(residual, fn, scale, base, sinkhorn_repeat=5)
    for got, want in zip(got_pre, want_pre):
        assert torch.equal(got, want)

    post_mix, comb_mix, _ = want_pre
    got_post = mhc_post(y, residual, post_mix, comb_mix)
    assert torch.equal(got_post, NativeMHCPostOp()(y, residual, post_mix, comb_mix))

    got_head = mhc_head(residual, head_fn, head_scale, head_base)
    assert torch.equal(got_head, NativeMHCHeadOp()(residual, head_fn, head_scale, head_base))

    assert get_op_calls == [("mhc_pre", "cpu"), ("mhc_post", "cpu"), ("mhc_head", "cpu")]


def test_entry_passes_the_keyword_arguments():
    residual, fn, scale, base, head_fn, head_scale, head_base, _ = _inputs(1)
    kwargs = {
        "rms_eps": 1e-3,
        "hc_pre_eps": 1e-2,
        "hc_sinkhorn_eps": 1e-3,
        "hc_post_mult_value": 1.5,
        "sinkhorn_repeat": 3,
    }
    for got, want in zip(
        mhc_pre(residual, fn, scale, base, **kwargs),
        NativeMHCPreOp()(residual, fn, scale, base, **kwargs),
    ):
        assert torch.equal(got, want)

    eps = {"rms_eps": 1e-3, "hc_pre_eps": 1e-2}
    got = mhc_head(residual, head_fn, head_scale, head_base, **eps)
    want = NativeMHCHeadOp()(residual, head_fn, head_scale, head_base, **eps)
    assert torch.equal(got, want)


@pytest.mark.parametrize("platform", ["cuda", "musa", "npu"])
def test_entry_raises_where_there_is_no_backend(monkeypatch, platform):
    residual, fn, scale, base, head_fn, head_scale, head_base, y = _inputs()
    monkeypatch.setattr(kernel_registry, "_platform_for_device", lambda device: platform)

    with pytest.raises(RuntimeError, match="No functional backend found for mhc_pre"):
        mhc_pre(residual, fn, scale, base)
    with pytest.raises(RuntimeError, match="No functional backend found for mhc_post"):
        mhc_post(y, residual, torch.zeros(_M, _N), torch.zeros(_M, _N, _N))
    with pytest.raises(RuntimeError, match="No functional backend found for mhc_head"):
        mhc_head(residual, head_fn, head_scale, head_base)
