# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""SM90 routed-expert backward (profile p5-sm90-bwd-bf16-v1, frozen MXFP4 base).

Accuracy against an FP32 reference that follows the oracle chain on the same
z (BF16 dh, oracle clamp-SwiGLU backward, BF16 dz); byte checks for batch
invariance (a row's gradients do not depend on the other rows) and run to run.
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.moe import oracle
from rl_engine.moe.contract import ExpertBatch
from rl_engine.moe.mx_format import mx_dequantize, mx_quantize


@pytest.fixture(scope="module")
def backend():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("SM90 required")
    from rl_engine.moe.backends.sm90_fused_mlp import Sm90FusedMoeMlp

    try:
        b = Sm90FusedMoeMlp()
    except NotImplementedError as exc:
        pytest.skip(f"SM90 MoE kernels unavailable: {exc}")
    if not b.capabilities()["backward"]:
        pytest.skip("backward kernels not built")
    return b


def _batch(counts, H, F, seed=0):
    from rl_engine.moe.backends.sm90_fused_mlp import PROFILE

    E = len(counts)
    M = sum(counts)
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(M, H, generator=g, device="cuda") * 0.5).bfloat16()
    w1 = torch.randn(E, 2 * F, H, generator=g, device="cuda") / H**0.5
    w2 = torch.randn(E, H, F, generator=g, device="cuda") / F**0.5
    p_s = torch.rand(M, generator=g, device="cuda")
    offsets = torch.tensor([0] + list(torch.tensor(counts).cumsum(0)), dtype=torch.int32)
    batch = ExpertBatch(
        x=x,
        expert_offsets=offsets.cuda(),
        p_s=p_s,
        w1=mx_quantize(w1, "e2m1"),
        w2=mx_quantize(w2, "e2m1"),
        lora=None,
        output_slot=torch.arange(M, dtype=torch.int32, device="cuda"),
        numeric_profile=PROFILE,
    )
    dy = torch.randn(M, H, generator=g, device="cuda").bfloat16()
    return batch, mx_quantize(x, "e4m3"), dy


def _reference(batch, z, dy):
    w1 = mx_dequantize(batch.w1)
    w2 = mx_dequantize(batch.w2)
    offs = batch.expert_offsets.tolist()
    F = batch.ffn
    dx = torch.zeros(batch.rows, batch.hidden, device="cuda")
    dz_all = torch.zeros(batch.rows, 2 * F, device="cuda", dtype=torch.bfloat16)
    dps = torch.zeros(batch.rows, device="cuda")
    for e in range(len(offs) - 1):
        lo, hi = offs[e], offs[e + 1]
        if lo == hi:
            continue
        dh = (dy[lo:hi].float() @ w2[e]).bfloat16()
        _, saved = oracle.clamp_swiglu_weighted_fwd(z[lo:hi, :F], z[lo:hi, F:], batch.p_s[lo:hi])
        dgate, dup, dp = oracle.clamp_swiglu_weighted_bwd(dh, saved)
        dz = torch.cat([dgate, dup], 1).bfloat16()
        dz_all[lo:hi] = dz
        dps[lo:hi] = dp
        dx[lo:hi] = dz.float() @ w1[e]
    return dx, dz_all, dps


def _rel(a, b):
    return ((a.float() - b.float()).abs().max() / (b.float().abs().max() + 1e-30)).item()


@pytest.mark.parametrize(
    "counts,H,F",
    [([64, 0, 37, 130], 1024, 256), ([1, 200], 1024, 512), ([97, 33, 0, 5], 4096, 2048)],
)
def test_backward_close_to_reference(backend, counts, H, F):
    batch, x_q, dy = _batch(counts, H, F)
    dx, dp_s = backend.backward(batch, x_q, dy)
    z = backend.fc1_z(batch, x_q)
    ref_dx, ref_dz, ref_dps = _reference(batch, z, dy)
    assert dx.dtype == torch.float32 and dx.shape == (batch.rows, H)
    assert _rel(dx, ref_dx) < 2e-2, _rel(dx, ref_dx)
    assert _rel(dp_s, ref_dps) < 1e-2, _rel(dp_s, ref_dps)


def test_backward_batch_invariant_and_deterministic(backend):
    from rl_engine.moe.backends.sm90_fused_mlp import PROFILE

    counts, H, F = [70, 0, 129, 3], 1024, 256
    batch, x_q, dy = _batch(counts, H, F, seed=1)
    dx, dp_s = backend.backward(batch, x_q, dy)
    dx2, dp_s2 = backend.backward(batch, x_q, dy)
    assert torch.equal(dx, dx2) and torch.equal(dp_s, dp_s2)
    # Rows 75..80 of expert 2 alone (expert 2's rows start at 70).
    lo, hi = 75, 81
    sub_off = torch.tensor([0, 0, 0, hi - lo, hi - lo], dtype=torch.int32, device="cuda")
    sub = ExpertBatch(
        x=batch.x[lo:hi].contiguous(),
        expert_offsets=sub_off,
        p_s=batch.p_s[lo:hi].contiguous(),
        w1=batch.w1,
        w2=batch.w2,
        lora=None,
        output_slot=torch.arange(hi - lo, dtype=torch.int32, device="cuda"),
        numeric_profile=PROFILE,
    )
    sub_q = mx_quantize(sub.x, "e4m3")
    sdx, sdp = backend.backward(sub, sub_q, dy[lo:hi].contiguous())
    assert torch.equal(sdx, dx[lo:hi]) and torch.equal(sdp, dp_s[lo:hi])


def test_backward_empty(backend):
    batch, x_q, dy = _batch([0, 0], 1024, 256)
    dx, dp_s = backend.backward(batch, x_q, dy)
    assert dx.shape == (0, 1024) and dp_s.shape == (0,)


def test_dequant_t_matches_mx_dequantize(backend):
    batch, _, _ = _batch([5, 0, 9], 1024, 256, seed=2)
    for w in (batch.w1, batch.w2):
        wt = backend._ext.sm90_moe_dequant_t(w.codes, w.scales, batch.expert_offsets)
        ref = mx_dequantize(w).transpose(1, 2).bfloat16()
        assert wt.shape == ref.shape
        for e in (0, 2):  # expert 1 has no rows and is not written
            assert torch.equal(wt[e].view(torch.int16), ref[e].view(torch.int16))


@pytest.mark.parametrize(
    "counts,H,F", [([64, 0, 300, 130], 1024, 256), ([257, 1, 0, 128], 4096, 2048)]
)
def test_backward_preexpand_same_bytes(counts, H, F):
    """The pre-expanded weights and the in-kernel expansion give the same bytes."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("SM90 required")
    from rl_engine.moe.backends.sm90_fused_mlp import Sm90FusedMoeMlp

    try:
        inline, pre = Sm90FusedMoeMlp(preexpand=False), Sm90FusedMoeMlp(preexpand=True)
    except NotImplementedError as exc:
        pytest.skip(f"SM90 MoE kernels unavailable: {exc}")
    batch, x_q, dy = _batch(counts, H, F, seed=3)
    dx_a, dp_a = inline.backward(batch, x_q, dy)
    dx_b, dp_b = pre.backward(batch, x_q, dy)
    assert torch.equal(dx_a.view(torch.int32), dx_b.view(torch.int32))
    assert torch.equal(dp_a.view(torch.int32), dp_b.view(torch.int32))
