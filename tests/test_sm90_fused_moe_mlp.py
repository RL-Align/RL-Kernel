# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""SM90 fused routed-expert MLP (profile p5-sm90-fused-mlp-v1).

Not oracle-aligned: accuracy is checked against the FP32 oracle with a bound
derived from FP8 tensor-core accumulation; byte-level checks cover batch
invariance, run-to-run identity, fused == two-launch, and fail-closed paths.
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.moe import oracle
from rl_engine.moe.contract import ExpertBatch
from rl_engine.moe.mx_format import MXTensor, mx_dequantize, mx_quantize

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module")
def backend():
    from rl_engine.moe.backends.sm90_fused_mlp import Sm90FusedMoeMlp

    try:
        return Sm90FusedMoeMlp()
    except NotImplementedError as exc:
        pytest.skip(f"SM90 fused MoE MLP unavailable: {exc}")


def _batch(M, E, H, F, offsets=None, seed=0, profile=None, device="cuda"):
    from rl_engine.moe.backends.sm90_fused_mlp import PROFILE

    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(M, H, generator=g) * 0.5).to(torch.bfloat16)
    w1 = (torch.randn(E, 2 * F, H, generator=g) / H**0.5).to(torch.bfloat16)
    w2 = (torch.randn(E, H, F, generator=g) / F**0.5).to(torch.bfloat16)
    p_s = torch.rand(M, generator=g)
    if offsets is None:
        offsets = torch.tensor([0] + [M * (i + 1) // E for i in range(E)], dtype=torch.int32)
    batch = ExpertBatch(
        x=x.to(device),
        expert_offsets=offsets.to(device=device, dtype=torch.int32),
        p_s=p_s.to(device),
        w1=mx_quantize(w1, "e2m1").to(device),
        w2=mx_quantize(w2, "e2m1").to(device),
        lora=None,
        output_slot=torch.arange(M, dtype=torch.int32, device=device),
        numeric_profile=profile or PROFILE,
    )
    return batch, mx_quantize(batch.x, "e4m3")


def _torch_ref(batch: ExpertBatch, x_q: MXTensor) -> torch.Tensor:
    """Fast FP32 reference on exactly dequantized operands (same math as the oracle)."""
    a = mx_dequantize(x_q)
    w1 = mx_dequantize(batch.w1)
    w2 = mx_dequantize(batch.w2)
    offs = batch.expert_offsets.tolist()
    F = batch.ffn
    y = torch.zeros(batch.rows, batch.hidden, dtype=torch.float32, device=a.device)
    for e in range(len(offs) - 1):
        lo, hi = offs[e], offs[e + 1]
        if lo == hi:
            continue
        z = a[lo:hi] @ w1[e].t()
        g = torch.clamp(z[:, :F], max=10.0)
        u = torch.clamp(z[:, F:], min=-10.0, max=10.0)
        h = ((g * torch.sigmoid(g)) * u * batch.p_s[lo:hi, None]).to(torch.bfloat16)
        hq = mx_quantize(h, "e4m3")
        y[lo:hi] = mx_dequantize(hq) @ w2[e].t()
    return y.to(torch.bfloat16)


def _rel(a, b):
    return ((a.float() - b.float()).abs().max() / (b.float().abs().max() + 1e-30)).item()


@requires_cuda
def test_fc1_matches_fp32_reference(backend):
    batch, x_q = _batch(777, 4, 4096, 2048, offsets=torch.tensor([0, 100, 100, 700, 777]))
    z = backend.fc1_z(batch, x_q)
    a, w1 = mx_dequantize(x_q), mx_dequantize(batch.w1)
    ref = torch.zeros_like(z)
    for e, (lo, hi) in enumerate(zip(batch.expert_offsets.tolist(), batch.expert_offsets.tolist()[1:])):
        if lo < hi:
            ref[lo:hi] = a[lo:hi] @ w1[e].t()
    assert torch.isfinite(z).all()
    assert _rel(z, ref) < 2e-4  # FP32-accumulate-level agreement (32-term blocks on tensor cores)


@requires_cuda
@pytest.mark.parametrize("path", ["two_launch", "fused"])
def test_forward_close_to_reference(backend, path):
    batch, x_q = _batch(777, 4, 4096, 2048, offsets=torch.tensor([0, 100, 100, 700, 777]))
    y = backend.forward(batch, x_q, path=path)
    ref = _torch_ref(batch, x_q)
    assert y.dtype == torch.bfloat16 and y.shape == (777, 4096)
    # Differences come from FP8 accumulation order feeding the E4M3 re-quantization
    # (about 0.2% of h codes move by one FP8 ulp) plus the BF16 output round.
    assert _rel(y, ref) < 3e-2


@requires_cuda
def test_forward_close_to_p5_oracle(backend):
    """Small shape so the serial FP32 oracle stays fast; same bound as above."""
    batch, x_q = _batch(96, 2, 512, 256, offsets=torch.tensor([0, 40, 96]))
    y_gold, _ = oracle.routed_expert_forward(
        ExpertBatch(**{**batch.__dict__, "numeric_profile": "oracle-fp32-serial-v1"})
    )
    y = backend.forward(batch, x_q, path="two_launch")
    assert _rel(y, y_gold) < 3e-2


@requires_cuda
def test_fused_equals_two_launch_bitwise(backend):
    for shape in [(130, 3, 512, 128, torch.tensor([0, 64, 64, 130])), (777, 4, 4096, 2048, None), (1, 2, 4096, 2048, torch.tensor([0, 0, 1]))]:
        batch, x_q = _batch(*shape[:4], offsets=shape[4])
        assert torch.equal(backend.forward(batch, x_q, "fused"), backend.forward(batch, x_q, "two_launch"))


@requires_cuda
@pytest.mark.parametrize("path", ["two_launch", "fused"])
def test_batch_invariance_across_expert_boundaries(backend, path):
    """fwd(x)[t] == fwd(x[t:t+1]) byte-for-byte, rows on both sides of every boundary."""
    M, E = 777, 4
    batch, x_q = _batch(M, E, 4096, 2048, offsets=torch.tensor([0, 100, 100, 700, 777]))
    y = backend.forward(batch, x_q, path=path)
    for m in (0, 1, 63, 64, 99, 100, 101, 699, 700, 776):
        e = int((batch.expert_offsets[1:] <= m).sum().item())
        one = ExpertBatch(
            **{
                **batch.__dict__,
                "x": batch.x[m : m + 1].contiguous(),
                "p_s": batch.p_s[m : m + 1].contiguous(),
                "expert_offsets": torch.tensor([0] * (e + 1) + [1] * (E - e), dtype=torch.int32, device="cuda"),
                "output_slot": batch.output_slot[m : m + 1].contiguous(),
            }
        )
        xq1 = MXTensor(codes=x_q.codes[m : m + 1].contiguous(), scales=x_q.scales[m : m + 1].contiguous(),
                       elem_format="e4m3", shape=(1, batch.hidden))
        assert torch.equal(backend.forward(one, xq1, path=path)[0], y[m]), f"row {m} ({path}) diverged"


@requires_cuda
def test_run_to_run_identity(backend):
    batch, x_q = _batch(512, 8, 4096, 2048)
    a = backend.forward(batch, x_q)
    b = backend.forward(batch, x_q)
    assert torch.equal(a, b)


@requires_cuda
def test_fail_closed(backend):
    from rl_engine.moe.backends.sm90_fused_mlp import PROFILE

    batch, x_q = _batch(64, 1, 512, 256, profile="oracle-fp32-serial-v1")
    with pytest.raises(NotImplementedError, match="fail-closed"):
        backend.forward(batch, x_q)
    cpu, xq_cpu = _batch(64, 1, 512, 256, device="cpu")
    with pytest.raises(NotImplementedError):
        backend.forward(cpu, xq_cpu)
    # weight columns whose block scales span more than 8 binades are rejected
    bad = batch.w1.scales.clone()
    bad[0, 0, 0] = 200
    bad[0, 0, 1] = 100
    with pytest.raises(RuntimeError, match="binades"):
        backend._ext.sm90_moe_prepare_weight_ref(bad)
    assert PROFILE == backend.numeric_profile
