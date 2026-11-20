# SPDX-License-Identifier: Apache-2.0
"""CDNA native FP8 selection and real-device codec/gradient regressions."""

from types import SimpleNamespace

import pytest
import torch

from rl_engine.backends.rocm.fp8 import fp8_dtype


@pytest.mark.parametrize(
    "arch,dtype",
    [
        ("gfx942:sramecc+:xnack-", torch.float8_e4m3fnuz),
        ("gfx950:sramecc+:xnack-", torch.float8_e4m3fn),
    ],
)
def test_native_fp8_encoding(monkeypatch, arch, dtype):
    monkeypatch.setattr(torch.version, "hip", "test")
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(gcnArchName=arch)
    )
    assert fp8_dtype(torch.device("cuda", 0)) == dtype


def test_other_architecture_is_not_silently_enabled(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "test")
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(gcnArchName="gfx90a")
    )
    with pytest.raises(ValueError, match="gfx90a"):
        fp8_dtype(torch.device("cuda", 0))


requires_gpu = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm GPU"
)


@requires_gpu
def test_qnorm_native_codes_batch_invariance_and_vjp():
    from aiter.ops.fused_qk_rmsnorm_group_quant import fused_qk_rmsnorm_group_quant
    from rl_engine.backends.rocm.norm.qnorm_fp8 import FusedQNorm, fused_kv_norm

    torch.manual_seed(42)
    original = torch.nn.RMSNorm(1024, eps=1e-6, dtype=torch.bfloat16, device="cuda")
    norm = FusedQNorm(original, 1e-6)
    x = torch.randn(2, 1024, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    y = norm(x)
    codes, scales = norm.quantized_input(y)
    assert codes.dtype == fp8_dtype(x.device)
    # Compare the wrapper boundary with the same production joint Q/K operator
    # used by vLLM, including its FP32 normalization before quantization.
    ref_codes, ref_scales = torch.empty_like(codes), torch.empty_like(scales)
    ref = torch.empty_like(y)
    fused_qk_rmsnorm_group_quant(
        q=x.detach(),
        q_weight=original.weight.detach(),
        q_out_quantized=ref_codes,
        q_out_scale=ref_scales,
        q_out_unquantized=ref,
        group_size=128,
        transpose_scale=False,
    )
    assert torch.equal(codes.view(torch.uint8), ref_codes.view(torch.uint8))
    assert torch.equal(scales, ref_scales)
    assert torch.equal(y, ref)
    for rows in (1, 17):
        norm(x.detach()[:1].expand(rows, -1).contiguous())
        assert torch.equal(norm._activation[1][0].view(torch.uint8), codes[0].view(torch.uint8))
        assert torch.equal(norm._activation[1][-1].view(torch.uint8), codes[0].view(torch.uint8))
        assert torch.equal(norm._activation[2][-1], scales[0])
    grad = torch.randn_like(y)
    y.backward(grad)
    xx = x.detach().cpu().double().requires_grad_()
    ww = original.weight.detach().cpu().double().requires_grad_()
    yy = xx * (xx.square().mean(-1, keepdim=True) + 1e-6).rsqrt() * ww
    yy.backward(grad.cpu().double())
    torch.testing.assert_close(x.grad.cpu().float(), xx.grad.float(), rtol=0.008, atol=0.008)
    torch.testing.assert_close(
        original.weight.grad.cpu().float(), ww.grad.float(), rtol=0.008, atol=0.008
    )
    kv = torch.randn(2, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    kw = torch.ones(512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = fused_kv_norm(kv, kw, x.detach(), original.weight.detach(), 1e-6, 1e-6)
    out.float().square().sum().backward()
    assert bool(torch.isfinite(out).all())
    assert bool(kv.grad.isfinite().all()) and bool(kw.grad.isfinite().all())


@requires_gpu
def test_swa_cache_uses_native_encoding():
    from rl_engine.backends.rocm.attention.qkv_preprocess import _QKV

    torch.manual_seed(43)
    q = torch.randn(3, 1, 64, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    kv = torch.ones(3, 1, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    positions = torch.zeros((3, 1), device="cuda", dtype=torch.int64)
    cache = torch.cat((torch.ones(1, 32), torch.zeros(1, 32)), -1).cuda()
    out, decoded = _QKV.apply(q, kv, positions, cache, 1e-6)
    # Unit values at position zero are exact in both encodings. Decoding an
    # OCP cache as FNUZ halves these values, even though all values are finite.
    torch.testing.assert_close(decoded, kv, rtol=0, atol=0)
    (out.float().sum() + decoded.float().sum()).backward()
    assert bool(q.grad.isfinite().all()) and bool(kv.grad.isfinite().all())
