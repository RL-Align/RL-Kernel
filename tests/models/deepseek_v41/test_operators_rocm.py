# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch

from rl_engine.backends.rocm.engram.gated_residual import engram_gated_residual
from rl_engine.backends.rocm.attention.v41_compressor_pool import compressor_pool
from rl_engine.models.deepseek_v41.bitwise import compare_bits
from rl_engine.models.deepseek_v41.reference import (
    compressor_pool_reference,
    engram_gated_residual_reference,
)


@pytest.fixture(scope="module", autouse=True)
def require_rocm():
    if not torch.version.hip or not torch.cuda.is_available():
        if os.environ.get("RLK_REQUIRE_ROCM") == "1":
            pytest.fail("ROCm GPU required")
        pytest.skip("ROCm GPU required")


def _run(fn, inputs, grad):
    xs = [x.detach().clone().requires_grad_(True) for x in inputs]
    y = fn(*xs)
    grads = torch.autograd.grad(y, xs, grad)
    return y.detach(), tuple(g.detach() for g in grads)


def _check(fn, ref, inputs, grad, rtol, atol):
    y, gs = _run(fn, inputs, grad)
    ref_y, ref_gs = _run(ref, inputs, grad)
    torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)
    for g, w in zip(gs, ref_gs, strict=True):
        assert torch.isfinite(g).all()
        torch.testing.assert_close(g, w, rtol=rtol, atol=atol)
    y2, gs2 = _run(fn, inputs, grad)
    for a, b in zip((y, *gs), (y2, *gs2), strict=True):
        assert compare_bits(a, b)["mismatched_elements"] == 0
    with torch.inference_mode():
        infer = fn(*inputs)
    assert compare_bits(y, infer)["mismatched_elements"] == 0


@pytest.mark.parametrize("d", [1, 17, 128, 5120])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_engram_gate_outputs_all_gradients_replay_parity(d, dtype):
    torch.manual_seed(91)
    inputs = [torch.randn(2, 3, 4, d, device="cuda", dtype=dtype) for _ in range(2)]
    inputs += [torch.randn(2, 3, d, device="cuda", dtype=dtype)]
    inputs += [torch.randn(4, d, device="cuda", dtype=torch.float32) for _ in range(2)]
    grad = torch.randn_like(inputs[0])
    mask = torch.tensor([[True, False, True], [False, True, True]], device="cuda")

    def fn(*xs):
        return engram_gated_residual(*xs, mask)

    def ref(*xs):
        return engram_gated_residual_reference(*xs, mask)

    _check(
        fn,
        ref,
        inputs,
        grad,
        1e-3 if dtype == torch.float32 else 3e-2,
        3e-5 if dtype == torch.float32 else 3e-2,
    )


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("d", [1, 17, 128, 512])
def test_pool_outputs_gradients_replay_parity(ratio, d):
    torch.manual_seed(11)
    inputs = [torch.randn(2, 3, ratio, d, device="cuda") for _ in range(2)]
    _check(
        compressor_pool,
        compressor_pool_reference,
        inputs,
        torch.randn(2, 3, d, device="cuda"),
        1e-4,
        2e-6,
    )


def test_masked_engram_preserves_residual_bits_and_zero_gate_gradients():
    x = torch.zeros(1, 3, 4, 17, device="cuda", requires_grad=True)
    key = torch.zeros_like(x, requires_grad=True)
    value = torch.ones(1, 3, 17, device="cuda", requires_grad=True)
    qw = torch.ones(4, 17, device="cuda", requires_grad=True)
    kw = torch.ones_like(qw, requires_grad=True)
    mask = torch.zeros(1, 3, device="cuda", dtype=torch.bool)
    y = engram_gated_residual(x, key, value, qw, kw, mask)
    y.sum().backward()
    assert compare_bits(y, x)["status"] == "PASS"
    assert torch.equal(x.grad, torch.ones_like(x))
    for t in (key, value, qw, kw):
        assert torch.equal(t.grad, torch.zeros_like(t))


def test_empty_pool_and_engram_backward():
    kv = torch.empty(0, 2, 512, device="cuda", requires_grad=True)
    score = torch.empty_like(kv, requires_grad=True)
    compressor_pool(kv, score).sum().backward()
    assert kv.grad.numel() == score.grad.numel() == 0
    xs = [torch.empty(0, 4, 128, device="cuda", requires_grad=True) for _ in range(2)]
    xs += [torch.empty(0, 128, device="cuda", requires_grad=True)]
    xs += [torch.ones(4, 128, device="cuda", requires_grad=True) for _ in range(2)]
    engram_gated_residual(*xs).sum().backward()
    assert torch.equal(xs[-1].grad, torch.zeros_like(xs[-1]))


@pytest.mark.parametrize("d", [1, 17, 512, 1280, 5120])
@pytest.mark.parametrize("eps", [1e-20, 1e-6])
def test_rmsnorm_explicit_epsilon_and_gradients(d, eps):
    from rl_engine.backends.rocm.norm.v41_rmsnorm import rmsnorm

    torch.manual_seed(12)
    x = torch.randn(2, 3, d, device="cuda")
    w = torch.randn(d, device="cuda")

    def fn(x, w):
        return rmsnorm(x, w, eps=eps)

    def ref(x, w):
        return (
            (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)) * w.float()
        ).to(x.dtype)

    _check(fn, ref, [x, w], torch.randn_like(x), 3e-4, 3e-5)


def test_rmsnorm_does_not_use_v4_epsilon():
    from rl_engine.backends.rocm.norm.v41_rmsnorm import rmsnorm

    x = torch.full((1, 128), 1e-10, device="cuda")
    w = torch.ones(128, device="cuda")
    actual = rmsnorm(x, w, eps=1e-20)
    v4 = rmsnorm(x, w, eps=1e-6)
    assert actual.min() > 0.7 and v4.max() < 1e-6


@pytest.mark.parametrize("width", [32, 96, 5120])
def test_quantizer_matches_official_ceil_scale_and_ocp_codes(width):
    from rl_engine.backends.rocm.gemm.v41_fp8 import quantize_fp8_block32

    torch.manual_seed(80)
    x = torch.randn(3, width, device="cuda", dtype=torch.bfloat16)
    x[0] = 0
    x[1, :32] = 1.75
    blocks = x.float().reshape(3, -1, 32)
    scale = 2.0 ** torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(1e-4) / 448))
    codes = (blocks / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(x.shape)
    q = quantize_fp8_block32(x)
    assert compare_bits(q.codes, codes.view(torch.uint8))["status"] == "PASS"
    assert (
        compare_bits(q.scales, scale.to(torch.float8_e8m0fnu).view(torch.uint8))["status"] == "PASS"
    )


@pytest.mark.parametrize("n,k", [(17, 32), (64, 96), (1280, 5120)])
def test_block32_projection_correctness_and_frozen_input_ste(n, k):
    from rl_engine.backends.rocm.gemm.v41_fp8 import V41FP8Linear, quantize_fp8_block32

    torch.manual_seed(18)
    w = (torch.randn(n, k, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    scales = (2.0 ** torch.randint(-2, 2, ((n + 31) // 32, k // 32), device="cuda").float()).to(
        torch.float8_e8m0fnu
    )
    mod = V41FP8Linear(w, scales)
    x = torch.randn(3, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn(3, n, device="cuda", dtype=torch.bfloat16)
    y = mod(x)
    dx = torch.autograd.grad(y, x, dy)[0]
    q = quantize_fp8_block32(x.detach())
    a = q.codes.view(torch.float8_e4m3fn).float().reshape(3, -1, 32)
    a = (a * q.scales.view(torch.float8_e8m0fnu).float().unsqueeze(-1)).reshape(3, k)
    sw = scales.float().repeat_interleave(32, 0)[:n].repeat_interleave(32, 1)
    weight = w.float() * sw
    reference = torch.zeros(3, n, device="cuda")
    # Ascending block accumulation is the numerical contract.
    for start in range(0, k, 32):
        reference = reference + a[:, start : start + 32] @ weight[:, start : start + 32].t()
    torch.testing.assert_close(y, reference.bfloat16(), rtol=1.5e-2, atol=0.02)
    torch.testing.assert_close(dx, (dy.float() @ weight).bfloat16(), rtol=2e-2, atol=0.05)
    x2 = x.detach().clone().requires_grad_(True)
    y2 = mod(x2)
    dx2 = torch.autograd.grad(y2, x2, dy)[0]
    assert compare_bits(y, y2)["status"] == compare_bits(dx, dx2)["status"] == "PASS"
    with torch.inference_mode():
        assert compare_bits(y, mod(x.detach()))["status"] == "PASS"


def test_hash_full_prefill_matches_decode_reset_and_scalar_reference():
    from pathlib import Path
    from rl_engine.models.deepseek_v41.config import DeepSeekV41Config
    from rl_engine.backends.rocm.engram.ngram_hash import EngramHash

    c = DeepSeekV41Config.from_json(Path(__file__).parent / "fixtures/config.json")
    # Explicit synthetic token map for an isolated integer hashing test.
    token_map = (
        torch.arange(c.text["vocab_size"], device="cuda") % c.text["engram_compressed_vocab_size"]
    )
    mod = EngramHash(c, token_map, max_batch_size=2, max_seq_len=16)
    ids = torch.tensor([[0, 1, 2, 3, 4, 5], [10, 129279, 4, 8, 14, 20]], device="cuda")
    mask = torch.tensor([[1, 1, 0, 1, 1, 1], [1, 0, 1, 1, 1, 1]], device="cuda", dtype=torch.bool)
    full = mod(ids, token_mask=mask)
    compressed = token_map[ids].cpu().tolist()
    alive = mask.cpu().tolist()
    multipliers = mod.multipliers.cpu().tolist()
    primes = mod.primes.cpu().tolist()
    offsets = mod.offsets.cpu().tolist()
    want = torch.empty_like(full, device="cpu")
    for b in range(2):
        for s in range(6):
            for layer in range(2):
                rolling = 0
                blocked = False
                for shift in range(4):
                    p = s - shift
                    blocked = blocked or p < 0 or not alive[b][max(p, 0)]
                    token = mod.pad if blocked else compressed[b][p]
                    rolling ^= token * multipliers[layer][shift]
                    if shift:
                        for h in range(8):
                            col = (shift - 1) * 8 + h
                            want[b, s, layer, col] = (
                                rolling % primes[layer][col] + offsets[layer][col]
                            )
    assert torch.equal(full.cpu(), want)
    chunks = [mod(ids[:, :2], token_mask=mask[:, :2])]
    for s in range(2, 6):
        chunks.append(mod(ids[:, s : s + 1], s, mask[:, s : s + 1]))
    assert compare_bits(full, torch.cat(chunks, dim=1))["status"] == "PASS"
    assert compare_bits(full, mod(ids, token_mask=mask))["status"] == "PASS"
    with pytest.raises(ValueError, match="contiguous"):
        mod(ids[:, :1], 9)
