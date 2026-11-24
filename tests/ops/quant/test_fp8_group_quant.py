# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Golden semantics of the DSv4 per-token-group FP8 activation quantization.

Two orthogonal properties: the golden reproduces vLLM's
``per_token_group_fp8_quant`` (group 128, eps 1e-10, UE8M0 scales) on
hand-built cases, and it is row-local, hence batch-invariant.
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.ops.pytorch.quant.fp8_group_quant import (
    FP8_MAX,
    GROUP_SIZE,
    fp8_group_quant,
    fp8_group_quant_tokens,
    tma_aligned_rows,
    ue8m0_ceil,
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def _block(values: list[float], dtype=torch.bfloat16) -> torch.Tensor:
    row = torch.zeros(GROUP_SIZE, dtype=torch.float32)
    row[: len(values)] = torch.tensor(values)
    return row.to(dtype).unsqueeze(0)


def _reference_scale(absmax: float) -> float:
    """vLLM's formula with exp2/ceil/log2 (exact for BF16/FP16 inputs)."""
    y = max(max(absmax, 1e-10) / FP8_MAX, 1e-10)
    y32 = torch.tensor(y, dtype=torch.float32)
    return float(torch.exp2(torch.ceil(torch.log2(y32))))


@pytest.mark.parametrize("device", DEVICES)
class TestSemantics:
    def test_scale_is_power_of_two_ceiling(self, device):
        for absmax in (448.0, 450.0, 1.0, 3.0, 1.0e-3, 1.0e-5, 6.0e4):
            x = _block([absmax, -absmax / 3]).to(device)
            _, s = fp8_group_quant(x)
            want = _reference_scale(float(x.float().abs().max()))
            assert s.item() == want, absmax

    def test_exact_power_of_two_scale_is_kept(self, device):
        _, s = fp8_group_quant(_block([448.0]).to(device))
        assert s.item() == 1.0
        _, s = fp8_group_quant(_block([450.0]).to(device))  # BF16 has no 449
        assert s.item() == 2.0

    def test_zero_block_uses_the_scale_floor(self, device):
        q, s = fp8_group_quant(torch.zeros(1, GROUP_SIZE, dtype=torch.bfloat16, device=device))
        assert s.item() == 2.0**-33  # ceil(log2(1e-10))
        assert not _bytes(q).any()

    def test_tiny_block_is_not_clamped_to_1e_minus_4(self, device):
        # DeepGEMM's helper clamps amax to 1e-4; vLLM keeps the real amax.
        x = _block([1.0e-6, -5.0e-7]).to(device)
        q, s = fp8_group_quant(x)
        assert s.item() == _reference_scale(float(x.float().abs().max()))
        assert q.float()[0, 0].item() != 0.0

    def test_values_dequantize_within_half_ulp(self, device):
        gen = torch.Generator().manual_seed(0)
        x = (torch.randn(8, 4 * GROUP_SIZE, generator=gen) * 3).to(torch.bfloat16).to(device)
        q, s = fp8_group_quant(x)
        deq = q.float().view(8, 4, GROUP_SIZE) * s.unsqueeze(2)
        err = (deq - x.float().view(8, 4, GROUP_SIZE)).abs()
        # E4M3 keeps 3 mantissa bits: relative half-ULP is 1/16 of the value,
        # and the smallest subnormal step is 2^-9 of the scale.
        bound = x.float().view(8, 4, GROUP_SIZE).abs() / 16 + s.unsqueeze(2) * 2.0**-10
        assert bool((err <= bound).all())

    def test_nan_and_inf_follow_vllm(self, device):
        x = _block([float("nan"), 2.0, -3.0]).to(device)
        q, s = fp8_group_quant(x)
        assert s.item() == _reference_scale(3.0)  # NaN does not count toward amax
        assert q.float()[0, 0].item() == -FP8_MAX  # fmax(NaN, -448) = -448
        x = _block([float("inf"), 2.0]).to(device)
        q, s = fp8_group_quant(x)
        assert s.item() == float("inf")
        assert q.float()[0, 0].item() == -FP8_MAX  # inf / inf = NaN -> -448
        assert q.float()[0, 1].item() == 0.0

    def test_scale_layout_is_tma_aligned_column_major(self, device):
        for rows in (1, 3, 4, 5, 129):
            x = torch.randn(rows, 3 * GROUP_SIZE, device=device).to(torch.bfloat16)
            q, s = fp8_group_quant(x)
            assert q.shape == x.shape and q.dtype == torch.float8_e4m3fn and q.is_contiguous()
            assert s.shape == (rows, 3) and s.dtype == torch.float32
            assert s.stride() == (1, tma_aligned_rows(rows))

    def test_rows_are_independent(self, device):
        gen = torch.Generator().manual_seed(1)
        x = torch.randn(300, 2 * GROUP_SIZE, generator=gen).to(torch.bfloat16).to(device)
        q, s = fp8_group_quant(x)
        for start, stop in ((0, 1), (7, 8), (128, 257), (299, 300)):
            q_part, s_part = fp8_group_quant(x[start:stop].contiguous())
            assert torch.equal(_bytes(q_part), _bytes(q[start:stop]))
            assert torch.equal(s_part, s[start:stop])

    def test_fp16_input(self, device):
        x = torch.randn(4, GROUP_SIZE, device=device).to(torch.float16)
        q, s = fp8_group_quant(x)
        assert torch.equal(_bytes(q), _bytes(fp8_group_quant(x.clone())[0]))
        assert s.dtype == torch.float32

    def test_rejects_unsupported_inputs(self, device):
        with pytest.raises(TypeError, match="BF16 or FP16"):
            fp8_group_quant(torch.zeros(1, GROUP_SIZE, device=device))
        with pytest.raises(ValueError, match="multiple of 128"):
            fp8_group_quant(torch.zeros(1, 100, dtype=torch.bfloat16, device=device))
        with pytest.raises(ValueError, match="2-D"):
            fp8_group_quant(torch.zeros(GROUP_SIZE, dtype=torch.bfloat16, device=device))


@pytest.mark.parametrize("device", DEVICES)
class TestTokenDirection:
    def test_matches_row_quant_of_the_padded_transpose(self, device):
        gen = torch.Generator().manual_seed(2)
        for tokens in (1, 128, 300):
            x = torch.randn(tokens, 64, generator=gen).to(torch.bfloat16).to(device)
            q, s = fp8_group_quant_tokens(x)
            padded = -(-tokens // GROUP_SIZE) * GROUP_SIZE
            xt = torch.zeros(64, padded, dtype=torch.bfloat16, device=device)
            xt[:, :tokens] = x.t()
            q_ref, s_ref = fp8_group_quant(xt)
            assert q.shape == (64, padded)
            assert torch.equal(_bytes(q), _bytes(q_ref)) and torch.equal(s, s_ref)
            assert not _bytes(q[:, tokens:]).any()


def test_ue8m0_ceil_matches_exp2_ceil_log2_for_bf16_and_fp16_amax():
    for dtype in (torch.bfloat16, torch.float16):
        values = torch.arange(0, 2**16, dtype=torch.int32).to(torch.int16).view(dtype).float()
        values = values[torch.isfinite(values) & (values > 0)]
        y = torch.clamp_min(torch.clamp_min(values, 1e-10) / FP8_MAX, 1e-10)
        assert torch.equal(ue8m0_ceil(y), torch.exp2(torch.ceil(torch.log2(y))))


@pytest.mark.parametrize("scale_ue8m0", [True, False])
def test_cpu_and_cuda_golden_agree(scale_ue8m0):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    gen = torch.Generator().manual_seed(3)
    x = (torch.randn(512, 4 * GROUP_SIZE, generator=gen) * 10).to(torch.bfloat16)
    x[3, 5] = float("nan")
    q_cpu, s_cpu = fp8_group_quant(x, scale_ue8m0=scale_ue8m0)
    q_gpu, s_gpu = fp8_group_quant(x.cuda(), scale_ue8m0=scale_ue8m0)
    assert torch.equal(_bytes(q_cpu), _bytes(q_gpu.cpu()))
    assert torch.equal(s_cpu, s_gpu.cpu())


def test_matches_deepgemm_helper_when_amax_is_not_tiny():
    """DeepGEMM's helper differs only for blocks with amax < 1e-4."""
    deep_gemm = pytest.importorskip("deep_gemm")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    gen = torch.Generator().manual_seed(4)
    x = (torch.randn(257, 8 * GROUP_SIZE, generator=gen) + 0.01).to(torch.bfloat16).cuda()
    q, s = fp8_group_quant(x)
    q_dg, s_dg = deep_gemm.utils.per_token_cast_to_fp8(x, use_ue8m0=True)
    assert torch.equal(_bytes(q), _bytes(q_dg))
    assert torch.equal(s, s_dg)
    # The scale tensor is already in the layout DeepGEMM wants.
    laid_out = deep_gemm.get_mn_major_tma_aligned_tensor(s)
    assert laid_out.stride() == s.stride() and torch.equal(laid_out, s)
