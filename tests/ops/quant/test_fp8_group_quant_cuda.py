# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CUDA per-token-group FP8 quantization: byte-exact with the PyTorch golden.

Two orthogonal properties: the CUDA kernel emits the golden's bytes (FP8 values
and FP32 scales, including the DeepGEMM scale layout) on edge cases and odd
shapes, and its output for a row does not depend on the batch around it.
"""

from __future__ import annotations

import importlib

import pytest
import torch

from rl_engine.kernels.ops.pytorch.quant.fp8_group_quant import (
    GROUP_SIZE,
    fp8_group_quant,
    fp8_group_quant_tokens,
)


def _kernel_available() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return False
    try:
        ext = importlib.import_module("rl_engine._C")
    except ImportError:
        return False
    return hasattr(ext, "fp8_group_quant_forward")


requires_kernel = pytest.mark.skipif(
    not _kernel_available(), reason="CUDA fp8_group_quant kernel not built"
)
DTYPES = [torch.bfloat16, torch.float16]


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def _assert_same(got, want, label: str) -> None:
    q, s = got
    q_ref, s_ref = want
    assert q.shape == q_ref.shape and q.dtype == q_ref.dtype, label
    assert s.shape == s_ref.shape and s.stride() == s_ref.stride(), label
    diff = (_bytes(q) != _bytes(q_ref)).sum().item()
    assert diff == 0, f"{label}: {diff} FP8 bytes differ"
    assert torch.equal(s.view(torch.int32), s_ref.view(torch.int32)), f"{label}: scales differ"


def _edge_rows(dtype: torch.dtype, cols: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(24, cols, generator=gen)
    x[0] = 0.0  # zero groups: the scale floor
    x[1] = -0.0
    x[2, :GROUP_SIZE] = torch.randn(GROUP_SIZE, generator=gen) * 1.0e-6  # amax below 1e-4
    x[3, 0] = float("nan")
    x[4, 1] = float("inf")
    x[5, 2] = float("-inf")
    x[6] *= 1.0e4  # large values (FP16 still finite)
    x[7, 0] = 448.0
    x[8, 0] = -450.0
    x[9] = torch.randn(cols, generator=gen) * 1.0e-38  # subnormal-scale inputs
    finfo = torch.finfo(dtype)
    x[10, :GROUP_SIZE] = finfo.tiny  # smallest normal of the input dtype
    x[11, :GROUP_SIZE] = finfo.smallest_normal * 2.0**-3  # input-dtype subnormals
    x[12, 0] = finfo.max
    x[13, ::2] = -x[13, ::2].abs()
    return x.to(dtype)


@requires_kernel
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("scale_ue8m0", [True, False])
class TestRowsMatchGolden:
    def test_edge_values(self, dtype, scale_ue8m0):
        from rl_engine.kernels.ops.cuda.quant import fp8_group_quant_cuda

        x = _edge_rows(dtype, 4 * GROUP_SIZE).cuda()
        _assert_same(
            fp8_group_quant_cuda(x, scale_ue8m0=scale_ue8m0),
            fp8_group_quant(x, scale_ue8m0=scale_ue8m0),
            "edge",
        )

    @pytest.mark.parametrize("rows", [1, 3, 4, 5, 31, 32, 33, 300, 4096])
    @pytest.mark.parametrize("groups", [1, 3, 8, 32, 64])
    def test_shapes(self, dtype, scale_ue8m0, rows, groups):
        from rl_engine.kernels.ops.cuda.quant import fp8_group_quant_cuda

        gen = torch.Generator().manual_seed(rows * 131 + groups)
        x = (torch.randn(rows, groups * GROUP_SIZE, generator=gen) * 4).to(dtype).cuda()
        _assert_same(
            fp8_group_quant_cuda(x, scale_ue8m0=scale_ue8m0),
            fp8_group_quant(x, scale_ue8m0=scale_ue8m0),
            f"{rows}x{groups}",
        )


@requires_kernel
@pytest.mark.parametrize("dtype", DTYPES)
class TestTokensMatchGolden:
    @pytest.mark.parametrize("tokens", [1, 15, 16, 127, 128, 129, 1000, 4096])
    @pytest.mark.parametrize("channels", [1, 31, 32, 33, 512])
    def test_shapes(self, dtype, tokens, channels):
        from rl_engine.kernels.ops.cuda.quant import fp8_group_quant_tokens_cuda

        gen = torch.Generator().manual_seed(tokens * 7 + channels)
        x = (torch.randn(tokens, channels, generator=gen) * 4).to(dtype).cuda()
        _assert_same(fp8_group_quant_tokens_cuda(x), fp8_group_quant_tokens(x), "tokens")

    def test_edge_values(self, dtype):
        from rl_engine.kernels.ops.cuda.quant import fp8_group_quant_tokens_cuda

        x = _edge_rows(dtype, 160).t().contiguous().cuda()  # [160 tokens, 24 channels]
        _assert_same(fp8_group_quant_tokens_cuda(x), fp8_group_quant_tokens(x), "tokens edge")


@requires_kernel
class TestBatchInvariance:
    def test_rows_do_not_depend_on_the_batch(self):
        from rl_engine.kernels.ops.cuda.quant import fp8_group_quant_cuda

        gen = torch.Generator().manual_seed(11)
        x = torch.randn(4096, 4096, generator=gen).to(torch.bfloat16).cuda()
        q, s = fp8_group_quant_cuda(x)
        for start, stop in ((0, 1), (1, 2), (127, 130), (1000, 1129), (4095, 4096)):
            q_part, s_part = fp8_group_quant_cuda(x[start:stop].contiguous())
            assert torch.equal(_bytes(q_part), _bytes(q[start:stop]))
            assert torch.equal(s_part, s[start:stop])
        q2, s2 = fp8_group_quant_cuda(x)
        assert torch.equal(_bytes(q2), _bytes(q)) and torch.equal(s2, s)

    def test_unaligned_view_is_handled(self):
        from rl_engine.kernels.ops.cuda.quant import fp8_group_quant_cuda

        base = torch.randn(GROUP_SIZE * 3 + 4, device="cuda").to(torch.bfloat16)
        x = base[1 : 1 + GROUP_SIZE * 3].view(1, -1)  # 2-byte offset
        _assert_same(fp8_group_quant_cuda(x), fp8_group_quant(x), "unaligned")


@requires_kernel
def test_deepgemm_consumes_the_scales_directly():
    deep_gemm = pytest.importorskip("deep_gemm")
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("SM90 required")
    from deep_gemm.utils import per_block_cast_to_fp8

    from rl_engine.kernels.ops.cuda.quant import fp8_group_quant_cuda

    gen = torch.Generator().manual_seed(12)
    x = torch.randn(257, 1024, generator=gen).to(torch.bfloat16).cuda()
    w = torch.randn(512, 1024, generator=gen).to(torch.bfloat16).cuda()
    b = per_block_cast_to_fp8(w, use_ue8m0=True)
    a = fp8_group_quant_cuda(x)
    assert deep_gemm.get_mn_major_tma_aligned_tensor(a[1]).data_ptr() == a[1].data_ptr()
    d = torch.empty(257, 512, device="cuda", dtype=torch.bfloat16)
    deep_gemm.fp8_gemm_nt(a, b, d, disable_ue8m0_cast=True)
    d_ref = torch.empty_like(d)
    deep_gemm.fp8_gemm_nt(fp8_group_quant(x), b, d_ref, disable_ue8m0_cast=True)
    assert torch.equal(d.view(torch.int16), d_ref.view(torch.int16))


@requires_kernel
def test_rejects_unsupported_inputs():
    from rl_engine.kernels.ops.cuda.quant import fp8_group_quant_cuda

    with pytest.raises(TypeError, match="BF16 or FP16"):
        fp8_group_quant_cuda(torch.zeros(1, GROUP_SIZE, device="cuda"))
    with pytest.raises(ValueError, match="multiple of 128"):
        fp8_group_quant_cuda(torch.zeros(1, 100, dtype=torch.bfloat16, device="cuda"))
    with pytest.raises(RuntimeError, match="CUDA tensor"):
        fp8_group_quant_cuda(torch.zeros(1, GROUP_SIZE, dtype=torch.bfloat16))
