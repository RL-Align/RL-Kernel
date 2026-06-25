# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Tests for NativeMatmulOp, the PyTorch fp32 GEMM reference."""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.ops.pytorch.linear.matmul import NativeMatmulOp

QWEN3_HIDDEN = 4096
QWEN3_INTERMEDIATE = 12288


def _make_inputs(
    batch: int,
    seq: int,
    k: int,
    n: int,
    *,
    dtype: torch.dtype = torch.float32,
    seed: int = 123,
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(batch, seq, k, generator=gen, dtype=dtype)
    b = torch.randn(k, n, generator=gen, dtype=dtype)
    return a, b


class TestNativeMatmulOpCorrectness:
    def test_output_shape_matches_matmul_contract(self):
        op = NativeMatmulOp()
        a, b = _make_inputs(2, 16, 64, 32)
        out = op.forward_fp32(a, b)
        assert out.shape == (2, 16, 32)

    def test_forward_fp32_returns_fp32(self):
        op = NativeMatmulOp()
        a, b = _make_inputs(2, 16, 64, 32, dtype=torch.bfloat16)
        out = op.forward_fp32(a, b)
        assert out.dtype == torch.float32

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_forward_returns_input_dtype(self, dtype):
        op = NativeMatmulOp()
        a, b = _make_inputs(2, 16, 64, 32, dtype=dtype)
        out = op.forward(a, b)
        assert out.dtype == dtype

    def test_call_equals_forward(self):
        op = NativeMatmulOp()
        a, b = _make_inputs(2, 16, 64, 32)
        assert torch.equal(op(a, b), op.forward(a, b))

    def test_matches_single_fp32_torch_matmul(self):
        op = NativeMatmulOp()
        a, b = _make_inputs(2, 16, 64, 32)
        out = op.forward_fp32(a, b)
        ref = torch.matmul(a.float(), b.float())
        assert torch.equal(out, ref)

    def test_pure_function_no_inplace(self):
        op = NativeMatmulOp()
        a, b = _make_inputs(2, 16, 64, 32)
        a_orig = a.clone()
        b_orig = b.clone()
        _ = op.forward_fp32(a, b)
        assert torch.equal(a, a_orig)
        assert torch.equal(b, b_orig)

    def test_op_class_is_reduction(self):
        assert NativeMatmulOp.op_class == "reduction"


class TestNativeMatmulOpBatchInvariance:
    def test_batch1_vs_batchN_bitwise(self):
        op = NativeMatmulOp()
        a, b = _make_inputs(4, 16, 64, 32, seed=321)
        full_out = op.forward_fp32(a, b)
        for row in range(a.shape[0]):
            single_out = op.forward_fp32(a[row : row + 1], b)
            assert torch.equal(
                full_out[row], single_out[0]
            ), f"Batch invariance broken at row {row}"

    def test_batch_invariance_with_padding(self):
        op = NativeMatmulOp()
        a_valid, b = _make_inputs(2, 16, 64, 32, seed=456)
        gen = torch.Generator().manual_seed(789)
        padding = torch.randn(3, 16, 64, generator=gen)
        a_padded = torch.cat([a_valid, padding], dim=0)
        out_valid = op.forward_fp32(a_valid, b)
        out_padded = op.forward_fp32(a_padded, b)
        assert torch.equal(out_valid[0], out_padded[0])
        assert torch.equal(out_valid[1], out_padded[1])


class TestNativeMatmulOpAccuracy:
    @pytest.mark.parametrize(
        "dtype, atol, rtol",
        [
            (torch.float32, 1e-4, 1e-4),
            (torch.bfloat16, 5e-2, 2e-2),
            (torch.float16, 1e-3, 1e-3),
        ],
    )
    def test_forward_vs_fp32_within_tolerance(self, dtype, atol, rtol):
        op = NativeMatmulOp()
        a, b = _make_inputs(2, 16, 64, 32, dtype=dtype)
        out_typed = op.forward(a, b).float()
        out_fp32 = op.forward_fp32(a, b)
        diff = (out_typed - out_fp32).abs().max().item()
        assert torch.allclose(out_typed, out_fp32, atol=atol, rtol=rtol), (
            f"dtype={dtype}, max_abs_error={diff:.3e} exceeds " f"atol={atol}, rtol={rtol}"
        )


class TestNativeMatmulOpQwen3Shapes:
    @pytest.mark.parametrize(
        "k, n, label",
        [
            (QWEN3_HIDDEN, QWEN3_HIDDEN, "q_proj/o_proj"),
            (QWEN3_HIDDEN, 1024, "k_proj/v_proj"),
            (QWEN3_HIDDEN, QWEN3_INTERMEDIATE, "gate_proj/up_proj"),
            (QWEN3_INTERMEDIATE, QWEN3_HIDDEN, "down_proj"),
        ],
    )
    def test_qwen3_projection_reduction_dims(self, k, n, label):
        del label
        op = NativeMatmulOp()
        a, b = _make_inputs(1, 2, k, n, seed=42)
        out = op.forward_fp32(a, b)
        assert out.shape == (1, 2, n)
        assert out.dtype == torch.float32


class TestNativeMatmulOpRegistry:
    def test_registry_returns_matmul_op(self):
        from rl_engine.kernels.registry import kernel_registry

        op = kernel_registry.get_op("matmul")
        assert isinstance(op, NativeMatmulOp)
