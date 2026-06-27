# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Tests for NativeRoPEOp — fp32 gold standard RoPE (HF rotate-half convention).

Validates:
- Axis A: batch invariance (bitwise torch.equal between batch=1 slice and batch=N slice)
- Axis B: accuracy (forward vs forward_fp32 under tolerance_contract thresholds)
- Functional correctness: pure function, dtype, shape, positions [S] vs [B,S]
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.ops.pytorch.rotary_embedding.rope import NativeRoPEOp

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

QWEN3_HEAD_DIM = 128
QWEN3_THETA = 1_000_000.0


def _make_inputs(batch: int, n_heads: int, seq: int, head_dim: int, seed: int = 42):
    """Deterministic RoPE inputs."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, n_heads, seq, head_dim, generator=gen)
    positions = torch.arange(seq, dtype=torch.long)
    return x, positions


def _hf_rotate_half_reference(x, positions, theta=1e6):
    """Independent HF rotate-half reference (from ISSUE_108_OPS_DEV §5)."""
    D = x.shape[-1]
    half = D // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    freqs = positions.float()[:, None] * inv_freq[None, :]
    cos = torch.cat([freqs.cos()] * 2, dim=-1)
    sin = torch.cat([freqs.sin()] * 2, dim=-1)
    a, b = x.float()[..., :half], x.float()[..., half:]
    rotated = torch.cat([-b, a], dim=-1)
    return x.float() * cos + rotated * sin


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


class TestNativeRoPEOpCorrectness:
    """Basic correctness: shape, dtype, purity, HF reference match."""

    def test_output_shape_matches_input(self):
        op = NativeRoPEOp()
        x, pos = _make_inputs(2, 32, 16, QWEN3_HEAD_DIM)
        out = op.forward_fp32(x, pos, theta=QWEN3_THETA)
        assert out.shape == x.shape

    def test_forward_fp32_returns_fp32(self):
        op = NativeRoPEOp()
        x, pos = _make_inputs(2, 32, 16, QWEN3_HEAD_DIM)
        out = op.forward_fp32(x, pos)
        assert out.dtype == torch.float32

    def test_forward_fp32_returns_fp32_even_with_bf16_input(self):
        op = NativeRoPEOp()
        x, pos = _make_inputs(2, 32, 16, QWEN3_HEAD_DIM)
        out = op.forward_fp32(x.bfloat16(), pos)
        assert out.dtype == torch.float32

    def test_call_equals_forward(self):
        op = NativeRoPEOp()
        x, pos = _make_inputs(2, 32, 16, QWEN3_HEAD_DIM)
        assert torch.equal(op(x, pos, theta=QWEN3_THETA), op.forward(x, pos, theta=QWEN3_THETA))

    def test_pure_function_no_inplace(self):
        op = NativeRoPEOp()
        x, pos = _make_inputs(2, 32, 16, QWEN3_HEAD_DIM)
        x_orig = x.clone()
        _ = op.forward_fp32(x, pos)
        assert torch.equal(x, x_orig), "forward_fp32 modified input in-place"

    def test_matches_hf_reference_bitwise(self):
        """NativeRoPEOp must be bitwise identical to the ISSUE_108_OPS_DEV §5 reference."""
        op = NativeRoPEOp()
        x, pos = _make_inputs(2, 32, 16, QWEN3_HEAD_DIM)
        our = op.forward_fp32(x, pos, theta=QWEN3_THETA)
        ref = _hf_rotate_half_reference(x, pos, theta=QWEN3_THETA)
        assert torch.equal(our, ref), (
            f"Not bitwise match with HF reference, max diff: "
            f"{(our - ref).abs().max().item():.2e}"
        )

    def test_positions_1d_and_2d_equivalent(self):
        """positions [S] and [B, S] (with identical values) must produce same output."""
        op = NativeRoPEOp()
        B, H, S, D = 3, 32, 16, QWEN3_HEAD_DIM
        x, pos_1d = _make_inputs(B, H, S, D)
        pos_2d = pos_1d.unsqueeze(0).expand(B, -1)
        out_1d = op.forward_fp32(x, pos_1d)
        out_2d = op.forward_fp32(x, pos_2d)
        assert torch.equal(out_1d, out_2d)

    def test_op_class_is_elementwise(self):
        assert NativeRoPEOp.op_class == "elementwise"

    def test_theta_affects_output(self):
        """Different theta must produce different results."""
        op = NativeRoPEOp()
        x, pos = _make_inputs(2, 32, 16, QWEN3_HEAD_DIM)
        out_1e6 = op.forward_fp32(x, pos, theta=1_000_000.0)
        out_1e4 = op.forward_fp32(x, pos, theta=10_000.0)
        assert not torch.equal(out_1e6, out_1e4)

    def test_position_zero_is_identity_for_cos(self):
        """At position 0, cos=1 and sin=0, so output should equal input."""
        op = NativeRoPEOp()
        x = torch.randn(1, 1, 1, QWEN3_HEAD_DIM)
        pos = torch.zeros(1, dtype=torch.long)
        out = op.forward_fp32(x, pos)
        # cos(0)=1, sin(0)=0 → out = x*1 + rotate_half(x)*0 = x
        assert torch.allclose(out, x.float(), atol=1e-7)


# ---------------------------------------------------------------------------
# Axis A — Batch invariance (bitwise)
# ---------------------------------------------------------------------------


class TestNativeRoPEOpBatchInvariance:
    """Axis A: forward_fp32 must be bitwise batch-invariant.

    Golden rule from ISSUE_108: compute on full input first, then slice —
    compare against computing on the single-batch slice alone.
    """

    def test_batch1_vs_batchN_bitwise(self):
        op = NativeRoPEOp()
        x, pos = _make_inputs(4, 32, 16, QWEN3_HEAD_DIM, seed=99)
        full_out = op.forward_fp32(x, pos)
        for i in range(x.shape[0]):
            single_out = op.forward_fp32(x[i : i + 1], pos)
            assert torch.equal(full_out[i], single_out[0]), f"Batch invariance broken at row {i}"

    def test_batch_invariance_with_padding(self):
        """Padded batch (extra rows) must not affect valid rows."""
        op = NativeRoPEOp()
        x_valid, pos = _make_inputs(2, 32, 16, QWEN3_HEAD_DIM, seed=77)
        # Pad with garbage
        x_padded = torch.cat([x_valid, torch.randn(3, 32, 16, QWEN3_HEAD_DIM)], dim=0)
        out_valid = op.forward_fp32(x_valid, pos)
        out_padded = op.forward_fp32(x_padded, pos)
        assert torch.equal(out_valid[0], out_padded[0])
        assert torch.equal(out_valid[1], out_padded[1])

    def test_batch_invariance_bf16(self):
        """Axis A must hold for bf16 inputs too (forward_fp32 path)."""
        op = NativeRoPEOp()
        x, pos = _make_inputs(4, 32, 16, QWEN3_HEAD_DIM, seed=55)
        x_bf16 = x.bfloat16()
        full_out = op.forward_fp32(x_bf16, pos)
        single_out = op.forward_fp32(x_bf16[0:1], pos)
        assert torch.equal(full_out[0], single_out[0])

    def test_batch_invariance_positions_2d(self):
        """Axis A with per-batch positions [B, S]."""
        op = NativeRoPEOp()
        B, H, S, D = 3, 32, 16, QWEN3_HEAD_DIM
        x, _ = _make_inputs(B, H, S, D, seed=33)
        # Different position offsets per batch item
        pos_2d = torch.stack([torch.arange(S) + i * 100 for i in range(B)])
        full_out = op.forward_fp32(x, pos_2d)
        for i in range(B):
            single_out = op.forward_fp32(x[i : i + 1], pos_2d[i : i + 1])
            assert torch.equal(
                full_out[i], single_out[0]
            ), f"Batch invariance broken at row {i} with 2D positions"


# ---------------------------------------------------------------------------
# Axis B — Accuracy (forward vs forward_fp32)
# ---------------------------------------------------------------------------


class TestNativeRoPEOpAccuracy:
    """Axis B: forward(input_dtype) vs forward_fp32 under tolerance thresholds.

    RoPE is elementwise → expected tolerance from tolerance_contract.yaml:
        float32:  atol=1e-5
        bfloat16: atol=2e-2
        float16:  atol=1e-3
    """

    @pytest.mark.parametrize(
        "dtype, atol, rtol",
        [
            (torch.float32, 1e-5, 1e-5),
            (torch.bfloat16, 2e-2, 1.6e-2),
            (torch.float16, 1e-3, 1e-3),
        ],
    )
    def test_forward_vs_fp32_within_tolerance(self, dtype, atol, rtol):
        op = NativeRoPEOp()
        x, pos = _make_inputs(2, 32, 16, QWEN3_HEAD_DIM)
        x_typed = x.to(dtype)
        out_typed = op.forward(x_typed, pos).float()
        out_fp32 = op.forward_fp32(x_typed, pos)
        diff = (out_typed - out_fp32).abs().max().item()
        assert torch.allclose(out_typed, out_fp32, atol=atol, rtol=rtol), (
            f"dtype={dtype}, max_abs_error={diff:.3e} exceeds " f"atol={atol}, rtol={rtol}"
        )


# ---------------------------------------------------------------------------
# Qwen3-8B specific shapes
# ---------------------------------------------------------------------------


class TestNativeRoPEOpQwen3Shapes:
    """Verify with Qwen3-8B actual dimensions."""

    @pytest.mark.parametrize(
        "batch, seq, label",
        [(2, 16, "SMALL"), (4, 512, "MEDIUM")],
    )
    def test_qwen3_shape(self, batch, seq, label):
        op = NativeRoPEOp()
        x, pos = _make_inputs(batch, 32, seq, 128, seed=12)
        out = op.forward_fp32(x, pos, theta=1_000_000.0)
        assert out.shape == (batch, 32, seq, 128)
        assert out.dtype == torch.float32

    def test_qwen3_kv_heads_shape(self):
        """RoPE is also applied to K with n_kv_heads=8."""
        op = NativeRoPEOp()
        x_k, pos = _make_inputs(2, 8, 16, 128, seed=13)
        out = op.forward_fp32(x_k, pos, theta=1_000_000.0)
        assert out.shape == (2, 8, 16, 128)
