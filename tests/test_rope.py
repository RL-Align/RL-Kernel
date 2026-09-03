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
from rl_engine.platforms.device import _npu_available

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
        assert torch.equal(
            op(x, pos, theta=QWEN3_THETA), op.forward(x, pos, theta=QWEN3_THETA)
        )

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
            assert torch.equal(
                full_out[i], single_out[0]
            ), f"Batch invariance broken at row {i}"

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

    def test_backward_batch_invariance_bitwise(self):
        """Axis A backward: full-batch x.grad slice must match batch=1 x.grad."""
        op = NativeRoPEOp()
        x, pos = _make_inputs(4, 32, 16, QWEN3_HEAD_DIM, seed=101)
        grad_out = torch.randn(*x.shape, generator=torch.Generator().manual_seed(202))

        full_x = x.clone().requires_grad_(True)
        op.forward_fp32(full_x, pos).backward(grad_out)
        assert full_x.grad is not None

        for i in range(x.shape[0]):
            single_x = x[i : i + 1].clone().requires_grad_(True)
            single_grad_out = grad_out[i : i + 1]
            op.forward_fp32(single_x, pos).backward(single_grad_out)
            assert single_x.grad is not None
            assert torch.equal(
                full_x.grad[i],
                single_x.grad[0],
            ), f"Backward batch invariance broken at row {i}"


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
            f"dtype={dtype}, max_abs_error={diff:.3e} exceeds "
            f"atol={atol}, rtol={rtol}"
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


class TestRoPEPackedPositionReset:
    """C5: packed tokens must keep logical positions, not packed 0..T-1."""

    def test_packed_logical_positions_match_per_sample_rope(self):
        op = NativeRoPEOp()
        heads, dim = 4, QWEN3_HEAD_DIM
        seqs = [torch.randn(heads, length, dim) for length in (3, 5)]
        logical_pos = [torch.tensor([2, 5, 9]), torch.tensor([1, 4, 6, 8, 11])]
        per_sample = [
            op.forward_fp32(seq.unsqueeze(0), pos, theta=QWEN3_THETA).squeeze(0)
            for seq, pos in zip(seqs, logical_pos)
        ]

        packed = torch.cat(seqs, dim=1).unsqueeze(0)
        packed_pos = torch.cat(logical_pos)
        packed_out = op.forward_fp32(packed, packed_pos, theta=QWEN3_THETA).squeeze(0)
        assert torch.equal(packed_out[:, :3], per_sample[0])
        assert torch.equal(packed_out[:, 3:], per_sample[1])

        naive_idx = torch.arange(packed_pos.numel())
        naive_out = op.forward_fp32(packed, naive_idx, theta=QWEN3_THETA).squeeze(0)
        assert not torch.equal(packed_out, naive_out)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="candidate RoPE requires CUDA"
)
class TestCandidateRoPELayouts:
    def _candidates(self):
        from rl_engine.kernels.ops.triton.rotary_embedding.rope import TritonRoPEOp

        ops = [("triton", TritonRoPEOp())]
        try:
            from rl_engine.kernels.ops.cuda.rotary_embedding.rope import RoPESM90Op

            # SM90 kernel may be present in a prebuilt extension on SM86 hosts;
            # only add the candidate when the active device can actually launch it.
            if torch.cuda.get_device_capability(0)[0] == 9:
                ops.append(("cuda-sm90", RoPESM90Op()))
        except RuntimeError:
            pass
        return ops

    def test_positions_1d_and_2d_match_native(self):
        native = NativeRoPEOp()
        x, pos_1d = _make_inputs(3, 8, 16, QWEN3_HEAD_DIM, seed=7)
        pos_2d = torch.stack([pos_1d + i * 17 for i in range(3)])
        x_bf16 = x.bfloat16().cuda()
        pos_1d = pos_1d.cuda()
        pos_2d = pos_2d.cuda()
        gold_1d = native.forward_fp32(x_bf16, pos_1d, theta=QWEN3_THETA)
        gold_2d = native.forward_fp32(x_bf16, pos_2d, theta=QWEN3_THETA)
        for name, op in self._candidates():
            got_1d = op.forward(x_bf16, pos_1d, theta=QWEN3_THETA).float()
            got_2d = op.forward(x_bf16, pos_2d, theta=QWEN3_THETA).float()
            assert torch.allclose(got_1d, gold_1d, atol=2e-2, rtol=1.6e-2), name
            assert torch.allclose(got_2d, gold_2d, atol=2e-2, rtol=1.6e-2), name

    def test_packed_logical_positions_on_candidates(self):
        native = NativeRoPEOp()
        heads, dim = 4, QWEN3_HEAD_DIM
        seqs = [torch.randn(heads, length, dim) for length in (3, 5)]
        logical_pos = [torch.tensor([2, 5, 9]), torch.tensor([1, 4, 6, 8, 11])]
        packed = torch.cat(seqs, dim=1).unsqueeze(0).bfloat16().cuda()
        packed_pos = torch.cat(logical_pos).cuda()
        gold = native.forward_fp32(packed, packed_pos, theta=QWEN3_THETA)
        for name, op in self._candidates():
            got = op.forward(packed, packed_pos, theta=QWEN3_THETA).float()
            assert torch.allclose(got, gold, atol=2e-2, rtol=1.6e-2), name


# ---------------------------------------------------------------------------
# Ascend C candidate
# ---------------------------------------------------------------------------


def _ascend_rope_available() -> bool:
    if not _npu_available():
        return False
    try:
        from rl_engine.kernels.ops.ascend.rotary_embedding.rope import (
            _C_npu,
            _NPU_EXT_AVAILABLE,
        )
    except Exception:
        return False
    return _NPU_EXT_AVAILABLE and hasattr(_C_npu, "rope_apply_ascend")


requires_ascend_rope = pytest.mark.skipif(
    not _ascend_rope_available(),
    reason="rope_apply_ascend is not compiled (requires an Ascend NPU build).",
)


def _make_ascend_inputs(batch: int, heads: int, seq: int, dim: int, dtype: torch.dtype):
    x, positions = _make_inputs(batch, heads, seq, dim, seed=2026)
    return x.to(dtype=dtype, device="npu"), positions.to(device="npu")


def test_ascend_per_batch_table_layout_round_trips_without_an_npu():
    from rl_engine.kernels.ops.ascend.rotary_embedding.rope import (
        _restore_rope,
        _rope_table,
    )

    x = torch.arange(2 * 3 * 4 * 8, dtype=torch.float32).reshape(2, 3, 4, 8)
    positions = torch.stack([torch.arange(4), torch.arange(4) + 17])
    x_2d, cos, sin = _rope_table(x, positions, QWEN3_THETA)

    assert torch.equal(x_2d, x.permute(1, 0, 2, 3).reshape(-1, 8))
    assert cos.shape == sin.shape == (8, 4)
    assert torch.equal(_restore_rope(x_2d, x, positions), x)


def test_ascend_per_batch_table_rejects_incompatible_shapes_without_an_npu():
    from rl_engine.kernels.ops.ascend.rotary_embedding.rope import _rope_table

    with pytest.raises(ValueError, match="incompatible"):
        _rope_table(
            torch.randn(2, 3, 4, 8),
            torch.zeros(3, 4, dtype=torch.long),
            QWEN3_THETA,
        )


@requires_ascend_rope
class TestRoPEAscend:
    def _op(self):
        from rl_engine.kernels.ops.ascend.rotary_embedding.rope import RoPEAscendOp

        return RoPEAscendOp()

    @pytest.mark.parametrize(
        "dtype,atol,rtol",
        [
            (torch.float32, 1e-5, 1e-5),
            (torch.float16, 1e-3, 1e-3),
            (torch.bfloat16, 2e-2, 1.6e-2),
        ],
    )
    def test_forward_matches_fp32_reference(self, dtype, atol, rtol):
        x, positions = _make_ascend_inputs(2, 8, 17, 128, dtype)
        actual = self._op()(x, positions, theta=QWEN3_THETA)
        expected = NativeRoPEOp().forward_fp32(x, positions, theta=QWEN3_THETA)
        assert actual.shape == x.shape
        assert actual.dtype == dtype
        assert torch.allclose(actual.float(), expected, atol=atol, rtol=rtol)

    def test_per_batch_positions_match_reference(self):
        x, positions = _make_ascend_inputs(3, 8, 11, 128, torch.bfloat16)
        positions = torch.stack([positions + batch * 97 for batch in range(x.shape[0])])
        actual = self._op()(x, positions, theta=QWEN3_THETA)
        expected = NativeRoPEOp().forward_fp32(x, positions, theta=QWEN3_THETA)
        assert torch.allclose(actual.float(), expected, atol=2e-2, rtol=1.6e-2)

    def test_batch_invariance_is_bitwise(self):
        x, positions = _make_ascend_inputs(4, 8, 13, 128, torch.bfloat16)
        full = self._op()(x, positions, theta=QWEN3_THETA)
        for batch in range(x.shape[0]):
            single = self._op()(x[batch : batch + 1], positions, theta=QWEN3_THETA)
            assert torch.equal(full[batch], single[0])

    def test_backward_matches_transposed_reference_rotation(self):
        x, positions = _make_ascend_inputs(2, 4, 9, 128, torch.float32)
        grad_out = torch.randn_like(x)

        actual_x = x.detach().clone().requires_grad_(True)
        self._op()(actual_x, positions, theta=QWEN3_THETA).backward(grad_out)

        expected_x = x.detach().clone().requires_grad_(True)
        NativeRoPEOp().forward_fp32(expected_x, positions, theta=QWEN3_THETA).backward(
            grad_out
        )
        assert actual_x.grad is not None
        assert expected_x.grad is not None
        assert torch.allclose(actual_x.grad, expected_x.grad, atol=1e-5, rtol=1e-5)

    def test_empty_batch(self):
        x = torch.empty(0, 8, 0, 128, device="npu", dtype=torch.bfloat16)
        positions = torch.empty(0, device="npu", dtype=torch.long)
        out = self._op()(x, positions)
        assert out.shape == x.shape
        assert out.dtype == x.dtype
