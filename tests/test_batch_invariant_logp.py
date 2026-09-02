# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Tests for batch-invariant selected-logprob (issue #148).

The test suite validates two orthogonal properties:
1. **Correctness** - output matches ``log_softmax + gather`` reference.
2. **Batch-invariance** - the result for a given row is bitwise identical
   regardless of batch size, batch position, padding, or mixed-batch layout.
"""

import pytest
import torch

from rl_engine.kernels.ops.pytorch.loss.batch_invariant_logp import NativeBatchInvariantLogpOp
from rl_engine.kernels.ops.pytorch.loss.logp import NativeLogpOp
from rl_engine.platforms.device import _npu_available, device_ctx

_V = 300

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA device required.",
)

try:
    import triton  # noqa: F401

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

requires_triton_cuda = pytest.mark.skipif(
    not (_HAS_TRITON and torch.cuda.is_available()),
    reason="Triton batch-invariant logp requires CUDA device and Triton.",
)

requires_triton = pytest.mark.skipif(
    not _HAS_TRITON,
    reason="Triton package required.",
)


def _sm90_kernel_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
    except Exception:
        return False
    return (
        _EXT_AVAILABLE
        and hasattr(_C, "batch_invariant_logp_sm90")
        and torch.cuda.get_device_capability()[0] == 9
    )


requires_sm90 = pytest.mark.skipif(
    not _sm90_kernel_available(),
    reason="batch_invariant_logp_sm90 kernel not compiled "
    "(needs KERNEL_ALIGN_FORCE_SM90=1 on an SM90/Hopper device).",
)


def _ascend_kernel_available() -> bool:
    if not _npu_available():
        return False
    try:
        from rl_engine.kernels.ops.ascend.loss.batch_invariant_logp import (
            _NPU_EXT_AVAILABLE,
            _C_npu,
        )
    except Exception:
        return False
    return _NPU_EXT_AVAILABLE and hasattr(_C_npu, "batch_invariant_logp_ascend")


requires_ascend = pytest.mark.skipif(
    not _ascend_kernel_available(),
    reason="batch_invariant_logp_ascend kernel not compiled "
    "(needs KERNEL_ALIGN_FORCE_ASCEND=1 on an Ascend NPU host).",
)

# 16-byte-aligned vocab so the TMA forward runs directly (not the fallback):
# bf16 needs V % 8 == 0, fp32 needs V % 4 == 0.
_VC = 1024


def _reference_logp(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Canonical reference: log_softmax(fp32) + gather."""
    logits_2d = logits.reshape(-1, logits.size(-1)).float()
    target_1d = target_ids.reshape(-1).long()
    log_probs = torch.log_softmax(logits_2d, dim=-1)
    selected = torch.gather(log_probs, dim=-1, index=target_1d.unsqueeze(1)).squeeze(1)
    return selected.reshape(target_ids.shape)


def _make_row(seed: int, vocab: int = _V, device: str = "cpu") -> torch.Tensor:
    """Generate a single deterministic logit row from a seed."""
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(1, vocab, generator=gen, device=device)


# ---------------------------------------------------------------------------
# 1. Correctness tests
# ---------------------------------------------------------------------------


class TestCorrectness:
    """Output must match the canonical ``log_softmax + gather`` reference."""

    def test_matches_reference_basic(self):
        op = NativeBatchInvariantLogpOp()
        logits = torch.randn(8, _V)
        target = torch.randint(0, _V, (8,))
        out = op(logits, target)
        ref = _reference_logp(logits, target)
        assert out.dtype == torch.float32
        assert torch.allclose(out, ref, atol=1e-6)

    def test_matches_native_logp_op(self):
        bi_op = NativeBatchInvariantLogpOp()
        native_op = NativeLogpOp()
        logits = torch.randn(16, _V)
        target = torch.randint(0, _V, (16,))
        out_bi = bi_op(logits, target)
        out_native = native_op.apply_fp32(logits, target)
        assert torch.allclose(out_bi, out_native, atol=1e-6)

    def test_leading_shape_preserved(self):
        op = NativeBatchInvariantLogpOp()
        logits = torch.randn(3, 5, _V)
        target = torch.randint(0, _V, (3, 5))
        out = op(logits, target)
        assert out.shape == (3, 5)
        ref = _reference_logp(logits, target)
        assert torch.allclose(out, ref, atol=1e-6)

    def test_bf16_input_fp32_output(self):
        op = NativeBatchInvariantLogpOp()
        logits = torch.randn(8, _V, dtype=torch.bfloat16)
        target = torch.randint(0, _V, (8,))
        out = op(logits, target)
        assert out.dtype == torch.float32
        ref = _reference_logp(logits.float(), target)
        assert torch.allclose(out, ref, atol=1e-5)

    def test_fp16_input_fp32_output(self):
        op = NativeBatchInvariantLogpOp()
        logits = torch.randn(8, _V, dtype=torch.float16)
        target = torch.randint(0, _V, (8,))
        out = op(logits, target)
        assert out.dtype == torch.float32
        ref = _reference_logp(logits.float(), target)
        assert torch.allclose(out, ref, atol=1e-5)

    def test_single_token(self):
        op = NativeBatchInvariantLogpOp()
        logits = torch.randn(1, _V)
        target = torch.randint(0, _V, (1,))
        out = op(logits, target)
        ref = _reference_logp(logits, target)
        assert torch.allclose(out, ref, atol=1e-6)

    def test_vocab_size_1(self):
        op = NativeBatchInvariantLogpOp()
        logits = torch.randn(4, 1)
        target = torch.zeros(4, dtype=torch.long)
        out = op(logits, target)
        assert torch.allclose(out, torch.zeros(4), atol=1e-6)

    def test_large_vocab(self):
        op = NativeBatchInvariantLogpOp()
        logits = torch.randn(4, 128256)
        target = torch.randint(0, 128256, (4,))
        out = op(logits, target)
        ref = _reference_logp(logits, target)
        assert torch.allclose(out, ref, atol=1e-5)


# ---------------------------------------------------------------------------
# 2. Batch-invariance sweep tests - the core of issue #148
# ---------------------------------------------------------------------------


class TestBatchInvariance:
    """Same row must produce bitwise-identical output regardless of batch context."""

    def _get_row_result_in_batch(self, row_logits, row_target, batch_size, position):
        """Embed ``row_logits`` at *position* in a random batch of *batch_size*
        and return the selected logprob for that row."""
        op = NativeBatchInvariantLogpOp()
        vocab = row_logits.size(-1)
        batch_logits = torch.randn(batch_size, vocab)
        batch_target = torch.randint(0, vocab, (batch_size,))
        batch_logits[position] = row_logits.squeeze(0)
        batch_target[position] = row_target.squeeze(0)
        out = op(batch_logits, batch_target)
        return out[position]

    def test_batch_size_1_vs_n(self):
        """Same row in batch=1 vs batch=N must be bitwise equal."""
        op = NativeBatchInvariantLogpOp()
        row = _make_row(42)
        target = torch.tensor([7])

        result_alone = op(row, target).item()

        for batch_size in [2, 4, 8, 16, 32, 64, 128]:
            result_in_batch = self._get_row_result_in_batch(
                row, target, batch_size, position=0
            ).item()
            assert result_alone == result_in_batch, (
                f"Drift at batch_size={batch_size}: "
                f"alone={result_alone}, in_batch={result_in_batch}"
            )

    def test_different_positions_in_batch(self):
        """Same row at different positions in the same batch must be bitwise equal."""
        row = _make_row(99)
        target = torch.tensor([13])

        batch_size = 16
        results = []
        for pos in range(batch_size):
            val = self._get_row_result_in_batch(row, target, batch_size, pos).item()
            results.append(val)

        assert all(
            r == results[0] for r in results
        ), f"Position-dependent drift detected: unique values = {set(results)}"

    def test_mixed_batch_content(self):
        """Changing *other* rows in the batch must not affect our row's result."""
        op = NativeBatchInvariantLogpOp()
        row = _make_row(77)
        target = torch.tensor([25])

        batch_size = 8
        results = []
        for trial_seed in range(20):
            torch.manual_seed(trial_seed * 1000)
            batch_logits = torch.randn(batch_size, _V)
            batch_target = torch.randint(0, _V, (batch_size,))
            batch_logits[3] = row.squeeze(0)
            batch_target[3] = target.squeeze(0)
            out = op(batch_logits, batch_target)
            results.append(out[3].item())

        assert all(
            r == results[0] for r in results
        ), f"Mixed-batch drift: unique values = {set(results)}"

    def test_padding_layout_invariance(self):
        """Left-padding vs right-padding must not affect real rows."""
        op = NativeBatchInvariantLogpOp()
        row = _make_row(55)
        target = torch.tensor([42])

        pad_logits = torch.zeros(1, _V)
        pad_target = torch.tensor([0])

        batch_left = torch.cat([pad_logits, pad_logits, row], dim=0)
        target_left = torch.cat([pad_target, pad_target, target], dim=0)

        batch_right = torch.cat([row, pad_logits, pad_logits], dim=0)
        target_right = torch.cat([target, pad_target, pad_target], dim=0)

        out_left = op(batch_left, target_left)
        out_right = op(batch_right, target_right)

        assert out_left[2].item() == out_right[0].item(), "Padding layout changed the result"

    def test_repeated_runs_deterministic(self):
        """Same input repeated N times must produce bitwise-identical output."""
        op = NativeBatchInvariantLogpOp()
        logits = torch.randn(16, _V)
        target = torch.randint(0, _V, (16,))

        results = [op(logits, target) for _ in range(50)]
        for i, r in enumerate(results[1:], 1):
            assert torch.equal(r, results[0]), f"Run {i} differs from run 0"

    def test_batch_invariance_with_ignore_index(self):
        """Ignored positions must not affect other rows and must output 0.0."""
        op = NativeBatchInvariantLogpOp()
        row = _make_row(33)
        target_val = 10

        batch_a = torch.cat([row, torch.randn(3, _V)], dim=0)
        target_a = torch.tensor([target_val, 5, 8, 2])
        out_a = op(batch_a, target_a)

        target_b = torch.tensor([target_val, -100, -100, -100])
        out_b = op(batch_a, target_b)

        assert out_a[0].item() == out_b[0].item(), "ignore_index on other rows changed row 0"
        assert out_b[1].item() == 0.0
        assert out_b[2].item() == 0.0
        assert out_b[3].item() == 0.0


# ---------------------------------------------------------------------------
# 3. Shape / validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_rejects_1d_logits(self):
        op = NativeBatchInvariantLogpOp()
        with pytest.raises(ValueError, match="at least 2-D"):
            op(torch.randn(10), torch.tensor([0]))

    def test_rejects_shape_mismatch(self):
        op = NativeBatchInvariantLogpOp()
        with pytest.raises(ValueError, match="must match"):
            op(torch.randn(4, _V), torch.randint(0, _V, (5,)))

    def test_rejects_negative_target(self):
        op = NativeBatchInvariantLogpOp()
        logits = torch.randn(4, _V)
        target = torch.tensor([0, -1, 2, 3])
        with pytest.raises(ValueError, match="outside"):
            op(logits, target)

    def test_rejects_target_ge_vocab(self):
        op = NativeBatchInvariantLogpOp()
        logits = torch.randn(4, _V)
        target = torch.tensor([0, 1, _V, 3])
        with pytest.raises(ValueError, match="outside"):
            op(logits, target)

    def test_negative_target_with_ignore_index_ok(self):
        op = NativeBatchInvariantLogpOp()
        logits = torch.randn(4, _V)
        target = torch.tensor([0, -100, 2, 3])
        out = op(logits, target)
        assert out[1].item() == 0.0

    def test_3d_logits(self):
        op = NativeBatchInvariantLogpOp()
        logits = torch.randn(2, 3, _V)
        target = torch.randint(0, _V, (2, 3))
        out = op(logits, target)
        assert out.shape == (2, 3)
        ref = _reference_logp(logits, target)
        assert torch.allclose(out, ref, atol=1e-6)


def test_ascend_backward_formula_matches_reference_without_full_onehot():
    from rl_engine.kernels.ops.ascend.loss.batch_invariant_logp import (
        _batch_invariant_logp_backward,
    )

    logits = torch.randn(4, 17)
    original_logits = logits.clone()
    target = torch.tensor([0, -100, 8, 16])
    valid = target != -100
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    lse = torch.logsumexp(logits, dim=1)
    grad_output = torch.randn(4)

    actual = _batch_invariant_logp_backward(logits, target, lse, grad_output, -100)

    reference_logits = logits.clone().requires_grad_(True)
    selected = reference_logits.gather(1, safe_target.unsqueeze(1)).squeeze(1)
    reference = selected - torch.logsumexp(reference_logits, dim=1)
    reference = torch.where(valid, reference, torch.zeros_like(reference))
    (reference * grad_output).sum().backward()

    assert torch.equal(logits, original_logits)
    assert torch.allclose(actual, reference_logits.grad, atol=1e-6)


def test_ascend_backward_formula_accepts_empty_batch():
    from rl_engine.kernels.ops.ascend.loss.batch_invariant_logp import (
        _batch_invariant_logp_backward,
    )

    grad = _batch_invariant_logp_backward(
        torch.empty(0, 17),
        torch.empty(0, dtype=torch.long),
        torch.empty(0),
        torch.empty(0),
        -100,
    )

    assert grad.shape == (0, 17)


# ---------------------------------------------------------------------------
# 4. Backward / gradient tests
# ---------------------------------------------------------------------------


class TestBackward:
    """Gradient must match the reference log_softmax + gather backward."""

    def test_backward_matches_reference(self):
        op = NativeBatchInvariantLogpOp()
        logits = torch.randn(4, _V, requires_grad=True)
        target = torch.randint(0, _V, (4,))

        out = op(logits, target).sum()
        out.backward()
        grad = logits.grad.detach().clone()

        ref_logits = logits.detach().clone().requires_grad_(True)
        ref = _reference_logp(ref_logits, target).sum()
        ref.backward()

        assert torch.allclose(grad, ref_logits.grad, atol=1e-6)

    def test_gradient_batch_invariance(self):
        """Same row's gradient must be bitwise equal in batch=1 vs batch=N."""
        op = NativeBatchInvariantLogpOp()
        row = _make_row(42)
        target = torch.tensor([7])

        logits_alone = row.clone().requires_grad_(True)
        op(logits_alone, target).sum().backward()
        grad_alone = logits_alone.grad.detach().clone()

        for batch_size in [4, 16, 64]:
            batch_logits = torch.randn(batch_size, _V)
            batch_logits[0] = row.squeeze(0)
            batch_logits.requires_grad_(True)
            batch_target = torch.randint(0, _V, (batch_size,))
            batch_target[0] = target.squeeze(0)
            op(batch_logits, batch_target).sum().backward()
            grad_in_batch = batch_logits.grad[0:1].detach().clone()
            assert torch.equal(
                grad_alone, grad_in_batch
            ), f"Gradient drift at batch_size={batch_size}"


# ---------------------------------------------------------------------------
# 5. Edge cases: all-ignore and custom ignore_index
# ---------------------------------------------------------------------------


class TestIgnoreEdgeCases:
    def test_all_ignore_index_outputs_zero(self):
        op = NativeBatchInvariantLogpOp()
        logits = torch.randn(4, _V)
        target = torch.full((4,), -100)
        out = op(logits, target)
        assert torch.equal(out, torch.zeros_like(out))

    def test_custom_ignore_index(self):
        op = NativeBatchInvariantLogpOp()
        logits = torch.randn(4, _V)
        target = torch.tensor([0, -1, 2, 3])
        out = op(logits, target, ignore_index=-1)
        assert out[1].item() == 0.0
        valid_idx = [0, 2, 3]
        ref = _reference_logp(logits[valid_idx], target[valid_idx])
        assert torch.allclose(out[valid_idx], ref, atol=1e-6)


# ---------------------------------------------------------------------------
# 6. CUDA tests - same logic on GPU
# ---------------------------------------------------------------------------


@requires_cuda
class TestCUDACorrectness:
    """Correctness on CUDA device."""

    def test_matches_reference_cuda(self):
        op = NativeBatchInvariantLogpOp()
        logits = torch.randn(8, _V, device="cuda")
        target = torch.randint(0, _V, (8,), device="cuda")
        out = op(logits, target)
        ref = _reference_logp(logits, target)
        assert out.device.type == "cuda"
        assert out.dtype == torch.float32
        assert torch.allclose(out, ref, atol=1e-6)

    def test_bf16_cuda(self):
        op = NativeBatchInvariantLogpOp()
        logits = torch.randn(8, _V, device="cuda", dtype=torch.bfloat16)
        target = torch.randint(0, _V, (8,), device="cuda")
        out = op(logits, target)
        assert out.dtype == torch.float32
        ref = _reference_logp(logits.float(), target)
        assert torch.allclose(out, ref, atol=1e-5)

    def test_large_vocab_cuda(self):
        op = NativeBatchInvariantLogpOp()
        logits = torch.randn(4, 128256, device="cuda")
        target = torch.randint(0, 128256, (4,), device="cuda")
        out = op(logits, target)
        ref = _reference_logp(logits, target)
        assert torch.allclose(out, ref, atol=1e-5)


@requires_cuda
class TestCUDABatchInvariance:
    """Batch-invariance on CUDA - the most important GPU validation."""

    def test_batch_size_1_vs_n_cuda(self):
        op = NativeBatchInvariantLogpOp()
        row = _make_row(42, device="cuda")
        target = torch.tensor([7], device="cuda")
        result_alone = op(row, target).item()

        for batch_size in [2, 4, 8, 16, 32, 64, 128]:
            batch_logits = torch.randn(batch_size, _V, device="cuda")
            batch_target = torch.randint(0, _V, (batch_size,), device="cuda")
            batch_logits[0] = row.squeeze(0)
            batch_target[0] = target.squeeze(0)
            result_in_batch = op(batch_logits, batch_target)[0].item()
            assert result_alone == result_in_batch, (
                f"CUDA drift at batch_size={batch_size}: "
                f"alone={result_alone}, in_batch={result_in_batch}"
            )

    def test_different_positions_cuda(self):
        op = NativeBatchInvariantLogpOp()
        row = _make_row(99, device="cuda")
        target = torch.tensor([13], device="cuda")
        batch_size = 16
        results = []
        for pos in range(batch_size):
            batch_logits = torch.randn(batch_size, _V, device="cuda")
            batch_target = torch.randint(0, _V, (batch_size,), device="cuda")
            batch_logits[pos] = row.squeeze(0)
            batch_target[pos] = target.squeeze(0)
            results.append(op(batch_logits, batch_target)[pos].item())
        assert all(
            r == results[0] for r in results
        ), f"CUDA position drift: unique = {set(results)}"

    def test_repeated_runs_cuda(self):
        op = NativeBatchInvariantLogpOp()
        logits = torch.randn(16, _V, device="cuda")
        target = torch.randint(0, _V, (16,), device="cuda")
        results = [op(logits, target) for _ in range(50)]
        for i, r in enumerate(results[1:], 1):
            assert torch.equal(r, results[0]), f"CUDA run {i} differs from run 0"

    def test_cpu_gpu_cross_check(self):
        """Same input on CPU vs CUDA should match within tolerance."""
        op = NativeBatchInvariantLogpOp()
        logits_cpu = torch.randn(8, _V)
        target_cpu = torch.randint(0, _V, (8,))
        out_cpu = op(logits_cpu, target_cpu)
        out_cuda = op(logits_cpu.cuda(), target_cpu.cuda())
        assert torch.allclose(
            out_cpu, out_cuda.cpu(), atol=1e-6, rtol=1e-6
        ), "CPU vs CUDA result mismatch"


# ---------------------------------------------------------------------------
# 7. Triton backend tests
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Correctness: Triton vs reference
# ---------------------------------------------------------------------------


@requires_triton_cuda
class TestTritonCorrectness:
    """Triton kernel output must match log_softmax + gather reference."""

    def _get_op(self):
        from rl_engine.kernels.ops.triton.loss.batch_invariant_logp import (
            TritonBatchInvariantLogpOp,
        )

        return TritonBatchInvariantLogpOp()

    def test_matches_reference_fp32(self):
        op = self._get_op()
        logits = torch.randn(8, _V, device="cuda")
        target = torch.randint(0, _V, (8,), device="cuda")
        out = op(logits, target)
        ref = _reference_logp(logits, target)
        assert out.dtype == torch.float32
        assert torch.allclose(out, ref, atol=1e-5)

    def test_matches_reference_bf16(self):
        op = self._get_op()
        logits = torch.randn(8, _V, device="cuda", dtype=torch.bfloat16)
        target = torch.randint(0, _V, (8,), device="cuda")
        out = op(logits, target)
        ref = _reference_logp(logits.float(), target)
        assert out.dtype == torch.float32
        assert torch.allclose(out, ref, atol=1e-4)

    def test_matches_reference_fp16(self):
        op = self._get_op()
        logits = torch.randn(8, _V, device="cuda", dtype=torch.float16)
        target = torch.randint(0, _V, (8,), device="cuda")
        out = op(logits, target)
        ref = _reference_logp(logits.float(), target)
        assert out.dtype == torch.float32
        assert torch.allclose(out, ref, atol=1e-4)

    def test_large_vocab(self):
        op = self._get_op()
        logits = torch.randn(4, 128256, device="cuda")
        target = torch.randint(0, 128256, (4,), device="cuda")
        out = op(logits, target)
        ref = _reference_logp(logits, target)
        assert torch.allclose(out, ref, atol=1e-5)

    def test_single_token(self):
        op = self._get_op()
        logits = torch.randn(1, _V, device="cuda")
        target = torch.randint(0, _V, (1,), device="cuda")
        out = op(logits, target)
        ref = _reference_logp(logits, target)
        assert torch.allclose(out, ref, atol=1e-5)

    def test_3d_logits(self):
        op = self._get_op()
        logits = torch.randn(2, 3, _V, device="cuda")
        target = torch.randint(0, _V, (2, 3), device="cuda")
        out = op(logits, target)
        assert out.shape == (2, 3)
        ref = _reference_logp(logits, target)
        assert torch.allclose(out, ref, atol=1e-5)

    def test_matches_pytorch_op(self):
        """Triton and PyTorch ops should agree within tolerance."""
        from rl_engine.kernels.ops.pytorch.loss.batch_invariant_logp import (
            NativeBatchInvariantLogpOp,
        )

        triton_op = self._get_op()
        pytorch_op = NativeBatchInvariantLogpOp()
        logits = torch.randn(16, _V, device="cuda")
        target = torch.randint(0, _V, (16,), device="cuda")
        out_triton = triton_op(logits, target)
        out_pytorch = pytorch_op(logits, target)
        assert torch.allclose(out_triton, out_pytorch, atol=1e-5)


# ---------------------------------------------------------------------------
# Batch-invariance on GPU via Triton
# ---------------------------------------------------------------------------


@requires_triton_cuda
class TestTritonBatchInvariance:
    """Triton kernel must be bitwise batch-invariant."""

    def _get_op(self):
        from rl_engine.kernels.ops.triton.loss.batch_invariant_logp import (
            TritonBatchInvariantLogpOp,
        )

        return TritonBatchInvariantLogpOp()

    def test_batch_size_1_vs_n(self):
        op = self._get_op()
        row = _make_row(42, device="cuda")
        target = torch.tensor([7], device="cuda")
        result_alone = op(row, target).item()

        for batch_size in [2, 4, 8, 16, 32, 64, 128]:
            batch_logits = torch.randn(batch_size, _V, device="cuda")
            batch_target = torch.randint(0, _V, (batch_size,), device="cuda")
            batch_logits[0] = row.squeeze(0)
            batch_target[0] = target.squeeze(0)
            result_in_batch = op(batch_logits, batch_target)[0].item()
            assert result_alone == result_in_batch, (
                f"Triton drift at batch_size={batch_size}: "
                f"alone={result_alone}, in_batch={result_in_batch}"
            )

    def test_different_positions(self):
        op = self._get_op()
        row = _make_row(99, device="cuda")
        target = torch.tensor([13], device="cuda")
        batch_size = 16
        results = []
        for pos in range(batch_size):
            batch_logits = torch.randn(batch_size, _V, device="cuda")
            batch_target = torch.randint(0, _V, (batch_size,), device="cuda")
            batch_logits[pos] = row.squeeze(0)
            batch_target[pos] = target.squeeze(0)
            results.append(op(batch_logits, batch_target)[pos].item())
        assert all(
            r == results[0] for r in results
        ), f"Triton position drift: unique = {set(results)}"

    def test_repeated_runs(self):
        op = self._get_op()
        logits = torch.randn(16, _V, device="cuda")
        target = torch.randint(0, _V, (16,), device="cuda")
        results = [op(logits, target) for _ in range(50)]
        for i, r in enumerate(results[1:], 1):
            assert torch.equal(r, results[0]), f"Triton run {i} differs from run 0"

    def test_mixed_batch_content(self):
        op = self._get_op()
        row = _make_row(77, device="cuda")
        target = torch.tensor([25], device="cuda")
        batch_size = 8
        results = []
        for trial_seed in range(20):
            torch.manual_seed(trial_seed * 1000)
            batch_logits = torch.randn(batch_size, _V, device="cuda")
            batch_target = torch.randint(0, _V, (batch_size,), device="cuda")
            batch_logits[3] = row.squeeze(0)
            batch_target[3] = target.squeeze(0)
            results.append(op(batch_logits, batch_target)[3].item())
        assert all(
            r == results[0] for r in results
        ), f"Triton mixed-batch drift: unique = {set(results)}"


# ---------------------------------------------------------------------------
# Backward / gradient
# ---------------------------------------------------------------------------


@requires_triton_cuda
class TestTritonBackward:
    """Gradient through the Triton op must match reference."""

    def _get_op(self):
        from rl_engine.kernels.ops.triton.loss.batch_invariant_logp import (
            TritonBatchInvariantLogpOp,
        )

        return TritonBatchInvariantLogpOp()

    def test_backward_matches_reference(self):
        op = self._get_op()
        logits = torch.randn(4, _V, device="cuda", requires_grad=True)
        target = torch.randint(0, _V, (4,), device="cuda")

        out = op(logits, target).sum()
        out.backward()
        grad = logits.grad.detach().clone()

        ref_logits = logits.detach().clone().requires_grad_(True)
        ref = _reference_logp(ref_logits, target).sum()
        ref.backward()

        assert torch.allclose(grad, ref_logits.grad, atol=1e-5)

    def test_gradient_batch_invariance(self):
        op = self._get_op()
        row = _make_row(42, device="cuda")
        target = torch.tensor([7], device="cuda")

        logits_alone = row.clone().requires_grad_(True)
        op(logits_alone, target).sum().backward()
        grad_alone = logits_alone.grad.detach().clone()

        for batch_size in [4, 16, 64]:
            batch_logits = torch.randn(batch_size, _V, device="cuda")
            batch_logits[0] = row.squeeze(0)
            batch_logits.requires_grad_(True)
            batch_target = torch.randint(0, _V, (batch_size,), device="cuda")
            batch_target[0] = target.squeeze(0)
            op(batch_logits, batch_target).sum().backward()
            grad_in_batch = batch_logits.grad[0:1].detach().clone()
            assert torch.allclose(
                grad_alone, grad_in_batch, atol=1e-5
            ), f"Triton gradient drift at batch_size={batch_size}"

    def test_ignored_row_grad_is_zero(self):
        """Ignored rows must have zero gradient across the entire vocab."""
        op = self._get_op()
        logits = torch.randn(4, _V, device="cuda", requires_grad=True)
        target = torch.tensor([0, -100, 2, -100], device="cuda")
        op(logits, target).sum().backward()
        assert torch.equal(logits.grad[1], torch.zeros(_V, device="cuda"))
        assert torch.equal(logits.grad[3], torch.zeros(_V, device="cuda"))

    def test_backward_bf16_input(self):
        """Backward with bf16 logits should match fp32 reference within tolerance."""
        op = self._get_op()
        logits = torch.randn(8, _V, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        target = torch.randint(0, _V, (8,), device="cuda")

        op(logits, target).sum().backward()
        grad = logits.grad.detach().clone()

        ref_logits = logits.detach().float().requires_grad_(True)
        _reference_logp(ref_logits, target).sum().backward()

        assert torch.allclose(grad.float(), ref_logits.grad, atol=1e-2)

    def test_backward_fp16_input(self):
        """Backward with fp16 logits should match fp32 reference within tolerance."""
        op = self._get_op()
        logits = torch.randn(8, _V, device="cuda", dtype=torch.float16, requires_grad=True)
        target = torch.randint(0, _V, (8,), device="cuda")

        op(logits, target).sum().backward()
        grad = logits.grad.detach().clone()

        ref_logits = logits.detach().float().requires_grad_(True)
        _reference_logp(ref_logits, target).sum().backward()

        assert torch.allclose(grad.float(), ref_logits.grad, atol=1e-2)


# ---------------------------------------------------------------------------
# ignore_index handling
# ---------------------------------------------------------------------------


@requires_triton_cuda
class TestTritonIgnoreIndex:
    def _get_op(self):
        from rl_engine.kernels.ops.triton.loss.batch_invariant_logp import (
            TritonBatchInvariantLogpOp,
        )

        return TritonBatchInvariantLogpOp()

    def test_ignore_outputs_zero(self):
        op = self._get_op()
        logits = torch.randn(4, _V, device="cuda")
        target = torch.tensor([0, -100, 2, -100], device="cuda")
        out = op(logits, target)
        assert out[1].item() == 0.0
        assert out[3].item() == 0.0
        ref = _reference_logp(logits[[0, 2]], target[[0, 2]])
        assert torch.allclose(out[[0, 2]], ref, atol=1e-5)

    def test_all_ignore(self):
        op = self._get_op()
        logits = torch.randn(4, _V, device="cuda")
        target = torch.full((4,), -100, device="cuda")
        out = op(logits, target)
        assert torch.equal(out, torch.zeros(4, device="cuda"))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@requires_triton
class TestTritonCPUValidation:
    """Tests that only need Triton importable, not a GPU."""

    def _get_op(self):
        from rl_engine.kernels.ops.triton.loss.batch_invariant_logp import (
            TritonBatchInvariantLogpOp,
        )

        return TritonBatchInvariantLogpOp()

    def test_rejects_cpu_tensor(self):
        op = self._get_op()
        with pytest.raises(RuntimeError, match="requires a GPU"):
            op(torch.randn(4, _V), torch.randint(0, _V, (4,)))


@requires_triton_cuda
class TestTritonValidation:
    def _get_op(self):
        from rl_engine.kernels.ops.triton.loss.batch_invariant_logp import (
            TritonBatchInvariantLogpOp,
        )

        return TritonBatchInvariantLogpOp()

    def test_rejects_1d_logits(self):
        op = self._get_op()
        with pytest.raises(ValueError, match="at least 2-D"):
            op(torch.randn(10, device="cuda"), torch.tensor([0], device="cuda"))

    def test_rejects_invalid_target(self):
        op = self._get_op()
        logits = torch.randn(4, _V, device="cuda")
        target = torch.tensor([0, -1, 2, 3], device="cuda")
        with pytest.raises(ValueError, match="outside"):
            op(logits, target, validate=True)


# ---------------------------------------------------------------------------
# 7b. CUDA SM90 TMA kernel backend
# ---------------------------------------------------------------------------


@requires_sm90
class TestCudaSM90Correctness:
    """Compiled SM90 TMA kernel output must match log_softmax + gather."""

    def _get_op(self):
        from rl_engine.kernels.ops.cuda.loss.batch_invariant_logp import BatchInvariantLogpSM90Op

        return BatchInvariantLogpSM90Op()

    def test_matches_reference_fp32(self):
        op = self._get_op()
        logits = torch.randn(8, _VC, device="cuda")
        target = torch.randint(0, _VC, (8,), device="cuda")
        out = op(logits, target)
        ref = _reference_logp(logits, target)
        assert out.dtype == torch.float32
        assert torch.allclose(out, ref, atol=1e-4)

    def test_matches_reference_bf16(self):
        op = self._get_op()
        logits = torch.randn(8, _VC, device="cuda", dtype=torch.bfloat16)
        target = torch.randint(0, _VC, (8,), device="cuda")
        out = op(logits, target)
        ref = _reference_logp(logits.float(), target)
        assert out.dtype == torch.float32
        assert torch.allclose(out, ref, atol=1e-3)

    def test_large_vocab(self):
        op = self._get_op()
        logits = torch.randn(4, 128256, device="cuda", dtype=torch.bfloat16)
        target = torch.randint(0, 128256, (4,), device="cuda")
        out = op(logits, target)
        ref = _reference_logp(logits.float(), target)
        assert torch.allclose(out, ref, atol=2e-3)

    def test_unaligned_vocab(self):
        # V is 16-byte aligned (TMA row stride) but not a TMA box multiple,
        # so the kernel's global-read tail path runs. 50257 is *not* aligned
        # (V*2 % 16 != 0) and is rejected before launch.
        op = self._get_op()
        vocab = 50264
        logits = torch.randn(8, vocab, device="cuda", dtype=torch.bfloat16)
        target = torch.randint(0, vocab, (8,), device="cuda")
        out = op(logits, target)
        ref = _reference_logp(logits.float(), target)
        assert torch.allclose(out, ref, atol=2e-3)

    def test_unaligned_row_stride_is_rejected(self):
        op = self._get_op()
        logits = torch.randn(8, 50257, device="cuda", dtype=torch.bfloat16)
        target = torch.randint(0, 50257, (8,), device="cuda")
        with pytest.raises(RuntimeError, match="16-byte-aligned"):
            op(logits, target)

    def test_single_token(self):
        op = self._get_op()
        logits = torch.randn(1, _VC, device="cuda")
        target = torch.randint(0, _VC, (1,), device="cuda")
        out = op(logits, target)
        ref = _reference_logp(logits, target)
        assert torch.allclose(out, ref, atol=1e-4)

    def test_3d_logits(self):
        op = self._get_op()
        logits = torch.randn(2, 3, _VC, device="cuda", dtype=torch.bfloat16)
        target = torch.randint(0, _VC, (2, 3), device="cuda")
        out = op(logits, target)
        assert out.shape == (2, 3)
        ref = _reference_logp(logits.float(), target)
        assert torch.allclose(out, ref, atol=1e-3)

    def test_matches_pytorch_op(self):
        op = self._get_op()
        pytorch_op = NativeBatchInvariantLogpOp()
        logits = torch.randn(16, _VC, device="cuda")
        target = torch.randint(0, _VC, (16,), device="cuda")
        assert torch.allclose(op(logits, target), pytorch_op(logits, target), atol=1e-4)


@requires_sm90
class TestCudaSM90BatchInvariance:
    """SM90 kernel must be bitwise batch-invariant (one CTA per row)."""

    def _get_op(self):
        from rl_engine.kernels.ops.cuda.loss.batch_invariant_logp import BatchInvariantLogpSM90Op

        return BatchInvariantLogpSM90Op()

    def test_batch_size_1_vs_n(self):
        op = self._get_op()
        row = _make_row(42, vocab=_VC, device="cuda")
        target = torch.tensor([7], device="cuda")
        result_alone = op(row, target).item()

        for batch_size in [2, 4, 8, 16, 32, 64, 128]:
            batch_logits = torch.randn(batch_size, _VC, device="cuda")
            batch_target = torch.randint(0, _VC, (batch_size,), device="cuda")
            batch_logits[0] = row.squeeze(0)
            batch_target[0] = target.squeeze(0)
            result_in_batch = op(batch_logits, batch_target)[0].item()
            assert result_alone == result_in_batch, (
                f"SM90 drift at batch_size={batch_size}: "
                f"alone={result_alone}, in_batch={result_in_batch}"
            )

    def test_different_positions(self):
        op = self._get_op()
        row = _make_row(99, vocab=_VC, device="cuda")
        target = torch.tensor([13], device="cuda")
        batch_size = 16
        results = []
        for pos in range(batch_size):
            batch_logits = torch.randn(batch_size, _VC, device="cuda")
            batch_target = torch.randint(0, _VC, (batch_size,), device="cuda")
            batch_logits[pos] = row.squeeze(0)
            batch_target[pos] = target.squeeze(0)
            results.append(op(batch_logits, batch_target)[pos].item())
        assert all(
            r == results[0] for r in results
        ), f"SM90 position drift: unique = {set(results)}"

    def test_repeated_runs(self):
        op = self._get_op()
        logits = torch.randn(16, _VC, device="cuda", dtype=torch.bfloat16)
        target = torch.randint(0, _VC, (16,), device="cuda")
        results = [op(logits, target) for _ in range(50)]
        for i, r in enumerate(results[1:], 1):
            assert torch.equal(r, results[0]), f"SM90 run {i} differs from run 0"


@requires_sm90
class TestCudaSM90Backward:
    """Gradient through the SM90 op must match reference."""

    def _get_op(self):
        from rl_engine.kernels.ops.cuda.loss.batch_invariant_logp import BatchInvariantLogpSM90Op

        return BatchInvariantLogpSM90Op()

    def test_backward_matches_reference(self):
        op = self._get_op()
        logits = torch.randn(4, _VC, device="cuda", requires_grad=True)
        target = torch.randint(0, _VC, (4,), device="cuda")
        op(logits, target).sum().backward()
        grad = logits.grad.detach().clone()

        ref_logits = logits.detach().clone().requires_grad_(True)
        _reference_logp(ref_logits, target).sum().backward()
        assert torch.allclose(grad, ref_logits.grad, atol=1e-4)

    def test_ignored_row_grad_is_zero(self):
        op = self._get_op()
        logits = torch.randn(4, _VC, device="cuda", requires_grad=True)
        target = torch.tensor([0, -100, 2, -100], device="cuda")
        op(logits, target).sum().backward()
        assert torch.equal(logits.grad[1], torch.zeros(_VC, device="cuda"))
        assert torch.equal(logits.grad[3], torch.zeros(_VC, device="cuda"))


@requires_sm90
class TestCudaSM90IgnoreIndex:
    def _get_op(self):
        from rl_engine.kernels.ops.cuda.loss.batch_invariant_logp import BatchInvariantLogpSM90Op

        return BatchInvariantLogpSM90Op()

    def test_ignore_outputs_zero(self):
        op = self._get_op()
        logits = torch.randn(4, _VC, device="cuda")
        target = torch.tensor([0, -100, 2, -100], device="cuda")
        out = op(logits, target)
        assert out[1].item() == 0.0
        assert out[3].item() == 0.0
        ref = _reference_logp(logits[[0, 2]], target[[0, 2]])
        assert torch.allclose(out[[0, 2]], ref, atol=1e-4)


@requires_sm90
class TestCudaSM90UnsupportedInputs:
    """WS1 SM90 logp does not silently fall back to Triton or Native."""

    def _get_op(self):
        from rl_engine.kernels.ops.cuda.loss.batch_invariant_logp import BatchInvariantLogpSM90Op

        return BatchInvariantLogpSM90Op()

    def test_fp16_is_rejected(self):
        op = self._get_op()
        logits = torch.randn(8, _VC, device="cuda", dtype=torch.float16)
        target = torch.randint(0, _VC, (8,), device="cuda")
        with pytest.raises(RuntimeError, match="fallback is forbidden"):
            op(logits, target)


# ---------------------------------------------------------------------------
# 7c. Ascend C kernel backend
# ---------------------------------------------------------------------------


@requires_ascend
class TestAscendCorrectness:
    """Compiled Ascend C kernel output must match log_softmax + gather."""

    def _get_op(self):
        from rl_engine.kernels.ops.ascend.loss.batch_invariant_logp import (
            BatchInvariantLogpAscendOp,
        )

        return BatchInvariantLogpAscendOp()

    def test_matches_reference_fp32(self):
        op = self._get_op()
        logits = torch.randn(8, _VC, device="npu")
        target = torch.randint(0, _VC, (8,), device="npu")
        out = op(logits, target)
        ref = _reference_logp(logits, target)
        assert out.dtype == torch.float32
        assert torch.allclose(out, ref, atol=1e-4)

    def test_matches_reference_bf16(self):
        op = self._get_op()
        logits = torch.randn(8, _VC, device="npu", dtype=torch.bfloat16)
        target = torch.randint(0, _VC, (8,), device="npu")
        out = op(logits, target)
        ref = _reference_logp(logits.float(), target)
        assert out.dtype == torch.float32
        assert torch.allclose(out, ref, atol=1e-3)

    def test_large_vocab(self):
        op = self._get_op()
        logits = torch.randn(4, 128256, device="npu", dtype=torch.bfloat16)
        target = torch.randint(0, 128256, (4,), device="npu")
        out = op(logits, target)
        ref = _reference_logp(logits.float(), target)
        assert torch.allclose(out, ref, atol=2e-3)

    def test_single_token(self):
        op = self._get_op()
        logits = torch.randn(1, _VC, device="npu")
        target = torch.randint(0, _VC, (1,), device="npu")
        out = op(logits, target)
        ref = _reference_logp(logits, target)
        assert torch.allclose(out, ref, atol=1e-4)

    def test_3d_logits(self):
        op = self._get_op()
        logits = torch.randn(2, 3, _VC, device="npu", dtype=torch.bfloat16)
        target = torch.randint(0, _VC, (2, 3), device="npu")
        out = op(logits, target)
        assert out.shape == (2, 3)
        ref = _reference_logp(logits.float(), target)
        assert torch.allclose(out, ref, atol=1e-3)

    def test_matches_pytorch_op(self):
        op = self._get_op()
        pytorch_op = NativeBatchInvariantLogpOp()
        logits = torch.randn(16, _VC, device="npu")
        target = torch.randint(0, _VC, (16,), device="npu")
        assert torch.allclose(op(logits, target), pytorch_op(logits, target), atol=1e-4)

    def test_empty_batch(self):
        op = self._get_op()
        logits = torch.empty(0, _VC, device="npu", requires_grad=True)
        target = torch.empty(0, device="npu", dtype=torch.long)
        out = op(logits, target)
        assert out.shape == (0,)
        assert out.dtype == torch.float32
        out.sum().backward()
        assert logits.grad.shape == logits.shape


@requires_ascend
class TestAscendBatchInvariance:
    """Ascend kernel must be bitwise batch-invariant (one block per row)."""

    def _get_op(self):
        from rl_engine.kernels.ops.ascend.loss.batch_invariant_logp import (
            BatchInvariantLogpAscendOp,
        )

        return BatchInvariantLogpAscendOp()

    def test_batch_size_1_vs_n(self):
        op = self._get_op()
        row = _make_row(42, vocab=_VC, device="npu")
        target = torch.tensor([7], device="npu")
        result_alone = op(row, target).item()

        for batch_size in [2, 4, 8, 16, 32, 64, 128]:
            batch_logits = torch.randn(batch_size, _VC, device="npu")
            batch_target = torch.randint(0, _VC, (batch_size,), device="npu")
            batch_logits[0] = row.squeeze(0)
            batch_target[0] = target.squeeze(0)
            result_in_batch = op(batch_logits, batch_target)[0].item()
            assert result_alone == result_in_batch, (
                f"Ascend drift at batch_size={batch_size}: "
                f"alone={result_alone}, in_batch={result_in_batch}"
            )

    def test_different_positions(self):
        op = self._get_op()
        row = _make_row(99, vocab=_VC, device="npu")
        target = torch.tensor([13], device="npu")
        batch_size = 16
        results = []
        for pos in range(batch_size):
            batch_logits = torch.randn(batch_size, _VC, device="npu")
            batch_target = torch.randint(0, _VC, (batch_size,), device="npu")
            batch_logits[pos] = row.squeeze(0)
            batch_target[pos] = target.squeeze(0)
            results.append(op(batch_logits, batch_target)[pos].item())
        assert all(
            r == results[0] for r in results
        ), f"Ascend position drift: unique = {set(results)}"

    def test_repeated_runs(self):
        op = self._get_op()
        logits = torch.randn(16, _VC, device="npu", dtype=torch.bfloat16)
        target = torch.randint(0, _VC, (16,), device="npu")
        results = [op(logits, target) for _ in range(50)]
        for i, r in enumerate(results[1:], 1):
            assert torch.equal(r, results[0]), f"Ascend run {i} differs from run 0"


@requires_ascend
class TestAscendBackward:
    """Gradient through the Ascend op must match reference."""

    def _get_op(self):
        from rl_engine.kernels.ops.ascend.loss.batch_invariant_logp import (
            BatchInvariantLogpAscendOp,
        )

        return BatchInvariantLogpAscendOp()

    def test_backward_matches_reference(self):
        op = self._get_op()
        logits = torch.randn(4, _VC, device="npu", requires_grad=True)
        target = torch.randint(0, _VC, (4,), device="npu")
        op(logits, target).sum().backward()
        grad = logits.grad.detach().clone()

        ref_logits = logits.detach().clone().requires_grad_(True)
        _reference_logp(ref_logits, target).sum().backward()
        assert torch.allclose(grad, ref_logits.grad, atol=1e-4)

    def test_ignored_row_grad_is_zero(self):
        op = self._get_op()
        logits = torch.randn(4, _VC, device="npu", requires_grad=True)
        target = torch.tensor([0, -100, 2, -100], device="npu")
        op(logits, target).sum().backward()
        assert torch.equal(logits.grad[1], torch.zeros(_VC, device="npu"))
        assert torch.equal(logits.grad[3], torch.zeros(_VC, device="npu"))


@requires_ascend
class TestAscendIgnoreIndex:
    def _get_op(self):
        from rl_engine.kernels.ops.ascend.loss.batch_invariant_logp import (
            BatchInvariantLogpAscendOp,
        )

        return BatchInvariantLogpAscendOp()

    def test_ignore_outputs_zero(self):
        op = self._get_op()
        logits = torch.randn(4, _VC, device="npu")
        target = torch.tensor([0, -100, 2, -100], device="npu")
        out = op(logits, target)
        assert out[1].item() == 0.0
        assert out[3].item() == 0.0
        ref = _reference_logp(logits[[0, 2]], target[[0, 2]])
        assert torch.allclose(out[[0, 2]], ref, atol=1e-4)


@requires_ascend
class TestAscendValidation:
    def test_rejects_invalid_target_when_requested(self):
        from rl_engine.kernels.ops.ascend.loss.batch_invariant_logp import (
            BatchInvariantLogpAscendOp,
        )

        op = BatchInvariantLogpAscendOp()
        logits = torch.randn(4, _VC, device="npu")
        target = torch.tensor([0, -1, 2, 3], device="npu")
        with pytest.raises(ValueError, match="outside"):
            op(logits, target, validate=True)


@requires_ascend
class TestAscendFallback:
    """Inputs the Ascend path can't take must silently fall back and stay correct."""

    def _get_op(self):
        from rl_engine.kernels.ops.ascend.loss.batch_invariant_logp import (
            BatchInvariantLogpAscendOp,
        )

        return BatchInvariantLogpAscendOp()

    def test_fp16_falls_back(self):
        op = self._get_op()
        logits = torch.randn(8, _VC, device="npu", dtype=torch.float16)
        target = torch.randint(0, _VC, (8,), device="npu")
        out = op(logits, target)
        ref = _reference_logp(logits.float(), target)
        assert torch.allclose(out, ref, atol=1e-3)


# ---------------------------------------------------------------------------
# 8. Registry dispatch test
# ---------------------------------------------------------------------------


def test_registry_dispatches_correctly():
    from rl_engine.kernels.registry import kernel_registry

    op = kernel_registry.get_op("batch_invariant_logp")
    assert (
        isinstance(op, NativeBatchInvariantLogpOp)
        or type(op).__name__ == "TritonBatchInvariantLogpOp"
        or type(op).__name__ == "BatchInvariantLogpSM90Op"
        or type(op).__name__ == "BatchInvariantLogpAscendOp"
    )
    logits = torch.randn(4, _V, device=device_ctx.device)
    target = torch.randint(0, _V, (4,), device=logits.device)
    out = op(logits, target)
    ref = _reference_logp(logits, target)
    assert torch.allclose(out, ref, atol=1e-6)


def test_benchmark_selects_accelerated_op_for_active_device(monkeypatch):
    from benchmarks import benchmark_batch_invariant_logp as benchmark

    cuda_op = object()
    ascend_op = object()
    monkeypatch.setattr(benchmark, "_maybe_sm90_op", lambda: cuda_op)
    monkeypatch.setattr(benchmark, "_maybe_ascend_op", lambda: ascend_op)

    monkeypatch.setattr(benchmark.device_ctx, "device_type", "npu")
    assert benchmark._maybe_accelerated_op() == ("ascend", ascend_op)

    monkeypatch.setattr(benchmark.device_ctx, "device_type", "cuda")
    assert benchmark._maybe_accelerated_op() == ("cuda", cuda_op)
