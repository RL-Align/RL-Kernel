# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Tests for fused logp CUDA backward support."""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.testing.reference_ops import selected_logprobs_reference

pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available() and _EXT_AVAILABLE and hasattr(_C, "fused_logp_backward_indexed")
    ),
    reason="requires CUDA and the compiled fused logp backward extension",
)


def _make_inputs(
    rows: int,
    vocab: int,
    *,
    dtype: torch.dtype = torch.float32,
    seed: int = 123,
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randn(rows, vocab, generator=gen, dtype=dtype).cuda()
    token_ids = torch.randint(0, vocab, (rows,), generator=gen, dtype=torch.long).cuda()
    return logits, token_ids


def _reference_grad(
    logits: torch.Tensor, token_ids: torch.Tensor, grad_out: torch.Tensor
) -> torch.Tensor:
    ref_logits = logits.detach().clone().requires_grad_(True)
    (selected_logprobs_reference(ref_logits, token_ids) * grad_out.float()).sum().backward()
    assert ref_logits.grad is not None
    return ref_logits.grad


def _make_row_indices(rows: int, *, seed: int = 321) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randperm(rows, generator=gen)[: rows // 2].cuda()


def _mask_rows(grad: torch.Tensor, row_indices: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros(grad.size(0), dtype=torch.bool, device=grad.device)
    mask[row_indices] = True
    return grad * mask.unsqueeze(1)


_GRAD_TOLERANCES = [
    (torch.float32, 1e-5),
    (torch.bfloat16, 2e-2),
    (torch.float16, 5e-3),
]


class TestFusedLogpForwardWithLse:
    def test_logp_matches_reference(self):
        logits, token_ids = _make_inputs(8, 257)
        logp, _, _ = _C.fused_logp_forward_with_lse(logits, token_ids)
        ref = selected_logprobs_reference(logits, token_ids)
        assert logp.dtype == torch.float32
        assert torch.allclose(logp, ref, atol=1e-5, rtol=1e-5)

    def test_lse_components_match_reference(self):
        # Keep LSE decomposed so a large row_max cannot round away log_sum in float32.
        logits, token_ids = _make_inputs(8, 257)
        _, row_max, log_sum = _C.fused_logp_forward_with_lse(logits, token_ids)
        assert row_max.dtype == torch.float32
        assert log_sum.dtype == torch.float32
        assert torch.equal(row_max, logits.float().amax(dim=-1))
        ref_lse = torch.logsumexp(logits.float(), dim=-1)
        assert torch.allclose(row_max + log_sum, ref_lse, atol=1e-5, rtol=1e-5)

    def test_matches_existing_fp32_forward_bitwise(self):
        logits, token_ids = _make_inputs(8, 257)
        logp, _, _ = _C.fused_logp_forward_with_lse(logits, token_ids)
        legacy = _C.fused_logp_forward_fp32(logits, token_ids)
        assert torch.equal(logp, legacy)


class TestFusedLogpVariantForwardWithLse:
    def test_indexed_matches_legacy_bitwise(self):
        logits, token_ids = _make_inputs(8, 257)
        row_indices = _make_row_indices(8)
        logp, _, _ = _C.fused_logp_forward_indexed_with_lse(logits, token_ids, row_indices)
        legacy = _C.fused_logp_forward_indexed_fp32(logits, token_ids, row_indices)
        assert torch.equal(logp, legacy)

    def test_indexed_unselected_rows_are_zero(self):
        logits, token_ids = _make_inputs(8, 257)
        row_indices = _make_row_indices(8)
        logp, row_max, log_sum = _C.fused_logp_forward_indexed_with_lse(
            logits, token_ids, row_indices
        )
        selected = torch.zeros(8, dtype=torch.bool, device=logp.device)
        selected[row_indices] = True
        assert torch.equal(logp[~selected], torch.zeros_like(logp[~selected]))
        assert torch.equal(row_max[~selected], torch.zeros_like(row_max[~selected]))
        assert torch.equal(log_sum[~selected], torch.zeros_like(log_sum[~selected]))
        ref_lse = torch.logsumexp(logits.float(), dim=-1)
        lse = row_max + log_sum
        assert torch.allclose(lse[selected], ref_lse[selected], atol=1e-5, rtol=1e-5)

    def test_online_matches_legacy_bitwise(self):
        logits, token_ids = _make_inputs(8, 257)
        logp, _, _ = _C.fused_logp_forward_online_with_lse(logits, token_ids)
        legacy = _C.fused_logp_forward_online_fp32(logits, token_ids)
        assert torch.equal(logp, legacy)

    def test_online_lse_components_match_reference(self):
        logits, token_ids = _make_inputs(8, 257)
        _, row_max, log_sum = _C.fused_logp_forward_online_with_lse(logits, token_ids)
        ref_lse = torch.logsumexp(logits.float(), dim=-1)
        assert torch.allclose(row_max + log_sum, ref_lse, atol=1e-5, rtol=1e-5)

    def test_online_indexed_matches_legacy_bitwise(self):
        logits, token_ids = _make_inputs(8, 257)
        row_indices = _make_row_indices(8)
        logp, _, _ = _C.fused_logp_forward_online_indexed_with_lse(logits, token_ids, row_indices)
        legacy = _C.fused_logp_forward_online_indexed_fp32(logits, token_ids, row_indices)
        assert torch.equal(logp, legacy)


class TestFusedLogpBackwardKernel:
    @pytest.mark.parametrize("dtype, atol", _GRAD_TOLERANCES)
    def test_grad_matches_autograd_reference(self, dtype, atol):
        logits, token_ids = _make_inputs(8, 257, dtype=dtype)
        gen = torch.Generator().manual_seed(456)
        grad_out = torch.randn(8, generator=gen).cuda()

        _, row_max, log_sum = _C.fused_logp_forward_with_lse(logits, token_ids)
        grad_logits = _C.fused_logp_backward(grad_out, logits, token_ids, row_max, log_sum)

        ref_grad = _reference_grad(logits, token_ids, grad_out)
        assert grad_logits.dtype == dtype
        assert torch.allclose(grad_logits.float(), ref_grad.float(), atol=atol, rtol=0.0)

    def test_grad_rows_sum_to_zero(self):
        logits, token_ids = _make_inputs(8, 257)
        grad_out = torch.randn(8, generator=torch.Generator().manual_seed(7)).cuda()
        _, row_max, log_sum = _C.fused_logp_forward_with_lse(logits, token_ids)
        grad_logits = _C.fused_logp_backward(grad_out, logits, token_ids, row_max, log_sum)
        row_sums = grad_logits.sum(dim=-1)
        assert torch.allclose(row_sums, torch.zeros_like(row_sums), atol=1e-4)

    def test_invalid_target_rows_get_zero_grad(self):
        logits, token_ids = _make_inputs(4, 33)
        token_ids = token_ids.clone()
        token_ids[1] = -1
        grad_out = torch.ones(4).cuda()
        _, row_max, log_sum = _C.fused_logp_forward_with_lse(logits, token_ids)
        grad_logits = _C.fused_logp_backward(grad_out, logits, token_ids, row_max, log_sum)
        assert torch.equal(grad_logits[1], torch.zeros_like(grad_logits[1]))
        assert not torch.equal(grad_logits[0], torch.zeros_like(grad_logits[0]))

    @pytest.mark.parametrize("shift", [-1e4, 1e4, 1e8])
    def test_grad_stable_under_constant_shift(self, shift):
        # Separate (row_max, log_sum) statistics preserve shift-invariant gradients.
        logits, token_ids = _make_inputs(8, 257)
        logits = logits + shift
        grad_out = torch.randn(8, generator=torch.Generator().manual_seed(456)).cuda()

        _, row_max, log_sum = _C.fused_logp_forward_with_lse(logits, token_ids)
        grad_logits = _C.fused_logp_backward(grad_out, logits, token_ids, row_max, log_sum)

        ref_grad = _reference_grad(logits, token_ids, grad_out)
        assert torch.allclose(grad_logits, ref_grad, atol=1e-5, rtol=0.0)


class TestFusedLogpBackwardIndexedKernel:
    def test_grad_matches_masked_reference(self):
        logits, token_ids = _make_inputs(8, 257)
        row_indices = _make_row_indices(8)
        grad_out = torch.randn(8, generator=torch.Generator().manual_seed(456)).cuda()

        _, row_max, log_sum = _C.fused_logp_forward_indexed_with_lse(logits, token_ids, row_indices)
        grad_logits = _C.fused_logp_backward_indexed(
            grad_out, logits, token_ids, row_max, log_sum, row_indices
        )

        ref_grad = _mask_rows(_reference_grad(logits, token_ids, grad_out), row_indices)
        assert torch.allclose(grad_logits, ref_grad, atol=1e-5, rtol=0.0)

    def test_unselected_rows_get_exactly_zero_grad(self):
        logits, token_ids = _make_inputs(8, 257)
        row_indices = _make_row_indices(8)
        grad_out = torch.ones(8).cuda()
        _, row_max, log_sum = _C.fused_logp_forward_indexed_with_lse(logits, token_ids, row_indices)
        grad_logits = _C.fused_logp_backward_indexed(
            grad_out, logits, token_ids, row_max, log_sum, row_indices
        )
        selected = torch.zeros(8, dtype=torch.bool, device=logits.device)
        selected[row_indices] = True
        assert torch.equal(grad_logits[~selected], torch.zeros_like(grad_logits[~selected]))
        assert not torch.equal(grad_logits[selected], torch.zeros_like(grad_logits[selected]))


class TestFusedLogpOpAutogradRouting:
    def _op(self):
        from rl_engine.kernels.ops.cuda.loss.logp import FusedLogpGenericOp

        return FusedLogpGenericOp()

    @pytest.mark.parametrize("dtype, atol", _GRAD_TOLERANCES)
    def test_apply_backward_matches_reference(self, dtype, atol):
        logits, token_ids = _make_inputs(8, 257, dtype=dtype)
        logits.requires_grad_(True)
        gen = torch.Generator().manual_seed(654)
        grad_out = torch.randn(8, generator=gen).cuda().to(dtype)

        out = self._op().apply(logits, token_ids)
        assert out.dtype == dtype
        out.backward(grad_out)

        ref_grad = _reference_grad(logits, token_ids, grad_out)
        assert logits.grad is not None
        assert torch.allclose(logits.grad.float(), ref_grad.float(), atol=atol, rtol=0.0)

    def test_apply_fp32_backward_matches_reference(self):
        logits, token_ids = _make_inputs(8, 257)
        logits.requires_grad_(True)
        grad_out = torch.randn(8, generator=torch.Generator().manual_seed(11)).cuda()

        out = self._op().apply_fp32(logits, token_ids)
        out.backward(grad_out)

        ref_grad = _reference_grad(logits, token_ids, grad_out)
        assert logits.grad is not None
        assert torch.allclose(logits.grad, ref_grad, atol=1e-5, rtol=0.0)

    def test_apply_fp32_grad_stable_under_constant_shift(self):
        logits, token_ids = _make_inputs(8, 257)
        logits = (logits + 1e8).requires_grad_(True)
        grad_out = torch.randn(8, generator=torch.Generator().manual_seed(17)).cuda()

        out = self._op().apply_fp32(logits, token_ids)
        out.backward(grad_out)

        ref_grad = _reference_grad(logits, token_ids, grad_out)
        assert logits.grad is not None
        assert torch.allclose(logits.grad, ref_grad, atol=1e-5, rtol=0.0)

    def test_3d_inputs_flow_gradients(self):
        gen = torch.Generator().manual_seed(99)
        logits = torch.randn(2, 4, 65, generator=gen).cuda().requires_grad_(True)
        token_ids = torch.randint(0, 65, (2, 4), generator=gen).cuda()

        out = self._op().apply(logits, token_ids)
        assert out.shape == (2, 4)
        out.sum().backward()
        assert logits.grad is not None
        assert logits.grad.shape == logits.shape

    def test_no_grad_path_keeps_input_dtype(self):
        logits, token_ids = _make_inputs(8, 257, dtype=torch.bfloat16)
        with torch.no_grad():
            out = self._op().apply(logits, token_ids)
        assert out.dtype == torch.bfloat16

    def test_no_grad_path_bitwise_unchanged(self):
        logits, token_ids = _make_inputs(8, 257)
        with torch.no_grad():
            routed = self._op().apply(logits, token_ids)
        legacy = _C.fused_logp(logits, token_ids)
        assert torch.equal(routed, legacy)

    def test_grad_and_no_grad_forward_bitwise_equal(self):
        logits, token_ids = _make_inputs(8, 257)
        op = self._op()
        with torch.no_grad():
            rollout = op.apply_fp32(logits, token_ids)
        train = op.apply_fp32(logits.clone().requires_grad_(True), token_ids)
        assert torch.equal(train.detach(), rollout)

    def test_online_fp32_backward_matches_reference(self):
        logits, token_ids = _make_inputs(8, 257)
        logits.requires_grad_(True)
        grad_out = torch.randn(8, generator=torch.Generator().manual_seed(21)).cuda()

        out = self._op().online_fp32(logits, token_ids)
        out.backward(grad_out)

        ref_grad = _reference_grad(logits, token_ids, grad_out)
        assert logits.grad is not None
        assert torch.allclose(logits.grad, ref_grad, atol=1e-5, rtol=0.0)

    @pytest.mark.parametrize("variant", ["indexed_fp32", "online_indexed_fp32"])
    def test_indexed_variants_backward_matches_masked_reference(self, variant):
        logits, token_ids = _make_inputs(8, 257)
        logits.requires_grad_(True)
        row_indices = _make_row_indices(8)
        grad_out = torch.randn(8, generator=torch.Generator().manual_seed(31)).cuda()

        out = getattr(self._op(), variant)(logits, token_ids, row_indices)
        out.backward(grad_out)

        ref_grad = _mask_rows(_reference_grad(logits, token_ids, grad_out), row_indices)
        assert logits.grad is not None
        assert torch.allclose(logits.grad, ref_grad, atol=1e-5, rtol=0.0)

    def test_online_fp32_grad_and_no_grad_forward_bitwise_equal(self):
        # The online reduction order differs from two-pass, so consistency must
        # hold within the online path itself.
        logits, token_ids = _make_inputs(8, 257)
        op = self._op()
        with torch.no_grad():
            rollout = op.online_fp32(logits, token_ids)
        train = op.online_fp32(logits.clone().requires_grad_(True), token_ids)
        assert torch.equal(train.detach(), rollout)

    def test_out_variants_reject_grad_logits(self):
        # out= style variants write a caller-provided buffer and stay
        # non-differentiable, matching PyTorch's own out= convention.
        logits, token_ids = _make_inputs(8, 257)
        op = self._op()
        with torch.no_grad():
            op.out(logits, token_ids, torch.empty(8).cuda())
        logits.requires_grad_(True)
        with pytest.raises(RuntimeError, match="forward-only"):
            op.out(logits, token_ids, torch.empty(8).cuda())
        with pytest.raises(RuntimeError, match="forward-only"):
            op.online_out(logits, token_ids, torch.empty(8).cuda())

    def test_stale_extension_raises_rebuild_hint(self, monkeypatch):
        logits, token_ids = _make_inputs(8, 257)
        logits.requires_grad_(True)
        op = self._op()
        monkeypatch.delattr(_C, "fused_logp_backward")
        with pytest.raises(RuntimeError, match="pip install -e ."):
            op.apply(logits, token_ids)

    def _sm90_op(self, monkeypatch):
        from rl_engine.kernels.ops.cuda.loss.logp import FusedLogpSM90Op

        # Bypass __init__ (the SM90 kernel may not be compiled) to test routing only.
        op = FusedLogpSM90Op.__new__(FusedLogpSM90Op)
        op._fallback = None
        op.op = lambda *_: pytest.fail("no-grad SM90 kernel entry must not run under grad")
        monkeypatch.setenv("RL_KERNEL_ENABLE_EXPERIMENTAL_SM90_LOGP", "1")
        return op

    def test_sm90_grad_path_keeps_fp32_contract_and_matches_reference(self, monkeypatch):
        # Use the generic statistics-returning forward to isolate SM90 autograd routing.
        op = self._sm90_op(monkeypatch)
        monkeypatch.setattr(
            _C,
            "fused_logp_sm90_with_lse",
            lambda logits, labels: _C.fused_logp_forward_with_lse(logits, labels.long()),
            raising=False,
        )

        gen = torch.Generator().manual_seed(3)
        logits = torch.randn(4, 72, generator=gen).cuda().to(torch.bfloat16).requires_grad_(True)
        token_ids = torch.randint(0, 72, (4,), generator=gen).cuda()

        out = op.apply(logits, token_ids)
        assert out.dtype == torch.float32
        assert out.grad_fn is not None

        grad_out = torch.randn(4, generator=gen).cuda()
        out.backward(grad_out)
        ref_grad = _reference_grad(logits, token_ids, grad_out)
        assert logits.grad is not None
        assert torch.allclose(logits.grad.float(), ref_grad.float(), atol=2e-2, rtol=0.0)

    def test_sm90_no_grad_path_still_runs_tma_kernel(self, monkeypatch):
        from rl_engine.kernels.ops.cuda.loss.logp import FusedLogpSM90Op

        op = FusedLogpSM90Op.__new__(FusedLogpSM90Op)
        op._fallback = None
        seen = {}

        def fake_kernel(logits, labels):
            seen["labels_dtype"] = labels.dtype
            return torch.zeros(logits.size(0), device=logits.device)

        op.op = fake_kernel
        monkeypatch.setenv("RL_KERNEL_ENABLE_EXPERIMENTAL_SM90_LOGP", "1")

        gen = torch.Generator().manual_seed(3)
        logits = torch.randn(4, 72, generator=gen).cuda().to(torch.bfloat16)
        token_ids = torch.randint(0, 72, (4,), generator=gen).cuda()
        with torch.no_grad():
            out = op.apply(logits, token_ids)
        assert seen["labels_dtype"] == torch.int32
        assert out.shape == (4,)

    def test_sm90_grad_path_stale_extension_raises_rebuild_hint(self, monkeypatch):
        op = self._sm90_op(monkeypatch)
        monkeypatch.delattr(_C, "fused_logp_sm90_with_lse", raising=False)

        gen = torch.Generator().manual_seed(3)
        logits = torch.randn(4, 72, generator=gen).cuda().to(torch.bfloat16).requires_grad_(True)
        token_ids = torch.randint(0, 72, (4,), generator=gen).cuda()

        with pytest.raises(RuntimeError, match="pip install -e ."):
            op.apply(logits, token_ids)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_apply_grad_and_no_grad_forward_bitwise_equal(self, dtype):
        # The fp32 grad path must round exactly like the typed forward kernel.
        logits, token_ids = _make_inputs(8, 257, dtype=dtype)
        op = self._op()
        with torch.no_grad():
            rollout = op.apply(logits, token_ids)
        train = op.apply(logits.clone().requires_grad_(True), token_ids)
        assert train.dtype == rollout.dtype == dtype
        assert torch.equal(train.detach(), rollout)


_SM90_HOPPER_AVAILABLE = (
    torch.cuda.is_available()
    and _EXT_AVAILABLE
    and hasattr(_C, "fused_logp_sm90_with_lse")
    and torch.cuda.get_device_capability()[0] == 9
)


@pytest.mark.skipif(
    not _SM90_HOPPER_AVAILABLE,
    reason="requires a Hopper GPU and the KERNEL_ALIGN_FORCE_SM90 build",
)
class TestFusedLogpSM90Kernel:
    def _inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        gen = torch.Generator().manual_seed(5)
        # Exercise repeated TMA phase transitions across multiple tiles.
        logits = torch.randn(4, 8192, generator=gen).to(torch.bfloat16).cuda()
        token_ids = torch.randint(0, 8192, (4,), generator=gen, dtype=torch.long).cuda()
        return logits, token_ids

    def test_with_lse_logp_matches_plain_sm90_bitwise(self):
        logits, token_ids = self._inputs()
        labels = token_ids.to(torch.int32)
        logp, _, _ = _C.fused_logp_sm90_with_lse(logits, labels)
        assert torch.equal(logp, _C.fused_logp_sm90(logits, labels))

    def test_stats_match_reference(self):
        logits, token_ids = self._inputs()
        _, row_max, log_sum = _C.fused_logp_sm90_with_lse(logits, token_ids.to(torch.int32))
        assert torch.equal(row_max, logits.float().amax(dim=-1))
        ref_lse = torch.logsumexp(logits.float(), dim=-1)
        assert torch.allclose(row_max + log_sum, ref_lse, atol=1e-3, rtol=1e-5)

    def test_partial_last_tile_matches_reference(self):
        logits, token_ids = _make_inputs(4, 520, dtype=torch.bfloat16, seed=5)
        logp, row_max, log_sum = _C.fused_logp_sm90_with_lse(logits, token_ids.to(torch.int32))
        ref = selected_logprobs_reference(logits, token_ids)
        assert torch.allclose(logp, ref, atol=1e-3, rtol=1e-5)
        assert torch.allclose(
            row_max + log_sum,
            torch.logsumexp(logits.float(), dim=-1),
            atol=1e-3,
            rtol=1e-5,
        )

    def test_invalid_labels_return_zero(self):
        logits, token_ids = _make_inputs(4, 520, dtype=torch.bfloat16, seed=5)
        labels = token_ids.to(torch.int32)
        labels[0] = -1
        labels[1] = logits.size(1)

        logp, _, _ = _C.fused_logp_sm90_with_lse(logits, labels)
        assert torch.equal(logp[:2], torch.zeros_like(logp[:2]))

        ref = selected_logprobs_reference(logits, token_ids)
        assert torch.allclose(logp[2:], ref[2:], atol=1e-3, rtol=1e-5)

    def test_unaligned_row_stride_uses_generic_fallback(self, monkeypatch):
        from rl_engine.kernels.ops.cuda.loss.logp import FusedLogpSM90Op

        monkeypatch.setenv("RL_KERNEL_ENABLE_EXPERIMENTAL_SM90_LOGP", "1")
        op = FusedLogpSM90Op()
        op.op = lambda *_: pytest.fail("unaligned row stride must not use the TMA kernel")
        logits, token_ids = _make_inputs(4, 257, dtype=torch.bfloat16, seed=5)
        with torch.no_grad():
            out = op.apply(logits, token_ids)
        ref = selected_logprobs_reference(logits, token_ids)
        assert torch.allclose(out.float(), ref, atol=2e-2, rtol=0.0)

    def test_op_grad_and_no_grad_forward_bitwise_equal(self, monkeypatch):
        from rl_engine.kernels.ops.cuda.loss.logp import FusedLogpSM90Op

        monkeypatch.setenv("RL_KERNEL_ENABLE_EXPERIMENTAL_SM90_LOGP", "1")
        op = FusedLogpSM90Op()
        logits, token_ids = self._inputs()
        with torch.no_grad():
            rollout = op.apply(logits, token_ids)
        train = op.apply(logits.clone().requires_grad_(True), token_ids)
        assert train.dtype == rollout.dtype == torch.float32
        assert torch.equal(train.detach(), rollout)

    def test_op_grad_matches_reference(self, monkeypatch):
        from rl_engine.kernels.ops.cuda.loss.logp import FusedLogpSM90Op

        monkeypatch.setenv("RL_KERNEL_ENABLE_EXPERIMENTAL_SM90_LOGP", "1")
        op = FusedLogpSM90Op()
        logits, token_ids = self._inputs()
        logits.requires_grad_(True)
        grad_out = torch.randn(4, generator=torch.Generator().manual_seed(9)).cuda()

        out = op.apply(logits, token_ids)
        out.backward(grad_out)

        ref_grad = _reference_grad(logits, token_ids, grad_out)
        assert logits.grad is not None
        assert torch.allclose(logits.grad.float(), ref_grad.float(), atol=2e-2, rtol=0.0)
