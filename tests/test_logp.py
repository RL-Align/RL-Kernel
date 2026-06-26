# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Tests for NativeLogpOp, the PyTorch selected-logprob reference."""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.ops.pytorch.loss.logp import NativeLogpOp


def _make_inputs(
    batch: int,
    seq: int,
    vocab: int,
    *,
    dtype: torch.dtype = torch.float32,
    seed: int = 123,
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, seq, vocab, generator=gen, dtype=dtype)
    token_ids = torch.randint(0, vocab, (batch, seq), generator=gen, dtype=torch.long)
    return logits, token_ids


def _reference_selected_logp(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return torch.gather(log_probs, dim=-1, index=token_ids.long().unsqueeze(-1)).squeeze(-1)


class TestNativeLogpOpCorrectness:
    def test_output_shape_matches_token_ids(self):
        op = NativeLogpOp()
        logits, token_ids = _make_inputs(2, 16, 257)
        out = op.forward_fp32(logits, token_ids)
        assert out.shape == token_ids.shape

    def test_forward_fp32_returns_fp32(self):
        op = NativeLogpOp()
        logits, token_ids = _make_inputs(2, 16, 257, dtype=torch.bfloat16)
        out = op.forward_fp32(logits, token_ids)
        assert out.dtype == torch.float32

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_forward_returns_input_dtype(self, dtype):
        op = NativeLogpOp()
        logits, token_ids = _make_inputs(2, 16, 257, dtype=dtype)
        out = op.forward(logits, token_ids)
        assert out.dtype == dtype

    def test_call_and_apply_alias_forward(self):
        op = NativeLogpOp()
        logits, token_ids = _make_inputs(2, 16, 257)
        forward = op.forward(logits, token_ids)
        assert torch.equal(op(logits, token_ids), forward)
        assert torch.equal(op.apply(logits, token_ids), forward)

    def test_apply_fp32_alias_forward_fp32(self):
        op = NativeLogpOp()
        logits, token_ids = _make_inputs(2, 16, 257)
        assert torch.equal(op.apply_fp32(logits, token_ids), op.forward_fp32(logits, token_ids))

    def test_matches_fp32_reference_bitwise(self):
        op = NativeLogpOp()
        logits, token_ids = _make_inputs(2, 16, 257)
        out = op.forward_fp32(logits, token_ids)
        ref = _reference_selected_logp(logits, token_ids)
        assert torch.equal(out, ref)

    def test_pure_function_no_inplace(self):
        op = NativeLogpOp()
        logits, token_ids = _make_inputs(2, 16, 257)
        logits_orig = logits.clone()
        token_ids_orig = token_ids.clone()
        _ = op.forward_fp32(logits, token_ids)
        assert torch.equal(logits, logits_orig)
        assert torch.equal(token_ids, token_ids_orig)

    def test_op_class_is_logprob(self):
        assert NativeLogpOp.op_class == "logprob"

    def test_rejects_mismatched_shapes(self):
        op = NativeLogpOp()
        logits = torch.randn(2, 3, 5)
        token_ids = torch.randint(0, 5, (2, 4))
        with pytest.raises(ValueError, match="must match"):
            op.forward_fp32(logits, token_ids)


class TestNativeLogpOpBatchInvariance:
    def test_batch1_vs_batchN_bitwise(self):
        op = NativeLogpOp()
        logits, token_ids = _make_inputs(4, 16, 257, seed=321)
        full_out = op.forward_fp32(logits, token_ids)
        for row in range(logits.shape[0]):
            single_out = op.forward_fp32(logits[row : row + 1], token_ids[row : row + 1])
            assert torch.equal(
                full_out[row], single_out[0]
            ), f"Batch invariance broken at row {row}"

    def test_batch_invariance_with_padding(self):
        op = NativeLogpOp()
        logits_valid, token_ids_valid = _make_inputs(2, 16, 257, seed=456)
        gen = torch.Generator().manual_seed(789)
        logits_padding = torch.randn(3, 16, 257, generator=gen)
        token_padding = torch.randint(0, 257, (3, 16), generator=gen)
        logits_padded = torch.cat([logits_valid, logits_padding], dim=0)
        token_ids_padded = torch.cat([token_ids_valid, token_padding], dim=0)

        out_valid = op.forward_fp32(logits_valid, token_ids_valid)
        out_padded = op.forward_fp32(logits_padded, token_ids_padded)
        assert torch.equal(out_valid[0], out_padded[0])
        assert torch.equal(out_valid[1], out_padded[1])


class TestNativeLogpOpAccuracy:
    @pytest.mark.parametrize(
        "dtype, atol",
        [
            (torch.float32, 1e-5),
            (torch.bfloat16, 2e-2),
            (torch.float16, 5e-3),
        ],
    )
    def test_forward_vs_fp32_within_tolerance(self, dtype, atol):
        op = NativeLogpOp()
        logits, token_ids = _make_inputs(2, 16, 17, dtype=dtype)
        out_typed = op.forward(logits, token_ids).float()
        out_fp32 = op.forward_fp32(logits, token_ids)
        diff = (out_typed - out_fp32).abs().max().item()
        assert torch.allclose(
            out_typed, out_fp32, atol=atol, rtol=0.0
        ), f"dtype={dtype}, max_abs_error={diff:.3e} exceeds atol={atol}"


class TestNativeLogpOpRegistry:
    @pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA dispatch may select fused logp")
    def test_registry_returns_logp_op(self):
        from rl_engine.kernels.registry import kernel_registry

        op = kernel_registry.get_op("logp")
        assert isinstance(op, NativeLogpOp)
