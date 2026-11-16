# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""NativeCausalConv1dOp tests (Qwen3-Next short-conv reference).

Covers:
- Correctness vs an independent fp64 triple-loop golden
- Cache carry: prefill + continuation == one-shot; decode single step
- Bias / no-bias, initial-state passthrough, guards
- Axis A batch invariance (slice + padding)
- Registry dispatch
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.reference.linear_attn.causal_conv1d_sequence import NativeCausalConv1dOp
from rl_engine.runtime.registry import OpBackend, kernel_registry

# Qwen3-Next short conv: kernel size 4, causal (left pad 3).
_KERNEL = 4


def _golden_conv_fp64(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    cache: torch.Tensor | None,
) -> torch.Tensor:
    batch, seqlen, channels = x.shape
    kernel = weight.shape[1]
    padded = torch.zeros(batch, seqlen + kernel - 1, channels, dtype=torch.float64, device=x.device)
    if cache is not None:
        padded[:, : kernel - 1] = cache.double()
    padded[:, kernel - 1 :] = x.double()
    out = torch.zeros(batch, seqlen, channels, dtype=torch.float64, device=x.device)
    for b in range(batch):
        for t in range(seqlen):
            for c in range(channels):
                acc = bias[c].double() if bias is not None else 0.0
                for j in range(kernel):
                    acc += weight[c, j].double() * padded[b, t + j, c]
                out[b, t, c] = acc
    return out


@pytest.fixture
def op() -> NativeCausalConv1dOp:
    return NativeCausalConv1dOp()


def test_matches_golden(op: NativeCausalConv1dOp) -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 11, 8)
    weight = torch.randn(8, _KERNEL)
    bias = torch.randn(8)
    expected = _golden_conv_fp64(x, weight, bias, None)
    torch.testing.assert_close(
        op.forward_fp32(x, weight, bias), expected.float(), rtol=0, atol=1e-4
    )


def test_no_bias_matches_golden(op: NativeCausalConv1dOp) -> None:
    torch.manual_seed(1)
    x = torch.randn(2, 7, 5)
    weight = torch.randn(5, _KERNEL)
    expected = _golden_conv_fp64(x, weight, None, None)
    torch.testing.assert_close(
        op.forward_fp32(x, weight, None), expected.float(), rtol=0, atol=1e-4
    )


def test_cache_carry_prefill_continuation(op: NativeCausalConv1dOp) -> None:
    """Splitting the sequence must reproduce the one-shot result."""
    torch.manual_seed(2)
    x = torch.randn(2, 16, 6)
    weight = torch.randn(6, _KERNEL)
    bias = torch.randn(6)

    one_shot = op(x, weight, bias)
    first, cache = op(x[:, :10], weight, bias, None, return_cache=True)
    second, _ = op(x[:, 10:], weight, bias, cache, return_cache=True)
    torch.testing.assert_close(torch.cat([first, second], dim=1), one_shot)


def test_decode_single_token(op: NativeCausalConv1dOp) -> None:
    torch.manual_seed(3)
    x = torch.randn(2, 12, 6)
    weight = torch.randn(6, _KERNEL)
    bias = torch.randn(6)

    _, cache = op(x[:, :-1], weight, bias, None, return_cache=True)
    step, _ = op(x[:, -1:], weight, bias, cache, return_cache=True)
    full = op(x, weight, bias)
    torch.testing.assert_close(step, full[:, -1:])


def test_batch_invariance_slice_and_padding(op: NativeCausalConv1dOp) -> None:
    torch.manual_seed(4)
    x = torch.randn(5, 9, 4)
    weight = torch.randn(4, _KERNEL)
    bias = torch.randn(4)
    full = op(x, weight, bias)

    sliced = op(x[1:4], weight, bias)
    torch.testing.assert_close(sliced, full[1:4])

    padded = op(torch.cat([x, torch.zeros(2, 9, 4)], dim=0), weight, bias)
    torch.testing.assert_close(padded[:5], full)


def test_dtype_cast_back(op: NativeCausalConv1dOp) -> None:
    x = torch.randn(2, 8, 4, dtype=torch.bfloat16)
    weight = torch.randn(4, _KERNEL, dtype=torch.bfloat16)
    out = op(x, weight, None)
    assert out.dtype == torch.bfloat16


def test_guards(op: NativeCausalConv1dOp) -> None:
    with pytest.raises(ValueError):
        op(torch.randn(2, 8), torch.randn(4, _KERNEL))
    with pytest.raises(ValueError):
        op(torch.randn(2, 8, 4), torch.randn(5, _KERNEL))
    with pytest.raises(ValueError):
        op(torch.randn(2, 8, 4), torch.randn(4, _KERNEL), torch.randn(5))
    with pytest.raises(ValueError):
        op(
            torch.randn(2, 8, 4),
            torch.randn(4, _KERNEL),
            None,
            torch.randn(2, 2, 4),
        )


def test_registry_dispatch_cpu() -> None:
    resolved = kernel_registry.get_op("causal_conv1d", device="cpu")
    assert isinstance(resolved, NativeCausalConv1dOp)
    candidates = kernel_registry._priority_map["cpu"]["causal_conv1d"]
    assert candidates == [OpBackend.PYTORCH_CAUSAL_CONV1D]


def test_cache_has_independent_compact_storage(op):
    x = torch.randn(2, 1024, 8)
    _, cache = op(x, torch.randn(8, 4), return_cache=True)
    assert cache.shape == (2, 3, 8)
    assert cache.untyped_storage().nbytes() == cache.numel() * cache.element_size()


@pytest.mark.parametrize("kernel", [1, 4])
def test_empty_input_preserves_cache(op, kernel):
    x = torch.empty(2, 0, 8)
    cache = torch.randn(2, kernel - 1, 8)
    out, updated = op(x, torch.randn(8, kernel), initial_cache=cache, return_cache=True)
    assert out.shape == x.shape
    torch.testing.assert_close(updated, cache, rtol=0, atol=0)


def test_zero_width_kernel_is_rejected(op):
    with pytest.raises(ValueError, match="positive"):
        op(torch.randn(1, 2, 3), torch.empty(3, 0))
