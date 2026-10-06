# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch

from rl_engine.kernels.ops.pytorch.mhc import NativeMHCPreHAggregateOp

HIDDEN_SIZE = 4096


def _kernel_available() -> bool:
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability()[0] < 8:
        return False
    try:
        from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
    except Exception:
        return False
    required = ("mhc_pre_h_aggregate", "mhc_pre_h_aggregate_backward")
    return _EXT_AVAILABLE and all(hasattr(_C, name) for name in required)


requires_mhc_kernel = pytest.mark.skipif(
    not _kernel_available(),
    reason="mhc_pre_h_aggregate requires the CUDA extension on SM80 or newer",
)


def _same_bytes(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.dtype == right.dtype and torch.equal(
        left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)
    )


def _make_inputs(
    num_tokens: int, *, device: str, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    residual = torch.randn(
        num_tokens,
        4,
        HIDDEN_SIZE,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    pre = torch.rand(
        num_tokens,
        4,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    grad_output = torch.randn(
        num_tokens,
        HIDDEN_SIZE,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    return residual, pre, grad_output


def test_native_mhc_pre_h_aggregate_backward_is_explicit_fp32():
    residual, pre, grad_output = _make_inputs(3, device="cpu", seed=7)

    grad_residual, grad_pre = NativeMHCPreHAggregateOp().backward_fp32(
        grad_output, residual, pre
    )

    assert grad_residual.dtype is torch.float32
    assert grad_pre.dtype is torch.float32
    expected_grad_residual = grad_output.float()[:, None, :] * pre[:, :, None]
    expected_grad_pre = torch.sum(
        grad_output.float()[:, None, :] * residual.float(), dim=-1
    )
    assert _same_bytes(grad_residual, expected_grad_residual)
    torch.testing.assert_close(grad_pre, expected_grad_pre, atol=5e-4, rtol=1e-5)


@requires_mhc_kernel
def test_mhc_pre_h_aggregate_forward_matches_fixed_tree_oracle_and_recomputes():
    from rl_engine.kernels.ops.base import _C

    residual, pre, _grad_output = _make_inputs(129, device="cuda", seed=0)
    reference = NativeMHCPreHAggregateOp()(residual, pre)

    first = _C.mhc_pre_h_aggregate(residual, pre)

    assert first.dtype is torch.bfloat16
    assert _same_bytes(first, reference)
    for _ in range(9):
        assert _same_bytes(first, _C.mhc_pre_h_aggregate(residual, pre))


@requires_mhc_kernel
def test_mhc_pre_h_aggregate_forward_is_batch_and_padding_invariant():
    from rl_engine.kernels.ops.base import _C

    residual, pre, _grad_output = _make_inputs(127, device="cuda", seed=1)
    pad_residual, pad_pre, _pad_grad_output = _make_inputs(2, device="cuda", seed=2)
    padded_residual = torch.cat((pad_residual[:1], residual, pad_residual[1:]))
    padded_pre = torch.cat((pad_pre[:1], pre, pad_pre[1:]))

    unpadded = _C.mhc_pre_h_aggregate(residual, pre)
    padded = _C.mhc_pre_h_aggregate(padded_residual, padded_pre)[1:-1]
    token_by_token = torch.cat(
        [
            _C.mhc_pre_h_aggregate(
                residual[token : token + 1], pre[token : token + 1]
            )
            for token in range(residual.shape[0])
        ]
    )

    assert _same_bytes(unpadded, padded)
    assert _same_bytes(unpadded, token_by_token)


@requires_mhc_kernel
def test_mhc_pre_h_aggregate_backward_matches_explicit_fixed_tree_oracle():
    from rl_engine.kernels.ops.base import _C

    residual, pre, grad_output = _make_inputs(129, device="cuda", seed=3)
    reference = NativeMHCPreHAggregateOp().backward_fp32(
        grad_output, residual, pre
    )

    first = _C.mhc_pre_h_aggregate_backward(grad_output, residual, pre)

    assert first[0].dtype is torch.float32
    assert first[1].dtype is torch.float32
    assert _same_bytes(first[0], reference[0])
    assert _same_bytes(first[1], reference[1])
    for _ in range(9):
        repeated = _C.mhc_pre_h_aggregate_backward(grad_output, residual, pre)
        assert _same_bytes(first[0], repeated[0])
        assert _same_bytes(first[1], repeated[1])


@requires_mhc_kernel
def test_mhc_pre_h_aggregate_cuda_wrapper_has_no_autograd_fallback():
    from rl_engine.kernels.ops.cuda.mhc import MHCPreHAggregateCudaOp

    residual, pre, _grad_output = _make_inputs(2, device="cuda", seed=8)
    residual.requires_grad_(True)
    pre.requires_grad_(True)

    with pytest.raises(RuntimeError, match="does not expose standalone autograd"):
        MHCPreHAggregateCudaOp()(residual, pre)


@requires_mhc_kernel
def test_mhc_pre_h_aggregate_backward_is_batch_and_padding_invariant():
    from rl_engine.kernels.ops.base import _C

    residual, pre, grad_output = _make_inputs(127, device="cuda", seed=4)
    pad_residual, pad_pre, pad_grad_output = _make_inputs(2, device="cuda", seed=5)
    padded_residual = torch.cat((pad_residual[:1], residual, pad_residual[1:]))
    padded_pre = torch.cat((pad_pre[:1], pre, pad_pre[1:]))
    padded_grad_output = torch.cat(
        (pad_grad_output[:1], grad_output, pad_grad_output[1:])
    )

    unpadded = _C.mhc_pre_h_aggregate_backward(grad_output, residual, pre)
    padded = _C.mhc_pre_h_aggregate_backward(
        padded_grad_output, padded_residual, padded_pre
    )
    split = [
        _C.mhc_pre_h_aggregate_backward(
            grad_output[token : token + 1],
            residual[token : token + 1],
            pre[token : token + 1],
        )
        for token in range(residual.shape[0])
    ]
    split_grad_residual = torch.cat([grads[0] for grads in split])
    split_grad_pre = torch.cat([grads[1] for grads in split])

    assert _same_bytes(unpadded[0], padded[0][1:-1])
    assert _same_bytes(unpadded[1], padded[1][1:-1])
    assert _same_bytes(unpadded[0], split_grad_residual)
    assert _same_bytes(unpadded[1], split_grad_pre)


@requires_mhc_kernel
def test_mhc_pre_h_aggregate_fails_closed_for_unsupported_contracts():
    from rl_engine.kernels.ops.base import _C

    residual, pre, grad_output = _make_inputs(2, device="cuda", seed=6)
    wrong_hidden = residual[:, :, :2048].contiguous()
    noncontiguous_residual = torch.empty(
        2, 4, HIDDEN_SIZE * 2, dtype=torch.bfloat16, device="cuda"
    )[:, :, ::2]
    noncontiguous_grad_output = torch.empty(
        2, HIDDEN_SIZE * 2, dtype=torch.bfloat16, device="cuda"
    )[:, ::2]

    with pytest.raises(RuntimeError, match=r"\[num_tokens, 4, 4096\]"):
        _C.mhc_pre_h_aggregate(wrong_hidden, pre)
    with pytest.raises(RuntimeError, match="residual must be bfloat16"):
        _C.mhc_pre_h_aggregate(residual.float(), pre)
    with pytest.raises(RuntimeError, match="pre must be float32"):
        _C.mhc_pre_h_aggregate(residual, pre.to(torch.bfloat16))
    with pytest.raises(RuntimeError, match="must be contiguous"):
        _C.mhc_pre_h_aggregate(noncontiguous_residual, pre)
    with pytest.raises(RuntimeError, match="grad_output and residual must be bfloat16"):
        _C.mhc_pre_h_aggregate_backward(grad_output.float(), residual, pre)
    with pytest.raises(RuntimeError, match="must be contiguous"):
        _C.mhc_pre_h_aggregate_backward(noncontiguous_grad_output, residual, pre)
