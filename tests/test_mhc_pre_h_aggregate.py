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
    # Check FP32 gradient outputs against the analytic aggregate derivative.
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


def test_native_mhc_pre_h_aggregate_matches_p1_oracle_bytes():
    # Check the native forward and backward against the P1 oracle byte-for-byte.
    """The operator oracle must be byte-equal to the P1 start-kit oracle."""
    from rl_engine.mhc import oracle

    residual, pre, grad_output = _make_inputs(5, device="cpu", seed=11)
    op = NativeMHCPreHAggregateOp()

    assert _same_bytes(op(residual, pre), oracle.h_aggregate_fwd(pre, residual))

    grad_residual, grad_pre = op.backward_fp32(grad_output, residual, pre)
    want_dr, want_dpre = oracle.h_aggregate_bwd(grad_output, pre, residual)
    assert _same_bytes(grad_residual, want_dr)
    assert _same_bytes(grad_pre, want_dpre)


@requires_mhc_kernel
def test_mhc_pre_h_aggregate_forward_matches_fixed_tree_oracle_and_recomputes():
    # Check the CUDA forward tree and repeatability against the native oracle.
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
    # Check CUDA forward bytes across padded, packed, and individual tokens.
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
def test_mhc_pre_h_aggregate_backward_matches_explicit_fixed_sum_oracle():
    # Check CUDA backward bytes and repeatability against ascending FP32 accumulation.
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
    # Reject standalone autograd requests instead of silently dropping gradients.
    from rl_engine.kernels.ops.cuda.mhc import MHCPreHAggregateCudaOp

    residual, pre, _grad_output = _make_inputs(2, device="cuda", seed=8)
    residual.requires_grad_(True)
    pre.requires_grad_(True)

    with pytest.raises(RuntimeError, match="does not expose standalone autograd"):
        MHCPreHAggregateCudaOp()(residual, pre)


@requires_mhc_kernel
def test_mhc_pre_h_aggregate_backward_is_batch_and_padding_invariant():
    # Check CUDA backward bytes across padded, packed, and individual tokens.
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
    # Reject unsupported CUDA shapes, dtypes, and noncontiguous tensors.
    from rl_engine.kernels.ops.base import _C

    residual, pre, grad_output = _make_inputs(2, device="cuda", seed=6)
    wrong_hidden = residual[:, :, :2048].contiguous()
    noncontiguous_residual = torch.empty(
        2, 4, HIDDEN_SIZE * 2, dtype=torch.bfloat16, device="cuda"
    )[:, :, ::2]
    noncontiguous_grad_output = torch.empty(
        2, HIDDEN_SIZE * 2, dtype=torch.bfloat16, device="cuda"
    )[:, ::2]

    with pytest.raises(RuntimeError, match="hidden_size must be 128 or 4096"):
        _C.mhc_pre_h_aggregate(wrong_hidden, pre)
    with pytest.raises(RuntimeError, match="residual must be bfloat16"):
        _C.mhc_pre_h_aggregate(residual.float(), pre)
    with pytest.raises(RuntimeError, match="pre must be float32"):
        _C.mhc_pre_h_aggregate(residual, pre.to(torch.bfloat16))
    with pytest.raises(RuntimeError, match="must be contiguous"):
        _C.mhc_pre_h_aggregate(noncontiguous_residual, pre)
    with pytest.raises(RuntimeError, match="grad_output must be bfloat16 or float32"):
        _C.mhc_pre_h_aggregate_backward(grad_output.half(), residual, pre)
    with pytest.raises(RuntimeError, match="must be contiguous"):
        _C.mhc_pre_h_aggregate_backward(noncontiguous_grad_output, residual, pre)


@pytest.fixture(params=("pytorch", "cuda", "triton"))
def aggregate_op(request):
    if request.param == "pytorch":
        return NativeMHCPreHAggregateOp(), "cpu"
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("NVIDIA SM80 or newer is required")
    if request.param == "cuda":
        if not _kernel_available():
            pytest.skip("Rebuild the CUDA extension")
        from rl_engine.kernels.ops.cuda.mhc import MHCPreHAggregateCudaOp
        return MHCPreHAggregateCudaOp(), "cuda"
    pytest.importorskip("triton")
    from rl_engine.kernels.ops.triton.mhc import MHCPreHAggregateTritonOp
    return MHCPreHAggregateTritonOp(), "cuda"


@pytest.mark.parametrize("hidden", (128, 4096))
@pytest.mark.parametrize("grad_dtype", (torch.bfloat16, torch.float32))
def test_aggregate_oracle_bytes(aggregate_op, hidden, grad_dtype):
    # Check both input-gradient dtypes against the P1 oracle without rounding FP32 dH.
    from rl_engine.mhc import oracle

    op, device = aggregate_op
    torch.manual_seed(366)
    residual = torch.randn(7, 4, hidden, device=device, dtype=torch.bfloat16)
    pre = torch.randn(7, 4, device=device)
    dh = torch.randn(7, hidden, device=device, dtype=grad_dtype)
    assert _same_bytes(op(residual, pre), oracle.h_aggregate_fwd(pre, residual))
    actual = op.backward_fp32(dh, residual, pre)
    expected = oracle.h_aggregate_bwd(dh, pre, residual)
    for got, want in zip(actual, expected, strict=True):
        assert got.dtype == torch.float32
        assert _same_bytes(got, want)


@pytest.mark.parametrize("hidden", (128, 4096))
def test_aggregate_batch_invariance(aggregate_op, hidden):
    # Check forward and backward bytes across batch sizes, padding, and repeated runs.
    op, device = aggregate_op
    residual = torch.randn(129, 4, hidden, device=device, dtype=torch.bfloat16)
    pre = torch.randn(129, 4, device=device)
    dh = torch.randn(129, hidden, device=device)
    full = (op(residual, pre), *op.backward_fp32(dh, residual, pre))
    for start, stop in ((0, 1), (1, 128), (0, 128), (0, 129)):
        part = (op(residual[start:stop], pre[start:stop]),
                *op.backward_fp32(dh[start:stop], residual[start:stop], pre[start:stop]))
        for got, want in zip(part, full, strict=True):
            assert _same_bytes(got, want[start:stop])


def test_aggregate_ascending_sum(aggregate_op):
    # Detect tree reductions that change cancellation in the ascending FP32 dPRE fold.
    op, device = aggregate_op
    residual = torch.zeros(1, 4, 128, device=device, dtype=torch.bfloat16)
    residual[:, :, :4] = torch.tensor([1e8, 1, -1e8, 1], device=device)
    dh = torch.ones(1, 128, device=device)
    pre = torch.ones(1, 4, device=device)
    _, grad_pre = op.backward_fp32(dh, residual, pre)
    assert _same_bytes(grad_pre, torch.ones_like(pre))


@pytest.mark.parametrize("hidden", (128, 4096))
def test_aggregate_empty_batch(aggregate_op, hidden):
    # Empty batches return correctly shaped tensors without launching a kernel.
    op, device = aggregate_op
    residual = torch.empty(0, 4, hidden, device=device, dtype=torch.bfloat16)
    pre = torch.empty(0, 4, device=device)
    dh = torch.empty(0, hidden, device=device)
    assert op(residual, pre).shape == (0, hidden)
    dr, dp = op.backward_fp32(dh, residual, pre)
    assert dr.shape == residual.shape and dr.dtype == torch.float32
    assert dp.shape == pre.shape and dp.dtype == torch.float32


def test_aggregate_invalid_inputs(aggregate_op):
    # All backends reject unsupported dimensions, dtypes, layouts, and gradient shapes.
    op, device = aggregate_op
    residual = torch.zeros(2, 4, 128, device=device, dtype=torch.bfloat16)
    pre = torch.ones(2, 4, device=device)
    dh = torch.ones(2, 128, device=device)
    for bad_residual, bad_pre in (
        (residual[:, :, :64].contiguous(), pre), (residual.float(), pre),
        (residual, pre.half()), (residual[:, :, ::2], pre), (residual, pre[:1]),
    ):
        with pytest.raises((ValueError, TypeError, RuntimeError)):
            op(bad_residual, bad_pre)
    for bad_grad in (dh[:, :64], dh.half(), dh.t().contiguous().t()):
        with pytest.raises((ValueError, TypeError, RuntimeError)):
            op.backward_fp32(bad_grad, residual, pre)


def test_aggregate_registry_order_and_fallback(monkeypatch):
    # Registry selection follows CUDA, Triton, then PyTorch when imports are unavailable.
    from rl_engine.kernels.registry import KernelRegistry, OpBackend

    registry = KernelRegistry()
    order = [OpBackend.CUDA_MHC_PRE_H_AGGREGATE, OpBackend.TRITON_MHC_PRE_H_AGGREGATE,
             OpBackend.PYTORCH_MHC_PRE_H_AGGREGATE]
    assert registry._priority_map["cuda"]["mhc_pre_h_aggregate"] == order
    for chosen in order:
        visited = []
        instance = object()

        def create(backend, chosen=chosen, instance=instance, visited=visited):
            visited.append(backend)
            if backend == chosen:
                return instance
            return None

        monkeypatch.setattr(registry, "_get_or_create_backend", create)
        assert registry.get_op("mhc_pre_h_aggregate", "cuda") is instance
        assert visited == order[:order.index(chosen) + 1]
    monkeypatch.undo()
    assert isinstance(registry.get_op("mhc_pre_h_aggregate", "cpu"), NativeMHCPreHAggregateOp)


@pytest.mark.parametrize("backend", ("Cuda", "Triton"))
def test_aggregate_provider_uses_candidate(backend, monkeypatch):
    # P1 composite execution reaches the candidate forward and FP32 backward methods.
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if backend == "Cuda" and not _kernel_available():
        pytest.skip("Rebuild the CUDA extension")
    if backend == "Triton":
        pytest.importorskip("triton")
    from rl_engine.mhc import fixtures, oracle
    from rl_engine.mhc.provider import resolve_provider

    provider = resolve_provider(f"rl_engine.mhc.h_aggregate_provider:{backend}HAggregateProvider")
    calls = []
    forward, backward = provider.op.forward, provider.op.backward_fp32

    def record_forward(*args):
        calls.append("forward")
        return forward(*args)

    def record_backward(*args):
        calls.append("backward")
        assert args[0].dtype == torch.float32
        return backward(*args)

    monkeypatch.setattr(provider.op, "forward", record_forward)
    monkeypatch.setattr(provider.op, "backward_fp32", record_backward)
    cpu_batch = fixtures.make_batch("one_row")
    batch = cpu_batch.to("cuda")
    grads = fixtures.make_grads("one_row", cpu_batch).to("cuda")
    _, saved = oracle.mhc_block_forward(batch, ops=provider)
    oracle.mhc_block_backward(batch, saved, grads, ops=provider)
    assert calls == ["forward", "backward"]
    assert provider.provenance()["actual_backend"] == f"{provider.name}+oracle-rest"
