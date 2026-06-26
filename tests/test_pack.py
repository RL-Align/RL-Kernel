# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Tests for the fused masking + pack-and-pad op (issue #42).

The PyTorch-native op is the portable reference that defines the numerical
contract for the Triton / CUDA / ROCm native kernels. Native correctness is
checked against ``SyntheticRLKernelBatch.compact_completion_values`` (the
canonical compaction already used elsewhere in the repo) and a plain
index-based reference; the Triton op is validated against the native op.
Native tests run on CPU; Triton tests require a CUDA/ROCm device.
"""

import pytest
import torch

from rl_engine.kernels.ops.pytorch.packing.pack import NativePackOp
from rl_engine.kernels.ops.triton.packing.pack import TritonPackOp
from rl_engine.testing import make_synthetic_rl_kernel_batch

try:
    import triton  # noqa: F401

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False

requires_triton_cuda = pytest.mark.skipif(
    not (_HAS_TRITON and torch.cuda.is_available()),
    reason="Triton pack op requires a CUDA device and Triton.",
)

_NUM_PROMPTS = 3
_SPP = 4
_COMP_LEN = 6
_VOCAB = 64


def _batch(seed=0, *, device="cpu", valid_density=0.8):
    return make_synthetic_rl_kernel_batch(
        num_prompts=_NUM_PROMPTS,
        samples_per_prompt=_SPP,
        prompt_len=0,
        completion_len=_COMP_LEN,
        vocab_size=_VOCAB,
        valid_density=valid_density,
        device=device,
        seed=seed,
    )


def _dense(batch, seed, *, vocab=_VOCAB, device="cpu"):
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(batch.batch_size, batch.completion_len, vocab, generator=gen, device=device)


# forward correctness
def test_pack_matches_batch_compaction():
    """Packed output must equal the repo's canonical compact_completion_values."""
    batch = _batch(seed=0)
    x = _dense(batch, seed=100)
    op = NativePackOp()

    packed, cu_seqlens = op(x, batch.completion_mask)
    expected = batch.compact_completion_values(x)

    assert packed.shape == expected.shape
    assert torch.equal(packed, expected)
    # Total active tokens equals the mask sum and the last cu_seqlens entry.
    assert int(cu_seqlens[-1].item()) == int(batch.completion_mask.sum().item())
    assert cu_seqlens.numel() == batch.batch_size + 1


def test_pack_matches_index_reference():
    batch = _batch(seed=1, valid_density=0.5)
    x = _dense(batch, seed=101)
    op = NativePackOp()

    packed, _ = op(x, batch.completion_mask)
    flat_mask = batch.completion_mask.reshape(-1)
    ref = x.reshape(-1, x.shape[-1])[flat_mask]
    assert torch.equal(packed, ref)


def test_cu_seqlens_is_per_row_prefix_sum():
    batch = _batch(seed=2, valid_density=0.6)
    x = _dense(batch, seed=102)
    op = NativePackOp()

    _, cu_seqlens = op(x, batch.completion_mask)
    per_row = batch.completion_mask.reshape(batch.batch_size, -1).sum(dim=1)
    expected = torch.zeros(batch.batch_size + 1, dtype=torch.int64)
    torch.cumsum(per_row.to(torch.int64), dim=0, out=expected[1:])
    assert torch.equal(cu_seqlens, expected)


def test_non_bool_mask_cu_seqlens_matches_packed_rows():
    """A non-bool mask (nonzero == active) must not over-count cu_seqlens.

    Counting active rows from the raw integer mask (e.g. values in {0, 2})
    would inflate cu_seqlens beyond the number of rows actually packed.
    """
    op = NativePackOp()
    mask = torch.tensor([[0, 2, 0], [2, 2, 0]], dtype=torch.int32)
    x = torch.randn(2, 3, 4)

    packed, cu_seqlens = op(x, mask)
    # 3 nonzero positions -> 3 packed rows; cu_seqlens must end at 3, not 6.
    assert packed.shape[0] == 3
    assert cu_seqlens.tolist() == [0, 1, 3]
    assert torch.equal(packed, x.reshape(-1, 4)[mask.reshape(-1).to(torch.bool)])


def test_pack_all_active_is_identity_flatten():
    batch = _batch(seed=3, valid_density=1.0)
    x = _dense(batch, seed=103)
    op = NativePackOp()

    packed, _ = op(x, batch.completion_mask)
    assert packed.shape[0] == batch.batch_size * batch.completion_len
    assert torch.equal(packed, x.reshape(-1, x.shape[-1]))


def test_pack_none_active_is_empty():
    batch = _batch(seed=4, valid_density=0.0)
    x = _dense(batch, seed=104)
    op = NativePackOp()

    packed, cu_seqlens = op(x, batch.completion_mask)
    assert packed.shape[0] == 0
    assert int(cu_seqlens[-1].item()) == 0


# unpack / round-trip
def test_unpack_round_trip_zeros_inactive():
    batch = _batch(seed=5, valid_density=0.7)
    x = _dense(batch, seed=105)
    op = NativePackOp()

    packed, _ = op(x, batch.completion_mask)
    restored = op.unpack(packed, batch.completion_mask)

    mask = batch.completion_mask
    active = mask.unsqueeze(-1).expand_as(x)
    # Active positions are restored exactly; inactive positions are zeroed.
    assert torch.equal(restored[active], x[active])
    assert torch.all(restored[~active] == 0.0)


# backward (scatter) correctness
def test_backward_scatters_grad_to_active_rows():
    batch = _batch(seed=6, valid_density=0.7)
    x = _dense(batch, seed=106).requires_grad_(True)
    op = NativePackOp()

    packed, _ = op(x, batch.completion_mask)
    g = torch.randn_like(packed)
    packed.backward(g)

    # The gradient w.r.t. x is the scatter of g back to the active rows.
    expected_grad = op.unpack(g, batch.completion_mask)
    assert x.grad is not None
    assert torch.equal(x.grad, expected_grad)
    # Inactive positions receive zero gradient.
    inactive = ~batch.completion_mask.unsqueeze(-1).expand_as(x)
    assert torch.all(x.grad[inactive] == 0.0)


def test_backward_gradcheck_double():
    """Analytic scatter backward must match numerical gradients (float64)."""
    batch = _batch(seed=7, valid_density=0.6)
    mask = batch.completion_mask
    x = torch.randn(batch.batch_size, batch.completion_len, 3, dtype=torch.float64).requires_grad_(
        True
    )
    op = NativePackOp()

    # Only the packed tensor is differentiable; cu_seqlens is integer.
    assert torch.autograd.gradcheck(lambda t: op(t, mask)[0], (x,), eps=1e-6, atol=1e-6)


# multi-dim tail and validation
def test_pack_supports_multidim_tail():
    mask = torch.tensor([[True, False, True], [False, True, True]])
    x = torch.randn(2, 3, 4, 5)
    op = NativePackOp()

    packed, cu_seqlens = op(x, mask)
    assert packed.shape == (4, 4, 5)
    assert torch.equal(packed, x.reshape(-1, 4, 5)[mask.reshape(-1)])
    assert cu_seqlens.tolist() == [0, 2, 4]


def test_pack_rejects_mismatched_mask_shape():
    x = torch.randn(2, 3, 4)
    bad_mask = torch.ones(2, 5, dtype=torch.bool)
    op = NativePackOp()
    with pytest.raises(ValueError):
        op(x, bad_mask)


# registry dispatch
def test_registry_dispatches_pack():
    from rl_engine.kernels.registry import kernel_registry

    op = kernel_registry.get_op("pack")
    if _HAS_TRITON and torch.cuda.is_available():
        assert isinstance(op, TritonPackOp)
    else:
        assert isinstance(op, NativePackOp)


# Triton fused op (validated against the native reference)
@requires_triton_cuda
@pytest.mark.parametrize("valid_density", [1.0, 0.7, 0.0])
def test_triton_forward_matches_native(valid_density):
    batch = _batch(seed=10, device="cuda", valid_density=valid_density)
    x = _dense(batch, seed=110, device="cuda")
    packed_t, cu_t = TritonPackOp()(x, batch.completion_mask)
    packed_n, cu_n = NativePackOp()(x, batch.completion_mask)
    assert torch.equal(packed_t, packed_n)
    assert torch.equal(cu_t, cu_n)


@requires_triton_cuda
def test_triton_backward_matches_native():
    batch = _batch(seed=11, device="cuda", valid_density=0.7)
    x0 = _dense(batch, seed=111, device="cuda")
    g = torch.randn(int(batch.completion_mask.sum()), _VOCAB, device="cuda")

    xt = x0.clone().requires_grad_(True)
    pt, _ = TritonPackOp()(xt, batch.completion_mask)
    pt.backward(g)

    xn = x0.clone().requires_grad_(True)
    pn, _ = NativePackOp()(xn, batch.completion_mask)
    pn.backward(g)

    assert xt.grad is not None
    assert torch.equal(xt.grad, xn.grad)


@requires_triton_cuda
def test_triton_supports_multidim_tail():
    mask = torch.tensor([[True, False, True], [False, True, True]], device="cuda")
    x = torch.randn(2, 3, 4, 5, device="cuda")
    packed_t, cu_t = TritonPackOp()(x, mask)
    packed_n, cu_n = NativePackOp()(x, mask)
    assert packed_t.shape == (4, 4, 5)
    assert torch.equal(packed_t, packed_n)
    assert cu_t.tolist() == [0, 2, 4]


@requires_triton_cuda
def test_triton_inactive_rows_do_not_leak():
    """Garbage values at masked rows must not appear in the packed output."""
    batch = _batch(seed=12, device="cuda", valid_density=0.6)
    x = _dense(batch, seed=112, device="cuda")
    inactive = ~batch.completion_mask.unsqueeze(-1).expand_as(x)
    x_pert = x.clone()
    x_pert[inactive] = 1e9

    base, _ = TritonPackOp()(x, batch.completion_mask)
    pert, _ = TritonPackOp()(x_pert, batch.completion_mask)
    assert torch.equal(base, pert)


@requires_triton_cuda
def test_triton_requires_gpu_tensor():
    op = TritonPackOp()
    x = torch.randn(2, 3, 4)
    mask = torch.ones(2, 3, dtype=torch.bool)
    with pytest.raises(RuntimeError):
        op(x, mask)
