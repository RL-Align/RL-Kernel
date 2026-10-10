# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Tests for the fused masking + pack-and-pad op (issue #42).

The PyTorch-native op is checked against
``SyntheticRLKernelBatch.compact_completion_values`` (the canonical compaction
already used elsewhere in the repo) and a plain index-based reference. The
same implementation is registered on CPU, CUDA, and ROCm because the removed
custom Triton path was slower than PyTorch's optimized indexing primitives.
"""

import pytest
import torch

from rl_engine.reference.packing.pack import NativePackOp
from rl_engine.validation.common import make_synthetic_rl_kernel_batch

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

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
    assert packed.data_ptr() != x.data_ptr()


def test_pack_none_active_is_empty():
    batch = _batch(seed=4, valid_density=0.0)
    x = _dense(batch, seed=104)
    op = NativePackOp()

    packed, cu_seqlens = op(x, batch.completion_mask)
    assert packed.shape[0] == 0
    assert int(cu_seqlens[-1].item()) == 0


@pytest.mark.parametrize("valid_density", [0.0, 1.0])
def test_pack_fast_paths_backward(valid_density):
    batch = _batch(seed=40, valid_density=valid_density)
    x = _dense(batch, seed=140).requires_grad_(True)
    packed, _ = NativePackOp()(x, batch.completion_mask)

    packed.sum().backward()

    expected = batch.completion_mask.unsqueeze(-1).expand_as(x).to(x.dtype)
    assert x.grad is not None
    assert torch.equal(x.grad, expected)


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


def test_unpack_rejects_mismatched_explicit_tail_shape():
    packed = torch.randn(3, 4)
    mask = torch.tensor([[True, False], [True, True]])

    with pytest.raises(ValueError, match="tail_shape"):
        NativePackOp.unpack(packed, mask, tail_shape=(2, 2))


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


@pytest.mark.parametrize("bad_mask", [torch.ones(6, dtype=torch.bool), torch.ones(2, 3, 1)])
def test_pack_requires_2d_mask(bad_mask):
    x = torch.randn(2, 3, 4)

    with pytest.raises(ValueError, match=r"\[B, S\]"):
        NativePackOp()(x, bad_mask)


@requires_cuda
def test_pack_rejects_device_mismatch():
    x = torch.randn(2, 3, 4, device="cuda")
    mask = torch.ones(2, 3, dtype=torch.bool)

    with pytest.raises(ValueError, match="same device"):
        NativePackOp()(x, mask)


@requires_cuda
def test_unpack_rejects_device_mismatch():
    packed = torch.randn(3, 4, device="cuda")
    mask = torch.tensor([[True, False], [True, True]])

    with pytest.raises(ValueError, match="same device"):
        NativePackOp.unpack(packed, mask)


# registry dispatch
def test_registry_dispatches_pack():
    from rl_engine.runtime.registry import kernel_registry

    op = kernel_registry.get_op("pack")
    assert isinstance(op, NativePackOp)
