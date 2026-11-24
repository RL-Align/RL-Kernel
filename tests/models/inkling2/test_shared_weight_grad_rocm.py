# SPDX-License-Identifier: Apache-2.0
"""INK-21 ROCm shared-loop weight-gradient tests (native HIP fold)."""

import pytest
import torch

from rl_engine.backends.extension import _C, _EXT_AVAILABLE
from rl_engine.models.inkling2 import (
    accumulate_shared_weight_gradients,
    accumulate_shared_weight_gradients_rocm,
)

pytestmark = pytest.mark.skipif(
    torch.version.hip is None
    or not torch.cuda.is_available()
    or not _EXT_AVAILABLE
    or not hasattr(_C, "inkling2_shared_weight_grad_forward"),
    reason="requires ROCm PyTorch, a GPU, and the native shared-weight-grad extension",
)

DTYPES = (torch.float32, torch.float16, torch.bfloat16)
ALL_ORDERS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]


def _grads(shape, dtype, seed=0, requires_grad=False):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return {
        v: torch.randn(shape, device="cuda", generator=gen, dtype=torch.float32)
        .to(dtype)
        .requires_grad_(requires_grad)
        for v in range(3)
    }


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    ("shape", "shard"),
    [
        ((4, 3), None),
        ((4, 3), (1, 3)),
        ((1, 1), None),
        ((7, 5), (2, 7)),
        ((3, 2304), (0, 3)),  # hidden
        ((128, 1152), (32, 64)),  # experts x intermediate
        ((130, 129), (1, 100)),  # awkward: not a multiple of 64 or the vector width
        ((201024, 5), (100000, 100003)),  # vocab-sized rows
        ((6, 3, 8, 128), (2, 5)),  # more than 2D, Q/KV head layout
    ],
)
def test_forward_bitwise_matches_reference(dtype, shape, shard):
    gradients = _grads(shape, dtype)
    native = accumulate_shared_weight_gradients_rocm(gradients, shard=shard)
    reference = accumulate_shared_weight_gradients(gradients, shard=shard)
    assert native.dtype == dtype and native.shape == reference.shape
    assert torch.equal(native, reference)


def test_empty_shard_and_empty_rows():
    gradients = _grads((4, 3), torch.bfloat16)
    out = accumulate_shared_weight_gradients_rocm(gradients, shard=(2, 2))
    assert out.shape == (0, 3)
    empty = {v: torch.empty((0, 3), device="cuda", dtype=torch.float32) for v in range(3)}
    assert accumulate_shared_weight_gradients_rocm(empty).shape == (0, 3)


@pytest.mark.parametrize("order", ALL_ORDERS)
def test_declared_order_is_exactly_what_the_kernel_computes(order):
    # 1e8 + 1 == 1e8 in FP32, so the fold result depends on the visit order.
    g = {
        0: torch.full((5, 7), 1.0e8, device="cuda"),
        1: torch.full((5, 7), 1.0, device="cuda"),
        2: torch.full((5, 7), -1.0e8, device="cuda"),
    }
    native = accumulate_shared_weight_gradients_rocm(g, visit_order=order)
    declared = accumulate_shared_weight_gradients(g, visit_order=order)
    assert torch.equal(native, declared)
    distinct = 0
    for other in ALL_ORDERS:
        expected = accumulate_shared_weight_gradients(g, visit_order=other)
        if not torch.equal(expected, declared):
            distinct += 1
            assert not torch.equal(native, expected)
    assert distinct > 0  # the probe really is order sensitive


def test_different_visit_order_is_not_what_kernel_computes_on_random_data():
    gradients = _grads((64, 2304), torch.float32, seed=5)
    native = accumulate_shared_weight_gradients_rocm(gradients, visit_order=(0, 1, 2))
    reordered = accumulate_shared_weight_gradients(gradients, visit_order=(2, 1, 0))
    assert torch.equal(native, accumulate_shared_weight_gradients(gradients, visit_order=(0, 1, 2)))
    assert not torch.equal(native, reordered)
    swapped = accumulate_shared_weight_gradients_rocm(gradients, visit_order=(2, 1, 0))
    assert torch.equal(swapped, reordered)
    assert not torch.equal(swapped, native)


def test_run_to_run_bitwise_determinism():
    gradients = _grads((257, 1152), torch.bfloat16, seed=9)
    first = accumulate_shared_weight_gradients_rocm(gradients, shard=(3, 250))
    for _ in range(8):
        again = accumulate_shared_weight_gradients_rocm(gradients, shard=(3, 250))
        assert torch.equal(first, again)


@pytest.mark.parametrize("dtype", DTYPES)
def test_batch_and_shard_invariance(dtype):
    gradients = _grads((40, 1152), dtype, seed=3)
    full = accumulate_shared_weight_gradients_rocm(gradients)
    # A shard equals the same rows of the full fold.
    for start, end in [(0, 1), (5, 6), (7, 33), (39, 40)]:
        shard = accumulate_shared_weight_gradients_rocm(gradients, shard=(start, end))
        assert torch.equal(shard, full[start:end])
    # Changing other rows leaves a row's result bit-identical.
    perturbed = {v: g.clone() for v, g in gradients.items()}
    for g in perturbed.values():
        g[10:20] = torch.randn_like(g[10:20])
        g[30:] = 0
    again = accumulate_shared_weight_gradients_rocm(perturbed)
    assert torch.equal(again[:10], full[:10])
    assert torch.equal(again[20:30], full[20:30])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shard", [None, (1, 3), (0, 4), (2, 2)])
def test_backward_matches_reference_autograd(dtype, shard):
    native_in = _grads((4, 5), dtype, seed=1, requires_grad=True)
    ref_in = {v: g.detach().clone().requires_grad_(True) for v, g in native_in.items()}
    native = accumulate_shared_weight_gradients_rocm(native_in, shard=shard)
    reference = accumulate_shared_weight_gradients(ref_in, shard=shard)
    upstream = torch.randn(reference.shape, device="cuda").to(dtype)
    native.backward(upstream)
    reference.backward(upstream)
    for v in range(3):
        assert native_in[v].grad is not None
        assert torch.equal(native_in[v].grad, ref_in[v].grad)


def test_input_validation():
    g = _grads((4, 3), torch.float32)
    with pytest.raises(ValueError, match="missing"):
        accumulate_shared_weight_gradients_rocm({0: g[0]})
    with pytest.raises(ValueError, match="shard"):
        accumulate_shared_weight_gradients_rocm(g, shard=(0, 9))
    with pytest.raises(ValueError):
        accumulate_shared_weight_gradients_rocm({v: torch.ones(2, 2) for v in range(3)})
