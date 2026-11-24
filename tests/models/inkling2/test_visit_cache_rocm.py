# SPDX-License-Identifier: Apache-2.0
"""INK-22 ROCm per-visit cache tests (native HIP byte-copy kernel)."""

import pytest
import torch

from rl_engine.backends.extension import _C, _EXT_AVAILABLE
from rl_engine.models.inkling2 import clone_visit_state_rocm
from rl_engine.models.inkling2.visit_cache import VisitKey, VisitStateCache

pytestmark = pytest.mark.skipif(
    torch.version.hip is None
    or not torch.cuda.is_available()
    or not _EXT_AVAILABLE
    or not hasattr(_C, "inkling2_visit_state_clone"),
    reason="requires ROCm PyTorch, a GPU, and the native visit-state extension",
)

DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _bits(tensor):
    return tensor.contiguous().view(torch.uint8).flatten()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "shape",
    [
        (1,),
        (3,),
        (7, 5),
        (1, 1, 1),
        (3, 4),
        (8, 2304),  # hidden
        (1, 8, 4096, 128),  # KV8 / D128 state
        (2, 2304, 3),  # conv state, kernel 4 -> 3 history taps
        (255, 129),
    ],
)
def test_clone_is_bitwise_and_not_aliased(dtype, shape):
    torch.manual_seed(0)
    source = torch.randn(shape, device="cuda").to(dtype)
    clone = clone_visit_state_rocm(source)
    assert clone.dtype == source.dtype and clone.shape == source.shape
    assert clone.data_ptr() != source.data_ptr()
    assert torch.equal(_bits(clone), _bits(source))
    before = _bits(source).clone()
    clone.add_(1)
    assert torch.equal(_bits(source), before)


@pytest.mark.parametrize("dtype", DTYPES)
def test_special_values_preserved_exactly(dtype):
    values = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), -0.0, 0.0, 1e-40, 65504.0, -1.5],
        device="cuda",
    ).to(dtype)
    clone = clone_visit_state_rocm(values)
    assert torch.equal(_bits(clone), _bits(values))


def test_random_bit_patterns_preserved():
    raw = torch.randint(0, 256, (4099 * 2,), dtype=torch.uint8, device="cuda")
    source = raw.view(torch.bfloat16)
    assert torch.equal(_bits(clone_visit_state_rocm(source)), raw)


@pytest.mark.parametrize("dtype", DTYPES)
def test_misaligned_view_uses_exact_tail_path(dtype):
    base = torch.randn(1024, device="cuda").to(dtype)
    for offset in (1, 2, 3, 5):
        view = base[offset : offset + 37]
        clone = clone_visit_state_rocm(view)
        assert torch.equal(_bits(clone), _bits(view))


def test_empty_state():
    source = torch.empty((0, 4), device="cuda", dtype=torch.bfloat16)
    clone = clone_visit_state_rocm(source)
    assert clone.shape == source.shape and clone.numel() == 0


def test_noncontiguous_input_is_cloned_by_value():
    source = torch.randn(6, 8, device="cuda", dtype=torch.float16).t()
    clone = clone_visit_state_rocm(source)
    assert torch.equal(clone, source)


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        clone_visit_state_rocm(torch.ones(2))
    with pytest.raises(ValueError):
        clone_visit_state_rocm(torch.tensor(1.0, device="cuda"))
    with pytest.raises(TypeError):
        clone_visit_state_rocm(torch.ones(2, device="cuda", dtype=torch.int32))


def test_clone_deterministic_run_to_run():
    source = torch.randn(64, 2304, device="cuda", dtype=torch.bfloat16)
    first = clone_visit_state_rocm(source)
    for _ in range(5):
        assert torch.equal(_bits(clone_visit_state_rocm(source)), _bits(first))


def test_cache_isolates_visits_on_rocm_without_contamination():
    cache = VisitStateCache()
    first, second = VisitKey(6, 0, 6), VisitKey(6, 1, 18)
    kv_a = torch.randn(2, 1, 8, 5, 128, device="cuda", dtype=torch.bfloat16)
    kv_b = torch.randn_like(kv_a)
    conv_a = torch.randn(1, 2304, 3, device="cuda", dtype=torch.bfloat16)
    conv_b = torch.randn_like(conv_a)
    keep_a, keep_b = kv_a.clone(), kv_b.clone()
    cache.put(first, kv=kv_a, conv=conv_a)
    cache.put(second, kv=kv_b, conv=conv_b)
    # Mutating live tensors or fetched copies must not leak into either visit.
    kv_a.fill_(7)
    cache.require(first).kv.fill_(9)
    assert torch.equal(cache.require(first).kv, keep_a)
    assert torch.equal(cache.require(second).kv, keep_b)
    assert torch.equal(cache.require(first).conv, conv_a)
    assert torch.equal(cache.require(second).conv, conv_b)
    cache.clear_pass(0)
    assert cache.get(first) is None
    assert torch.equal(cache.require(second).kv, keep_b)
    assert cache.require(second).kv.is_cuda
