# SPDX-License-Identifier: Apache-2.0
"""INK-06 ROCm padded token embedding tests (native HIP kernel, MI300X)."""

import pytest
import torch
import torch.nn.functional as F

from rl_engine.models.inkling2.token_embedding import (
    PADDED_VOCAB_SIZE,
    VALID_VOCAB_SIZE,
    token_embedding,
    token_embedding_rocm,
)

pytestmark = pytest.mark.skipif(
    getattr(torch.version, "hip", None) is None or not torch.cuda.is_available(),
    reason="ROCm GPU is required",
)

HIDDEN = 2304
DEVICE = "cuda"


@pytest.fixture(scope="module", autouse=True)
def _binding():
    from rl_engine.backends.extension import _C, _EXT_AVAILABLE

    if not _EXT_AVAILABLE or not hasattr(_C, "inkling2_token_embedding_rocm"):
        pytest.skip("ROCm token embedding extension is unavailable")


def _grad_reference(ids, grad, vocab):
    """FP32 accumulation in ascending token order, one cast at the end (contract)."""
    flat_ids = ids.reshape(-1).cpu().tolist()
    flat_grad = grad.reshape(len(flat_ids), -1).cpu()
    hidden = flat_grad.shape[1]
    out = torch.zeros(vocab, hidden, dtype=grad.dtype)
    acc = {}
    for t, i in enumerate(flat_ids):
        prev = acc.get(i, torch.zeros(hidden, dtype=torch.float32))
        acc[i] = prev + flat_grad[t].float()
    for i, v in acc.items():
        out[i] = v.to(grad.dtype)
    return out


@pytest.fixture(scope="module")
def big_weight():
    gen = torch.Generator(device=DEVICE).manual_seed(606)
    return torch.randn(PADDED_VOCAB_SIZE, HIDDEN, device=DEVICE, generator=gen).to(torch.bfloat16)


@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_real_inkling_shape_bitwise_vs_reference_and_torch(big_weight, id_dtype):
    ids = torch.tensor(
        [[0, 1, VALID_VOCAB_SIZE - 1, VALID_VOCAB_SIZE, PADDED_VOCAB_SIZE - 1, 777]],
        device=DEVICE,
        dtype=id_dtype,
    )
    out = token_embedding_rocm(ids, big_weight)
    assert out.shape == (1, 6, HIDDEN) and out.dtype == torch.bfloat16
    assert torch.equal(out, token_embedding(ids, big_weight))
    assert torch.equal(out, F.embedding(ids.long(), big_weight))
    # Padded rows (>= 200058) are valid rows and are preserved verbatim.
    assert torch.equal(out[0, 3], big_weight[VALID_VOCAB_SIZE])
    assert torch.equal(out[0, 4], big_weight[PADDED_VOCAB_SIZE - 1])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden", [1, 3, 63, 64, 65, 130, 2304])
def test_awkward_hidden_sizes_all_dtypes_bitwise(dtype, hidden):
    torch.manual_seed(hidden)
    weight = torch.randn(257, hidden, device=DEVICE).to(dtype)
    ids = torch.randint(0, 257, (3, 11), device=DEVICE)
    out = token_embedding_rocm(ids, weight)
    assert out.dtype == dtype
    assert torch.equal(out, F.embedding(ids, weight))


def test_unaligned_weight_view_uses_byte_exact_fallback():
    base = torch.randn(40 * 7 + 1, device=DEVICE, dtype=torch.bfloat16)
    weight = base[1:].reshape(40, 7)  # data pointer is only 2-byte aligned
    ids = torch.tensor([39, 0, 5, 5], device=DEVICE)
    assert torch.equal(token_embedding_rocm(ids, weight), F.embedding(ids, weight))


def test_single_row_and_empty_inputs():
    weight = torch.randn(9, 5, device=DEVICE)
    one = token_embedding_rocm(torch.tensor([8], device=DEVICE), weight)
    assert torch.equal(one, weight[8:9])
    empty = token_embedding_rocm(torch.empty(0, dtype=torch.int64, device=DEVICE), weight)
    assert empty.shape == (0, 5) and empty.dtype == weight.dtype
    empty2d = token_embedding_rocm(torch.empty(2, 0, dtype=torch.int32, device=DEVICE), weight)
    assert empty2d.shape == (2, 0, 5)


def test_output_dtype_cast_matches_reference():
    weight = torch.randn(8, 4, device=DEVICE, dtype=torch.bfloat16)
    ids = torch.tensor([1, 7], device=DEVICE, dtype=torch.int32)
    out = token_embedding_rocm(ids, weight, output_dtype=torch.float32)
    assert out.dtype == torch.float32
    assert torch.equal(out, token_embedding(ids, weight, output_dtype=torch.float32))


@pytest.mark.parametrize("bad", [-1, 8, PADDED_VOCAB_SIZE])
def test_out_of_range_ids_raise_like_reference(bad):
    weight = torch.zeros(8, 2, device=DEVICE)
    ids = torch.tensor([0, bad], device=DEVICE)
    with pytest.raises(ValueError, match="must be in"):
        token_embedding(ids, weight)
    with pytest.raises(ValueError, match="must be in"):
        token_embedding_rocm(ids, weight)


def test_dtype_and_device_validation():
    weight = torch.zeros(8, 2, device=DEVICE)
    with pytest.raises(TypeError):
        token_embedding_rocm(torch.tensor([1], device=DEVICE, dtype=torch.int16), weight)
    with pytest.raises(ValueError, match="HIP device"):
        token_embedding_rocm(torch.tensor([1]), weight)
    with pytest.raises(ValueError):
        token_embedding_rocm(torch.tensor([1], device=DEVICE), weight.double())


def test_kernel_never_reads_out_of_range_rows():
    # Memory-safety backstop in the raw binding: invalid rows are zero-filled.
    from rl_engine.backends.extension import _C

    weight = torch.ones(4, 70, device=DEVICE)
    ids = torch.tensor([1, 4, -1, 3], device=DEVICE)
    out = _C.inkling2_token_embedding_rocm(ids, weight)
    assert torch.equal(out[0], weight[1]) and torch.equal(out[3], weight[3])
    assert not out[1].any() and not out[2].any()


def test_batch_invariance_and_run_to_run_determinism(big_weight):
    gen = torch.Generator(device=DEVICE).manual_seed(5)
    ids = torch.randint(0, PADDED_VOCAB_SIZE, (64,), device=DEVICE, generator=gen)
    full = token_embedding_rocm(ids, big_weight)
    assert torch.equal(full, token_embedding_rocm(ids, big_weight))
    other = ids.clone()
    other[::2] = (other[::2] + 12345) % PADDED_VOCAB_SIZE
    mixed = token_embedding_rocm(other, big_weight)
    assert torch.equal(mixed[1::2], full[1::2])
    for n in (1, 7, 33):
        assert torch.equal(token_embedding_rocm(ids[:n], big_weight), full[:n])
    assert torch.equal(token_embedding_rocm(ids.flip(0), big_weight), full.flip(0))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_backward_matches_ordered_fp32_contract(dtype):
    torch.manual_seed(11)
    vocab, hidden = 300, 130
    weight = torch.randn(vocab, hidden, device=DEVICE, dtype=dtype, requires_grad=True)
    ids = torch.randint(0, 20, (4, 50), device=DEVICE, dtype=torch.int32)  # heavy duplicates
    grad = torch.randn(4, 50, hidden, device=DEVICE).to(dtype)
    out = token_embedding_rocm(ids, weight)
    out.backward(grad)
    expected = _grad_reference(ids, grad, vocab)
    assert weight.grad.dtype == dtype
    assert torch.equal(weight.grad.cpu(), expected)
    first = weight.grad.clone()
    weight.grad = None
    token_embedding_rocm(ids, weight).backward(grad)
    assert torch.equal(first, weight.grad)


def test_backward_matches_torch_embedding_on_exact_values():
    weight = torch.randn(6, 3, device=DEVICE, requires_grad=True)
    ids = torch.tensor([2, 2, 4], device=DEVICE)
    token_embedding_rocm(ids, weight).sum().backward()
    assert torch.equal(weight.grad[2], torch.full((3,), 2.0, device=DEVICE))
    assert torch.equal(weight.grad[4], torch.ones(3, device=DEVICE))
    ref = torch.randn(6, 3, device=DEVICE, requires_grad=True)
    F.embedding(ids, ref).sum().backward()
    assert torch.equal(ref.grad[[0, 1, 3, 5]], weight.grad[[0, 1, 3, 5]])


def test_backward_real_shape_padded_rows_and_batch_invariance():
    torch.manual_seed(3)
    weight = torch.randn(PADDED_VOCAB_SIZE, HIDDEN, device=DEVICE).to(torch.bfloat16)
    weight.requires_grad_(True)
    ids = torch.tensor([5, VALID_VOCAB_SIZE, 5, PADDED_VOCAB_SIZE - 1, 5], device=DEVICE)
    grad = torch.randn(5, HIDDEN, device=DEVICE).to(torch.bfloat16)
    token_embedding_rocm(ids, weight).backward(grad)
    g = weight.grad
    assert torch.equal(g[VALID_VOCAB_SIZE], grad[1])
    assert torch.equal(g[PADDED_VOCAB_SIZE - 1], grad[3])
    expected5 = (grad[0].float() + grad[2].float() + grad[4].float()).to(torch.bfloat16)
    assert torch.equal(g[5], expected5)
    # The gradient of id 5 does not depend on what other ids/rows are present.
    weight.grad = None
    sub_ids = torch.tensor([5, 5, 5], device=DEVICE)
    token_embedding_rocm(sub_ids, weight).backward(grad[[0, 2, 4]])
    assert torch.equal(weight.grad[5], expected5)


def test_backward_empty():
    weight = torch.randn(5, 4, device=DEVICE, requires_grad=True)
    out = token_embedding_rocm(torch.empty(0, dtype=torch.int64, device=DEVICE), weight)
    out.sum().backward()
    assert weight.grad.shape == weight.shape and not weight.grad.any()
