# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Tests for NativeEmbeddingOp (ISSUE #108 WS1 ground-truth baseline).

Embedding is a pure row gather (weight[token_ids]) with no floating-point
accumulation, so unlike silu/swiglu there is no reduction-order drift: the
fp32 path and the dtype path differ only by a single cast. Correctness is
therefore asserted bitwise (torch.equal), not allclose.
"""

import pytest
import torch

from rl_engine.kernels.ops.pytorch.linear.embedding import NativeEmbeddingOp
from rl_engine.kernels.registry import kernel_registry

# Qwen3-8B architecture (synthetic tensors, no weight download). Most tests use
# a shrunk vocab/hidden -- the logic is identical and the full-size table is
# pointless to materialize for every case. The real Qwen3-8B dims are exercised
# separately by the GPU smoke test below.
_VOCAB = 128  # shrunk; real value: _QWEN3_VOCAB
_HIDDEN = 64  # shrunk; real value: _QWEN3_HIDDEN

# Real Qwen3-8B input-embedding table dims: 151936 x 4096 ~ 2.49 GB in fp32.
_QWEN3_VOCAB = 151936
_QWEN3_HIDDEN = 4096


# Shared helpers -- fixed-seed Generator for determinism / reproducibility.
def _rand_weight(vocab=_VOCAB, hidden=_HIDDEN, *, seed, dtype=torch.float32):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(vocab, hidden, generator=gen, dtype=dtype)


def _rand_ids(shape, *, seed, vocab=_VOCAB):
    # token ids are indices: int64, values in [0, vocab).
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, shape, generator=gen, dtype=torch.int64)


# Correctness: embedding == indexing weight by token_ids. All three dtypes
# tested. Output dtype follows *weight* (the float operand), never token_ids
# (which stay int64). The gather is lossless, so the fp32 reference equals
# weight.float()[token_ids] exactly.
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16, torch.float16))
def test_native_embedding_matches_gather_reference(dtype: torch.dtype):
    token_ids = _rand_ids((2, 5), seed=1)
    weight = _rand_weight(seed=1, dtype=dtype)

    fp32_reference = weight.float()[token_ids]
    result = NativeEmbeddingOp().forward(token_ids, weight)

    # forward: output dtype follows weight; lossless gather -> bitwise equal
    # to the reference cast back down.
    assert result.dtype == dtype
    assert torch.equal(result, fp32_reference.to(dtype))
    # forward_fp32: forced fp32 output == ground truth.
    assert torch.equal(NativeEmbeddingOp().forward_fp32(token_ids, weight), fp32_reference)


# Output shape must be token_ids.shape + (hidden,).
def test_native_embedding_output_shape():
    token_ids = _rand_ids((3, 7), seed=2)
    weight = _rand_weight(seed=2)
    out = NativeEmbeddingOp().forward(token_ids, weight)
    assert out.shape == (3, 7, _HIDDEN)


# Non-int64 ids (e.g. int32) must be tolerated: the op casts via .long().
def test_native_embedding_accepts_non_int64_ids():
    token_ids = _rand_ids((2, 4), seed=3).to(torch.int32)
    weight = _rand_weight(seed=3)
    out = NativeEmbeddingOp().forward(token_ids, weight)
    assert torch.equal(out, weight.float()[token_ids.long()].to(weight.dtype))


# Axis A -- batch invariance, bitwise (the WS1 "aligned" property). A token's
# embedding must not depend on how many other tokens share the batch. Compute
# on the full input once, then slice -- never compute a slice on its own (that
# would let the golden source carry its own batch dependence). Trivially true
# for a gather, but asserted explicitly to guard the contract.
def test_embedding_batch_invariance_slice():
    op = NativeEmbeddingOp()
    token_ids = _rand_ids((8, 32), seed=4)
    weight = _rand_weight(seed=4)
    full = op.forward_fp32(token_ids, weight)  # compute on full batch...
    assert torch.equal(op.forward_fp32(token_ids[:1], weight), full[:1])  # ...then slice
    assert torch.equal(op.forward_fp32(token_ids[3:5], weight), full[3:5])


def test_embedding_batch_invariance_with_padding():
    """Padding extra positions must not perturb the real ones (bitwise).

    Mimics a variable-length batch: real token ids followed by padding ids
    along the seq dim; the real prefix must match the no-padding result.
    """
    op = NativeEmbeddingOp()
    weight = _rand_weight(seed=5)
    real = _rand_ids((4, 10), seed=5)
    pad = _rand_ids((4, 6), seed=99)  # 6 extra padding positions
    padded = torch.cat([real, pad], dim=1)  # concat along seq
    assert torch.equal(op.forward_fp32(padded, weight)[:, :10], op.forward_fp32(real, weight))


# Purity -- neither token_ids nor weight may be mutated in place.
def test_embedding_inputs_not_mutated():
    op = NativeEmbeddingOp()
    token_ids = _rand_ids((2, 8), seed=6)
    weight = _rand_weight(seed=6)
    ids_c, w_c = token_ids.clone(), weight.clone()
    op.forward(token_ids, weight)
    op.forward_fp32(token_ids, weight)
    assert torch.equal(token_ids, ids_c) and torch.equal(weight, w_c)


# Gradient flows (fp32 autograd = backward golden source). Gradient is only
# defined for weight (token_ids are integer indices). The weight gradient is
# sparse: only the rows that were indexed are non-zero; unused rows stay 0.
#
# Reproducibility caveat: embedding backward is a scatter-add -- repeated token
# ids accumulate into the same weight.grad row. On CUDA that uses atomic adds
# (nondeterministic ordering), so the weight gradient is bit-exact across runs
# only under torch.use_deterministic_algorithms(True). Since forward_fp32 is the
# backward golden source, we enable it here and assert the gradient is bitwise
# reproducible across two independent backward passes. CPU is always deterministic.
def test_embedding_gradient_flows_to_weight():
    op = NativeEmbeddingOp()
    # Small vocab -> some rows unused, and repeated ids -> the scatter-add path
    # (multiple contributions into one row) is actually exercised.
    token_ids = _rand_ids((2, 4), seed=7, vocab=10)

    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        weight_a = _rand_weight(vocab=10, seed=7).requires_grad_(True)
        op.forward_fp32(token_ids, weight_a).sum().backward()
        grad_a = weight_a.grad

        weight_b = _rand_weight(vocab=10, seed=7).requires_grad_(True)
        op.forward_fp32(token_ids, weight_b).sum().backward()
        grad_b = weight_b.grad
    finally:
        torch.use_deterministic_algorithms(prev)

    assert torch.isfinite(grad_a).all()
    # Backward golden source must be bitwise reproducible (Axis-A for gradients).
    assert torch.equal(grad_a, grad_b)
    # Sparse gradient: rows never indexed stay exactly zero.
    used = torch.unique(token_ids).tolist()
    unused = torch.tensor([i for i in range(10) if i not in used])
    if len(unused):
        assert torch.equal(grad_a[unused], torch.zeros_like(grad_a[unused]))


# Registry dispatch -- "embedding" resolves to NativeEmbeddingOp (matches the
# PYTORCH_NATIVE_EMBEDDING entry + the per-platform priority-map additions).
def test_registry_dispatches_native_embedding_op():
    assert isinstance(kernel_registry.get_op("embedding"), NativeEmbeddingOp)


# --------------------------------------------------------------------------- #
# Qwen3-8B real-shape smoke test
# --------------------------------------------------------------------------- #
# Exercises the actual embedding table dims (vocab=151936, hidden=4096). The
# fp32 weight is ~2.5 GB, so this is GPU-only and skips when CUDA is absent or
# there is not enough free memory. The shrunk-dim tests above already cover the
# logic; this one validates the real index range (incl. boundary ids 0 and
# vocab-1) and the real hidden width, with a small (batch, seq) load point.
def _enough_gpu_memory(num_bytes: int) -> bool:
    if not torch.cuda.is_available():
        return False
    free, _ = torch.cuda.mem_get_info()
    return free > int(num_bytes * 1.5)  # headroom for the gathered output


@pytest.mark.skipif(
    not _enough_gpu_memory(_QWEN3_VOCAB * _QWEN3_HIDDEN * 4),
    reason="needs a CUDA GPU with room for the ~2.5 GB fp32 Qwen3-8B embedding table",
)
def test_native_embedding_qwen3_8b_real_shape():
    device = torch.device("cuda")
    op = NativeEmbeddingOp()

    # SMALL load point (batch=2, seq=16) at the real model dims.
    gen = torch.Generator(device=device).manual_seed(0)
    token_ids = torch.randint(
        0, _QWEN3_VOCAB, (2, 16), generator=gen, dtype=torch.int64, device=device
    )
    # Pin boundary ids so the full vocab range is actually indexed.
    token_ids[0, 0] = 0
    token_ids[0, 1] = _QWEN3_VOCAB - 1
    weight = torch.randn(
        _QWEN3_VOCAB, _QWEN3_HIDDEN, generator=gen, dtype=torch.float32, device=device
    )

    out = op.forward_fp32(token_ids, weight)
    assert out.shape == (2, 16, _QWEN3_HIDDEN)
    assert out.dtype == torch.float32
    # Lossless gather: bitwise equal to direct indexing.
    assert torch.equal(out, weight[token_ids])
    # Axis A: compute on full batch, then slice (no per-slice recompute).
    assert torch.equal(op.forward_fp32(token_ids[:1], weight), out[:1])
