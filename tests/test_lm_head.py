# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Tests for NativeLMHeadOp (ISSUE #108 WS1 ground-truth baseline).

The lm_head projects hidden states to vocab logits: out = hidden @ weight.t()
(+ bias). Unlike embedding (a lossless gather), this is a *reduction* over the
hidden dimension, so:

  * Axis-B (accuracy): the low-precision ``forward`` path accumulates in the
    input dtype and drifts from the fp32 ``forward_fp32`` ground truth. It is
    checked with a tolerance (relative to the output peak magnitude), not
    bitwise -- elementwise rtol is useless here because many logits are near
    zero while the accumulated error tracks the reduction length, not the
    output value.
  * Axis-A (batch invariance): still bitwise within a single dtype, but only
    once the CPU reduction order is pinned. Multi-threaded CPU GEMM splits the
    K (=hidden) reduction across threads differently depending on the M
    (=batch*seq) dimension, which silently breaks bitwise batch invariance for
    large hidden. ``_single_thread`` fixes the reduction order; this is the
    local stand-in for the planned testing/determinism.py::deterministic_context.
"""

import contextlib

import pytest
import torch

from rl_engine.kernels.ops.pytorch.linear.lm_head import NativeLMHeadOp
from rl_engine.kernels.registry import kernel_registry

# Qwen3-8B architecture (synthetic tensors, no weight download). The vocab is
# shrunk -- it is just the number of independent output dot products, so the
# logic is identical at any vocab. The reduction dim (hidden) is kept at the
# *real* 4096 so the Axis-B drift is representative: weight is only
# [vocab=128, hidden=4096] ~ 2 MB, trivial on CPU. The full vocab is exercised
# by the GPU smoke test below.
_VOCAB = 128  # shrunk; real value: _QWEN3_VOCAB
_HIDDEN = 4096  # real Qwen3-8B reduction dim -- kept real for representative drift

# Real Qwen3-8B output-projection dims: 151936 x 4096 ~ 2.49 GB in fp32.
_QWEN3_VOCAB = 151936
_QWEN3_HIDDEN = 4096

# Axis-B: max abs error as a fraction of the output peak magnitude. Calibrated
# from measured SMALL drift (bf16 ~0.3% of peak, fp16 ~0.04%) with headroom.
_DTYPE_REL_PEAK = {torch.bfloat16: 1.0e-2, torch.float16: 2.0e-3}


@contextlib.contextmanager
def _single_thread():
    """Pin CPU GEMM to one thread so the K reduction order is M-independent."""
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(prev)


# Shared helpers -- fixed-seed Generator for determinism / reproducibility.
def _rand_hidden(batch, seq, hidden=_HIDDEN, *, seed, dtype=torch.float32):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(batch, seq, hidden, generator=gen, dtype=dtype)


def _rand_weight(vocab=_VOCAB, hidden=_HIDDEN, *, seed, dtype=torch.float32):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(vocab, hidden, generator=gen, dtype=dtype)


# Correctness of the fp32 ground truth: forward_fp32 == naive fp32 matmul,
# bitwise. The fp32 dtype path (forward) is identical to the ground truth, so
# it too is bitwise equal -- only bf16/fp16 introduce drift.
def test_native_lm_head_fp32_matches_naive_matmul():
    hidden = _rand_hidden(2, 5, seed=1)
    weight = _rand_weight(seed=1)

    naive = hidden.float() @ weight.float().t()
    assert torch.equal(NativeLMHeadOp().forward_fp32(hidden, weight), naive)
    # fp32 forward path computes in fp32 too -> bitwise equal to ground truth.
    assert torch.equal(NativeLMHeadOp().forward(hidden, weight), naive)


# Axis-B accuracy: the low-precision dtype path drifts from the fp32 reference
# by a bounded fraction of the output peak. Errors/stats are printed for the PR.
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_native_lm_head_dtype_path_accuracy(dtype: torch.dtype):
    op = NativeLMHeadOp()
    hidden = _rand_hidden(2, 16, seed=2)
    weight = _rand_weight(seed=2)

    ref = op.forward_fp32(hidden, weight)  # fp32 ground truth
    cand = op.forward(hidden.to(dtype), weight.to(dtype))  # dtype path
    assert cand.dtype == dtype

    err = (cand.float() - ref).abs()
    peak = ref.abs().max()
    max_abs, mean_abs = err.max().item(), err.mean().item()
    print(f"\n[lm_head {dtype}] max_abs={max_abs:.4g} mean_abs={mean_abs:.4g} peak={peak:.4g}")
    assert max_abs <= _DTYPE_REL_PEAK[dtype] * peak.item()


# Output shape must be hidden.shape[:-1] + (vocab,).
def test_native_lm_head_output_shape():
    hidden = _rand_hidden(3, 7, seed=3)
    weight = _rand_weight(seed=3)
    out = NativeLMHeadOp().forward(hidden, weight)
    assert out.shape == (3, 7, _VOCAB)


# Bias: None (Qwen3 default) is a plain matmul; a provided [vocab] bias is added.
def test_native_lm_head_bias():
    op = NativeLMHeadOp()
    hidden = _rand_hidden(2, 4, seed=4)
    weight = _rand_weight(seed=4)
    gen = torch.Generator().manual_seed(4)
    bias = torch.randn(_VOCAB, generator=gen)

    no_bias = op.forward_fp32(hidden, weight)
    with_bias = op.forward_fp32(hidden, weight, bias=bias)
    assert torch.equal(with_bias, no_bias + bias.float())
    # default is bias=None (== no bias term).
    assert torch.equal(op.forward_fp32(hidden, weight, bias=None), no_bias)


# Axis A -- batch invariance, bitwise (the WS1 "aligned" property). A row's
# logits must not depend on how many other rows share the batch. Compute on the
# full input once, then slice -- never compute a slice on its own. Requires the
# pinned single-thread reduction order (see module docstring).
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16, torch.float16))
def test_lm_head_batch_invariance_slice(dtype: torch.dtype):
    op = NativeLMHeadOp()
    hidden = _rand_hidden(8, 32, seed=5).to(dtype)
    weight = _rand_weight(seed=5).to(dtype)
    with _single_thread():
        full = op.forward(hidden, weight)  # compute on full batch...
        assert torch.equal(op.forward(hidden[:1], weight), full[:1])  # ...then slice
        assert torch.equal(op.forward(hidden[3:5], weight), full[3:5])


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16, torch.float16))
def test_lm_head_batch_invariance_with_padding(dtype: torch.dtype):
    """Padding extra seq positions must not perturb the real ones (bitwise)."""
    op = NativeLMHeadOp()
    weight = _rand_weight(seed=6).to(dtype)
    real = _rand_hidden(4, 10, seed=6).to(dtype)
    pad = _rand_hidden(4, 6, seed=99).to(dtype)  # 6 extra padding positions
    padded = torch.cat([real, pad], dim=1)  # concat along seq
    with _single_thread():
        assert torch.equal(op.forward(padded, weight)[:, :10], op.forward(real, weight))


# Purity -- no input may be mutated in place.
def test_lm_head_inputs_not_mutated():
    op = NativeLMHeadOp()
    hidden = _rand_hidden(2, 8, seed=7)
    weight = _rand_weight(seed=7)
    gen = torch.Generator().manual_seed(7)
    bias = torch.randn(_VOCAB, generator=gen)
    h_c, w_c, b_c = hidden.clone(), weight.clone(), bias.clone()
    op.forward(hidden, weight, bias=bias)
    op.forward_fp32(hidden, weight, bias=bias)
    assert torch.equal(hidden, h_c) and torch.equal(weight, w_c) and torch.equal(bias, b_c)


# Gradient (fp32 autograd = backward golden source). For out.sum() the grads
# have closed forms: dL/dhidden = weight.sum(0) per row; dL/dweight = sum of
# hidden over (batch, seq) per vocab row.
def test_lm_head_gradient_flows():
    op = NativeLMHeadOp()
    hidden = _rand_hidden(2, 4, seed=8).requires_grad_(True)
    weight = _rand_weight(seed=8).requires_grad_(True)
    op.forward_fp32(hidden, weight).sum().backward()

    assert torch.isfinite(hidden.grad).all() and torch.isfinite(weight.grad).all()
    assert hidden.grad.shape == hidden.shape and weight.grad.shape == weight.shape
    exp_h = weight.detach().sum(dim=0).expand_as(hidden.grad)
    exp_w = hidden.detach().sum(dim=(0, 1)).expand_as(weight.grad)
    assert torch.allclose(hidden.grad, exp_h, atol=1e-4, rtol=1e-4)
    assert torch.allclose(weight.grad, exp_w, atol=1e-4, rtol=1e-4)


# Registry dispatch -- "lm_head" resolves to NativeLMHeadOp.
def test_registry_dispatches_native_lm_head_op():
    assert isinstance(kernel_registry.get_op("lm_head"), NativeLMHeadOp)


# --------------------------------------------------------------------------- #
# Qwen3-8B real-shape smoke test
# --------------------------------------------------------------------------- #
# Exercises the real output-projection dims (vocab=151936, hidden=4096). The
# fp32 weight is ~2.5 GB, so this is GPU-only and skips when CUDA is absent or
# there is not enough free memory. The shrunk-vocab tests above already cover
# the logic; this validates the real vocab width and hidden reduction length at
# a small (batch, seq) load point.
def _enough_gpu_memory(num_bytes: int) -> bool:
    if not torch.cuda.is_available():
        return False
    free, _ = torch.cuda.mem_get_info()
    return free > int(num_bytes * 1.5)  # headroom for the logits output


@pytest.mark.skipif(
    not _enough_gpu_memory(_QWEN3_VOCAB * _QWEN3_HIDDEN * 4),
    reason="needs a CUDA GPU with room for the ~2.5 GB fp32 Qwen3-8B lm_head weight",
)
def test_native_lm_head_qwen3_8b_real_shape():
    device = torch.device("cuda")
    op = NativeLMHeadOp()

    # SMALL load point (batch=2, seq=16) at the real model dims.
    gen = torch.Generator(device=device).manual_seed(0)
    hidden = torch.randn(2, 16, _QWEN3_HIDDEN, generator=gen, dtype=torch.float32, device=device)
    weight = torch.randn(
        _QWEN3_VOCAB, _QWEN3_HIDDEN, generator=gen, dtype=torch.float32, device=device
    )

    out = op.forward_fp32(hidden, weight)
    assert out.shape == (2, 16, _QWEN3_VOCAB)
    assert out.dtype == torch.float32
    # Bitwise equal to the naive fp32 matmul (same call, same inputs).
    assert torch.equal(out, hidden @ weight.t())
    # NB: Axis-A bitwise is asserted on CPU (single-thread reduction) above.
    # On GPU it is NOT free -- cuBLAS also splits K by the M dimension, so a
    # batch-invariant GEMM is a downstream kernel concern, not validated here.
