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


def _cpu_fp16_matmul_supported() -> bool:
    """Probe whether this CPU backend implements float16 matmul."""
    try:
        _ = torch.randn(2, 2, dtype=torch.float16) @ torch.randn(2, 2, dtype=torch.float16)
        return True
    except RuntimeError:
        return False


# CPU half-precision matmul is backend/ISA-dependent (AVX512_FP16, AMX) and may
# be unimplemented on some runners -- gate the fp16 axis so a missing kernel
# skips rather than fails the test.
_FP16_IF_CPU_MATMUL_SUPPORTED = pytest.param(
    torch.float16,
    marks=pytest.mark.skipif(
        not _cpu_fp16_matmul_supported(),
        reason="CPU float16 matmul unsupported on this backend",
    ),
)
_DTYPES_AXIS_B = (torch.bfloat16, _FP16_IF_CPU_MATMUL_SUPPORTED)
_DTYPES_AXIS_A = (torch.float32, torch.bfloat16, _FP16_IF_CPU_MATMUL_SUPPORTED)


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
    """Fixed-seed random hidden-state tensor for reproducibility."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(batch, seq, hidden, generator=gen, dtype=dtype)


def _rand_weight(vocab=_VOCAB, hidden=_HIDDEN, *, seed, dtype=torch.float32):
    """Fixed-seed random lm_head weight tensor for reproducibility."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(vocab, hidden, generator=gen, dtype=dtype)


# Correctness of the fp32 ground truth: forward_fp32 == naive fp32 matmul,
# bitwise. forward_fp32 disables autocast/TF32, so it is a true fp32 reference
# unconditionally. The fp32 dtype path (forward) follows the ambient precision
# context, so it is bitwise-equal to the ground truth only when TF32/autocast is
# off (the default here); only bf16/fp16 introduce drift.
def test_native_lm_head_fp32_matches_naive_matmul():
    """forward_fp32 (and fp32 forward, TF32 off) is bitwise-equal to a naive fp32 matmul."""
    hidden = _rand_hidden(2, 5, seed=1)
    weight = _rand_weight(seed=1)

    # Pin TF32 off so the fp32 forward path and the naive reference below are both
    # true fp32 regardless of the machine's global default, making this assertion
    # deterministic. forward_fp32 disables TF32 internally and does not need this.
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        naive = hidden.float() @ weight.float().t()
        assert torch.equal(NativeLMHeadOp().forward_fp32(hidden, weight), naive)
        # fp32 forward path computes in fp32 too -> bitwise equal to ground truth.
        assert torch.equal(NativeLMHeadOp().forward(hidden, weight), naive)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32


def test_forward_fp32_ignores_ambient_autocast_and_restores_tf32():
    """forward_fp32 is a strict fp32 reference under ambient autocast/TF32 settings."""
    op = NativeLMHeadOp()
    hidden = _rand_hidden(2, 5, hidden=32, seed=11)
    weight = _rand_weight(vocab=7, hidden=32, seed=11)
    gen = torch.Generator().manual_seed(11)
    bias = torch.randn(7, generator=gen)
    ref = hidden.float() @ weight.float().t() + bias.float()

    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = op.forward_fp32(hidden, weight, bias=bias)

        assert out.dtype == torch.float32
        assert torch.equal(out, ref)
        assert torch.backends.cuda.matmul.allow_tf32 is True
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32


# GPU counterpart of the test above: TF32 only changes the numbers on CUDA, so
# the "TF32 is disabled" half of forward_fp32 can only be exercised on a GPU.
# With TF32 forced on globally, forward_fp32 must stay true fp32 (tight against a
# high-precision fp64 reference) and never be worse than a plain TF32 matmul. The
# assertion holds whether or not the GPU actually uses TF32 for fp32 matmul.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU to exercise TF32")
def test_forward_fp32_disables_tf32_on_gpu():
    """On a TF32-enabled GPU, forward_fp32 stays true fp32, no worse than a TF32 matmul."""
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(21)
    hidden = torch.randn(2, 16, _HIDDEN, generator=gen, dtype=torch.float32, device=device)
    weight = torch.randn(256, _HIDDEN, generator=gen, dtype=torch.float32, device=device)
    ref = (hidden.double() @ weight.double().t()).float()  # high-precision reference

    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True  # hostile ambient setting
    try:
        strict = NativeLMHeadOp().forward_fp32(hidden, weight)  # must ignore TF32
        tf32 = hidden @ weight.t()  # plain matmul -> TF32 if the GPU supports it
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32

    peak = ref.abs().max().item()
    strict_err = (strict - ref).abs().max().item()
    tf32_err = (tf32 - ref).abs().max().item()
    print(
        f"\n[lm_head fp32-vs-tf32] strict_err={strict_err:.3g} "
        f"tf32_err={tf32_err:.3g} peak={peak:.3g}"
    )
    # forward_fp32 is fp32-tight (well under the TF32 ~10-bit-mantissa drift floor)...
    assert strict_err <= 1.0e-3 * peak
    # ...and never worse than a plain matmul (far tighter when the GPU uses TF32).
    assert strict_err <= tf32_err


# Axis-B accuracy: the low-precision dtype path drifts from the fp32 reference
# by a bounded fraction of the output peak. Errors/stats are printed for the PR.
@pytest.mark.parametrize("dtype", _DTYPES_AXIS_B)
def test_native_lm_head_dtype_path_accuracy(dtype: torch.dtype):
    """Axis-B: the low-precision path drifts from fp32 by a bounded fraction of the output peak."""
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
    """Output shape is hidden.shape[:-1] + (vocab,)."""
    hidden = _rand_hidden(3, 7, seed=3)
    weight = _rand_weight(seed=3)
    out = NativeLMHeadOp().forward(hidden, weight)
    assert out.shape == (3, 7, _VOCAB)


# Bias: None (Qwen3 default) is a plain matmul; a provided [vocab] bias is added.
def test_native_lm_head_bias():
    """bias=None is a plain matmul; a [vocab] bias is added elementwise."""
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
@pytest.mark.parametrize("dtype", _DTYPES_AXIS_A)
def test_lm_head_batch_invariance_slice(dtype: torch.dtype):
    """Axis-A: a row's logits are bitwise-independent of how many rows share the batch."""
    op = NativeLMHeadOp()
    hidden = _rand_hidden(8, 32, seed=5).to(dtype)
    weight = _rand_weight(seed=5).to(dtype)
    with _single_thread():
        full = op.forward(hidden, weight)  # compute on full batch...
        assert torch.equal(op.forward(hidden[:1], weight), full[:1])  # ...then slice
        assert torch.equal(op.forward(hidden[3:5], weight), full[3:5])


@pytest.mark.parametrize("dtype", _DTYPES_AXIS_A)
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
    """Purity: no input tensor is mutated in place."""
    op = NativeLMHeadOp()
    hidden = _rand_hidden(2, 8, seed=7)
    weight = _rand_weight(seed=7)
    gen = torch.Generator().manual_seed(7)
    bias = torch.randn(_VOCAB, generator=gen)
    h_c, w_c, b_c = hidden.clone(), weight.clone(), bias.clone()
    op.forward(hidden, weight, bias=bias)
    op.forward_fp32(hidden, weight, bias=bias)
    assert torch.equal(hidden, h_c) and torch.equal(weight, w_c) and torch.equal(bias, b_c)


# Gradient (fp32 autograd = backward golden source). Backprop a *random*
# cotangent, not out.sum(): an all-ones cotangent collapses both grads to column
# sums (every row of each grad identical), so a transposed / mis-contracted
# backward could still pass under that symmetry. A random dy exercises the full
# contraction. For out = hidden @ weight.t() (weight is HF [V, K]) the exact
# closed forms are dL/dhidden = dy @ weight and dL/dweight = dy^T @ hidden.
def test_lm_head_gradient_flows():
    """fp32 autograd matches the closed-form grads under a random cotangent."""
    op = NativeLMHeadOp()
    hidden = _rand_hidden(2, 4, seed=8).requires_grad_(True)
    weight = _rand_weight(seed=8).requires_grad_(True)
    out = op.forward_fp32(hidden, weight)

    gen = torch.Generator().manual_seed(8)
    dy = torch.randn(out.shape, generator=gen, dtype=out.dtype)
    out.backward(dy)

    assert torch.isfinite(hidden.grad).all() and torch.isfinite(weight.grad).all()
    assert hidden.grad.shape == hidden.shape and weight.grad.shape == weight.shape
    exp_h = dy @ weight.detach()  # [.., V] @ [V, K] -> [.., K]
    exp_w = dy.reshape(-1, _VOCAB).t() @ hidden.detach().reshape(-1, _HIDDEN)  # [V, K]
    torch.testing.assert_close(hidden.grad, exp_h, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(weight.grad, exp_w, rtol=1e-5, atol=1e-5)


# Registry dispatch -- "lm_head" resolves to NativeLMHeadOp.
def test_registry_dispatches_native_lm_head_op():
    """The registry resolves "lm_head" to NativeLMHeadOp."""
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
    """Return True only if CUDA is present and has free memory with headroom."""
    if not torch.cuda.is_available():
        return False
    try:
        free, _ = torch.cuda.mem_get_info()
    except RuntimeError:
        return False
    return free > int(num_bytes * 1.5)  # headroom for the logits output


@pytest.mark.skipif(
    not _enough_gpu_memory(_QWEN3_VOCAB * _QWEN3_HIDDEN * 4),
    reason="needs a CUDA GPU with room for the ~2.5 GB fp32 Qwen3-8B lm_head weight",
)
def test_native_lm_head_qwen3_8b_real_shape():
    """GPU smoke test at real Qwen3-8B dims (vocab=151936, hidden=4096)."""
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
