# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""The Triton backward of the ROCm mHC pre op (#508, #510).

Batch invariance of the input gradients is checked with the forward in
tests/backends/rocm/test_rocm_mhc_batch_invariance.py. This file holds the rest:

- the forward with grad, whose kernels also store the values the backward uses,
  has the bits of the forward without grad and in inference mode;
- input and parameter gradients against the PyTorch reference
  (rl_engine/ops/mhc/training.py), with bf16 inputs as in the model;
- parameter gradients through the canonical backward session, as in
  tests/models/deepseek_v4/test_o_projection_rocm.py: the same bits for every
  chunking of the tokens, row order and backward order, with a padding row;
  ``validate_complete``; trainable parameters without a session raise; frozen
  parameters (the first RL round) need no session;
- backward launch configs do not change the bits; empty batches; the backward
  that ran is recorded.
"""

import pytest
import torch

from rl_engine.ops.autograd.backward_runtime import (
    reset_backward_runtime,
    snapshot_backward_runtime,
)
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session
from rl_engine.ops.mhc import training
from rl_engine.validation.common.batch_invariance import bits_equal
from tests.helpers.contract_tolerance import assert_matches_fp32_reference
from tests.helpers.deepseek_v4_mhc import N, randn, rocm_gpu_available, weights

pytestmark = pytest.mark.skipif(not rocm_gpu_available(), reason="ROCm GPU required.")

_C = 4096  # DeepSeek-V4-Flash-0731 hidden_size
_MIX = 2 * N + N * N


def _triton_mhc():
    from rl_engine.backends.rocm.mhc import triton_mhc

    return triton_mhc


def _gen(seed):
    return torch.Generator(device="cuda").manual_seed(seed)


def _params(c=_C, requires_grad=False):
    w = weights(c)
    params = [w[f"pre_{name}"].clone() for name in ("fn", "scale", "base")]
    return [p.requires_grad_(requires_grad) for p in params]


def _keys(m, start=0):
    rows = torch.arange(start, start + m, device="cuda")
    return torch.stack([torch.zeros_like(rows), rows], 1)


def _close(actual, expected, name, rtol=1e-4):
    """Max error relative to the largest reference value; for the fp32 gradients."""
    scale = expected.abs().max().clamp_min(1e-30)
    error = (actual.float() - expected.float()).abs().max()
    assert error <= rtol * scale, f"{name}: max error {error:.3e}, max |reference| {scale:.3e}"


def _assert_gradients_close(ours, ref):
    """dResidual (bf16) with the WS1 tolerance; d fn, d scale, d base (fp32) with _close."""
    assert_matches_fp32_reference(
        ours[0], ref[0], judgment="gradient_accuracy", op_class="reduction", name="dResidual"
    )
    for name, a, b in zip(("dfn", "dscale", "dbase"), ours[1:], ref[1:]):
        _close(a, b, name)


# ---------------------------------------------------------------------------
# Forward bits: the stores of the saved values change nothing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", [1, 17, 1024])
def test_forward_with_grad_has_the_bits_of_inference(m):
    T = _triton_mhc()
    residual = randn(_gen(1), m, N, _C).to(torch.bfloat16)
    pre = _params()

    def run(residual):
        return list(T.mhc_pre_dsv4(residual, *pre))

    with torch.no_grad():
        plain = run(residual)
    with torch.inference_mode():
        inference = run(residual)
    saving = run(residual.clone().requires_grad_(True))
    assert all(o.grad_fn is not None for o in saving)
    names = ("post_mix", "comb_mix", "layer_input")
    for other in (inference, saving):
        differing = [n for n, a, b in zip(names, plain, other) if not bits_equal(a, b.detach())]
        assert not differing, f"M={m}: {differing}"


# ---------------------------------------------------------------------------
# Gradients against the PyTorch reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sinkhorn_repeat", [0, 1, 20])
def test_pre_gradients_match_reference(sinkhorn_repeat):
    T = _triton_mhc()
    m = 64
    gen = _gen(2)
    residual = randn(gen, m, N, _C).to(torch.bfloat16)
    grads = [randn(gen, m, N, 1), randn(gen, m, N, N), randn(gen, m, _C).to(torch.bfloat16)]
    params = _params(requires_grad=True)

    def run(op, **kwargs):
        leaf = residual.clone().requires_grad_(True)
        for p in params:
            p.grad = None
        outputs = op(leaf, *params, sinkhorn_repeat=sinkhorn_repeat, **kwargs)
        torch.autograd.backward(list(outputs), [g.view_as(o) for g, o in zip(grads, outputs)])
        return [leaf.grad] + [p.grad.clone() for p in params]

    with canonical_backward_session() as session:
        ours = run(T.mhc_pre_dsv4, logical_keys=_keys(m), parameter_id="pre")
        session.validate_complete()
    ref = run(lambda *x, **k: training.mhc_pre(*x, backend="rocm", **k))
    _assert_gradients_close(ours, ref)


# ---------------------------------------------------------------------------
# Parameter gradients through the canonical session (test_o_projection_rocm.py:58)
# ---------------------------------------------------------------------------


def _session_run(residual, grads, params, keys, chunks, perm, reverse_backward):
    """(outputs per row in perm order, dResidual, d fn, d scale, d base)."""
    T = _triton_mhc()
    op = T.TritonMHCPreOp()
    leaf = residual.clone().requires_grad_(True)
    for p in params:
        p.grad = None
    with canonical_backward_session() as session:
        pieces, start = [], 0
        for count in chunks:
            index = perm[start : start + count]
            outputs = op(leaf[index], *params, logical_keys=keys[index], parameter_id="mhc_pre")
            outputs = list(outputs)
            pieces.append((index, outputs))
            start += count
        for index, outputs in reversed(pieces) if reverse_backward else pieces:
            torch.autograd.backward(outputs, [g[index] for g in grads])
        session.validate_complete()
    forward = [torch.cat([o[i].detach() for _, o in pieces]) for i in range(len(pieces[0][1]))]
    return forward, [leaf.grad] + [p.grad.clone() for p in params]


@pytest.mark.parametrize("chunks", [[1, 1, 1, 1, 1, 1, 1], [3, 4], [7]])
@pytest.mark.parametrize("reverse_backward", [False, True])
def test_bi_chunks_permutation_and_backward_order(chunks, reverse_backward):
    m = 7
    gen = _gen(5)
    residual = randn(gen, m, N, _C).to(torch.bfloat16)
    grads = [randn(gen, m, N), randn(gen, m, N, N), randn(gen, m, _C).to(torch.bfloat16)]
    params = _params(requires_grad=True)
    keys = _keys(m, start=100)
    keys[5] = -1  # a padding row: zero gradient, no share of the parameter gradients
    identity = torch.arange(m, device="cuda")
    full_out, full = _session_run(residual, grads, params, keys, [m], identity, False)
    assert full[0][5].eq(0).all() and full[0][4].ne(0).any()

    perm = torch.tensor([6, 4, 0, 5, 1, 3, 2], device="cuda")
    out, got = _session_run(residual, grads, params, keys, chunks, perm, reverse_backward)
    for a, b in zip(out, full_out):
        assert bits_equal(a, b[perm])
    names = ("dResidual", "dfn", "dscale", "dbase")
    differing = [n for n, a, b in zip(names, got, full) if not bits_equal(a, b)]
    assert not differing, differing


def test_trainable_parameters_require_a_session():
    T = _triton_mhc()
    residual = randn(_gen(6), 3, N, _C).to(torch.bfloat16)
    params = _params()
    params[1].requires_grad_(True)
    op = T.mhc_pre_dsv4
    with pytest.raises(RuntimeError, match="canonical_backward_session"):
        op(residual, *params)
    with canonical_backward_session(), pytest.raises(ValueError, match="logical_keys"):
        op(residual, *params)
    with torch.no_grad():
        op(residual, *params)


def test_frozen_parameters_compute_no_parameter_gradient():
    T = _triton_mhc()
    leaf = randn(_gen(7), 5, N, _C).to(torch.bfloat16).requires_grad_(True)
    pre = _params()
    post_mix, comb_mix, layer_input = T.TritonMHCPreOp()(leaf, *pre)
    layer_input.float().sum().backward()
    assert leaf.grad is not None and leaf.grad.dtype == torch.bfloat16
    assert all(p.grad is None for p in pre)


# ---------------------------------------------------------------------------
# Launch configs, empty batch, record
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config",
    [
        {"MIX_BLOCK_M": 1},
        {"MIX_BLOCK_M": 32, "mix_num_warps": 8},
        {"DX_BLOCK_M": 8, "DX_BLOCK_C": 64, "dx_num_warps": 2},
        {"DX_BLOCK_M": 64, "DX_BLOCK_C": 256, "dx_num_warps": 8},
    ],
    ids=lambda c: "-".join(f"{k}={v}" for k, v in c.items()),
)
def test_backward_launch_config_does_not_change_bits(config):
    T = _triton_mhc()
    pre = _params()
    for m in (17, 1024):
        gen = _gen(8)
        residual = randn(gen, m, N, _C).to(torch.bfloat16)
        g_pre = [randn(gen, m, N, 1), randn(gen, m, N, N), randn(gen, m, _C).to(torch.bfloat16)]

        def run(backward_config):
            leaf = residual.clone().requires_grad_(True)
            outputs = T.mhc_pre_dsv4(leaf, *pre, backward_config=backward_config)
            torch.autograd.backward(list(outputs), g_pre)
            return [leaf.grad]

        names = ("dResidual",)
        differing = [n for n, a, b in zip(names, run(None), run(config)) if not bits_equal(a, b)]
        assert not differing, f"M={m}: {differing}"


def test_empty_batch_backward():
    T = _triton_mhc()
    leaf = torch.zeros(0, N, _C, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    _, _, layer_input = T.TritonMHCPreOp()(leaf, *_params())
    layer_input.float().sum().backward()
    assert leaf.grad.shape == (0, N, _C)


def test_backward_kernels_are_recorded():
    T = _triton_mhc()
    reset_backward_runtime()
    leaf = randn(_gen(9), 3, N, _C).to(torch.bfloat16).requires_grad_(True)
    _, _, layer_input = T.TritonMHCPreOp()(leaf, *_params())
    layer_input.float().sum().backward()
    events = snapshot_backward_runtime()
    for kind, kernel in (
        ("mhc_pre", "_mhc_pre_backward_mix_kernel"),
    ):
        assert events[kind]["execution_count"] == 1
        assert any(k.endswith(kernel) for k in events[kind]["kernel_ids"])
        assert events[kind]["family"] == "rocm-triton"
