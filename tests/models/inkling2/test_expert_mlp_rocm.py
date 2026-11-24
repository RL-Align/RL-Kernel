# SPDX-License-Identifier: Apache-2.0
"""INK-26 ROCm softcapped expert gate/up tests (MI300X)."""

import pytest
import torch

from rl_engine.backends.extension import _C, _EXT_AVAILABLE
from rl_engine.models.inkling2.expert_mlp import expert_gate_up, expert_gate_up_rocm

pytestmark = pytest.mark.skipif(
    torch.version.hip is None
    or not torch.cuda.is_available()
    or not _EXT_AVAILABLE
    or not hasattr(_C, "inkling2_expert_gate_up_forward"),
    reason="ROCm device and Inkling 2 expert gate/up extension required",
)

DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _make(rows, hidden, inter, dtype, seed=0, scale=1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def r(*shape, s=1.0):
        return (torch.randn(*shape, device="cuda", generator=g) * s).to(dtype)

    return (
        r(rows, hidden, s=scale),
        r(inter, hidden, s=hidden**-0.5),
        r(inter, hidden, s=hidden**-0.5),
        r(inter),
        r(inter),
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(1, 5, 4), (3, 70, 67), (65, 129, 130), (0, 8, 8)])
@pytest.mark.parametrize("use_bias", [True, False])
def test_forward_matches_reference(dtype, shape, use_bias):
    hidden, gw, uw, gb, ub = _make(*shape, dtype)
    if not use_bias:
        gb = ub = None
    out = expert_gate_up_rocm(hidden, gw, uw, gb, ub)
    ref = expert_gate_up(hidden, gw, uw, gb, ub)
    assert out.shape == ref.shape and out.dtype == dtype
    tol = 1e-4 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(out.float(), ref.float(), atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", DTYPES)
def test_real_inkling_shape_forward_and_determinism(dtype):
    # H2304 / I1152, gate softcap 4 / up softcap 25, large activations to exercise the caps
    hidden, gw, uw, gb, ub = _make(70, 2304, 1152, dtype, seed=4, scale=8.0)
    a = expert_gate_up_rocm(hidden, gw, uw, gb, ub)
    b = expert_gate_up_rocm(hidden, gw, uw, gb, ub)
    assert torch.equal(a, b)
    ref = expert_gate_up(hidden, gw, uw, gb, ub)
    tol = 1e-3 if dtype == torch.float32 else 5e-2
    torch.testing.assert_close(a.float(), ref.float(), atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", DTYPES)
def test_batch_invariance_bitwise(dtype):
    hidden, gw, uw, gb, ub = _make(75, 130, 131, dtype, seed=6)
    base = expert_gate_up_rocm(hidden, gw, uw, gb, ub)
    assert torch.equal(expert_gate_up_rocm(hidden[:9], gw, uw, gb, ub), base[:9])
    assert torch.equal(expert_gate_up_rocm(hidden[40:41], gw, uw, gb, ub), base[40:41])
    other = hidden.clone()
    other[30:] = torch.randn_like(other[30:])
    assert torch.equal(expert_gate_up_rocm(other, gw, uw, gb, ub)[:30], base[:30])
    perm = torch.randperm(75, device="cuda")
    shuf = expert_gate_up_rocm(hidden[perm], gw, uw, gb, ub)
    assert torch.equal(shuf, base[perm])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(5, 70, 67), (33, 129, 65)])
def test_backward_matches_reference_and_is_deterministic(dtype, shape):
    hidden, gw, uw, gb, ub = _make(*shape, dtype, seed=8, scale=3.0)
    leaves = [x.requires_grad_(True) for x in (hidden, gw, uw, gb, ub)]
    out = expert_gate_up_rocm(*leaves)
    dy = torch.randn_like(out)
    out.backward(dy)
    grads = [x.grad.clone() for x in leaves]
    for x in leaves:
        x.grad = None
    expert_gate_up_rocm(*leaves).backward(dy)
    for g, x in zip(grads, leaves, strict=True):
        assert torch.equal(g, x.grad)
        x.grad = None

    ref_leaves = [x.detach().clone().requires_grad_(True) for x in leaves]
    ref = expert_gate_up(*ref_leaves)
    ref.backward(dy)
    tol = 1e-3 if dtype == torch.float32 else 5e-2
    for g, r in zip(grads, ref_leaves, strict=True):
        torch.testing.assert_close(g.float(), r.grad.float(), atol=tol, rtol=tol)


def test_rejects_non_positive_caps_and_cpu():
    hidden, gw, uw, _, _ = _make(2, 8, 8, torch.float32)
    with pytest.raises(ValueError, match="positive"):
        expert_gate_up_rocm(hidden, gw, uw, gate_cap=0.0)
    with pytest.raises(ValueError):
        expert_gate_up_rocm(hidden.cpu(), gw.cpu(), uw.cpu())
