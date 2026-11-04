# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Checks specific to the ROCm mHC post op (DSV4 RFC task 4, #508).

Batch invariance and determinism are checked for all mHC ops together in
tests/backends/rocm/test_rocm_mhc_batch_invariance.py; the reference itself in
tests/reference/test_mhc_reference.py. This file holds what only applies to
the aiter op:

- it matches the fp32 reference in R_new, dy, dR_old, dPOST and dC;
- the old residual is added exactly once;
- empty batch, the registry selection and the semantic descriptor.
"""

import pytest
import torch

from rl_engine.backends.rocm.mhc.aiter_mhc import check_post_inputs
from rl_engine.reference.mhc import NativeMHCPostOp
from rl_engine.runtime.registry import kernel_registry
from rl_engine.runtime.semantic_registry import OperatorRequirements
from rl_engine.validation.common.batch_invariance import bits_equal
from tests.helpers.contract_tolerance import assert_matches_fp32_reference

_N, _C = 4, 4096  # DeepSeek-V4-Flash-0731: hc_mult, hidden_size
_GRAD_NAMES = ("dy", "dR_old", "dPOST", "dC")


def _aiter_post_op():
    if not (torch.cuda.is_available() and torch.version.hip is not None):
        return None
    try:
        from rl_engine.backends.rocm.mhc.aiter_mhc import AiterMHCPostOp

        return AiterMHCPostOp()
    except RuntimeError:
        return None


_AITER_OP = _aiter_post_op()
requires_aiter = pytest.mark.skipif(
    _AITER_OP is None, reason="ROCm GPU with aiter DeepSeek-V4 mHC ops required."
)


def _inputs(m, *, seed, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, device=device)

    y = randn(m, _C).to(torch.bfloat16)
    residual = randn(m, _N, _C).to(torch.bfloat16)
    post_mix = 2.0 * torch.sigmoid(randn(m, _N))
    comb_mix = torch.softmax(randn(m, _N, _N), dim=-1)
    grad_out = randn(m, _N, _C).to(torch.bfloat16)
    return [y, residual, post_mix, comb_mix], grad_out


def _forward_backward(op, inputs, grad_out):
    leaves = [t.detach().clone().requires_grad_(True) for t in inputs]
    out = op(*leaves)
    out.backward(grad_out)
    return out.detach(), [t.grad.detach() for t in leaves]


# Correctness: aiter output and gradients against the fp32 reference.
@requires_aiter
def test_matches_reference_forward_and_grads():
    inputs, grad_out = _inputs(64, seed=10)
    out, grads = _forward_backward(_AITER_OP, inputs, grad_out)
    ref_out, ref_grads = _forward_backward(NativeMHCPostOp().forward_fp32, inputs, grad_out)
    assert_matches_fp32_reference(
        out, ref_out, judgment="forward_accuracy", op_class="reduction", name="R_new"
    )
    for name, grad, ref_grad in zip(_GRAD_NAMES, grads, ref_grads):
        assert_matches_fp32_reference(
            grad, ref_grad, judgment="gradient_accuracy", op_class="reduction", name=name
        )


# Residual added once: with comb_mix = identity and post_mix = 0 the output must
# be exactly the old residual. A second residual add would double it.
@requires_aiter
def test_residual_added_once():
    (y, residual, post_mix, _), _ = _inputs(64, seed=11)
    comb_identity = torch.eye(_N, device="cuda").expand(64, _N, _N)
    out = _AITER_OP(y, residual, torch.zeros_like(post_mix), comb_identity)
    assert bits_equal(out, residual)


@requires_aiter
def test_empty_batch_forward():
    inputs, _ = _inputs(0, seed=60)
    assert _AITER_OP(*inputs).shape == (0, _N, _C)


@requires_aiter
def test_registry_selects_aiter_on_rocm():
    assert type(kernel_registry.get_op("mhc_post", device="cuda")).__name__ == "AiterMHCPostOp"


def test_semantic_descriptor_declares_forward_batch_invariance_on_rocm():
    resolution = kernel_registry.semantic.session().resolve(
        semantic_op="mhc_post",
        requested_backend="rlkernel.rocm.aiter_mhc_post_dsv4.batch_invariant.v1",
        target="rollout",
        requirements=OperatorRequirements(
            device="rocm", dtype="bfloat16", alignment_properties={"batch_invariant": True}
        ),
    )
    properties = resolution.descriptor.determinism_or_alignment_properties
    assert properties["backward_batch_invariant"] is False


# Unsupported inputs raise instead of running unvalidated (checked on CPU).
@pytest.mark.parametrize(
    "change,error",
    [
        ({"y": torch.zeros(2, 64)}, TypeError),  # fp32 y
        ({"residual": torch.zeros(2, _N, 64)}, TypeError),  # fp32 residual
        ({"y": torch.zeros(2, 32, dtype=torch.bfloat16)}, ValueError),
        ({"post_mix": torch.zeros(2, _N, dtype=torch.bfloat16)}, TypeError),
        ({"post_mix": torch.zeros(2, _N, 2)}, ValueError),
        ({"comb_mix": torch.zeros(2, _N, 2)}, ValueError),
    ],
    ids=["fp32_y", "fp32_residual", "y_shape", "bf16_post_mix", "post_mix_shape", "comb_shape"],
)
def test_unsupported_inputs_are_rejected(change, error):
    inputs = {
        "y": torch.zeros(2, 64, dtype=torch.bfloat16),
        "residual": torch.zeros(2, _N, 64, dtype=torch.bfloat16),
        "post_mix": torch.zeros(2, _N, 1),
        "comb_mix": torch.zeros(2, _N, _N),
    }
    check_post_inputs(**inputs)
    with pytest.raises(error):
        check_post_inputs(**dict(inputs, **change))
