# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Checks specific to the ROCm mHC head op (DSV4 RFC task 8, #508).

Batch invariance and determinism of mHC head are checked in
tests/backends/rocm/test_rocm_mhc_batch_invariance.py, those of the DeepSeek-V4
layer end in tests/models/deepseek_v4/test_layer_end_batch_invariance.py; the
reference itself in
tests/reference/test_mhc_reference.py. This file holds what only applies to
the aiter op: it matches the fp32 reference in its output and residual
gradient; empty batch, the shipped table, the registry selection and the
semantic descriptor.
"""

import pytest
import torch

from rl_engine.backends.rocm import config_utils
from rl_engine.backends.rocm.mhc.aiter_mhc import (
    FIXED_HEAD_CONFIG,
    HEAD_BIT_PARAMS,
    check_head_inputs,
)
from rl_engine.reference.mhc import NativeMHCHeadOp
from rl_engine.runtime.registry import kernel_registry
from rl_engine.runtime.semantic_registry import OperatorRequirements
from tests.helpers.contract_tolerance import assert_matches_fp32_reference

_N, _C = 4, 4096  # DeepSeek-V4-Flash-0731: hc_mult, hidden_size


def _aiter_head_op():
    if not (torch.cuda.is_available() and torch.version.hip is not None):
        return None
    try:
        from rl_engine.backends.rocm.mhc.aiter_mhc import AiterMHCHeadOp

        return AiterMHCHeadOp()
    except RuntimeError:
        return None


_AITER_OP = _aiter_head_op()
requires_aiter = pytest.mark.skipif(
    _AITER_OP is None, reason="ROCm GPU with aiter DeepSeek-V4 mHC ops required."
)


def _params():
    gen = torch.Generator(device="cuda").manual_seed(0)
    k = _N * _C
    fn = torch.randn(_N, k, generator=gen, device="cuda") / k**0.5
    scale = 0.5 + torch.rand(1, generator=gen, device="cuda")
    base = 0.1 * torch.randn(_N, generator=gen, device="cuda")
    return fn, scale, base


def _residual(m, *, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(m, _N, _C, generator=gen, device="cuda").to(torch.bfloat16)


def _forward_backward(op, residual, grad_out):
    leaf = residual.detach().clone().requires_grad_(True)
    out = op(leaf, *_params())
    out.backward(grad_out)
    return out.detach(), leaf.grad.detach()


# Correctness: aiter mHC head output and residual gradient against the fp32 reference.
# Only the residual gradient is checked: head parameters are frozen in the first
# round (LoRA-only training).
@requires_aiter
def test_matches_reference_forward_and_grad():
    residual = _residual(64, seed=10)
    grad_out = _residual(64, seed=11)[:, 0]
    out, grad = _forward_backward(_AITER_OP, residual, grad_out)
    ref_out, ref_grad = _forward_backward(NativeMHCHeadOp().forward_fp32, residual, grad_out)
    assert_matches_fp32_reference(
        out, ref_out, judgment="forward_accuracy", op_class="reduction", name="head"
    )
    assert_matches_fp32_reference(
        grad, ref_grad, judgment="gradient_accuracy", op_class="reduction", name="dResidual"
    )


@requires_aiter
def test_empty_batch_forward():
    assert _AITER_OP(_residual(0, seed=50), *_params()).shape == (0, _C)


@requires_aiter
def test_registry_selects_aiter_on_rocm():
    assert type(kernel_registry.get_op("mhc_head", device="cuda")).__name__ == "AiterMHCHeadOp"


def test_shipped_table_loads_and_is_used(monkeypatch):
    monkeypatch.setattr(config_utils, "_device_arch", lambda index: "gfx942")
    config_utils.load_table.cache_clear()
    config_utils._bucket_table.cache_clear()
    config_utils._SELECTED.clear()
    try:
        file_name = f"MHC_HEAD-n={_N}-C={_C}.json"
        path = config_utils.CONFIG_ROOT / "gfx942" / "mhc_head" / file_name
        table = config_utils.load_table(path, HEAD_BIT_PARAMS)
        assert table is not None and "any" in table
        chosen = config_utils.get_config(
            "mhc_head", file_name, 1, FIXED_HEAD_CONFIG, HEAD_BIT_PARAMS, torch.device("cuda", 0)
        )
        assert chosen == table["M_LEQ_1"]
    finally:
        config_utils.load_table.cache_clear()
        config_utils._bucket_table.cache_clear()
        config_utils._SELECTED.clear()


def test_semantic_descriptor_declares_forward_batch_invariance_on_rocm():
    resolution = kernel_registry.semantic.session().resolve(
        semantic_op="mhc_head",
        requested_backend="rlkernel.rocm.aiter_mhc_head_dsv4.batch_invariant.v1",
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
        ({"residual": torch.zeros(2, _N, 64, dtype=torch.float16)}, TypeError),
        ({"residual": torch.zeros(2, 3, 64, dtype=torch.bfloat16)}, ValueError),
        ({"fn": torch.zeros(_N, _N * 64, dtype=torch.bfloat16)}, TypeError),
        ({"scale": torch.zeros(3)}, ValueError),
        ({"base": torch.zeros(_N + 1)}, ValueError),
    ],
    ids=["fp16_residual", "three_streams", "bf16_fn", "scale_shape", "base_shape"],
)
def test_unsupported_inputs_are_rejected(change, error):
    inputs = {
        "residual": torch.zeros(2, _N, 64, dtype=torch.bfloat16),
        "fn": torch.zeros(_N, _N * 64),
        "scale": torch.zeros(1),
        "base": torch.zeros(_N),
    }
    check_head_inputs(**inputs)
    with pytest.raises(error):
        check_head_inputs(**dict(inputs, **change))
