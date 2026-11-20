# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Checks specific to the ROCm mHC head op (DSV4 RFC task 8, #508).

Batch invariance and determinism of mHC head are checked in
tests/backends/rocm/test_rocm_mhc_batch_invariance.py, those of the DeepSeek-V4
layer end in tests/models/deepseek_v4/test_layer_end_batch_invariance.py; the
PyTorch reference (rl_engine/ops/mhc/training.py) in
tests/ops/mhc/test_p1_training.py. This file holds what only applies to the
ROCm op (``TritonMHCHeadOp``): it matches the PyTorch reference in its output
and residual gradient; the entry's ``rocm_triton`` backend runs it; launch
configs do not change its bits; empty batch, the registry selection and the
semantic descriptor.
"""

import pytest
import torch

from rl_engine.backends.rocm.mhc.triton_mhc import check_head_inputs, mhc_head_dsv4
from rl_engine.ops.mhc import training
from rl_engine.runtime.registry import kernel_registry
from rl_engine.runtime.semantic_registry import OperatorRequirements
from rl_engine.validation.common.batch_invariance import bits_equal
from tests.helpers.contract_tolerance import assert_matches_fp32_reference

_N, _C = 4, 4096  # DeepSeek-V4-Flash-0731: hc_mult, hidden_size
# The descriptors declare one GPU without tensor parallelism.
_SINGLE_GPU = {"world_size": 1, "tensor_parallel_size": 1}


def _head_op():
    if not (torch.cuda.is_available() and torch.version.hip is not None):
        return None
    from rl_engine.backends.rocm.mhc.triton_mhc import TritonMHCHeadOp

    return TritonMHCHeadOp()


_OP = _head_op()
requires_rocm = pytest.mark.skipif(_OP is None, reason="ROCm GPU required.")


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


# Correctness: mHC head output and residual gradient against the PyTorch reference.
# Only the residual gradient is checked: head parameters are frozen in the first
# round (LoRA-only training).
@requires_rocm
def test_matches_reference_forward_and_grad():
    residual = _residual(64, seed=10)
    grad_out = _residual(64, seed=11)[:, 0]
    out, grad = _forward_backward(_OP, residual, grad_out)
    ref_out, ref_grad = _forward_backward(
        lambda *x: training.mhc_head(*x, backend="rocm"), residual, grad_out
    )
    assert_matches_fp32_reference(
        out, ref_out, judgment="forward_accuracy", op_class="reduction", name="head"
    )
    assert_matches_fp32_reference(
        grad, ref_grad, judgment="gradient_accuracy", op_class="reduction", name="dResidual"
    )


# No launch parameter changes the bits.
@requires_rocm
@pytest.mark.parametrize(
    "config",
    [
        {"BLOCK_M": 16},
        {"BLOCK_M": 64, "num_warps": 8},
        {"num_warps": 2, "num_stages": 2, "waves_per_eu": 2},
        {"APPLY_BLOCK_M": 64, "APPLY_BLOCK_C": 256, "apply_num_warps": 8},
        {"APPLY_BLOCK_M": 16, "APPLY_BLOCK_C": 64, "apply_num_warps": 2},
    ],
    ids=lambda c: "-".join(f"{k}={v}" for k, v in c.items()),
)
def test_launch_config_does_not_change_bits(config):
    for m in (17, 1024):
        residual = _residual(m, seed=40)
        default = mhc_head_dsv4(residual, *_params())
        assert bits_equal(mhc_head_dsv4(residual, *_params(), config=config), default), f"M={m}"


@requires_rocm
def test_empty_batch_forward():
    assert _OP(_residual(0, seed=50), *_params()).shape == (0, _C)


@requires_rocm
def test_entry_rocm_triton_runs_the_op():
    residual = _residual(17, seed=12)
    want = _OP(residual, *_params())
    assert bits_equal(training.mhc_head(residual, *_params(), backend="rocm_triton"), want)


@requires_rocm
def test_registry_selects_the_triton_op_on_rocm():
    assert type(kernel_registry.get_op("mhc_head", device="cuda")).__name__ == "TritonMHCHeadOp"


def test_semantic_descriptor_declares_forward_batch_invariance_on_rocm():
    resolution = kernel_registry.semantic.session().resolve(
        semantic_op="mhc_head",
        requested_backend="rlkernel.rocm.triton_mhc_head_dsv4.batch_invariant.v1",
        target="rollout",
        requirements=OperatorRequirements(
            device="rocm",
            dtype="bfloat16",
            alignment_properties={"batch_invariant": True},
            topology=_SINGLE_GPU,
        ),
    )
    properties = resolution.descriptor.determinism_or_alignment_properties
    assert properties["backward_batch_invariant"] is True


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
