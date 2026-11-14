# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""The DeepSeek-V4 layer end against its reference backend.

mHC head -> final RMSNorm -> lm_head -> raw logprob (DSV4 RFC task 8, #508). The
reference backend runs mHC head in fp32 with bf16 output, task 5's serial fp32
RMSNorm reference, lm_head in fp32 and the PyTorch logprob. Batch invariance of
the layer end is checked in test_layer_end_batch_invariance.py.

The CPU tests run the reference backend: input checks and the empty batch.
"""

import pytest
import torch

from rl_engine.contracts.operators.rmsnorm_residual import RMSNormResidualContractError
from rl_engine.models.deepseek_v4.layer_end import LAYER_END_STAGES, mhc_layer_end
from rl_engine.ops.norm.rmsnorm_residual import rmsnorm_residual
from tests.helpers.contract_tolerance import assert_matches_fp32_reference
from tests.helpers.deepseek_v4_mhc import N, V, forward_backward, rocm_gpu_available, weights

_C = 4096  # DeepSeek-V4-Flash-0731 hidden_size

requires_rocm = pytest.mark.skipif(not rocm_gpu_available(), reason="ROCm GPU required.")


def _rows(m, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    residual = torch.randn(m, N, _C, generator=gen, device="cuda").to(torch.bfloat16)
    return residual, torch.randint(0, V, (m,), generator=gen, device="cuda")


def _layer_end(residual, targets, w, backend):
    with torch.no_grad():
        return mhc_layer_end(
            residual,
            w["head_fn"],
            w["head_scale"],
            w["head_base"],
            w["norm_weight"],
            w["lm_head_weight"],
            targets,
            backend=backend,
        )


@requires_rocm
def test_layer_end_matches_reference():
    w = weights(_C)
    residual, targets = _rows(64, 1)
    stages = _layer_end(residual, targets, w, "rocm")
    reference = _layer_end(residual, targets, w, "reference")

    assert tuple(stages) == LAYER_END_STAGES
    assert [stages[s].dtype for s in LAYER_END_STAGES] == [
        torch.bfloat16,
        torch.bfloat16,
        torch.float32,
        torch.float32,
    ]
    for name in ("head", "final_norm", "logits"):
        assert_matches_fp32_reference(
            stages[name],
            reference[name],
            judgment="forward_accuracy",
            op_class="reduction",
            name=name,
        )
    assert_matches_fp32_reference(
        stages["logprob"],
        reference["logprob"],
        judgment="forward_accuracy",
        op_class="logprob",
        name="logprob",
    )


@requires_rocm
def test_empty_batch_forward_and_backward():
    w = weights(_C)
    residual, targets = _rows(0, 3)

    def run(residual, targets):
        stages = mhc_layer_end(
            residual,
            w["head_fn"],
            w["head_scale"],
            w["head_base"],
            w["norm_weight"],
            w["lm_head_weight"],
            targets,
            backend="rocm",
        )
        return [stages[s] for s in LAYER_END_STAGES]

    outputs, (grad,) = forward_backward(run, [residual, targets])
    assert [tuple(o.shape) for o in outputs] == [(0, _C), (0, _C), (0, V), (0,)]
    assert grad.shape == (0, N, _C)


@requires_rocm
def test_target_outside_the_vocabulary_raises():
    w = weights(_C)
    residual, targets = _rows(4, 4)
    targets[1] = V
    with pytest.raises(ValueError, match="outside"):
        _layer_end(residual, targets, w, "rocm")


# --- CPU: the reference backend ------------------------------------------------------------

_CPU_C, _CPU_V = 16, 32


def _cpu_inputs(m, seed=0):
    gen = torch.Generator().manual_seed(seed)
    residual = torch.randn(m, N, _CPU_C, generator=gen).to(torch.bfloat16)
    params = {
        "head_fn": torch.randn(N, N * _CPU_C, generator=gen) / (N * _CPU_C) ** 0.5,
        "head_scale": 0.5 + torch.rand(1, generator=gen),
        "head_base": 0.1 * torch.randn(N, generator=gen),
        "norm_weight": 1.0 + 0.1 * torch.randn(_CPU_C, generator=gen),
        "lm_head_weight": (0.02 * torch.randn(_CPU_V, _CPU_C, generator=gen)).to(torch.bfloat16),
    }
    return residual, torch.randint(0, _CPU_V, (m,), generator=gen), params


def _cpu_layer_end(residual, targets, p, **kwargs):
    return mhc_layer_end(
        residual,
        p["head_fn"],
        p["head_scale"],
        p["head_base"],
        p["norm_weight"],
        p["lm_head_weight"],
        targets,
        backend="reference",
        **kwargs,
    )


def test_reference_backend_on_cpu():
    residual, targets, p = _cpu_inputs(3)
    stages = _cpu_layer_end(residual, targets, p)
    assert tuple(stages) == LAYER_END_STAGES
    want, _ = rmsnorm_residual(stages["head"], p["norm_weight"], backend="reference")
    assert torch.equal(stages["final_norm"], want)
    logprob = torch.log_softmax(stages["logits"], dim=-1).gather(-1, targets[:, None])
    torch.testing.assert_close(stages["logprob"], logprob.squeeze(-1))


def test_reference_backend_empty_batch():
    residual, targets, p = _cpu_inputs(0)
    leaf = residual.clone().requires_grad_(True)
    stages = _cpu_layer_end(leaf, targets, p)
    assert [tuple(stages[s].shape) for s in LAYER_END_STAGES] == [
        (0, _CPU_C),
        (0, _CPU_C),
        (0, _CPU_V),
        (0,),
    ]
    stages["logprob"].sum().backward()
    assert leaf.grad.shape == (0, N, _CPU_C)


@pytest.mark.parametrize(
    "change,error",
    [
        ({"lm_head_weight": torch.zeros(_CPU_V, _CPU_C)}, TypeError),  # fp32 lm_head weight
        ({"norm_weight": torch.ones(_CPU_C, dtype=torch.bfloat16)}, RMSNormResidualContractError),
    ],
)
def test_inputs_outside_the_contract_raise(change, error):
    residual, targets, p = _cpu_inputs(2)
    with pytest.raises(error):
        _cpu_layer_end(residual, targets, {**p, **change})


def test_norm_eps_other_than_task5_raises():
    residual, targets, p = _cpu_inputs(2)
    with pytest.raises(RMSNormResidualContractError, match="eps"):
        _cpu_layer_end(residual, targets, p, norm_eps=1e-5)


def test_target_outside_the_vocabulary_raises_on_cpu():
    residual, targets, p = _cpu_inputs(2)
    targets[0] = _CPU_V
    with pytest.raises(ValueError, match="outside"):
        _cpu_layer_end(residual, targets, p)
    targets[0] = -100  # the ignore index is allowed
    assert _cpu_layer_end(residual, targets, p)["logprob"][0] == 0


def test_unknown_backend_raises():
    residual, targets, p = _cpu_inputs(1)
    with pytest.raises(ValueError, match="backend"):
        mhc_layer_end(
            residual,
            p["head_fn"],
            p["head_scale"],
            p["head_base"],
            p["norm_weight"],
            p["lm_head_weight"],
            targets,
            backend="cuda",
        )
