# SPDX-License-Identifier: Apache-2.0
"""The gate has one explicit ROCm implementation and no fallback."""

import pytest

from rl_engine.runtime.registry import _default_semantic_descriptors
from rl_engine.runtime.semantic_registry import (
    OperatorRequirements,
    OperatorResolutionError,
    SemanticOperatorCatalog,
)


def resolve(*, device="rocm", dtype="bfloat16", tp=1, graph=False, target="training"):
    session = SemanticOperatorCatalog(_default_semantic_descriptors()).session()
    resolution = session.resolve(
        semantic_op="router_gate_projection",
        requested_backend="rlkernel.aiter_fixed_bf16_fp32.v1",
        target=target,
        requirements=OperatorRequirements(
            device=device,
            dtype=dtype,
            topology={"tensor_parallel_size": tp},
            alignment_properties={"graph": graph},
        ),
    )
    return session, resolution


@pytest.mark.parametrize("target", ["training", "rollout"])
def test_gate_exact_dispatch_and_instantiation(target):
    session, resolution = resolve(target=target)
    op = session.instantiate(resolution)
    assert type(op).__name__ == "RouterGateProjectionOp"
    assert resolution.trace.fallback_attempts == ()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"device": "cpu"},
        {"device": "cuda"},
        {"dtype": "float32"},
        {"tp": 2},
        {"graph": True},
    ],
)
def test_gate_ineligible_dispatch_is_an_error(kwargs):
    with pytest.raises(OperatorResolutionError):
        resolve(**kwargs)
