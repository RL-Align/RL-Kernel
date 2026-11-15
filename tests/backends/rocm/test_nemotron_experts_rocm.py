# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""ROCm experts: the shared Triton provider against the same contract.

The ROCm column has no native implementation, so these cases cover the Triton
provider only. They were authored without ROCm hardware; the CUDA suite runs the
same `run_core_assertions` body, so the arithmetic, invariance and gradient
assertions are identical.
"""

from __future__ import annotations

import pytest
import torch

from tests.backends import nemotron_experts_cases as cases

pytestmark = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="ROCm device required",
)


def _provider():
    pytest.importorskip("triton")
    from rl_engine.backends.shared.triton.moe.nemotron_experts import TritonNemotronExpertsOp

    return TritonNemotronExpertsOp()


@pytest.mark.parametrize("concentrated", [False, True])
def test_core(concentrated):
    cases.run_core_assertions(_provider(), "cuda", concentrated=concentrated)


def test_contract_geometry():
    cases.run_core_assertions(_provider(), "cuda", geometry=cases.REAL)


def test_registry_selects_the_shared_provider():
    from rl_engine.backends.shared.triton.moe.nemotron_experts import TritonNemotronExpertsOp
    from rl_engine.runtime.registry import KernelRegistry

    op = KernelRegistry().get_op("moe_experts_combine", device="cuda")
    assert isinstance(op, TritonNemotronExpertsOp)
    assert op.provider_id == "rlkernel.nemotron_experts.triton.v1"


def test_provider_id_reports_triton():
    from rl_engine.ops.moe.nemotron_experts import nemotron_experts_provider_id

    assert nemotron_experts_provider_id(_provider()) == "rlkernel.nemotron_experts.triton.v1"
