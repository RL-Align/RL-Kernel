# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""End to end: the H3 conditioning chain on the pinned checkpoint, through the registry.

Runs every stage registered in ``rl_engine.testing.h3_chain.STAGES`` (each
RFC #420 row adds its own), chained exactly as the model calls them, and
checks every stage's promises:

* the registry dispatched the CUDA backend;
* a repeated run is bitwise equal;
* the chained output is within the stage's contract tolerance of the golden;
* stages that promise it are bitwise equal to diffusers on diffusers' inputs,
  so the first isolated drift can only be a stage that declares a reduction.
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.registry import KernelRegistry
from rl_engine.testing import h3_chain

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

# (T distinct timesteps, S packed rows): single row, short packed, realistic-size packed.
CASES = [(1, 3), (3, 257), (4, 4097)]


@pytest.fixture(scope="module")
def chain_weights(h3_weights_cpu):
    return {name: tensor.cuda() for name, tensor in h3_weights_cpu.items()}


@pytest.fixture(scope="module")
def registry():
    return KernelRegistry()


@pytest.mark.parametrize("num_timesteps, seq_len", CASES)
def test_chain_forward(registry, chain_weights, num_timesteps, seq_len):
    report = h3_chain.run_case(
        registry, chain_weights, num_timesteps=num_timesteps, seq_len=seq_len
    )
    stages = {stage.name: stage for stage in h3_chain.STAGES}
    for entry in report["stages"]:
        stage = stages[entry["stage"]]
        assert entry["backend"].endswith("CudaOp"), entry
        assert entry["repeat_bitwise_equal"], entry["stage"]
        assert entry["chained_vs_golden"]["within_tolerance"], entry
        if stage.provider_bitwise_isolated:
            assert entry["isolated_vs_provider"]["bitwise_equal"], entry["stage"]
    drift = report["first_isolated_drift"]
    assert drift is None or not stages[drift].provider_bitwise_isolated
