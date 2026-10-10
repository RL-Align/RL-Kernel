# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""End to end: the H3 conditioning chain on the pinned checkpoint, through the registry.

Runs every stage registered in ``rl_engine.validation.models.h3_chain.STAGES`` (each
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

from rl_engine.runtime.registry import KernelRegistry
from rl_engine.validation.models import h3_chain

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
    reason="needs an SM80+ CUDA device",
)

# (T distinct timesteps, S packed rows): single row, short packed, realistic-size packed.
CASES = [(1, 3), (3, 257), (4, 4097)]


@pytest.fixture(scope="module")
def chain_weights(h3_weights_cpu):
    """Move all pinned conditioning tensors to CUDA once for the end-to-end module."""

    return {name: tensor.cuda() for name, tensor in h3_weights_cpu.items()}


@pytest.fixture(scope="module")
def registry():
    """Provide the production registry used to dispatch every conditioning stage."""

    return KernelRegistry()


@pytest.mark.parametrize("num_timesteps, seq_len", CASES)
def test_chain_forward(registry, chain_weights, num_timesteps, seq_len):
    """Check CUDA dispatch, golden tolerance, repeatability, and declared provider parity."""

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


@pytest.mark.parametrize("num_timesteps, seq_len", [(1, 257), (3, 4097)])
def test_chain_backward(registry, chain_weights, num_timesteps, seq_len):
    """Parameter gradients of the whole chain: deterministic, and FP32-accurate when fused."""

    report = h3_chain.run_backward_case(
        registry, chain_weights, num_timesteps=num_timesteps, seq_len=seq_len
    )
    for name, entry in report["leaves"].items():
        for mode in ("candidate", "candidate_fused"):
            assert entry[mode]["repeat_bitwise_equal"], (mode, name)
        fused = entry["candidate_fused"]
        if name.startswith("time_embedder"):
            # FP32 parameters: no BF16 rounding anywhere on the fused path.
            assert fused["max_abs_vs_golden_over_absmax"] < 1e-5, (name, fused)
        else:
            # BF16 parameters: only their own final rounding remains.
            assert fused["correctly_rounded_fraction"] > 0.99, (name, fused)
