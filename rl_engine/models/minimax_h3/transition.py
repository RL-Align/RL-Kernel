# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RFC #420 ``stochastic_transition_contract``: the H3 flow-matching CPS step, layout invariant.

Targets the verl-omni H3 recipe: ``sde_type="cps"``, ``noise_level=0.8``
(``FlowMatchSDEDiscreteScheduler.sample_previous_step``). With ``s = sin(noise_level * pi / 2)``::

    std          = sigma_prev * s
    pred_x0      = x - sigma * v
    noise_est    = x + v * (1 - sigma)
    mean         = pred_x0 * (1 - sigma_prev) + noise_est * sqrt(sigma_prev**2 - std**2)
    x_prev       = mean + std * eps
    log_prob     = mean over non-batch dims of -(x_prev - mean)**2     (CPS has no normaliser)

The contract makes every number a function of the *sample*, never of how samples are batched:

* FP32 elementwise ops in exactly the order above (separate roundings, no fused contraction);
* ``eps`` comes from ``step_noise(seed, step, sample_id, modality, shape)``, a per-sample
  generator, not one stream consumed in batch order;
* the log-prob mean uses a fixed-order GEMM reduction (``h3-projection-gemm-v1``, FP32 ``ieee``),
  each sample's row independent of the batch;
* ``noise_level == 0`` (a non-selected step) gives ``std == 0``, ``x_prev == mean`` and
  ``log_prob == 0``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch

SDE_TYPE = "cps"
NOISE_LEVEL = 0.8
CONTRACT = "h3-cps-transition-v1"


def step_noise(
    seed: int,
    step: int,
    sample_id: int,
    modality: str,
    shape: tuple[int, ...],
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """FP32 standard normal noise for one sample at one step, independent of the batch layout."""

    digest = hashlib.sha256(f"{seed}/{step}/{sample_id}/{modality}".encode()).digest()
    gen = torch.Generator(device="cpu").manual_seed(int.from_bytes(digest[:8], "little") >> 1)
    return torch.randn(shape, generator=gen, dtype=torch.float32).to(device)


@dataclass
class Transition:
    prev_sample: torch.Tensor
    prev_sample_mean: torch.Tensor
    std_dev_t: torch.Tensor
    log_prob: torch.Tensor | None


def fixed_order_mean(x: torch.Tensor) -> torch.Tensor:
    """Per-sample mean over all non-batch elements with a fixed reduction order (``(B,)`` FP32)."""

    flat = x.reshape(x.shape[0], -1).contiguous().float()
    ones = torch.ones(flat.shape[1], 1, device=flat.device, dtype=torch.float32)
    from rl_engine.backends.shared.triton.gemm import h3_projection as g

    return g.gemm(flat, ones)[:, 0] / flat.shape[1]


def cps_transition(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigma: torch.Tensor | float,
    sigma_prev: torch.Tensor | float,
    *,
    noise_level: float = NOISE_LEVEL,
    noise: torch.Tensor | None = None,
    prev_sample: torch.Tensor | None = None,
    return_logprob: bool = True,
) -> Transition:
    """One CPS step. ``noise`` (or ``prev_sample`` when scoring a stored rollout) is required."""

    if sample.dtype != torch.float32 or velocity.dtype != torch.float32:
        raise TypeError("the CPS step runs in FP32 (the scheduler asserts the same)")
    if (noise is None) == (prev_sample is None):
        raise ValueError("pass exactly one of noise (sampling) or prev_sample (scoring)")
    sigma = torch.as_tensor(sigma, dtype=torch.float32, device=sample.device)
    sigma_prev = torch.as_tensor(sigma_prev, dtype=torch.float32, device=sample.device)
    sin_noise = math.sin(noise_level * math.pi / 2)
    std_dev_t = sigma_prev * sin_noise
    pred_original = sample - sigma * velocity
    noise_estimate = sample + velocity * (1 - sigma)
    mean = pred_original * (1 - sigma_prev) + noise_estimate * torch.sqrt(
        sigma_prev**2 - std_dev_t**2
    )
    if prev_sample is None:
        prev_sample = mean + std_dev_t * noise
    log_prob = None
    if return_logprob:
        log_prob = fixed_order_mean(-((prev_sample.detach() - mean) ** 2))
    return Transition(prev_sample, mean, std_dev_t, log_prob)
