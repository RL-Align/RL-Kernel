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
* the batch-invariant ``eps`` is ``step_noise(seed, step, sample_id, modality, shape)``, a
  per-sample generator, not one stream consumed in batch order (the strict runtime draws it so);
  :func:`h3_sample_previous_step`, the drop-in for verl-omni's ``sample_h3_transition``, takes
  ``noise=`` for such draws and otherwise draws from the caller's ``generator`` in batch order,
  as the scheduler does, so a stock rollout keeps its own noise; the drawn ``x_prev`` and its
  log-prob then depend on the batch position (as in the scheduler: the log-prob is a function of
  the drawn noise), while the score of a given stored ``x_prev`` depends only on that sample;
* the log-prob mean uses a fixed-order FP32 row reduction (``h3-row-mean-v1``: fixed-size chunks,
  then their sums), each sample's row independent of the batch;
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


class _FixedOrderMean(torch.autograd.Function):
    """Row means of a ``(B, K)`` FP32 matrix in a fixed order, differentiable.

    The sums come from ``h3-row-mean-v1`` (fixed-size chunks, then the chunk sums, in an order
    that depends only on ``K``), which has no autograd of its own. The gradient of a mean is
    ``grad / K`` for every element of the row: element-wise, no reduction, so it needs no
    fixed order and is the same bits ``torch.mean`` gives.
    """

    @staticmethod
    def forward(ctx, flat: torch.Tensor) -> torch.Tensor:
        from rl_engine.backends.shared.triton.reduction import row_mean

        ctx.shape = flat.shape
        return row_mean.row_sums(flat) / flat.shape[1]

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        rows, cols = ctx.shape
        return (grad / cols)[:, None].expand(rows, cols)


def fixed_order_mean(x: torch.Tensor) -> torch.Tensor:
    """Per-sample mean over all non-batch elements with a fixed reduction order (``(B,)`` FP32).

    Differentiable: the log-prob built from it carries the policy gradient to the velocity.
    """

    return _FixedOrderMean.apply(x.reshape(x.shape[0], -1).contiguous().float())


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


def _step_sigmas(sigmas: torch.Tensor, index: int, batch: int, ndim: int):
    sigmas = torch.as_tensor(sigmas, dtype=torch.float32)
    if sigmas.ndim != 1 or sigmas.numel() < 2 or float(sigmas[-1]) != 0.0:
        raise ValueError("the H3 sigma grid must be one-dimensional and end at zero")
    if not 0 <= index < sigmas.numel() - 1:
        raise IndexError(f"H3 sigma index {index} is out of range")
    view = (-1,) + (1,) * (ndim - 1)
    sigma = sigmas[index].expand(batch).reshape(view)
    sigma_prev = sigmas[index + 1].expand(batch).reshape(view)
    return sigma, sigma_prev


def h3_sample_previous_step(
    sample: torch.Tensor,
    h3_velocity: torch.Tensor,
    sigmas: torch.Tensor,
    sigma_index: int,
    *,
    noise_level: float = NOISE_LEVEL,
    sde_type: str = SDE_TYPE,
    generator: torch.Generator | None = None,
    noise: torch.Tensor | None = None,
    prev_sample: torch.Tensor | None = None,
    return_log_prob: bool = True,
):
    """Strict drop-in for verl-omni's ``sample_h3_transition`` (``sde_type="cps"``).

    Same inputs and outputs as ``sample_h3_transition(scheduler, sample, h3_velocity, step, ...)``
    with ``sigmas = scheduler.sigmas`` (the H3 grid ending at zero, as ``configure_flow_scheduler``
    installs it) and ``sigma_index = scheduler.index_for_timestep(scheduler.timesteps[step])``,
    the index the scheduler itself resolves (it equals ``step`` unless the grid repeats a
    timestep), and the same return tuple ``(prev_sample, log_prob, prev_sample_mean, std_dev_t,
    sqrt_dt)``:

    * the flow-matching model output is ``-h3_velocity`` (the H3 velocity's sign convention);
    * ``sigma`` and ``sigma_prev`` broadcast per sample, ``std_dev_t`` has shape ``(B, 1, ...)``
      and ``sqrt_dt`` shape ``(B,)``;
    * when sampling without ``noise``, the noise is ``randn(model_output.shape)`` drawn from
      ``generator`` (on the generator's device, then moved to the sample's device), as Diffusers'
      ``randn_tensor`` does for one generator, so a rollout draws exactly the scheduler's noise.
      That stream is consumed in batch order: a sample's draw depends on its batch position;
    * ``noise`` (FP32 standard normal of the sample's shape, e.g. one :func:`step_noise` per
      sample, stacked) replaces that draw, so ``x_prev`` is independent of the batch too; it
      excludes ``generator`` and ``prev_sample``;
    * ``prev_sample`` and ``prev_sample_mean`` are the scheduler's element-wise arithmetic in the
      same order; only the log-prob mean differs: it is :func:`fixed_order_mean` (independent of
      the batch), not ``Tensor.mean``.

    Only the CPS type the H3 recipe uses is accepted; anything else is refused rather than run
    on another formula.
    """

    if sde_type != "cps":
        raise ValueError(
            f"the strict H3 transition implements sde_type='cps' only, got {sde_type!r}"
        )
    if isinstance(noise_level, torch.Tensor):
        raise TypeError("noise_level must be a Python float (the H3 recipe passes 0.8)")
    model_output = -h3_velocity.float()
    sample = sample.float()
    sigma, sigma_prev = _step_sigmas(sigmas, sigma_index, sample.shape[0], sample.ndim)
    sigma = sigma.to(sample.device)
    sigma_prev = sigma_prev.to(sample.device)
    if noise is not None:
        if generator is not None or prev_sample is not None:
            raise ValueError("noise= excludes generator= and prev_sample=")
        if noise.shape != sample.shape:
            raise ValueError(
                f"noise has shape {tuple(noise.shape)}, the sample {tuple(sample.shape)}"
            )
        noise = noise.float().to(sample.device)
    elif prev_sample is None:
        rand_device = generator.device if generator is not None else sample.device
        noise = torch.randn(
            model_output.shape, generator=generator, device=rand_device, dtype=torch.float32
        ).to(sample.device)
    else:
        prev_sample = prev_sample.float()
    out = cps_transition(
        sample,
        model_output,
        sigma,
        sigma_prev,
        noise_level=noise_level,
        noise=noise,
        prev_sample=prev_sample,
        return_logprob=return_log_prob,
    )
    sqrt_dt = torch.sqrt(-1 * (sigma_prev - sigma)).reshape(sample.shape[0])
    return out.prev_sample, out.log_prob, out.prev_sample_mean, out.std_dev_t, sqrt_dt
