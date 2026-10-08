# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Independent FP32 Flow-GRPO SDE reference and shared input contract.

Formula source: yifan123/flow_grpo@879042cf5707f8b90daa98d147d7deac2317c5da,
flow_grpo/diffusers_patch/sd3_sde_with_logprob.py (sde branch only).
Contract v1 fixes all arithmetic boundaries and the latent-domain mean tree.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

CONTRACT_ID = "qwenimage.flow_sde_step_logp.v1"
REFERENCE_REVISION = "879042cf5707f8b90daa98d147d7deac2317c5da"
TILE_SIZE = 256
MAX_ELEMENTS = 1 << 24


class FlowSDEResult(NamedTuple):
    prev_sample: torch.Tensor
    logp: torch.Tensor
    mean: torch.Tensor
    std_dev: torch.Tensor


def _parameter(value, name: str, batch: int) -> torch.Tensor:
    # Schedule metadata must come from the pinned FP32 host table. GPU tensors
    # are rejected rather than introducing a device sync inside the operator.
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.dtype != torch.float32 or value.requires_grad:
            raise ValueError(f"{name} must be detached CPU FP32 metadata")
        result = value.reshape(-1)
    else:
        result = torch.tensor([value], dtype=torch.float32)
    if result.numel() not in (1, batch):
        raise ValueError(f"{name} must be scalar or contain one value per sample")
    result = result.expand(batch).contiguous()
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def prepare_inputs(
    sample,
    model_output,
    sigma,
    sigma_next,
    *,
    sigma_max,
    noise_level,
    noise=None,
    prev_sample=None,
):
    if sample.ndim < 2 or any(dim == 0 for dim in sample.shape):
        raise ValueError("sample must have nonempty [B, ...] geometry")
    if sample.shape[0] > 65535:
        raise ValueError("at most 65535 samples are supported")
    if sample.numel() // sample.shape[0] > MAX_ELEMENTS:
        raise ValueError(f"at most {MAX_ELEMENTS} latent elements per sample are supported")
    if (noise is None) == (prev_sample is None):
        raise ValueError("provide exactly one of explicit FP32 noise or prev_sample")
    for name, value in (("sample", sample), ("model_output", model_output)):
        if value.shape != sample.shape or value.device != sample.device:
            raise ValueError(f"{name} must have the sample shape and device")
        if value.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(f"{name} supports FP32/BF16 only")
    auxiliary = noise if noise is not None else prev_sample
    if auxiliary.shape != sample.shape or auxiliary.device != sample.device:
        raise ValueError("noise/prev_sample must have the sample shape and device")
    if auxiliary.dtype != torch.float32 or auxiliary.requires_grad:
        raise ValueError("noise/prev_sample must be detached FP32 tensors")
    batch = sample.shape[0]
    s = _parameter(sigma, "sigma", batch)
    sn = _parameter(sigma_next, "sigma_next", batch)
    sm = _parameter(sigma_max, "sigma_max", batch)
    level = _parameter(noise_level, "noise_level", batch)
    if not ((s > 0) & (s <= 1) & (sn >= 0) & (sn < s)).all():
        raise ValueError("require 0 <= sigma_next < sigma <= 1 and sigma > 0")
    if not ((sm > 0) & (sm < 1) & (level > 0)).all():
        raise ValueError("require 0 < sigma_max < 1 and noise_level > 0; no ODE logp")
    params = torch.stack((s, sn, sm, level), dim=1)
    # Reject FP32 overflow/underflow before launch, including nonzero Python
    # parameters that round to zero in the declared schedule dtype.
    coeff = reference_coefficients(params)
    if not torch.isfinite(coeff).all() or not (coeff[:, 2] > 0).all():
        raise ValueError("SDE coefficients must be finite with positive FP32 noise scale")
    variance = coeff[:, 2] * coeff[:, 2]
    if not (variance > 0).all() or not torch.isfinite(2.0 * variance).all():
        raise ValueError("FP32 transition variance underflows or overflows")
    return params.to(sample.device), auxiliary.detach().contiguous()


def reference_coefficients(params: torch.Tensor) -> torch.Tensor:
    s, sn, sm, level = params.unbind(dim=1)
    dt = sn - s
    std = torch.sqrt(s / (1.0 - torch.where(s == 1.0, sm, s))) * level
    variance_rate = std * std
    a = 1.0 + (variance_rate / (2.0 * s)) * dt
    b = (1.0 + (variance_rate * (1.0 - s)) / (2.0 * s)) * dt
    tau = std * torch.sqrt(-dt)
    return torch.stack((a, b, tau, std), dim=1)


def fixed_mean(values: torch.Tensor) -> torch.Tensor:
    """Adjacent-pair tree within 256-element tiles, then ascending tile fold."""
    rows = values.reshape(values.shape[0], -1)
    width = rows.shape[1]
    pad = (-width) % TILE_SIZE
    if pad:
        rows = torch.cat((rows, rows.new_zeros((rows.shape[0], pad))), dim=1)
    tree = rows.reshape(rows.shape[0], -1, TILE_SIZE)
    while tree.shape[-1] > 1:
        tree = tree[..., 0::2] + tree[..., 1::2]
    partial = tree.squeeze(-1)
    total = torch.zeros_like(partial[:, 0])
    for tile in range(partial.shape[1]):
        total = total + partial[:, tile]
    return total / float(width)


def execution_trace(backend: str, sample=None) -> dict:
    return {
        "contract": CONTRACT_ID,
        "reference_revision": REFERENCE_REVISION,
        "requested_backend": backend,
        "actual_backend": backend,
        "kernel_identity": f"{CONTRACT_ID}.{backend}.tile256",
        "fallback": False,
        "arithmetic": "FP32, separate mul/add, precise division/sqrt, no TF32/atomics",
        "reduction": "adjacent-pair tree in 256-element tiles; ascending tile fold; mean",
        "output_dtype": "float32",
        "noise": "external detached FP32; no RNG consumed",
        "shape": None if sample is None else list(sample.shape),
        "input_dtype": None if sample is None else str(sample.dtype),
        "device": None if sample is None else str(sample.device),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name(sample.device)
            if sample is not None and sample.is_cuda
            else None
        ),
        "compute_capability": (
            list(torch.cuda.get_device_capability(sample.device))
            if sample is not None and sample.is_cuda
            else None
        ),
        "execution_recorded": False,
    }


class NativeFlowSDEStepLogpOp:
    """FP32 reference; differentiates sample/model_output, never replay targets.

    Sigma/noise metadata are fixed, not learnable inputs. Outputs stay FP32.
    No hidden random generation, dtype cast, ODE substitution or backend fallback.
    """

    def __init__(self):
        self._last_trace = None

    def __call__(
        self,
        sample,
        model_output,
        sigma,
        sigma_next,
        *,
        sigma_max,
        noise_level=0.7,
        noise=None,
        prev_sample=None,
    ):
        return self.forward_fp32(
            sample,
            model_output,
            sigma,
            sigma_next,
            sigma_max=sigma_max,
            noise_level=noise_level,
            noise=noise,
            prev_sample=prev_sample,
        )

    apply = __call__
    forward = __call__

    def forward_fp32(
        self,
        sample,
        model_output,
        sigma,
        sigma_next,
        *,
        sigma_max,
        noise_level=0.7,
        noise=None,
        prev_sample=None,
    ):
        params, auxiliary = prepare_inputs(
            sample,
            model_output,
            sigma,
            sigma_next,
            sigma_max=sigma_max,
            noise_level=noise_level,
            noise=noise,
            prev_sample=prev_sample,
        )
        coeff = reference_coefficients(params)
        shape = (-1,) + (1,) * (sample.ndim - 1)
        a, b, tau, std = (x.reshape(shape) for x in coeff.unbind(dim=1))
        mean = sample.float() * a + model_output.float() * b
        target = mean + tau * auxiliary if noise is not None else auxiliary
        residual = target.detach() - mean
        density = -(residual * residual) / (2.0 * (tau * tau))
        # The source computes log(sqrt(2*pi)) in FP32. Pin that rounded
        # constant, rather than silently replacing it by a double expression.
        constant = torch.tensor(6.283185307179586, dtype=torch.float32, device=sample.device)
        log_normalizer = torch.log(torch.sqrt(constant))
        density = (density - torch.log(tau)) - log_normalizer
        result = FlowSDEResult(target, fixed_mean(density), mean, std.reshape(-1))
        self._last_trace = execution_trace("pytorch", sample)
        self._last_trace["execution_recorded"] = True
        self._last_trace["mode"] = "sampling" if noise is not None else "replay"
        return result

    def execution_trace(self, sample=None):
        return dict(self._last_trace or execution_trace("pytorch", sample))
