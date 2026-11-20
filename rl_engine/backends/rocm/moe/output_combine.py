# SPDX-License-Identifier: Apache-2.0
"""Explicit task 55 ROCm candidate for a validated task 51/52 mapping."""

import torch

from rl_engine.contracts.operators.fixed_output_combine import PROFILE

BLOCK = 256


def _require_device(device):
    if not torch.version.hip or device.type != "cuda":
        raise RuntimeError(f"{PROFILE} requires ROCm GPU tensors; no CPU fallback")
    arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
    if arch not in ("gfx942", "gfx950"):
        raise RuntimeError(f"{PROFILE} has no candidate for {arch}")


class RocmOutputCombine:
    profile = PROFILE

    @staticmethod
    @torch.no_grad()
    def forward(prepared, expert_output, shared_output, debug):
        prepared.validate_execution(expert_output, shared_output, prepared.mapping_run_id)
        _require_device(expert_output.device)
        from .output_combine_kernels import _combine

        t, h = prepared.num_tokens, prepared.plan.hidden_size
        out = torch.empty((t, h), device=expert_output.device, dtype=expert_output.dtype)
        steps = torch.empty((t, 6, h), device=out.device, dtype=torch.float32) if debug else None
        before = torch.empty((t, h), device=out.device, dtype=torch.float32) if debug else None
        with torch.cuda.device(out.device):
            _combine[(t, (h + BLOCK - 1) // BLOCK)](
                expert_output,
                prepared._rows,
                out if shared_output is None else shared_output,
                out,
                out if steps is None else steps,
                out if before is None else before,
                h,
                *expert_output.stride(),
                *((0, 0) if shared_output is None else shared_output.stride()),
                shared_output is not None,
                debug,
                BLOCK,
                num_warps=4,
                enable_fp_fusion=False,
            )
        return out, steps, before

    @staticmethod
    @torch.no_grad()
    def backward(dy, tokens, dtype):
        _require_device(dy.device)
        from .output_combine_kernels import _output_vjp

        h = dy.shape[1]
        dx = torch.empty((tokens.numel(), h), device=dy.device, dtype=dtype)
        with torch.cuda.device(dy.device):
            _output_vjp[(tokens.numel(), (h + BLOCK - 1) // BLOCK)](
                dy,
                tokens,
                dx,
                h,
                *dy.stride(),
                BLOCK,
                num_warps=4,
                enable_fp_fusion=False,
            )
        return dx
