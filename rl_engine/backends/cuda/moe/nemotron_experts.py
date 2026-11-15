# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Native SM90 provider for the Nemotron grouped expert MLP.

`csrc/cuda/moe/nemotron_experts.cu` owns the kernels; this module is the
registry entry point and the autograd wiring.  The operator is strict: when the
extension was built without `KERNEL_ALIGN_DET_GEMM_SM90=1`, `__init__` raises so
`KernelRegistry` can fall through to the shared Triton provider instead of
silently running a different arithmetic.
"""

from __future__ import annotations

import torch

from rl_engine.backends.extension import _C, _EXT_AVAILABLE

PROVIDER_ID = "rlkernel.nemotron_experts.sm90.v1"


def _require_sm90(device: torch.device) -> None:
    major, minor = torch.cuda.get_device_capability(device)
    # Exactly SM90, not ">= 9": the build emits `sm_90a` (wgmma/TMA) code and
    # architecture-specific `a` images are not forward compatible, so Blackwell
    # must not be admitted by this gate.
    if major != 9:
        raise RuntimeError(
            f"nemotron experts native kernel requires NVIDIA SM90, got sm_{major}{minor} "
            f"on {device}"
        )


class NemotronExpertsOp:
    """Registry entry point for the SM90 grouped expert operator."""

    provider_id = PROVIDER_ID

    def __init__(self) -> None:
        # A missing symbol means the extension predates this operator; a False
        # compile flag means the SM90 path is absent from this build. Both are
        # reported as instance failures so `get_op` can try the next candidate.
        if not _EXT_AVAILABLE or not hasattr(_C, "nemotron_experts_forward"):
            raise RuntimeError("nemotron experts native extension is not built")
        if not _C.nemotron_experts_compiled():
            raise RuntimeError(
                "nemotron experts native kernel requires KERNEL_ALIGN_DET_GEMM_SM90=1"
            )

    def __call__(
        self,
        packed: torch.Tensor,
        x: torch.Tensor | None,
        weights: torch.Tensor,
        permutation: torch.Tensor,
        offsets: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor,
        w_sh1: torch.Tensor | None = None,
        w_sh2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # The build flag says the kernel exists, not that `packed`'s device can
        # run it: `KERNEL_ALIGN_DET_GEMM_SM90=1` compiles the SM90 paths from
        # the build host's architecture, while the registry lists this provider
        # first for every CUDA device and caches the instance for the process.
        # The check therefore follows the payload device on every call, not
        # `torch.cuda.current_device()` once at construction, so a pre-Hopper
        # device fails closed and a Hopper device stays usable in one process.
        _require_sm90(packed.device)
        return _ExpertsCombine.apply(
            packed, x, weights, permutation, offsets, w_up, w_down, w_sh1, w_sh2
        )


class _ExpertsCombine(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        packed,
        x,
        weights,
        permutation,
        offsets,
        w_up,
        w_down,
        w_sh1,
        w_sh2,
    ):
        out, up, z, down, inverse, shared_up, shared_z, shared_down = _C.nemotron_experts_forward(
            packed, x, weights, permutation, offsets, w_up, w_down, w_sh1, w_sh2
        )
        ctx.save_for_backward(
            packed,
            x,
            weights,
            permutation,
            offsets,
            up,
            z,
            down,
            inverse,
            w_up,
            w_down,
            w_sh1,
            w_sh2,
            shared_up,
            shared_z,
            shared_down,
        )
        ctx.mark_non_differentiable(permutation, offsets)
        ctx.set_materialize_grads(False)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (
            packed,
            x,
            weights,
            permutation,
            offsets,
            up,
            z,
            down,
            inverse,
            w_up,
            w_down,
            w_sh1,
            w_sh2,
            shared_up,
            shared_z,
            shared_down,
        ) = ctx.saved_tensors
        needs = ctx.needs_input_grad
        if grad_out is None:
            return (None,) * 9
        shared = w_sh1 is not None
        need_shared_weight_grad = shared and (bool(needs[7]) or bool(needs[8]))
        need_any = (
            bool(needs[0])
            or (shared and bool(needs[1]))
            or bool(needs[2])
            or bool(needs[5])
            or bool(needs[6])
            or need_shared_weight_grad
        )
        if not need_any:
            return (None,) * 9
        # Autograd does not guarantee a contiguous incoming gradient (a narrowed
        # or transposed output produces a strided one); the kernel requires it.
        d_packed, d_x, d_weights, d_w_up, d_w_down, d_w_sh1, d_w_sh2 = _C.nemotron_experts_backward(
            grad_out.contiguous(),
            packed,
            x,
            weights,
            permutation,
            offsets,
            up,
            z,
            down,
            inverse,
            w_up,
            w_down,
            w_sh1,
            w_sh2,
            shared_up,
            shared_z,
            shared_down,
            bool(needs[0]),
            shared and bool(needs[1]),
            bool(needs[2]),
            bool(needs[5]) or bool(needs[6]),
            need_shared_weight_grad,
        )
        # Undefined extension results already arrive as `None`; a zero-element
        # tensor is a real gradient (the zero-token payload case) and keeps its
        # shape instead of collapsing to "absent".
        return (d_packed, d_x, d_weights, None, None, d_w_up, d_w_down, d_w_sh1, d_w_sh2)
