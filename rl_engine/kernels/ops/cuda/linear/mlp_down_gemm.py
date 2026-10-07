# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Hand-written CUDA backend for the MLP down projection: two contracts, two paths.

The row's CUDA backend serves the same operator with two *different*, both
frozen, arithmetic orders, and which one a call takes is reported:

* ``mlp-down-gemm-mma-v1`` -- the hardware order. The K reduction walks
  ascending k-chunks of 16, one fp32 accumulator chained in place per output
  element, no split-K, no atomics, bias added once in fp32, exactly one bf16
  cast at the store. Two kernels implement it and they are **byte-identical** to
  each other: ``cuda_wgmma_tma_hopper_pinned_schedule``
  (``csrc/cuda/gemm/mlp_down_gemm_sm90.cu``: Hopper TMA bulk-tensor staging
  driving ``wgmma.mma_async...m64n128k16``, built only with
  ``KERNEL_ALIGN_FORCE_SM90=1``, used only on compute capability 9.x) and the
  Triton backend, whose ``wgmma.m64n128k16`` lowering is byte-identical to it.
  On this contract the output is within the declared tolerance of the row's
  independent fp32 CPU reference, not equal to it.

* ``mlp-down-gemm-tree-v1`` -- the portable order, implemented by
  ``cuda_fp32_tree_schedule`` (``csrc/cuda/gemm/mlp_down_gemm.cu``): the fp32
  32-wide-leaf mid-split tree *is* the operator's contract, so this path is
  byte-equal to ``mlp_down_gemm_reference_forward`` /
  ``mlp_down_gemm_reference_backward`` on every shape (checked by the suite at
  the model's K = 12288, N = 3072 and at the RFC's synthetic tails). It runs on
  fp32 CUDA cores -- a tree cannot use tensor cores -- so it is far slower than
  the Hopper path; it needs no tensor cores and no TMA, which is why it serves
  every device and every operand layout the row supports, with no fallback.

``auto`` (the default) prefers the Hopper path when it is built and the device
is cc 9.x, and runs the portable tree otherwise. ``general`` forces the tree
even on a Hopper device (that is how the two are A/B compared on one machine),
and ``hopper`` refuses instead of silently changing the arithmetic order --
which is what a deployment that must not change contract should pin. Both
paths share the forward/backward entry points, the epilogue discipline (bias
once in fp32, one cast at the store) and the ``db`` left fold, so ``db`` is
bit-identical under either contract.

bf16 only: this is the model's runtime dtype, and an fp32 call fails closed
rather than silently running an SGEMM.
"""

from __future__ import annotations

import os
from threading import Lock

from typing import Optional

import torch

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.utils.logger import logger

MMA_CONTRACT_VERSION = "mlp-down-gemm-mma-v1"
TREE_CONTRACT_VERSION = "mlp-down-gemm-tree-v1"
# Advertised by the backend object: the hardware order is the shipping default
# (Hopper wherever its symbols are built, the portable tree everywhere else).
# The *per-call* contract is what :func:`mlp_down_gemm_contract_used` reports.
MLP_DOWN_GEMM_CONTRACT_VERSION = MMA_CONTRACT_VERSION
_REQUIRED_SYMBOLS = (
    "mlp_down_gemm_cuda_forward",
    "mlp_down_gemm_cuda_dx",
    "mlp_down_gemm_cuda_dw",
    "mlp_down_gemm_cuda_db",
)
# Hopper (SM90) TMA + wgmma backend for the mma contract.
_SM90_SYMBOLS = (
    "mlp_down_gemm_cuda_forward_sm90",
    "mlp_down_gemm_cuda_dx_sm90",
    "mlp_down_gemm_cuda_dw_sm90",
)
_HOPPER_IMPL = "cuda_wgmma_tma_hopper_pinned_schedule"
_TREE_IMPL = "cuda_fp32_tree_schedule"
_HOPPER_PIN_HINT = (
    "pin RL_KERNEL_MLP_DOWN_GEMM_BACKEND=hopper for the mlp-down-gemm-mma-v1 "
    "hardware order"
)


def mma_backend_available() -> bool:
    return _EXT_AVAILABLE and all(hasattr(_C, name) for name in _REQUIRED_SYMBOLS)


def sm90_backend_compiled() -> bool:
    """True when the extension contains the Hopper wgmma entry points."""

    return _EXT_AVAILABLE and all(hasattr(_C, name) for name in _SM90_SYMBOLS)


from rl_engine.runtime_mode import rl_kernel_mode, route_report_enabled

_ROUTE_REPORTED = False
_ROUTE_REPORT_LOCK = Lock()

_BACKEND_ENV = "RL_KERNEL_MLP_DOWN_GEMM_BACKEND"
_AUTO_BACKEND = "auto"
_HOPPER_BACKEND = "hopper"
_GENERAL_BACKEND = "general"
_VALID_BACKENDS = (_AUTO_BACKEND, _HOPPER_BACKEND, _GENERAL_BACKEND)


def mlp_down_gemm_backend() -> str:
    """Requested CUDA backend for this row: ``auto`` (default), ``hopper``, ``general``.

    Mirrors ``RL_KERNEL_DET_GEMM_BACKEND``. ``general`` forces the portable
    fp32 tree kernel (``mlp-down-gemm-tree-v1``) even on a Hopper device with the
    wgmma build present, which is how the two contracts are A/B compared on one
    machine; ``hopper`` forces the hardware order and refuses (rather than
    falling back to a different arithmetic order) when it cannot serve the
    operands. ``auto`` picks the Hopper path when it can serve the contraction
    and the portable tree otherwise.
    """

    raw = os.environ.get(_BACKEND_ENV, _AUTO_BACKEND).strip().lower()
    if raw not in _VALID_BACKENDS:
        raise ValueError(f"{_BACKEND_ENV} must be one of {_VALID_BACKENDS}; got {raw!r}")
    return raw


def _report_route_once(x: torch.Tensor, weight: torch.Tensor) -> None:
    """Emit this row's route report once, in the shape det_gemm uses.

    ``[RL-Kernel][route] mode=... module=mlp_down_gemm requested=... actual=...
    fallback=... contract=...`` answers the roadmap's operator-trace question
    (requested backend, actual backend, fallback state) for this row, and names
    the arithmetic contract the call took -- the two CUDA contracts are
    different reduction orders, so the contract id, not just the backend name,
    is what a deployment wants to see.
    """

    global _ROUTE_REPORTED
    if torch._dynamo.is_compiling() or not route_report_enabled():
        return
    with _ROUTE_REPORT_LOCK:
        if _ROUTE_REPORTED:
            return
        _ROUTE_REPORTED = True
    requested = mlp_down_gemm_backend()
    actual = mlp_down_gemm_backend_used(x, weight)
    contract = mlp_down_gemm_contract_used(x, weight)
    fallback = actual != _HOPPER_BACKEND
    if not fallback:
        reason = ""
    elif requested == _GENERAL_BACKEND:
        # A deliberate contract change, not a degraded Hopper run: the portable
        # tree is a different (also frozen) arithmetic order.
        reason = f" reason=general_is_the_portable_tree_contract ({_HOPPER_PIN_HINT})"
    else:
        reason = " reason=hopper_path_cannot_serve_these_operands"
    print(
        f"[RL-Kernel][route] mode={rl_kernel_mode().value} module=mlp_down_gemm "
        f"requested={requested} actual={actual} fallback={str(fallback).lower()}{reason} "
        f"contract={contract}",
        flush=True,
    )


def mlp_down_gemm_backend_used(x: torch.Tensor, weight: torch.Tensor) -> str:
    """Which CUDA path a call with these operands takes: ``"hopper"`` or ``"general"``.

    Together with :func:`mlp_down_gemm_backend` (what was requested) this is the
    requested/actual backend and fallback state the roadmap's operator-trace work
    item asks every row to be able to report;
    :func:`mlp_down_gemm_contract_used` names the arithmetic the path implements.
    """

    return "hopper" if _sm90_usable(x, weight) else "general"


def mlp_down_gemm_contract_used(x: torch.Tensor, weight: torch.Tensor) -> str:
    """Which arithmetic contract a call with these operands is under.

    ``mlp-down-gemm-mma-v1`` when the Hopper TMA + wgmma kernel serves the
    contraction (byte-identical to the Triton backend, within the declared
    tolerance of the fp32 CPU reference), ``mlp-down-gemm-tree-v1`` when the
    portable fp32 tree kernel does (byte-equal to that reference by
    construction). The two are different association orders of the same sum, so
    a consumer that needs a stable contract should pin
    ``RL_KERNEL_MLP_DOWN_GEMM_BACKEND=hopper``.
    """

    return MMA_CONTRACT_VERSION if _sm90_usable(x, weight) else TREE_CONTRACT_VERSION


def _sm90_usable(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether the Hopper path can serve this contraction.

    Requires the 90a entry points, a compute capability 9.x device, and the
    operand views the TMA tensor maps are built on: bf16 2-D operands whose
    row strides (the contiguous extents of x and weight) are multiples of 8
    elements (16 B) and whose bases are 16 B aligned. Anything else runs on the
    portable tree kernel instead -- which computes a different, also frozen,
    arithmetic order, so request ``auto``/``general`` knowingly, or
    ``hopper`` to make the change an error.
    """

    requested = mlp_down_gemm_backend()
    if requested == _GENERAL_BACKEND:
        return False
    if not sm90_backend_compiled() or not x.is_cuda:
        if requested == _HOPPER_BACKEND:
            raise RuntimeError(
                "mlp_down_gemm: backend 'hopper' was requested but the extension has no "
                "Hopper entry points; rebuild with KERNEL_ALIGN_FORCE_SM90=1, or request "
                "'general' to run the portable mlp-down-gemm-tree-v1 path"
            )
        return False
    if x.dtype != torch.bfloat16:
        if requested == _HOPPER_BACKEND:
            raise RuntimeError("mlp_down_gemm: backend 'hopper' requires bf16 operands")
        return False
    hopper_ok = (
        torch.cuda.get_device_capability(x.device)[0] >= 9
        and x.size(-1) % 8 == 0
        and weight.size(0) % 8 == 0
        and x.data_ptr() % 16 == 0
        and weight.data_ptr() % 16 == 0
    )
    if not hopper_ok and requested == _HOPPER_BACKEND:
        raise RuntimeError(
            "mlp_down_gemm: backend 'hopper' needs a compute capability 9.x device and "
            "bf16 operands whose row strides are multiples of 8 elements; it is the "
            f"mlp-down-gemm-mma-v1 contract -- {_HOPPER_PIN_HINT}, or request 'general' "
            "for the portable mlp-down-gemm-tree-v1 tree"
        )
    return hopper_ok


class _MlpDownGemmFunction(torch.autograd.Function):
    """Forward/backward through the requested contract.

    On a cc 9.x device with the Hopper entry points built in, the forward, ``dx``
    and ``dW`` run on the TMA + wgmma kernel (``mlp-down-gemm-mma-v1``);
    everywhere else they run on the portable fp32 tree kernel
    (``mlp-down-gemm-tree-v1``). Each path is its own frozen arithmetic order --
    the Hopper one is byte-identical to the Triton backend, the tree one is
    byte-equal to the row's fp32 CPU reference -- and ``db``, the ascending-row
    fp32 left fold, is shared and bit-identical under both.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        x2d = x.reshape(-1, x.size(-1)).contiguous()
        w = weight.contiguous()
        use_sm90 = _sm90_usable(x2d, w)
        ctx.use_sm90 = use_sm90
        if use_sm90:
            out = _C.mlp_down_gemm_cuda_forward_sm90(x2d, w, bias)
        else:
            out = _C.mlp_down_gemm_cuda_forward(x2d, w, bias)
        # Save the *contiguous* weight: both backends assume contiguous operand
        # strides, and the Hopper path's tensor maps are built from the extents.
        ctx.save_for_backward(x2d, w, bias)
        ctx.lead_shape = x.shape[:-1]
        return out.reshape(*ctx.lead_shape, weight.size(0))

    @staticmethod
    def backward(  # type: ignore[override]
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        from rl_engine.kernels.ops.backward_runtime import record_backward

        x2d, weight, bias = ctx.saved_tensors
        g = grad_output.reshape(-1, grad_output.size(-1)).contiguous()
        if ctx.use_sm90:
            grad_x = _C.mlp_down_gemm_cuda_dx_sm90(g, weight)
            grad_w = _C.mlp_down_gemm_cuda_dw_sm90(g, x2d)
        else:
            grad_x = _C.mlp_down_gemm_cuda_dx(g, weight)
            grad_w = _C.mlp_down_gemm_cuda_dw(g, x2d)
        grad_x = grad_x.reshape(*ctx.lead_shape, weight.size(1))
        grad_b: Optional[torch.Tensor] = None
        if bias is not None:
            # Ascending-row fp32 fold, shared by both CUDA paths.
            grad_b = _C.mlp_down_gemm_cuda_db(grad_output).to(bias.dtype)
        record_backward(
            "mlp_down_gemm",
            kernel_id=MMA_CONTRACT_VERSION if ctx.use_sm90 else TREE_CONTRACT_VERSION,
            impl=_HOPPER_IMPL if ctx.use_sm90 else _TREE_IMPL,
            family="cuda",
        )
        return grad_x, grad_w, grad_b


class CudaMlpDownGemmOp:
    """Evaluation backend: bf16 only, two frozen contracts, two paths.

    ``contract_version`` is the hardware-order contract this backend ships by
    default; the contract a *call* takes is
    :func:`mlp_down_gemm_contract_used` (``mlp-down-gemm-mma-v1`` on the Hopper
    path, ``mlp-down-gemm-tree-v1`` on the portable tree).
    """

    op_class = "reduction"
    is_batch_invariant = True  # both paths: verified by the CUDA tests
    backward_impl = _TREE_IMPL
    contract_version = MLP_DOWN_GEMM_CONTRACT_VERSION

    def __init__(self) -> None:
        if not mma_backend_available():
            raise RuntimeError(
                "mlp_down_gemm CUDA symbols are not compiled into the extension; rebuild with "
                "a CUDA toolchain that supports sm_80 or newer; refusing a non-strict fallback."
            )
        logger.info(
            "CudaMlpDownGemmOp ready (portable fp32 tree %s; Hopper wgmma %s when built).",
            TREE_CONTRACT_VERSION,
            MMA_CONTRACT_VERSION,
        )

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward(x, weight, bias=bias)

    def _grads_for_test(
        self, x: torch.Tensor, weight: torch.Tensor, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Backward entry used by the byte-equality tests against the spec."""

        x = x.detach().requires_grad_(True)
        weight = weight.detach().requires_grad_(True)
        self.forward(x, weight).backward(grad_output)
        return x.grad, weight.grad

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if torch.version.hip is not None:
            raise RuntimeError(
                "mlp_down_gemm: this CUDA backend has no ROCm build (its kernels are NVIDIA "
                "PTX); the registry dispatches ROCm to the Triton backend"
            )
        if not x.is_cuda or x.device != weight.device:
            raise ValueError("CudaMlpDownGemmOp requires CUDA tensors on one device")
        # Both paths are gated to sm_80 and up (the row's validated CUDA support
        # matrix); below that dispatching would run an unvalidated configuration,
        # so fail closed instead.
        if torch.cuda.get_device_capability(x.device)[0] < 8:
            raise ValueError(
                "mlp_down_gemm's CUDA backend requires a compute capability >= 8.0 "
                "(sm80+) device; below that use the PyTorch reference backend"
            )
        if x.dtype is not torch.bfloat16 or weight.dtype is not torch.bfloat16:
            raise ValueError(
                "mlp_down_gemm's CUDA backend is the model's runtime dtype: bf16 x/weight "
                f"required, got x={x.dtype}, weight={weight.dtype}"
            )
        if bias is not None and (bias.device != x.device or bias.dtype is not torch.bfloat16):
            raise ValueError("bias must be bf16 on the same device as x")
        if x.size(-1) != weight.size(-1):
            raise ValueError("x K must match weight K")
        _report_route_once(x, weight)
        return _MlpDownGemmFunction.apply(x, weight, bias)
