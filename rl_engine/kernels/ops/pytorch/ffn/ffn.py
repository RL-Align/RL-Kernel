# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Bias-free gated FFN assembled from deterministic GPU kernels."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from rl_engine.distributed.collectives import _COLLECTIVES as _SHARED_COLLECTIVES
from rl_engine.distributed.collectives import (
    collective_for_group,
    deterministic_all_reduce_inplace,
    deterministic_all_reduce_staged,
    deterministic_staging_reserve,
)
from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.kernels.ops.matmul.det_gemm import (
    det_gemm_linear,
    det_gemm_linear_input_gradient,
    det_gemm_linear_weight_gradient,
)

QWEN3_8B_HIDDEN_SIZE = 4096
QWEN3_8B_INTERMEDIATE_SIZE = 12288
BACKEND_ID = "rlkernel.ffn.qwen3.deterministic.v1"

_DET_GEMM_SYMBOLS = (
    "det_gemm_fwd",
    "det_gemm_fwd_rhs_transposed",
    "det_gemm_db_transposed",
)
_SWIGLU_SYMBOLS = (
    "swiglu_forward",
    "swiglu_backward",
)
_PACKED_SWIGLU_SYMBOLS = (
    "swiglu_packed_forward",
    "swiglu_packed_backward",
)
_REQUIRED_SYMBOLS = _DET_GEMM_SYMBOLS + _SWIGLU_SYMBOLS
_COLLECTIVE_MIN_CAPACITY_BYTES = 64 * 1024 * 1024
# Backward-compatible test hook; ownership lives in the shared communication layer.
_COLLECTIVES = _SHARED_COLLECTIVES
_PACKED_INFERENCE_OBSERVERS: list[Callable[[], None]] = []
_PACKED_INFERENCE_STAGING_BY_HANDLE: dict[int, tuple[int, Tensor, Tensor]] = {}
_PACKED_INFERENCE_SLOT_BY_RUNTIME_HANDLE: dict[int, int] = {}


def register_packed_inference_observer(callback: Callable[[], None]) -> None:
    """Arm one execution callback for the graph-captured rollout custom op."""

    if not callable(callback):
        raise TypeError("packed inference observer must be callable")
    _PACKED_INFERENCE_OBSERVERS.append(callback)


def _notify_packed_inference_observers() -> None:
    callbacks = tuple(_PACKED_INFERENCE_OBSERVERS)
    _PACKED_INFERENCE_OBSERVERS.clear()
    for callback in callbacks:
        callback()


@torch.library.custom_op("rl_kernel::qwen3_ffn_packed_inference", mutates_args=())
def _qwen3_ffn_packed_inference(
    rmsnorm_output: Tensor,
    fused_gate_up_weight: Tensor,
    down_weight: Tensor,
) -> Tensor:
    """Run the graph-safe compute portion of the strict rollout FFN."""

    _notify_packed_inference_observers()
    input_shape = rmsnorm_output.shape
    hidden_2d = rmsnorm_output.reshape(-1, input_shape[-1]).contiguous()
    gate_up = det_gemm_linear(
        hidden_2d,
        fused_gate_up_weight,
        native_op=_C.det_gemm_fwd_rhs_transposed,
    )
    activated = _C.swiglu_packed_forward(gate_up)
    output = det_gemm_linear(
        activated,
        down_weight,
        native_op=_C.det_gemm_fwd_rhs_transposed,
    )
    return output.reshape(*input_shape[:-1], output.size(-1))


@_qwen3_ffn_packed_inference.register_fake
def _qwen3_ffn_packed_inference_fake(
    rmsnorm_output: Tensor,
    fused_gate_up_weight: Tensor,
    down_weight: Tensor,
) -> Tensor:
    del fused_gate_up_weight
    return rmsnorm_output.new_empty((*rmsnorm_output.shape[:-1], down_weight.shape[0]))


@torch.library.custom_op(
    "rl_kernel::qwen3_ffn_packed_inference_to_staging",
    mutates_args={"output"},
)
def _qwen3_ffn_packed_inference_to_staging(
    rmsnorm_output: Tensor,
    fused_gate_up_weight: Tensor,
    down_weight: Tensor,
    output: Tensor,
) -> None:
    """Run strict FFN compute with the down projection targeting IPC staging."""

    _notify_packed_inference_observers()
    hidden_2d = rmsnorm_output.reshape(-1, rmsnorm_output.shape[-1]).contiguous()
    gate_up = det_gemm_linear(
        hidden_2d,
        fused_gate_up_weight,
        native_op=_C.det_gemm_fwd_rhs_transposed,
    )
    activated = _C.swiglu_packed_forward(gate_up)
    det_gemm_linear(
        activated,
        down_weight,
        native_op=_C.det_gemm_fwd_rhs_transposed,
        out=output,
    )


@_qwen3_ffn_packed_inference_to_staging.register_fake
def _qwen3_ffn_packed_inference_to_staging_fake(
    rmsnorm_output: Tensor,
    fused_gate_up_weight: Tensor,
    down_weight: Tensor,
    output: Tensor,
) -> None:
    del rmsnorm_output, fused_gate_up_weight, down_weight, output


@torch.library.custom_op(
    "rl_kernel::qwen3_ffn_packed_tp_inference_rocm",
    mutates_args=(),
)
def _qwen3_ffn_packed_tp_inference_rocm(
    rmsnorm_output: Tensor,
    fused_gate_up_weight: Tensor,
    down_weight: Tensor,
    collective_handle: int,
) -> Tensor:
    """Keep the ROCm TP FFN behind one eager graph-partition boundary."""

    binding = _PACKED_INFERENCE_STAGING_BY_HANDLE.get(collective_handle)
    if binding is None:
        raise RuntimeError("packed ROCm rollout FFN staging handle is not registered")
    runtime_handle, staging, stable_output = binding
    input_shape = rmsnorm_output.shape
    rows = rmsnorm_output.numel() // input_shape[-1]
    if rows <= staging.size(0):
        # Keep the output address stable across piecewise HIP-graph capture and
        # replay so the next captured partition reads the current invocation.
        direct_input = staging.narrow(0, 0, rows)
        output = stable_output.narrow(0, 0, rows)
        _C.deterministic_collective_rocm_ipc_prepare_staged(
            runtime_handle,
            direct_input,
        )
        _qwen3_ffn_packed_inference_to_staging(
            rmsnorm_output,
            fused_gate_up_weight,
            down_weight,
            direct_input,
        )
        _C.deterministic_collective_rocm_ipc_all_reduce_staged(runtime_handle, direct_input, output)
        return output.reshape(*input_shape[:-1], down_weight.shape[0])
    else:
        # Profiling and uncaptured prefill can exceed the decode capture bound.
        output = _qwen3_ffn_packed_inference(
            rmsnorm_output,
            fused_gate_up_weight,
            down_weight,
        )
    _C.deterministic_collective_rocm_ipc_all_reduce_input(
        runtime_handle,
        output,
        output,
    )
    return output.reshape(*input_shape[:-1], down_weight.shape[0])


@_qwen3_ffn_packed_tp_inference_rocm.register_fake
def _qwen3_ffn_packed_tp_inference_rocm_fake(
    rmsnorm_output: Tensor,
    fused_gate_up_weight: Tensor,
    down_weight: Tensor,
    collective_handle: int,
) -> Tensor:
    del fused_gate_up_weight, collective_handle
    return rmsnorm_output.new_empty((*rmsnorm_output.shape[:-1], down_weight.shape[0]))


def qwen3_ffn_packed_inference(
    rmsnorm_output: Tensor,
    fused_gate_up_weight: Tensor,
    down_weight: Tensor,
    *,
    collective_handle: int = 0,
    tp_world_size: int = 1,
    collective: Any | None = None,
) -> Tensor:
    """Inference-only packed FFN entry compatible with torch.compile."""

    if tp_world_size <= 1:
        return _qwen3_ffn_packed_inference(
            rmsnorm_output,
            fused_gate_up_weight,
            down_weight,
        )
    if collective_handle <= 0:
        raise RuntimeError("packed rollout FFN requires a bound TP collective")
    if (
        getattr(torch.version, "hip", None) is not None
        and collective is not None
        and collective_handle not in _PACKED_INFERENCE_STAGING_BY_HANDLE
    ):
        # Eager ROCm does not reserve graph staging.  Keep its established
        # in-place fixed-tree path; graph-enabled runs register the handle in
        # prepare_packed_inference and stay behind the opaque custom op below.
        output = _qwen3_ffn_packed_inference(
            rmsnorm_output,
            fused_gate_up_weight,
            down_weight,
        )
        return collective.all_reduce(output, out=output)
    if getattr(torch.version, "hip", None) is not None:
        return _qwen3_ffn_packed_tp_inference_rocm(
            rmsnorm_output,
            fused_gate_up_weight,
            down_weight,
            collective_handle,
        )
    input_shape = rmsnorm_output.shape
    output_shape_2d = (
        rmsnorm_output.numel() // input_shape[-1],
        down_weight.shape[0],
    )
    direct_staging = (
        None if collective is None else getattr(collective, "direct_staging_view", None)
    )
    direct_output = (
        None
        if not callable(direct_staging)
        else direct_staging(output_shape_2d, dtype=rmsnorm_output.dtype)
    )
    if direct_output is not None:
        deterministic_staging_reserve(
            direct_output,
            collective_handle=collective_handle,
        )
        _qwen3_ffn_packed_inference_to_staging(
            rmsnorm_output,
            fused_gate_up_weight,
            down_weight,
            direct_output,
        )
        reduced = deterministic_all_reduce_staged(
            direct_output,
            collective_handle=collective_handle,
        )
        return reduced.reshape(*input_shape[:-1], down_weight.shape[0])
    output = _qwen3_ffn_packed_inference(
        rmsnorm_output,
        fused_gate_up_weight,
        down_weight,
    )
    return deterministic_all_reduce_inplace(
        output,
        collective_handle=collective_handle,
    )


def _require_ffn_kernels(*, disable_split_k: bool, packed_gate_up: bool = False) -> None:
    required: tuple[str, ...] = _REQUIRED_SYMBOLS if disable_split_k else _SWIGLU_SYMBOLS
    if packed_gate_up:
        required = tuple(required) + tuple(_PACKED_SWIGLU_SYMBOLS)
    missing = [name for name in required if not hasattr(_C, name)]
    if not _EXT_AVAILABLE or _C is None or missing:
        suffix = f" Missing symbols: {', '.join(missing)}." if missing else ""
        needed = (
            "compiled deterministic GEMM and SwiGLU GPU kernels"
            if disable_split_k
            else "compiled SwiGLU GPU kernels"
        )
        raise RuntimeError(f"qwen3_ffn requires the {needed}.{suffix}")


def _linear_fwd(a: Tensor, weight: Tensor, *, disable_split_k: bool) -> Tensor:
    if disable_split_k:
        return det_gemm_linear(
            a,
            weight,
            native_op=_C.det_gemm_fwd_rhs_transposed,
        )
    # cuBLASLt / CUTLASS: may use split-K. Detach so Autograd.Function owns backward.
    with torch.no_grad():
        return torch.nn.functional.linear(a, weight)


def _linear_da(grad_output: Tensor, weight: Tensor, *, disable_split_k: bool) -> Tensor:
    if disable_split_k:
        return det_gemm_linear_input_gradient(
            grad_output,
            weight,
            native_op=_C.det_gemm_fwd,
        )
    with torch.no_grad():
        return torch.matmul(grad_output, weight)


def _linear_dw(a: Tensor, grad_output: Tensor, *, disable_split_k: bool) -> Tensor:
    if disable_split_k:
        return det_gemm_linear_weight_gradient(
            a,
            grad_output,
            native_op=_C.det_gemm_db_transposed,
        )
    with torch.no_grad():
        return torch.matmul(grad_output.t().contiguous(), a)


def _require_parallel_group(group: Any, name: str):
    if group is None:
        return None

    import torch.distributed as dist

    if not dist.is_available():
        raise RuntimeError(f"{name}-parallel FFN requires torch.distributed.")
    if not dist.is_initialized():
        raise RuntimeError(f"{name}-parallel FFN requires an initialized process group.")
    if dist.get_world_size(group=group) <= 1:
        raise ValueError(f"{name}_group must contain at least two ranks.")
    return dist


def _validate_ffn_inputs(
    rmsnorm_output: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
    fused_gate_up_weight: Tensor | None,
) -> None:
    tensors = {
        "rmsnorm_output": rmsnorm_output,
        "gate_weight": gate_weight,
        "up_weight": up_weight,
        "down_weight": down_weight,
    }
    if fused_gate_up_weight is not None:
        tensors["fused_gate_up_weight"] = fused_gate_up_weight
    for name, tensor in tensors.items():
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)!r}.")

    if rmsnorm_output.dim() < 1:
        raise ValueError("rmsnorm_output must have at least one dimension.")
    if rmsnorm_output.numel() == 0:
        raise ValueError("rmsnorm_output must contain at least one token.")
    for name, weight in (
        ("gate_weight", gate_weight),
        ("up_weight", up_weight),
        ("down_weight", down_weight),
    ):
        if weight.dim() != 2:
            raise ValueError(f"{name} must be 2-D, got shape {tuple(weight.shape)}.")

    hidden_size = rmsnorm_output.size(-1)
    intermediate_size = gate_weight.size(0)
    expected_shapes = {
        "gate_weight": (intermediate_size, hidden_size),
        "up_weight": (intermediate_size, hidden_size),
        "down_weight": (hidden_size, intermediate_size),
    }
    if fused_gate_up_weight is not None:
        expected_shapes["fused_gate_up_weight"] = (2 * intermediate_size, hidden_size)
    for name, expected in expected_shapes.items():
        actual = tuple(tensors[name].shape)
        if actual != expected:
            raise ValueError(f"{name} must have shape {expected}, got {actual}.")

    for name, tensor in tensors.items():
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} must have dtype bfloat16, got {tensor.dtype}.")
        if not tensor.is_cuda:
            # PyTorch exposes AMD GPU tensors through the torch.cuda API too.
            raise RuntimeError(f"{name} must be on a CUDA/ROCm GPU device, got '{tensor.device}'.")
        if tensor.device != rmsnorm_output.device:
            raise RuntimeError(
                f"all FFN inputs must be on {rmsnorm_output.device}, "
                f"got {name} on {tensor.device}."
            )


def _collective_for_group(group: Any, *, min_size_bytes: int):
    return collective_for_group(
        group=group,
        min_size_bytes=min_size_bytes,
        minimum_capacity_bytes=_COLLECTIVE_MIN_CAPACITY_BYTES,
    )


def _all_gather_tokens(tensor: Tensor, collective: Any) -> Tensor:
    return collective.all_gather(tensor.contiguous())


def _all_gather_packed_tokens(*tensors: Tensor, collective: Any) -> tuple[Tensor, ...]:
    """Gather same-row tensors with one handshake and no repacking copies."""

    return collective.all_gather_many(tuple(tensor.contiguous() for tensor in tensors))


def _reduce_scatter_tokens(tensor: Tensor, collective: Any) -> Tensor:
    world_size = collective.world_size
    if tensor.size(0) % world_size != 0:
        raise ValueError(
            "the gathered token count must be divisible by the tensor-parallel "
            f"world size, got {tensor.size(0)} and {world_size}."
        )
    return collective.reduce_scatter(tensor.contiguous())


def _all_reduce_inplace(tensor: Tensor, collective: Any) -> Tensor:
    return collective.all_reduce(tensor, out=tensor)


class _DeterministicFFNFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        rmsnorm_output: Tensor,
        gate_weight: Tensor,
        up_weight: Tensor,
        down_weight: Tensor,
        fused_gate_up_weight: Tensor | None,
        tp_group: Any,
        cp_group: Any,
        sequence_parallel: bool,
        disable_split_k: bool,
    ) -> Tensor:
        tp_dist = _require_parallel_group(tp_group, "tensor")
        _require_parallel_group(cp_group, "context")
        if sequence_parallel and tp_dist is None:
            raise ValueError("sequence_parallel requires a tensor-parallel group.")

        input_shape = rmsnorm_output.shape
        rmsnorm_output_2d = rmsnorm_output.reshape(-1, input_shape[-1]).contiguous()
        tp_world = tp_dist.get_world_size(group=tp_group) if tp_dist is not None else 1
        gemm_tokens = rmsnorm_output_2d.size(0) * (tp_world if sequence_parallel else 1)
        element_size = rmsnorm_output_2d.element_size()
        token_hidden_bytes = gemm_tokens * rmsnorm_output_2d.size(1) * element_size
        # Sequence-parallel backward reduces the gate and up input-gradient
        # lanes together. ``reduce_scatter_many`` packs those lanes along the
        # final dimension, so reserve capacity for both lanes in one transport
        # call rather than growing the collective (or failing) mid-backward.
        reduction_bytes = token_hidden_bytes * (2 if sequence_parallel else 1)
        min_size_bytes = max(
            reduction_bytes,
            gemm_tokens * gate_weight.size(0) * element_size,
            gate_weight.numel() * element_size,
            up_weight.numel() * element_size,
            down_weight.numel() * element_size,
        )
        if cp_group is not None:
            packed_width = 2 * rmsnorm_output_2d.size(1) + 3 * gate_weight.size(0)
            min_size_bytes = max(
                min_size_bytes,
                gemm_tokens * packed_width * element_size,
            )
        # Create TP before CP so every rank follows the same group order.
        tp_collective = _collective_for_group(tp_group, min_size_bytes=min_size_bytes)
        cp_collective = _collective_for_group(cp_group, min_size_bytes=min_size_bytes)

        if sequence_parallel:
            rmsnorm_output_2d = _all_gather_tokens(rmsnorm_output_2d, tp_collective)

        packed_gate_up = fused_gate_up_weight is not None and disable_split_k
        if packed_gate_up:
            assert fused_gate_up_weight is not None
            gate_up = _linear_fwd(
                rmsnorm_output_2d,
                fused_gate_up_weight,
                disable_split_k=True,
            )
            activated = _C.swiglu_packed_forward(gate_up)
        else:
            gate = _linear_fwd(
                rmsnorm_output_2d,
                gate_weight,
                disable_split_k=disable_split_k,
            )
            up = _linear_fwd(
                rmsnorm_output_2d,
                up_weight,
                disable_split_k=disable_split_k,
            )
            activated = _C.swiglu_forward(gate, up)
        output = _linear_fwd(activated, down_weight, disable_split_k=disable_split_k)

        if sequence_parallel:
            output = _reduce_scatter_tokens(output, tp_collective)
        elif tp_collective is not None:
            output = _all_reduce_inplace(output, tp_collective)

        if packed_gate_up:
            ctx.save_for_backward(
                rmsnorm_output_2d,
                gate_up,
                activated,
                gate_weight,
                up_weight,
                down_weight,
            )
        else:
            ctx.save_for_backward(
                rmsnorm_output_2d,
                gate,
                up,
                activated,
                gate_weight,
                up_weight,
                down_weight,
            )
        ctx.input_shape = input_shape
        ctx.tp_collective = tp_collective
        ctx.cp_collective = cp_collective
        ctx.sequence_parallel = sequence_parallel
        ctx.disable_split_k = disable_split_k
        ctx.packed_gate_up = packed_gate_up
        return output.reshape(*input_shape[:-1], output.size(-1))

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        if ctx.packed_gate_up:
            (
                rmsnorm_output,
                gate_up,
                activated,
                gate_weight,
                up_weight,
                down_weight,
            ) = ctx.saved_tensors
        else:
            (
                rmsnorm_output,
                gate,
                up,
                activated,
                gate_weight,
                up_weight,
                down_weight,
            ) = ctx.saved_tensors
        tp_collective = ctx.tp_collective
        cp_collective = ctx.cp_collective
        disable_split_k = ctx.disable_split_k
        grad_output = grad_output.reshape(-1, grad_output.size(-1)).contiguous()
        if ctx.sequence_parallel:
            grad_output = _all_gather_tokens(grad_output, tp_collective)

        # Down input-gradient shards concatenate across TP; no TP reduction.
        grad_activated = _linear_da(
            grad_output,
            down_weight,
            disable_split_k=disable_split_k,
        )
        if ctx.packed_gate_up:
            grad_gate, grad_up = _C.swiglu_packed_backward(grad_activated, gate_up)
        else:
            grad_gate, grad_up = _C.swiglu_backward(grad_activated, gate, up)

        # Weight gradients must see every CP token so the GEMM K-tree matches
        # CP=1. These payloads become available before any weight-gradient GEMM,
        # so one rank-ordered gather preserves the arithmetic contract while
        # avoiding four redundant collective handshakes per layer.
        if cp_collective is not None:
            (
                activated_full,
                grad_output_full,
                rmsnorm_full,
                grad_gate_full,
                grad_up_full,
            ) = _all_gather_packed_tokens(
                activated,
                grad_output,
                rmsnorm_output,
                grad_gate,
                grad_up,
                collective=cp_collective,
            )
            grad_down_weight = _linear_dw(
                activated_full,
                grad_output_full,
                disable_split_k=disable_split_k,
            )
            grad_gate_weight = _linear_dw(
                rmsnorm_full,
                grad_gate_full,
                disable_split_k=disable_split_k,
            )
            grad_up_weight = _linear_dw(
                rmsnorm_full,
                grad_up_full,
                disable_split_k=disable_split_k,
            )
        else:
            grad_down_weight = _linear_dw(
                activated,
                grad_output,
                disable_split_k=disable_split_k,
            )
            grad_gate_weight = _linear_dw(
                rmsnorm_output,
                grad_gate,
                disable_split_k=disable_split_k,
            )
            grad_up_weight = _linear_dw(
                rmsnorm_output,
                grad_up,
                disable_split_k=disable_split_k,
            )

        # Gate/Up input gradients reduce across TP, then add locally.
        grad_rmsnorm_from_gate = _linear_da(
            grad_gate,
            gate_weight,
            disable_split_k=disable_split_k,
        )
        grad_rmsnorm_from_up = _linear_da(
            grad_up,
            up_weight,
            disable_split_k=disable_split_k,
        )
        if ctx.sequence_parallel:
            # These are independent reduction lanes. Pack them into one
            # ReduceScatter while keeping each lane's balanced rank tree
            # separate; adding them before the collective would change the
            # floating-point parenthesization and break cross-TP bitwise
            # invariance.
            grad_rmsnorm_from_gate, grad_rmsnorm_from_up = tp_collective.reduce_scatter_many(
                (grad_rmsnorm_from_gate, grad_rmsnorm_from_up)
            )
        elif tp_collective is not None:
            grad_rmsnorm_from_gate = _all_reduce_inplace(
                grad_rmsnorm_from_gate,
                tp_collective,
            )
            grad_rmsnorm_from_up = _all_reduce_inplace(
                grad_rmsnorm_from_up,
                tp_collective,
            )

        grad_rmsnorm_output = grad_rmsnorm_from_gate.add_(grad_rmsnorm_from_up)
        return (
            grad_rmsnorm_output.reshape(ctx.input_shape),
            grad_gate_weight,
            grad_up_weight,
            grad_down_weight,
            None,
            None,
            None,
            None,
            None,
        )


def qwen3_ffn(
    rmsnorm_output: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
    *,
    fused_gate_up_weight: Tensor | None = None,
    tp_group: Any = None,
    cp_group: Any = None,
    sequence_parallel: bool = False,
    deterministic: bool | None = None,
    disable_split_k: bool | None = None,
) -> Tensor:
    """Apply a bias-free SiLU-gated FFN with deterministic backward kernels.

    Args:
        rmsnorm_output: RMSNorm output, shape ``[..., H]``.
        gate_weight: Gate projection weight in ``[out, in]`` layout, shape
            ``[I_local, H]``.
        up_weight: Up projection weight in ``[out, in]`` layout, shape
            ``[I_local, H]``.
        down_weight: Down projection weight in ``[out, in]`` layout, shape
            ``[H, I_local]``.
        fused_gate_up_weight: Optional existing framework weight in
            ``[2 * I_local, H]`` layout. Strict CUDA execution consumes this
            with one GEMM launch while returning gradients through the
            separate gate/up views, so the framework parameter layout stays
            unchanged.
        tp_group: Optional tensor-parallel process group. Gate and Up are
            column-parallel; Down is row-parallel. Reductions use the
            platform deterministic fixed-tree collectives. On ROCm, RCCL only
            transports rank inputs and the reduction tree executes locally.
        cp_group: Optional context-parallel process group. Each rank owns
            different token rows and the same local weight shards. Weight
            gradients AllGather tokens along CP and run the full-token
            ``det_gemm_db_transposed`` so they match CP=1 bitwise.
        sequence_parallel: Whether ``rmsnorm_output`` and the returned output
            are sharded on the flattened token dimension across ``tp_group``.
            Token gather/scatter use the deterministic AllGather and
            ReduceScatter.
        deterministic: Select the RL-Kernel fixed-reduction GEMM when True
            (default), or the production ``torch.matmul`` GEMM when False.
        disable_split_k: Compatibility alias for ``deterministic``. New code
            should use ``deterministic`` because Split-K is only one possible
            implementation detail of the production GEMM.

    Returns:
        FFN output with shape ``[..., H]``.
    """
    if not isinstance(sequence_parallel, bool):
        raise TypeError("sequence_parallel must be a bool.")
    deterministic = _resolve_deterministic_mode(deterministic, disable_split_k)
    _validate_ffn_inputs(
        rmsnorm_output,
        gate_weight,
        up_weight,
        down_weight,
        fused_gate_up_weight,
    )
    _require_ffn_kernels(
        disable_split_k=deterministic,
        packed_gate_up=fused_gate_up_weight is not None and deterministic,
    )
    return _DeterministicFFNFunction.apply(
        rmsnorm_output,
        gate_weight,
        up_weight,
        down_weight,
        fused_gate_up_weight,
        tp_group,
        cp_group,
        sequence_parallel,
        deterministic,
    )


def _resolve_deterministic_mode(
    deterministic: bool | None,
    disable_split_k: bool | None,
) -> bool:
    if deterministic is not None and not isinstance(deterministic, bool):
        raise TypeError("deterministic must be a bool or None.")
    if disable_split_k is not None and not isinstance(disable_split_k, bool):
        raise TypeError("disable_split_k must be a bool or None.")
    if (
        deterministic is not None
        and disable_split_k is not None
        and deterministic != disable_split_k
    ):
        raise ValueError("deterministic and disable_split_k select conflicting FFN backends.")
    if deterministic is not None:
        return deterministic
    if disable_split_k is not None:
        return disable_split_k
    return True


class Qwen3FFNOp:
    """Instantiable Qwen3 FFN wrapper for semantic operator dispatch."""

    op_class = "ffn"
    is_batch_invariant = True
    backend_id = BACKEND_ID

    def __init__(self) -> None:
        # Keep graph-bound IPC resources alive independently of the module-level
        # lookup cache. CUDA Graphs retain only the small opaque C++ handle.
        self._packed_inference_collectives: dict[int, Any] = {}

    def prepare_packed_inference(
        self,
        fused_gate_up_weight: Tensor,
        down_weight: Tensor,
        *,
        tp_group: Any,
    ) -> tuple[int, int]:
        """Create the rollout TP resource before Dynamo captures the model."""

        if tp_group is None:
            return 0, 1
        dist = _require_parallel_group(tp_group, "tensor")
        if dist is None:
            return 0, 1
        tp_world_size = int(dist.get_world_size(group=tp_group))
        if fused_gate_up_weight.size(0) % 2:
            raise ValueError("fused gate/up weight must contain two equal shards")
        element_size = fused_gate_up_weight.element_size()
        min_size_bytes = (
            max(
                fused_gate_up_weight.numel() // 2,
                down_weight.numel(),
            )
            * element_size
        )
        collective = _collective_for_group(
            tp_group,
            min_size_bytes=min_size_bytes,
        )
        max_capture = int(os.getenv("RL_KERNEL_VLLM_CUDAGRAPH_MAX_CAPTURE_SIZE", "0"))
        if max_capture <= 0:
            if getattr(torch.version, "hip", None) is not None:
                collective_handle = int(collective._handle)
                self._packed_inference_collectives[collective_handle] = collective
                return collective_handle, tp_world_size
            raise RuntimeError("packed rollout FFN requires a positive graph capture size")
        collective.prepare_direct_staging_views(
            ((batch, int(down_weight.shape[0])) for batch in range(1, max_capture + 1)),
            dtype=down_weight.dtype,
        )
        runtime_handle = int(collective._handle)
        collective_handle = runtime_handle
        if getattr(torch.version, "hip", None) is not None:
            staging = collective.direct_staging_view(
                (max_capture, int(down_weight.shape[0])),
                dtype=down_weight.dtype,
            )
            if staging is None:
                raise RuntimeError("packed ROCm rollout FFN staging allocation failed")
            collective_handle = _PACKED_INFERENCE_SLOT_BY_RUNTIME_HANDLE.get(runtime_handle, 0)
            if collective_handle == 0:
                # Keep the AOT graph identity stable across worker processes;
                # resolve its process-local C++ handle inside the custom op.
                collective_handle = len(_PACKED_INFERENCE_SLOT_BY_RUNTIME_HANDLE) + 1
                _PACKED_INFERENCE_SLOT_BY_RUNTIME_HANDLE[runtime_handle] = collective_handle
            binding = _PACKED_INFERENCE_STAGING_BY_HANDLE.get(collective_handle)
            stable_output = torch.empty_like(staging) if binding is None else binding[2]
            _PACKED_INFERENCE_STAGING_BY_HANDLE[collective_handle] = (
                runtime_handle,
                staging,
                stable_output,
            )
        self._packed_inference_collectives[collective_handle] = collective
        return collective_handle, tp_world_size

    def packed_inference_backend_id(self, collective_handle: int) -> str:
        collective = self._packed_inference_collectives.get(collective_handle)
        if collective is None:
            raise RuntimeError("packed rollout FFN collective is not bound")
        return str(
            getattr(
                collective,
                "backend_id",
                "deterministic_all_reduce.ipc_localized_fixed_tree.v1",
            )
        )

    def packed_inference(
        self,
        rmsnorm_output: Tensor,
        fused_gate_up_weight: Tensor,
        down_weight: Tensor,
        *,
        collective_handle: int,
        tp_world_size: int,
    ) -> Tensor:
        return qwen3_ffn_packed_inference(
            rmsnorm_output,
            fused_gate_up_weight,
            down_weight,
            collective_handle=collective_handle,
            tp_world_size=tp_world_size,
            collective=self._packed_inference_collectives.get(collective_handle),
        )

    def __call__(
        self,
        rmsnorm_output: Tensor,
        gate_weight: Tensor,
        up_weight: Tensor,
        down_weight: Tensor,
        *,
        fused_gate_up_weight: Tensor | None = None,
        tp_group: Any = None,
        cp_group: Any = None,
        sequence_parallel: bool = False,
        deterministic: bool | None = None,
        disable_split_k: bool | None = None,
    ) -> Tensor:
        return self.apply(
            rmsnorm_output,
            gate_weight,
            up_weight,
            down_weight,
            fused_gate_up_weight=fused_gate_up_weight,
            tp_group=tp_group,
            cp_group=cp_group,
            sequence_parallel=sequence_parallel,
            deterministic=deterministic,
            disable_split_k=disable_split_k,
        )

    def apply(
        self,
        rmsnorm_output: Tensor,
        gate_weight: Tensor,
        up_weight: Tensor,
        down_weight: Tensor,
        *,
        fused_gate_up_weight: Tensor | None = None,
        tp_group: Any = None,
        cp_group: Any = None,
        sequence_parallel: bool = False,
        deterministic: bool | None = None,
        disable_split_k: bool | None = None,
    ) -> Tensor:
        return qwen3_ffn(
            rmsnorm_output,
            gate_weight,
            up_weight,
            down_weight,
            fused_gate_up_weight=fused_gate_up_weight,
            tp_group=tp_group,
            cp_group=cp_group,
            sequence_parallel=sequence_parallel,
            deterministic=deterministic,
            disable_split_k=disable_split_k,
        )
