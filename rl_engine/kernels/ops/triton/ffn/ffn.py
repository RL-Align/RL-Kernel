# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""ROCm-native distributed deterministic Qwen3 FFN built with Triton.

BF16 is preserved at every GEMM/SwiGLU boundary, GEMMs use a canonical
FP32-leaf/BF16-node K tree, and TP reductions use the rank-ordered balanced
RCCL transport collective. CP gathers full token sequences before
weight-gradient GEMMs so their K tree is identical to CP=1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from rl_engine.kernels.ops.triton.activation.swiglu import (
    _launch_swiglu_bwd,
    _launch_swiglu_fwd,
)
from rl_engine.kernels.ops.triton.matmul.det_gemm import _triton_tree_gemm

QWEN3_8B_HIDDEN_SIZE = 4096
QWEN3_8B_INTERMEDIATE_SIZE = 12288

_COLLECTIVE_MIN_CAPACITY_BYTES = 64 * 1024 * 1024
_COLLECTIVES: dict[tuple[int, int, int, int], Any] = {}


@dataclass(eq=False)
class Qwen3FFNForwardWeights:
    """Stable-storage GEMM-ready copies produced after loading or synchronization."""

    gate_weight_t: Tensor
    up_weight_t: Tensor
    down_weight_t: Tensor
    _sources: tuple[Tensor, Tensor, Tensor] = field(repr=False)
    _source_data_ptrs: tuple[int, int, int] = field(repr=False)
    _source_versions: tuple[int | None, int | None, int | None] = field(repr=False)
    _packed_data_ptrs: tuple[int, int, int] = field(repr=False)
    _packed_versions: tuple[int, int, int] = field(repr=False)
    _source_shapes: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] = field(repr=False)
    _source_strides: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] = field(repr=False)

    def refresh_(
        self,
        gate_weight: Tensor,
        up_weight: Tensor,
        down_weight: Tensor,
    ) -> Qwen3FFNForwardWeights:
        """Refresh values in-place while preserving CUDA Graph-visible addresses."""

        return refresh_qwen3_ffn_forward_weights(
            self,
            gate_weight,
            up_weight,
            down_weight,
        )


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
) -> None:
    tensors = {
        "rmsnorm_output": rmsnorm_output,
        "gate_weight": gate_weight,
        "up_weight": up_weight,
        "down_weight": down_weight,
    }
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
    if intermediate_size == 0:
        raise ValueError("FFN intermediate size must be positive.")
    expected_shapes = {
        "gate_weight": (intermediate_size, hidden_size),
        "up_weight": (intermediate_size, hidden_size),
        "down_weight": (hidden_size, intermediate_size),
    }
    for name, expected in expected_shapes.items():
        actual = tuple(tensors[name].shape)
        if actual != expected:
            raise ValueError(f"{name} must have shape {expected}, got {actual}.")

    for name, tensor in tensors.items():
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} must have dtype bfloat16, got {tensor.dtype}.")
        if not tensor.is_cuda:
            raise RuntimeError(f"{name} must be on a CUDA/ROCm GPU device, got '{tensor.device}'.")
        if tensor.device != rmsnorm_output.device:
            raise RuntimeError(
                f"all FFN inputs must be on {rmsnorm_output.device}, "
                f"got {name} on {tensor.device}."
            )


def _tracked_tensor_version(tensor: Tensor) -> int | None:
    # Inference tensors deliberately have no version counter. Packed buffers are
    # allocated outside inference mode below, but a loader-owned source may be an
    # inference tensor and therefore relies on the explicit refresh lifecycle.
    return None if torch.is_inference(tensor) else int(tensor._version)


def _validate_forward_weight_sources(
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    weights = {
        "gate_weight": gate_weight,
        "up_weight": up_weight,
        "down_weight": down_weight,
    }
    for name, weight in weights.items():
        if not isinstance(weight, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(weight)!r}.")
        if weight.dim() != 2:
            raise ValueError(f"{name} must be 2-D, got shape {tuple(weight.shape)}.")
        if weight.dtype != torch.bfloat16:
            raise TypeError(f"{name} must have dtype bfloat16, got {weight.dtype}.")
        if not weight.is_cuda:
            raise RuntimeError(f"{name} must be on a CUDA/ROCm GPU device, got '{weight.device}'.")

    if tuple(up_weight.shape) != tuple(gate_weight.shape):
        raise ValueError(
            "up_weight must have the same shape as gate_weight, got "
            f"{tuple(up_weight.shape)} and {tuple(gate_weight.shape)}."
        )
    expected_down_shape = (gate_weight.size(1), gate_weight.size(0))
    if tuple(down_weight.shape) != expected_down_shape:
        raise ValueError(
            f"down_weight must have shape {expected_down_shape}, "
            f"got {tuple(down_weight.shape)}."
        )
    if any(weight.device != gate_weight.device for weight in weights.values()):
        raise RuntimeError("all FFN weights must be on the same device before packing.")
    return gate_weight, up_weight, down_weight


def pack_qwen3_ffn_forward_weights(
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
) -> Qwen3FFNForwardWeights:
    """Prepare detached forward-only transposes outside the FFN hot path.

    This is analogous to vLLM's backend-specific
    ``process_weights_after_loading``/kernel-format lifecycle: prepare the
    kernel-facing layout once per weight load/reload and reuse it at runtime.
    The canonical weights remain the source of truth for backward and
    optimization. Call ``refresh_`` after every optimizer update or external
    synchronization. Freshness checks are best-effort for inference tensors and
    external writers, so the loader/optimizer must refresh even when it mutates
    through ``.data``, DLPack, or a custom kernel.
    """

    sources = _validate_forward_weight_sources(gate_weight, up_weight, down_weight)
    # Ensure the cached storage has an ordinary version counter even when the
    # model loader calls us from torch.inference_mode(). A fresh explicit copy
    # also prevents degenerate transposes from aliasing source storage.
    with torch.inference_mode(False), torch.no_grad():
        packed = tuple(
            torch.empty(
                (weight.size(1), weight.size(0)),
                dtype=weight.dtype,
                device=weight.device,
            )
            for weight in sources
        )
        for packed_weight, source in zip(packed, sources, strict=True):
            packed_weight.copy_(source.t())
    return Qwen3FFNForwardWeights(
        gate_weight_t=packed[0],
        up_weight_t=packed[1],
        down_weight_t=packed[2],
        _sources=sources,
        _source_data_ptrs=tuple(weight.data_ptr() for weight in sources),
        _source_versions=tuple(_tracked_tensor_version(weight) for weight in sources),
        _packed_data_ptrs=tuple(weight.data_ptr() for weight in packed),
        _packed_versions=tuple(int(weight._version) for weight in packed),
        _source_shapes=tuple(tuple(weight.shape) for weight in sources),
        _source_strides=tuple(tuple(weight.stride()) for weight in sources),
    )


def refresh_qwen3_ffn_forward_weights(
    forward_weights: Qwen3FFNForwardWeights,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
) -> Qwen3FFNForwardWeights:
    """Refresh a forward cache in-place without changing any packed data pointer.

    Stable storage is required by already-captured CUDA Graphs: replacing either
    the canonical tensors or this bundle would leave graph nodes pointing at old
    values. Copy reloaded values into the original canonical tensors first, then
    call this function. Order both copies before graph replay, normally on the
    same stream or with an explicit stream dependency.
    """

    if not isinstance(forward_weights, Qwen3FFNForwardWeights):
        raise TypeError("forward_weights must be created by " "pack_qwen3_ffn_forward_weights.")
    sources = _validate_forward_weight_sources(gate_weight, up_weight, down_weight)
    if any(
        source is not original
        for source, original in zip(sources, forward_weights._sources, strict=True)
    ):
        raise ValueError(
            "stable refresh requires the original canonical weight tensors; "
            "copy new values into those tensors, or repack and recapture CUDA Graphs."
        )
    packed = (
        forward_weights.gate_weight_t,
        forward_weights.up_weight_t,
        forward_weights.down_weight_t,
    )
    for name, target, source, expected_ptr in zip(
        ("gate_weight", "up_weight", "down_weight"),
        packed,
        sources,
        forward_weights._packed_data_ptrs,
        strict=True,
    ):
        expected_shape = (source.size(1), source.size(0))
        if tuple(target.shape) != expected_shape:
            raise ValueError(
                f"packed {name} has shape {tuple(target.shape)}, but refresh "
                f"requires {expected_shape}; repack and recapture CUDA Graphs."
            )
        if target.dtype != source.dtype or target.device != source.device:
            raise RuntimeError(
                f"packed {name} dtype/device cannot change during stable refresh; "
                "repack and recapture CUDA Graphs."
            )
        if not target.is_contiguous() or target.data_ptr() != expected_ptr:
            raise RuntimeError(f"packed {name} storage changed; repack and recapture CUDA Graphs.")

    with torch.inference_mode(False), torch.no_grad():
        for target, source in zip(packed, sources, strict=True):
            target.copy_(source.t())

    forward_weights._source_versions = tuple(_tracked_tensor_version(weight) for weight in sources)
    forward_weights._packed_versions = tuple(int(weight._version) for weight in packed)
    return forward_weights


def _validate_forward_weights(
    forward_weights: Qwen3FFNForwardWeights,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
) -> None:
    if not isinstance(forward_weights, Qwen3FFNForwardWeights):
        raise TypeError("forward_weights must be created by " "pack_qwen3_ffn_forward_weights.")

    sources = (gate_weight, up_weight, down_weight)
    names = ("gate_weight", "up_weight", "down_weight")
    for index, (name, source) in enumerate(zip(names, sources, strict=True)):
        if forward_weights._sources[index] is not source:
            raise ValueError(f"forward_weights was not packed from this {name} tensor.")
        if forward_weights._source_data_ptrs[index] != source.data_ptr():
            raise RuntimeError(f"{name} storage changed after forward weights were packed.")
        if forward_weights._source_shapes[index] != tuple(source.shape):
            raise RuntimeError(f"{name} shape changed after forward weights were packed.")
        if forward_weights._source_strides[index] != tuple(source.stride()):
            raise RuntimeError(f"{name} strides changed after forward weights were packed.")
        if forward_weights._source_versions[index] != _tracked_tensor_version(source):
            raise RuntimeError(
                f"{name} changed after forward weights were packed; refresh before FFN."
            )

    expected = (
        (gate_weight.size(1), gate_weight.size(0)),
        (up_weight.size(1), up_weight.size(0)),
        (down_weight.size(1), down_weight.size(0)),
    )
    packed = (
        forward_weights.gate_weight_t,
        forward_weights.up_weight_t,
        forward_weights.down_weight_t,
    )
    for name, weight, shape in zip(names, packed, expected, strict=True):
        if not isinstance(weight, Tensor):
            raise TypeError(f"packed {name} must be a torch.Tensor.")
        if tuple(weight.shape) != shape:
            raise ValueError(f"packed {name} must have shape {shape}, got {tuple(weight.shape)}.")
        if weight.dtype != torch.bfloat16:
            raise TypeError(f"packed {name} must have dtype bfloat16, got {weight.dtype}.")
        if weight.device != gate_weight.device:
            raise RuntimeError(
                f"packed {name} must be on {gate_weight.device}, got {weight.device}."
            )
        if not weight.is_contiguous():
            raise ValueError(f"packed {name} must be contiguous.")
        if weight.requires_grad:
            raise ValueError(f"packed {name} must be detached from autograd.")
    for index, (name, weight) in enumerate(zip(names, packed, strict=True)):
        if forward_weights._packed_data_ptrs[index] != weight.data_ptr():
            raise RuntimeError(f"packed {name} storage changed; repack before FFN.")
        if forward_weights._packed_versions[index] != int(weight._version):
            raise RuntimeError(f"packed {name} changed; refresh before FFN.")


def _create_collective(*, group: Any, max_size_bytes: int):
    try:
        from rl_engine.distributed import create_deterministic_collective
    except ImportError as exc:
        raise RuntimeError(
            "parallel Triton FFN requires the deterministic collective factory"
        ) from exc
    return create_deterministic_collective(
        group=group,
        max_size_bytes=max_size_bytes,
    )


def _collective_for_group(group: Any, *, min_size_bytes: int):
    if group is None:
        return None

    import torch.distributed as dist

    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)
    device_index = torch.cuda.current_device()
    key = (id(group), rank, world_size, device_index)
    cached = _COLLECTIVES.get(key)
    if cached is not None and cached.max_size_bytes >= min_size_bytes:
        return cached
    if cached is not None:
        cached.close()

    collective = _create_collective(
        group=group,
        max_size_bytes=max(_COLLECTIVE_MIN_CAPACITY_BYTES, min_size_bytes),
    )
    _COLLECTIVES[key] = collective
    return collective


def _all_gather_tokens(tensor: Tensor, collective: Any) -> Tensor:
    return collective.all_gather(tensor.contiguous())


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


def _gemm(a: Tensor, b: Tensor) -> Tensor:
    return _triton_tree_gemm(a, b)


def _gemm_db(a: Tensor, grad_output: Tensor) -> Tensor:
    grad_weight = torch.empty(
        (grad_output.size(1), a.size(1)),
        dtype=torch.bfloat16,
        device=a.device,
    )
    # Mirror only the buffer placement used by Transformer Engine's
    # fuse_wgrad_accumulation and Megatron's gradient_accumulation_fusion: write
    # the canonical GEMM root into the final [out, in] gradient buffer. The
    # strict operand order, K tree, BF16 nodes, and rounding remain unchanged.
    # Unlike their fused main_grad paths, this returns one fresh autograd dW and
    # does not fuse or reorder microbatch accumulation.
    return _triton_tree_gemm(
        a.t(),
        grad_output,
        transpose_output=True,
        out=grad_weight,
        preserve_a_strides=True,
    )


class _TritonDeterministicFFNFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        rmsnorm_output: Tensor,
        gate_weight: Tensor,
        up_weight: Tensor,
        down_weight: Tensor,
        gate_weight_t: Tensor | None,
        up_weight_t: Tensor | None,
        down_weight_t: Tensor | None,
        tp_group: Any,
        cp_group: Any,
        sequence_parallel: bool,
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
        # Sequence-parallel backward reduces the independent gate/up input
        # gradient lanes in one fixed-tree transport call.
        reduction_bytes = token_hidden_bytes * (2 if sequence_parallel else 1)
        min_size_bytes = max(
            reduction_bytes,
            gemm_tokens * gate_weight.size(0) * element_size,
            gate_weight.numel() * element_size,
            up_weight.numel() * element_size,
            down_weight.numel() * element_size,
        )
        tp_collective = _collective_for_group(tp_group, min_size_bytes=min_size_bytes)
        cp_collective = _collective_for_group(cp_group, min_size_bytes=min_size_bytes)

        if sequence_parallel:
            rmsnorm_output_2d = _all_gather_tokens(rmsnorm_output_2d, tp_collective)

        # A loader integration can prepare/cache this backend-specific layout
        # once per load/reload, analogous to vLLM's
        # process_weights_after_loading lifecycle. Canonical weights stay
        # untouched and are saved below for backward.
        gate = _gemm(
            rmsnorm_output_2d,
            gate_weight.t().contiguous() if gate_weight_t is None else gate_weight_t,
        )
        up = _gemm(
            rmsnorm_output_2d,
            up_weight.t().contiguous() if up_weight_t is None else up_weight_t,
        )
        activated = _launch_swiglu_fwd(gate, up)
        output = _gemm(
            activated,
            down_weight.t().contiguous() if down_weight_t is None else down_weight_t,
        )

        if sequence_parallel:
            output = _reduce_scatter_tokens(output, tp_collective)
        elif tp_collective is not None:
            output = _all_reduce_inplace(output, tp_collective)

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
        return output.reshape(*input_shape[:-1], output.size(-1))

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Any, ...]:
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
        grad_output = grad_output.reshape(-1, grad_output.size(-1)).contiguous()
        if ctx.sequence_parallel:
            grad_output = _all_gather_tokens(grad_output, tp_collective)

        if cp_collective is not None:
            activated_full = _all_gather_tokens(activated, cp_collective)
            grad_output_full = _all_gather_tokens(grad_output, cp_collective)
            grad_down_weight = _gemm_db(activated_full, grad_output_full)
        else:
            grad_down_weight = _gemm_db(activated, grad_output)

        grad_activated = _gemm(grad_output, down_weight)
        grad_gate, grad_up = _launch_swiglu_bwd(grad_activated, gate, up)

        if cp_collective is not None:
            rmsnorm_full = _all_gather_tokens(rmsnorm_output, cp_collective)
            grad_gate_full = _all_gather_tokens(grad_gate, cp_collective)
            grad_up_full = _all_gather_tokens(grad_up, cp_collective)
            grad_gate_weight = _gemm_db(rmsnorm_full, grad_gate_full)
            grad_up_weight = _gemm_db(rmsnorm_full, grad_up_full)
        else:
            grad_gate_weight = _gemm_db(rmsnorm_output, grad_gate)
            grad_up_weight = _gemm_db(rmsnorm_output, grad_up)

        grad_rmsnorm_from_gate = _gemm(grad_gate, gate_weight)
        grad_rmsnorm_from_up = _gemm(grad_up, up_weight)
        if ctx.sequence_parallel:
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
            None,
        )


def qwen3_ffn(
    rmsnorm_output: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
    *,
    forward_weights: Qwen3FFNForwardWeights | None = None,
    tp_group: Any = None,
    cp_group: Any = None,
    sequence_parallel: bool = False,
) -> Tensor:
    """Apply the distributed deterministic Qwen3 FFN with Triton kernels.

    ``forward_weights`` is an optional, forward-only cache. Refresh it in-place
    after every optimizer update or external weight synchronization. The
    canonical weight arguments always remain the autograd/optimizer source of
    truth. Refresh is mandatory for inference tensors and external writers,
    whose mutations cannot always be discovered from a PyTorch version counter.
    """

    _validate_ffn_inputs(rmsnorm_output, gate_weight, up_weight, down_weight)
    if forward_weights is not None:
        _validate_forward_weights(
            forward_weights,
            gate_weight,
            up_weight,
            down_weight,
        )
    if not isinstance(sequence_parallel, bool):
        raise TypeError(f"sequence_parallel must be a bool, got {type(sequence_parallel)!r}.")
    return _TritonDeterministicFFNFunction.apply(
        rmsnorm_output,
        gate_weight,
        up_weight,
        down_weight,
        None if forward_weights is None else forward_weights.gate_weight_t,
        None if forward_weights is None else forward_weights.up_weight_t,
        None if forward_weights is None else forward_weights.down_weight_t,
        tp_group,
        cp_group,
        sequence_parallel,
    )


# Keep the explicit suffix for callers that select an implementation by name.
qwen3_ffn_triton = qwen3_ffn
