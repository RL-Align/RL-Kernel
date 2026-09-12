# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import socket
import threading
from collections.abc import Iterable
from types import TracebackType
from typing import Any

import torch
import torch.distributed as dist

_SUPPORTED_WORLD_SIZES = (1, 2, 4, 8)
_DEFAULT_MAX_SIZE_BYTES = 64 * 1024 * 1024
_COLLECTIVE_STAGING_FRAMES = 3
_COLLECTIVE_FRAME_METADATA_BYTES = 4 * 8
_DIRECT_STAGING_MAX_BYTES = 256 * 1024
_REDUCTION_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_REDUCTION_DTYPE_BYTES = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2}
_COLLECTIVES: dict[tuple[int, int, int, int], Any] = {}
DETERMINISTIC_ALL_REDUCE_OP = "rl_kernel::deterministic_all_reduce_"
DETERMINISTIC_STAGING_RESERVE_OP = "rl_kernel::deterministic_staging_reserve_"
DETERMINISTIC_STAGED_ALL_REDUCE_OP = "rl_kernel::deterministic_staged_all_reduce"


@torch.library.custom_op(DETERMINISTIC_ALL_REDUCE_OP, mutates_args={"input"})
def _deterministic_all_reduce_(input: torch.Tensor, collective_handle: int) -> None:
    """Expose the stateful IPC reduction as an explicit graph mutation."""

    from rl_engine import _C

    if torch.version.hip is not None:
        _C.deterministic_collective_rocm_ipc_all_reduce_input(collective_handle, input, input)
    else:
        _C.deterministic_collective_all_reduce_fused(collective_handle, input, input)


@_deterministic_all_reduce_.register_fake
def _deterministic_all_reduce_fake(
    input: torch.Tensor,
    collective_handle: int,
) -> None:
    del input, collective_handle


def deterministic_all_reduce_inplace(
    input: torch.Tensor,
    *,
    collective_handle: int,
) -> torch.Tensor:
    """Run the graph-visible deterministic all-reduce in place."""

    _deterministic_all_reduce_(input, collective_handle)
    return input


@torch.library.custom_op(DETERMINISTIC_STAGING_RESERVE_OP, mutates_args={"staging"})
def _deterministic_staging_reserve_(staging: torch.Tensor, collective_handle: int) -> None:
    """Wait until every peer has finished reading the shared direct-output slot."""

    from rl_engine import _C

    if torch.version.hip is not None:
        _C.deterministic_collective_rocm_ipc_prepare_staged(collective_handle, staging)
    else:
        _C.deterministic_collective_prepare_staged(collective_handle, staging)


@_deterministic_staging_reserve_.register_fake
def _deterministic_staging_reserve_fake(
    staging: torch.Tensor,
    collective_handle: int,
) -> None:
    del staging, collective_handle


@torch.library.custom_op(DETERMINISTIC_STAGED_ALL_REDUCE_OP, mutates_args=())
def _deterministic_staged_all_reduce(
    staging: torch.Tensor,
    collective_handle: int,
) -> torch.Tensor:
    """Reduce a GEMM result already resident in the local CUDA IPC payload."""

    from rl_engine import _C

    output = torch.empty_like(staging)
    if torch.version.hip is not None:
        _C.deterministic_collective_rocm_ipc_all_reduce_staged(collective_handle, staging, output)
    else:
        _C.deterministic_collective_all_reduce_staged(collective_handle, staging, output)
    return output


@_deterministic_staged_all_reduce.register_fake
def _deterministic_staged_all_reduce_fake(
    staging: torch.Tensor,
    collective_handle: int,
) -> torch.Tensor:
    del collective_handle
    return torch.empty_like(staging)


def deterministic_staging_reserve(
    staging: torch.Tensor,
    *,
    collective_handle: int,
) -> torch.Tensor:
    _deterministic_staging_reserve_(staging, collective_handle)
    return staging


def deterministic_all_reduce_staged(
    staging: torch.Tensor,
    *,
    collective_handle: int,
) -> torch.Tensor:
    return _deterministic_staged_all_reduce(staging, collective_handle)


class DeterministicCollective:
    """Correctness-first TP-invariant CUDA collectives for one eight-GPU node.

    TP sizes 1, 2, 4, and 8 use nested prefixes of the same balanced tree.
    A reduction is cross-TP bitwise invariant when every rank input is the
    corresponding contiguous subtree root of one canonical finest-grained
    reduction, as produced by a TBIK-compatible row-parallel kernel. Every
    node evaluates the lower logical subtree before the higher one.

    One instance owns a symmetric CUDA IPC staging buffer. All ranks must call
    its methods in the same order with matching shapes and dtypes. Device-side
    IPC sequence fences order staging and payload access without a steady-state
    host barrier and advance correctly during CUDA Graph replay.
    """

    backend_id = "cuda_ipc_fixed_tree"

    def __init__(
        self,
        group: dist.ProcessGroup | None = None,
        device: torch.device | str | int | None = None,
        *,
        max_size_bytes: int = _DEFAULT_MAX_SIZE_BYTES,
    ) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before collectives")
        if not torch.cuda.is_available():
            raise RuntimeError("deterministic collectives require CUDA")
        if max_size_bytes <= 0:
            raise ValueError("max_size_bytes must be positive")

        self.group = group if group is not None else dist.group.WORLD
        self.rank = dist.get_rank(group=self.group)
        self.world_size = dist.get_world_size(group=self.group)
        if self.world_size not in _SUPPORTED_WORLD_SIZES:
            raise ValueError(
                "deterministic collectives require world_size in "
                f"{_SUPPORTED_WORLD_SIZES}, got {self.world_size}"
            )

        if device is None:
            normalized_device = torch.device("cuda", torch.cuda.current_device())
        elif isinstance(device, int):
            normalized_device = torch.device("cuda", device)
        else:
            normalized_device = torch.device(device)
        if normalized_device.type != "cuda":
            raise ValueError(f"deterministic collectives require a CUDA device, got {device!r}")
        if normalized_device.index is None:
            normalized_device = torch.device("cuda", torch.cuda.current_device())
        if normalized_device.index != torch.cuda.current_device():
            raise ValueError(
                "the collective device must be the current CUDA device; call "
                f"torch.cuda.set_device({normalized_device.index}) first"
            )

        try:
            from rl_engine import _C
        except ImportError as exc:
            raise RuntimeError(
                "the RL-Kernel CUDA extension is required; rebuild with "
                "`pip install --no-build-isolation -e .`"
            ) from exc
        required_symbols = (
            "deterministic_collective_ipc_meta",
            "deterministic_collective_create",
            "deterministic_collective_destroy",
            "deterministic_collective_stage",
            "deterministic_collective_all_reduce",
            "deterministic_collective_prepare_staged",
            "deterministic_collective_all_reduce_staged",
            "deterministic_collective_reduce_scatter",
            "deterministic_collective_all_gather",
            "deterministic_collective_all_gather_many",
        )
        missing = [name for name in required_symbols if not hasattr(_C, name)]
        if missing:
            raise RuntimeError(
                "the RL-Kernel CUDA extension lacks deterministic collectives: "
                + ", ".join(missing)
            )

        self.device = normalized_device
        self.max_size_bytes = int(max_size_bytes)
        self._extension = _C
        self._lock = threading.Lock()
        self._handle = 0
        self._validated_signatures: set[tuple[Any, ...]] = set()
        self._direct_staging_views: dict[tuple[tuple[int, ...], torch.dtype], torch.Tensor] = {}
        self._staging = torch.zeros(
            _COLLECTIVE_STAGING_FRAMES * (self.max_size_bytes + _COLLECTIVE_FRAME_METADATA_BYTES),
            dtype=torch.uint8,
            device=self.device,
        )

        handle, offset = self._extension.deterministic_collective_ipc_meta(self._staging)
        local_meta = {
            "handle": handle,
            "offset": int(offset),
            "capacity": self.max_size_bytes,
            "hostname": socket.gethostname(),
        }
        gathered_meta: list[dict[str, Any] | None] = [None] * self.world_size
        dist.all_gather_object(gathered_meta, local_meta, group=self.group)
        if any(meta is None for meta in gathered_meta):
            raise RuntimeError("failed to exchange CUDA IPC metadata")
        complete_meta = [meta for meta in gathered_meta if meta is not None]
        hostnames = {meta["hostname"] for meta in complete_meta}
        if len(hostnames) != 1:
            raise ValueError("deterministic collectives require all ranks on one host")
        capacities = {meta["capacity"] for meta in complete_meta}
        if capacities != {self.max_size_bytes}:
            raise ValueError("all ranks must use the same max_size_bytes")

        handles = [meta["handle"] for meta in complete_meta]
        offsets = [meta["offset"] for meta in complete_meta]
        self._handle = self._extension.deterministic_collective_create(
            self._staging,
            handles,
            offsets,
            self.rank,
        )
        self._synchronize_ranks()

    def prepare_direct_staging_views(
        self,
        shapes: Iterable[tuple[int, ...]],
        *,
        dtype: torch.dtype,
    ) -> None:
        """Materialize stable, graph-capturable views of the local IPC payload."""

        if dtype not in _REDUCTION_DTYPE_BYTES:
            raise TypeError(f"unsupported direct-staging dtype {dtype}")
        element_size = _REDUCTION_DTYPE_BYTES[dtype]
        for raw_shape in shapes:
            shape = tuple(int(dim) for dim in raw_shape)
            numel = 1
            for dim in shape:
                if dim < 0:
                    raise ValueError(f"direct-staging dimensions must be non-negative, got {shape}")
                numel *= dim
            size_bytes = numel * element_size
            if size_bytes > min(self.max_size_bytes, _DIRECT_STAGING_MAX_BYTES):
                continue
            byte_view = self._staging.narrow(
                0,
                _COLLECTIVE_FRAME_METADATA_BYTES,
                size_bytes,
            )
            self._direct_staging_views[(shape, dtype)] = byte_view.view(dtype).view(shape)

    def direct_staging_view(
        self,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Return a pre-bound direct-output view, or ``None`` for an uncaptured shape."""

        return self._direct_staging_views.get((tuple(int(dim) for dim in shape), dtype))

    def all_reduce(
        self,
        input: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        validate_signature: bool = True,
    ) -> torch.Tensor:
        """Return the TBIK-compatible fixed-tree sum on every rank.

        Supported dtypes are float32, float16, and bfloat16. ``out`` may alias
        ``input``; the input is staged before the output kernel starts. Cross-TP
        invariance requires inputs to follow the class-level subtree contract.
        """

        self._check_open()
        self._validate_reduction_input(input)
        if out is None:
            out = torch.empty_like(input)
        self._validate_output(out, input)

        with self._lock:
            if validate_signature:
                self._validate_matching_signature("all_reduce", input)
            self._extension.deterministic_collective_all_reduce_fused(self._handle, input, out)
        return out

    def all_gather(
        self,
        input: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        validate_signature: bool = True,
    ) -> torch.Tensor:
        """Gather rank-ordered input bit patterns along dimension 0.

        The result is cross-TP invariant when every TP configuration partitions
        the same global dimension 0 into equal contiguous rank-ordered shards.
        """

        self._check_open()
        self._validate_gather_input(input)
        output_shape = (input.size(0) * self.world_size, *input.shape[1:])
        if out is None:
            out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
        self._validate_sharded_output(out, input, output_shape)

        with self._lock:
            if validate_signature:
                self._validate_matching_signature("all_gather", input)
            self._extension.deterministic_collective_all_gather_fused(self._handle, input, out)
        return out

    def all_gather_many(
        self,
        inputs: tuple[torch.Tensor, ...],
        *,
        validate_signature: bool = True,
    ) -> tuple[torch.Tensor, ...]:
        """Gather several tensors with one staging handshake and no packing."""

        if not inputs:
            raise ValueError("all_gather_many requires at least one input")
        outputs = []
        for input in inputs:
            self._validate_gather_input(input)
            output_shape = (input.size(0) * self.world_size, *input.shape[1:])
            output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
            self._validate_sharded_output(output, input, output_shape)
            outputs.append(output)
        with self._lock:
            if validate_signature:
                self._validate_matching_many_signature("all_gather_many", inputs)
            self._validate_many_capacity(inputs)
            self._extension.deterministic_collective_all_gather_many(
                self._handle,
                list(inputs),
                outputs,
            )
        return tuple(outputs)

    def reduce_scatter(
        self,
        input: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        validate_signature: bool = True,
    ) -> torch.Tensor:
        """TBIK-compatible fixed-tree sum, then a rank-ordered dimension-0 scatter.

        Concatenating all rank outputs is cross-TP invariant when inputs follow
        the class-level subtree contract and the global dimension-0 size is fixed.
        """

        self._check_open()
        self._validate_reduction_input(input)
        if input.dim() == 0:
            raise ValueError("reduce_scatter input must have at least one dimension")
        if input.size(0) % self.world_size != 0:
            raise ValueError(
                "reduce_scatter input.size(0) must be divisible by "
                f"world_size={self.world_size}; got {input.size(0)}"
            )
        output_shape = (input.size(0) // self.world_size, *input.shape[1:])
        if out is None:
            out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
        self._validate_sharded_output(out, input, output_shape)

        with self._lock:
            if validate_signature:
                self._validate_matching_signature("reduce_scatter", input)
            self._extension.deterministic_collective_stage(self._handle, input)
            self._extension.deterministic_collective_reduce_scatter(self._handle, out)
        return out

    def reduce_scatter_many(
        self,
        inputs: tuple[torch.Tensor, ...],
        *,
        validate_signature: bool = True,
    ) -> tuple[torch.Tensor, ...]:
        """Reduce-scatter several tensors through the single-tensor ABI."""

        if not inputs:
            raise ValueError("reduce_scatter_many requires at least one input")
        return tuple(
            self.reduce_scatter(input, validate_signature=validate_signature) for input in inputs
        )

    def close(self) -> None:
        """Release imported CUDA IPC mappings after the last collective call."""

        handle = getattr(self, "_handle", 0)
        if not handle:
            return
        torch.cuda.synchronize(self.device)
        self._handle = 0
        self._extension.deterministic_collective_destroy(handle)

    def __enter__(self) -> DeterministicCollective:
        self._check_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _check_open(self) -> None:
        if not getattr(self, "_handle", 0):
            raise RuntimeError("deterministic collective is closed")

    def _validate_reduction_input(self, input: torch.Tensor) -> None:
        if not input.is_cuda or input.device != self.device:
            raise ValueError(f"input must be on {self.device}, got {input.device}")
        if not input.is_contiguous():
            raise ValueError("input must be contiguous")
        if input.dtype not in _REDUCTION_DTYPES:
            raise TypeError(
                "deterministic reductions support float32, float16, and bfloat16; "
                f"got {input.dtype}"
            )
        input_bytes = input.numel() * input.element_size()
        if input_bytes > self.max_size_bytes:
            raise ValueError(
                f"input requires {input_bytes} bytes but max_size_bytes={self.max_size_bytes}"
            )

    def _validate_gather_input(self, input: torch.Tensor) -> None:
        if not input.is_cuda or input.device != self.device:
            raise ValueError(f"input must be on {self.device}, got {input.device}")
        if not input.is_contiguous():
            raise ValueError("input must be contiguous")
        if input.dim() == 0:
            raise ValueError("all_gather input must have at least one dimension")
        input_bytes = input.numel() * input.element_size()
        if input_bytes > self.max_size_bytes:
            raise ValueError(
                f"input requires {input_bytes} bytes but max_size_bytes={self.max_size_bytes}"
            )

    def _validate_output(self, output: torch.Tensor, input: torch.Tensor) -> None:
        if output.device != input.device:
            raise ValueError("out must be on the same device as input")
        if output.dtype != input.dtype:
            raise TypeError("out must have the same dtype as input")
        if output.shape != input.shape:
            raise ValueError("out must have the same shape as input")
        if not output.is_contiguous():
            raise ValueError("out must be contiguous")

    def _validate_sharded_output(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        output_shape: tuple[int, ...],
    ) -> None:
        if output.device != input.device:
            raise ValueError("out must be on the same device as input")
        if output.dtype != input.dtype:
            raise TypeError("out must have the same dtype as input")
        if output.shape != output_shape:
            raise ValueError(f"out must have shape {output_shape}, got {tuple(output.shape)}")
        if not output.is_contiguous():
            raise ValueError("out must be contiguous")

    def _validate_matching_signature(self, op_name: str, input: torch.Tensor) -> None:
        signature = (op_name, tuple(input.shape), str(input.dtype), input.numel())
        if signature in self._validated_signatures:
            return
        signatures: list[tuple[Any, ...] | None] = [None] * self.world_size
        dist.all_gather_object(signatures, signature, group=self.group)
        if any(peer_signature != signature for peer_signature in signatures):
            raise ValueError(
                f"all ranks must call {op_name} with matching shapes and dtypes; got {signatures}"
            )
        self._validated_signatures.add(signature)

    def _validate_matching_many_signature(
        self,
        op_name: str,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        signature = (
            op_name,
            tuple((tuple(input.shape), str(input.dtype), input.numel()) for input in inputs),
        )
        if signature in self._validated_signatures:
            return
        signatures: list[tuple[Any, ...] | None] = [None] * self.world_size
        dist.all_gather_object(signatures, signature, group=self.group)
        if any(peer_signature != signature for peer_signature in signatures):
            raise ValueError(
                f"all ranks must call {op_name} with matching tensors; got {signatures}"
            )
        self._validated_signatures.add(signature)

    def _validate_many_capacity(self, inputs: tuple[torch.Tensor, ...]) -> None:
        total_bytes = 0
        for input in inputs:
            total_bytes = (total_bytes + 15) & ~15
            total_bytes += input.numel() * input.element_size()
        if total_bytes > self.max_size_bytes:
            raise ValueError(
                f"inputs require {total_bytes} bytes but max_size_bytes={self.max_size_bytes}"
            )

    def _synchronize_ranks(self) -> None:
        torch.cuda.synchronize(self.device)
        backend = dist.get_backend(self.group)
        if backend == dist.Backend.NCCL or str(backend).lower() == "nccl":
            dist.barrier(group=self.group, device_ids=[self.device.index])
        else:
            dist.barrier(group=self.group)


def create_deterministic_collective(
    group: dist.ProcessGroup | None = None,
    device: torch.device | str | int | None = None,
    *,
    max_size_bytes: int = _DEFAULT_MAX_SIZE_BYTES,
) -> Any:
    """Create the deterministic collective for the active accelerator."""

    if getattr(torch.version, "hip", None) is not None:
        return RCCLDeterministicCollective(
            group=group,
            device=device,
            max_size_bytes=max_size_bytes,
        )

    return DeterministicCollective(
        group=group,
        device=device,
        max_size_bytes=max_size_bytes,
    )


def collective_for_group(
    group: dist.ProcessGroup | None,
    *,
    min_size_bytes: int = 0,
    minimum_capacity_bytes: int = _DEFAULT_MAX_SIZE_BYTES,
    device: torch.device | str | int | None = None,
) -> Any | None:
    """Return the process-local platform collective shared by hot-path ops."""

    if group is None:
        return None
    if min_size_bytes < 0:
        raise ValueError("min_size_bytes must be non-negative")
    if minimum_capacity_bytes <= 0:
        raise ValueError("minimum_capacity_bytes must be positive")

    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)
    if device is None:
        device_index = torch.cuda.current_device()
    else:
        normalized_device = (
            torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
        )
        device_index = (
            torch.cuda.current_device()
            if normalized_device.index is None
            else normalized_device.index
        )
    key = (id(group), rank, world_size, device_index)
    cached = _COLLECTIVES.get(key)
    if cached is not None and cached.max_size_bytes >= min_size_bytes:
        return cached
    # Borrowers such as Attention autograd contexts can outlive this cache
    # entry. Replacing an undersized entry must not invalidate those live
    # references; normal Python ownership closes it after the last borrower.

    collective = create_deterministic_collective(
        group=group,
        device=device_index,
        max_size_bytes=max(minimum_capacity_bytes, min_size_bytes),
    )
    _COLLECTIVES[key] = collective
    return collective


# Keep the public API provided by the ROCm test branch while leaving the CUDA
# implementation above identical to PR377. The ROCm classes own no device
# resources until constructed, so importing their definitions has no hot-path
# or accelerator side effect.
from rl_engine.distributed.rocm_collectives import (  # noqa: E402
    RCCLDeterministicCollective,
    TorchDistributedDeterministicCollective,
)

__all__ = [
    "DETERMINISTIC_ALL_REDUCE_OP",
    "DeterministicCollective",
    "RCCLDeterministicCollective",
    "TorchDistributedDeterministicCollective",
    "collective_for_group",
    "create_deterministic_collective",
    "deterministic_all_reduce_inplace",
]
