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
_PACKED_REDUCE_SCATTER_MAX_BYTES = 8 * 1024 * 1024
_ROCM_IPC_DIRECT_ALL_REDUCE_MAX_BYTES = 768 * 1024
_ROCM_IPC_SHARDED_ALL_REDUCE_MIN_BYTES = 2176 * 1024
_ROCM_IPC_ALL_GATHER_MAX_BYTES = 256 * 1024
_ROCM_IPC_CONTROL_BYTES = 256
_REDUCTION_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


class TorchDistributedDeterministicCollective:
    """Correctness-first collectives using AllGather as transport only.

    Rank inputs are gathered without arithmetic and reduced locally as the
    balanced tree ``((rank0 + rank1) + (rank2 + rank3)) + ...``. Consequently,
    all ranks execute the exact same floating-point expression. TP sizes 1,
    2, 4, and 8 are nested prefixes of that expression and match the existing
    CUDA IPC collective's ordering.

    The generic class also supports a CPU/Gloo process group, which is useful
    as an executable reference. Production ROCm callers should use
    :class:`RCCLDeterministicCollective` or
    :func:`create_deterministic_collective` so backend validation fails closed.
    All ranks must call methods in the same order with matching input shapes
    and dtypes, and construct the instance with the same ``max_size_bytes``.
    """

    backend_id = "torch_distributed_balanced_tree"
    transport_only = True
    reduction_order = "balanced_rank_tree"
    supports_async_overlap = False
    supports_compute_communication_fusion = False

    def __init__(
        self,
        group: dist.ProcessGroup | None = None,
        device: torch.device | str | int | None = None,
        *,
        max_size_bytes: int = _DEFAULT_MAX_SIZE_BYTES,
    ) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before collectives")
        if max_size_bytes <= 0:
            raise ValueError("max_size_bytes must be positive")

        self.group = group if group is not None else dist.group.WORLD
        self.rank = int(dist.get_rank(group=self.group))
        self.world_size = int(dist.get_world_size(group=self.group))
        if self.world_size not in _SUPPORTED_WORLD_SIZES:
            raise ValueError(
                "deterministic collectives require world_size in "
                f"{_SUPPORTED_WORLD_SIZES}, got {self.world_size}"
            )

        self.device = self._normalize_device(device)
        self.max_size_bytes = int(max_size_bytes)
        self._backend = str(dist.get_backend(self.group)).lower()
        self._lock = threading.Lock()
        self._closed = False
        # Keep a lifecycle marker for callers that historically inspected the
        # CUDA IPC collective's ``_handle`` while managing the cache. Concrete
        # transports own any native resource through their own state.
        self._handle = id(self)
        # One dtype-agnostic byte workspace is grown on demand and reused by
        # reduction collectives. AllGather writes directly into its output.
        self._workspace: torch.Tensor | None = None
        # A Python-object collective is useful for catching a mismatched new
        # signature, but running one on every hot-path call dominates small
        # message latency. Validate each local signature once and then rely on
        # the standard collective contract that ranks call operations in the
        # same order.
        self._validated_signatures: set[tuple[Any, ...]] = set()
        self._validate_matching_capacity()

    @staticmethod
    def _normalize_device(
        device: torch.device | str | int | None,
    ) -> torch.device:
        if device is None:
            if torch.cuda.is_available():
                normalized = torch.device("cuda", torch.cuda.current_device())
            else:
                normalized = torch.device("cpu")
        elif isinstance(device, int):
            normalized = torch.device("cuda", device)
        else:
            normalized = torch.device(device)

        if normalized.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("a CUDA/ROCm device was requested but none is available")
            current_device = torch.cuda.current_device()
            if normalized.index is None:
                normalized = torch.device("cuda", current_device)
            if normalized.index != current_device:
                raise ValueError(
                    "the collective device must be the current CUDA/ROCm device; call "
                    f"torch.cuda.set_device({normalized.index}) first"
                )
        return normalized

    @property
    def closed(self) -> bool:
        """Whether this instance rejects further collective calls."""

        return self._closed

    @property
    def workspace_size_bytes(self) -> int:
        """Currently retained reduction workspace size in bytes."""

        workspace = self._workspace
        return 0 if workspace is None else int(workspace.numel())

    def all_reduce(
        self,
        input: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        validate_signature: bool = True,
    ) -> torch.Tensor:
        """Return the fixed balanced-tree sum on every rank."""

        self._check_open()
        self._validate_reduction_input(input)
        if out is None:
            out = torch.empty_like(input)
        self._validate_output(out, input, tuple(input.shape))
        if self.world_size == 1:
            out.copy_(input)
            return out

        with self._lock:
            self._check_open()
            if validate_signature:
                self._validate_matching_signature("all_reduce", input)
            if self._direct_all_reduce(input, out):
                return out
            rank_inputs = self._all_gather_transport(input)
            if not self._fused_reduction(
                rank_inputs,
                out,
                operation="all_reduce",
            ):
                reduced = self._balanced_tree_sum(rank_inputs)
                out.copy_(reduced)
        return out

    def all_gather(
        self,
        input: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        validate_signature: bool = True,
    ) -> torch.Tensor:
        """Gather rank-ordered input bit patterns along dimension 0."""

        self._check_open()
        self._validate_gather_input(input)
        output_shape = (input.size(0) * self.world_size, *input.shape[1:])
        if out is None:
            out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
        self._validate_output(out, input, output_shape)
        if self.world_size == 1:
            out.copy_(input)
            return out

        with self._lock:
            self._check_open()
            if validate_signature:
                self._validate_matching_signature("all_gather", input)
            if self._direct_all_gather(input, out):
                return out
            self._all_gather_transport(input, gathered_flat=out.view(-1))
        return out

    def all_gather_many(
        self,
        inputs: tuple[torch.Tensor, ...] | list[torch.Tensor],
        *,
        validate_signature: bool = True,
    ) -> tuple[torch.Tensor, ...]:
        """Gather several tensors through the platform transport."""

        values = tuple(inputs)
        if not values:
            raise ValueError("all_gather_many requires at least one input")
        return tuple(
            self.all_gather(value, validate_signature=validate_signature) for value in values
        )

    def reduce_scatter(
        self,
        input: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        validate_signature: bool = True,
    ) -> torch.Tensor:
        """Fixed-tree sum followed by rank-ordered dimension-0 slicing."""

        self._check_open()
        self._validate_reduction_input(input)
        if input.dim() == 0:
            raise ValueError("reduce_scatter input must have at least one dimension")
        if input.size(0) % self.world_size != 0:
            raise ValueError(
                "reduce_scatter input.size(0) must be divisible by "
                f"world_size={self.world_size}; got {input.size(0)}"
            )
        rows_per_rank = input.size(0) // self.world_size
        output_shape = (rows_per_rank, *input.shape[1:])
        if out is None:
            out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
        self._validate_output(out, input, output_shape)
        if self.world_size == 1:
            out.copy_(input)
            return out

        with self._lock:
            self._check_open()
            if validate_signature:
                self._validate_matching_signature("reduce_scatter", input)
            if self._direct_reduce_scatter(input, out):
                return out
            rank_inputs = self._all_gather_transport(input)
            begin = self.rank * rows_per_rank
            # Only this rank's output shard participates in the reduction. The
            # previous implementation reduced every global row and sliced the
            # result afterwards, doing world_size times more arithmetic than
            # ReduceScatter needs. The fixed rank tree is unchanged.
            reduced = rank_inputs[:, begin : begin + rows_per_rank]
            if not self._fused_reduction(reduced, out, operation="reduce_scatter"):
                reduced = self._balanced_tree_sum(reduced)
                out.copy_(reduced)
        return out

    def reduce_scatter_many(
        self,
        inputs: tuple[torch.Tensor, ...] | list[torch.Tensor],
        *,
        outs: tuple[torch.Tensor, ...] | list[torch.Tensor] | None = None,
        validate_signature: bool = True,
    ) -> tuple[torch.Tensor, ...]:
        """Reduce-scatter independent tensors in one fixed-tree collective.

        The tensors are packed along their final dimension, so each tensor's
        element still follows the same balanced rank tree as an individual
        ``reduce_scatter`` call. This is useful for independent gradient lanes:
        packing them together removes one RCCL launch without changing the
        floating-point expression for either lane. Inputs must have matching
        shape/device/dtype except for the final dimension.
        """

        self._check_open()
        values = tuple(inputs)
        if not values:
            raise ValueError("reduce_scatter_many requires at least one input")
        if outs is not None and len(outs) != len(values):
            raise ValueError("reduce_scatter_many outs must match the number of inputs")
        if len(values) == 1:
            return (
                self.reduce_scatter(
                    values[0],
                    out=None if outs is None else outs[0],
                    validate_signature=validate_signature,
                ),
            )

        first = values[0]
        self._validate_reduction_input(first)
        if first.dim() < 2:
            raise ValueError(
                "reduce_scatter_many inputs must have at least two dimensions "
                "when packing independent lanes"
            )
        if first.size(0) % self.world_size != 0:
            raise ValueError("reduce_scatter_many inputs must have a divisible leading dimension")
        for value in values[1:]:
            self._validate_reduction_input(value)
            if value.dim() != first.dim() or value.shape[:-1] != first.shape[:-1]:
                raise ValueError(
                    "reduce_scatter_many inputs must match in rank and all dimensions "
                    "except the final dimension"
                )
            if value.device != first.device or value.dtype != first.dtype:
                raise ValueError("reduce_scatter_many inputs must share device and dtype")
        lane_sizes = tuple(int(value.size(-1)) for value in values)
        rows_per_rank = first.size(0) // self.world_size
        output_shape = (rows_per_rank, *first.shape[1:-1])
        if outs is not None:
            for lane_size, out in zip(lane_sizes, outs, strict=True):
                self._validate_output(
                    out,
                    first,
                    (*output_shape, lane_size),
                )

        packed_bytes = sum(value.numel() * value.element_size() for value in values)
        if self._can_direct_reduce_scatter_many():
            if packed_bytes > self.max_size_bytes:
                raise ValueError(
                    "reduce_scatter_many packed input requires "
                    f"{packed_bytes} bytes but max_size_bytes={self.max_size_bytes}"
                )
            direct_outputs = tuple(
                (
                    outs[index]
                    if outs is not None
                    else torch.empty(
                        (*output_shape, lane_size),
                        dtype=first.dtype,
                        device=first.device,
                    )
                )
                for index, lane_size in enumerate(lane_sizes)
            )
            with self._lock:
                self._check_open()
                if validate_signature:
                    self._validate_matching_signature(
                        f"reduce_scatter_many:{lane_sizes}",
                        first,
                    )
                if self._direct_reduce_scatter_many(values, direct_outputs):
                    return direct_outputs

        if packed_bytes > _PACKED_REDUCE_SCATTER_MAX_BYTES:
            # A single packed AllGather moves the same bytes as two separate
            # calls but loses RCCL's smaller-message algorithm. Use the
            # established per-lane path above the measured crossover; this
            # keeps the convenience API from regressing large FFN gradients.
            return tuple(
                self.reduce_scatter(
                    value,
                    out=None if outs is None else outs[index],
                    validate_signature=validate_signature,
                )
                for index, value in enumerate(values)
            )

        if packed_bytes > self.max_size_bytes:
            raise ValueError(
                "reduce_scatter_many packed input requires "
                f"{packed_bytes} bytes but max_size_bytes={self.max_size_bytes}"
            )
        packed = torch.cat(values, dim=-1)
        packed_out = torch.empty(
            (packed.size(0) // self.world_size, *packed.shape[1:]),
            dtype=packed.dtype,
            device=packed.device,
        )
        with self._lock:
            self._check_open()
            # Include lane boundaries in the signature. Equal packed shapes
            # alone do not guarantee that every rank will split the result the
            # same way, which could silently associate gradients with the
            # wrong lane.
            if validate_signature:
                self._validate_matching_signature(
                    f"reduce_scatter_many:{lane_sizes}",
                    packed,
                )
            if self._direct_reduce_scatter(packed, packed_out):
                pieces = tuple(packed_out.split(tuple(value.size(-1) for value in values), dim=-1))
                if outs is None:
                    return pieces
                result: list[torch.Tensor] = []
                for piece, out in zip(pieces, outs, strict=True):
                    out.copy_(piece)
                    result.append(out)
                return tuple(result)
            rank_inputs = self._all_gather_transport(packed)
            begin = self.rank * rows_per_rank
            reduced = rank_inputs[:, begin : begin + rows_per_rank]
            if not self._fused_reduction(
                reduced,
                packed_out,
                operation="reduce_scatter",
            ):
                reduced = self._balanced_tree_sum(reduced)
                packed_out.copy_(reduced)

        pieces = tuple(packed_out.split(tuple(value.size(-1) for value in values), dim=-1))
        if outs is None:
            return pieces
        result: list[torch.Tensor] = []
        for piece, out in zip(pieces, outs, strict=True):
            out.copy_(piece)
            result.append(out)
        return tuple(result)

    def close(self) -> None:
        """Close the instance.

        Closing releases the lazily allocated reduction workspace and marks
        the lifecycle boundary. Collective calls are blocking at this API.
        """

        with self._lock:
            self._workspace = None
            self._validated_signatures.clear()
            self._closed = True
            self._handle = 0

    def __enter__(self) -> TorchDistributedDeterministicCollective:
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
        if getattr(self, "_closed", True):
            raise RuntimeError("deterministic collective is closed")

    def _validate_tensor(self, input: torch.Tensor) -> None:
        if not isinstance(input, torch.Tensor):
            raise TypeError(f"input must be a torch.Tensor, got {type(input)!r}")
        if input.device != self.device:
            raise ValueError(f"input must be on {self.device}, got {input.device}")
        if not input.is_contiguous():
            raise ValueError("input must be contiguous")
        input_bytes = input.numel() * input.element_size()
        if input_bytes > self.max_size_bytes:
            raise ValueError(
                f"input requires {input_bytes} bytes but max_size_bytes={self.max_size_bytes}"
            )

    def _validate_reduction_input(self, input: torch.Tensor) -> None:
        self._validate_tensor(input)
        if input.dtype not in _REDUCTION_DTYPES:
            raise TypeError(
                "deterministic reductions support float32, float16, and bfloat16; "
                f"got {input.dtype}"
            )

    def _validate_gather_input(self, input: torch.Tensor) -> None:
        self._validate_tensor(input)
        if input.dim() == 0:
            raise ValueError("all_gather input must have at least one dimension")

    @staticmethod
    def _validate_output(
        output: torch.Tensor,
        input: torch.Tensor,
        output_shape: tuple[int, ...],
    ) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"out must be a torch.Tensor, got {type(output)!r}")
        if output.device != input.device:
            raise ValueError("out must be on the same device as input")
        if output.dtype != input.dtype:
            raise TypeError("out must have the same dtype as input")
        if tuple(output.shape) != output_shape:
            raise ValueError(f"out must have shape {output_shape}, got {tuple(output.shape)}")
        if not output.is_contiguous():
            raise ValueError("out must be contiguous")

    def _validate_matching_signature(self, op_name: str, input: torch.Tensor) -> None:
        if self.world_size == 1:
            return
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

    def _validate_matching_capacity(self) -> None:
        if self.world_size == 1:
            return
        capacities: list[int | None] = [None] * self.world_size
        dist.all_gather_object(capacities, self.max_size_bytes, group=self.group)
        if any(peer_capacity != self.max_size_bytes for peer_capacity in capacities):
            raise ValueError(f"all ranks must use the same max_size_bytes; got {capacities}")

    def _all_gather_transport(
        self,
        input: torch.Tensor,
        *,
        gathered_flat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Flattening makes the output contract independent of whether a given
        # ProcessGroup implements the concatenation or stacking form of AG.
        if self.world_size == 1:
            if gathered_flat is None:
                gathered_flat = input.clone().view(-1)
            else:
                gathered_flat.copy_(input.view(-1))
            return gathered_flat.reshape((1, *input.shape))
        input_flat = input.view(-1)
        required_elements = self.world_size * input_flat.numel()
        if gathered_flat is None:
            gathered_flat = self._workspace_for(input, required_elements)
        elif (
            gathered_flat.numel() != required_elements
            or gathered_flat.dtype != input.dtype
            or gathered_flat.device != input.device
            or not gathered_flat.is_contiguous()
        ):
            raise ValueError("gathered transport output has an invalid layout")
        if "nccl" in self._backend:
            # PyTorch exposes RCCL through the NCCL ProcessGroup API. Keep this
            # as a tensor-only transport; reduction happens below.
            dist.all_gather_into_tensor(gathered_flat, input_flat, group=self.group)
        else:
            # Some reference backends (notably Gloo versions without
            # all_gather_into_tensor) only implement the list API.
            gathered_chunks = list(
                gathered_flat.reshape(self.world_size, input_flat.numel()).unbind(0)
            )
            dist.all_gather(gathered_chunks, input_flat, group=self.group)
        return gathered_flat.reshape((self.world_size, *input.shape))

    def _workspace_for(self, input: torch.Tensor, required_elements: int) -> torch.Tensor:
        required_bytes = required_elements * input.element_size()
        workspace = self._workspace
        if workspace is None or workspace.numel() < required_bytes:
            workspace = torch.empty(required_bytes, dtype=torch.uint8, device=self.device)
            self._workspace = workspace
        # Tensor.view(dtype) reinterprets the aligned byte allocation without
        # an allocation or copy. Restrict the view to the current operation.
        return workspace[:required_bytes].view(input.dtype)

    def _direct_all_reduce(self, input: torch.Tensor, output: torch.Tensor) -> bool:
        return False

    def _direct_reduce_scatter(self, input: torch.Tensor, output: torch.Tensor) -> bool:
        return False

    def _can_direct_reduce_scatter_many(self) -> bool:
        return False

    def _direct_reduce_scatter_many(
        self,
        inputs: tuple[torch.Tensor, ...],
        outputs: tuple[torch.Tensor, ...],
    ) -> bool:
        return False

    def _direct_all_gather(self, input: torch.Tensor, output: torch.Tensor) -> bool:
        return False

    @staticmethod
    def _balanced_tree_sum(rank_inputs: torch.Tensor) -> torch.Tensor:
        world_size = rank_inputs.size(0)
        if world_size not in _SUPPORTED_WORLD_SIZES:
            raise ValueError(
                "balanced reduction requires rank inputs for world_size in "
                f"{_SUPPORTED_WORLD_SIZES}, got {world_size}"
            )
        # ``rank_inputs`` is the private transport workspace for reductions,
        # so fixed-tree nodes can be accumulated in place. This preserves the
        # exact pairings while avoiding one temporary allocation per tree node.
        stride = 1
        while stride < world_size:
            for index in range(0, world_size, 2 * stride):
                rank_inputs[index].add_(rank_inputs[index + stride])
            stride *= 2
        return rank_inputs[0]

    @staticmethod
    def _fused_reduction(
        rank_inputs: torch.Tensor,
        output: torch.Tensor,
        *,
        operation: str,
    ) -> bool:
        """Use the optional ROCm fused fixed-tree kernel when available.

        The extension is deliberately optional: CPU/Gloo reference collectives
        and installations built without the ROCm kernel retain the executable
        Python implementation above.
        """

        if getattr(torch.version, "hip", None) is None or not rank_inputs.is_cuda:
            return False
        try:
            from rl_engine import _C
        except ImportError:
            return False
        if operation == "all_reduce":
            fn = getattr(_C, "deterministic_collective_rocm_all_reduce", None)
            if fn is not None:
                fn(rank_inputs, output)
                return True
        elif operation == "reduce_scatter":
            fn = getattr(_C, "deterministic_collective_rocm_reduce_scatter", None)
            if fn is not None:
                fn(rank_inputs, output)
                return True
        return False


class RCCLDeterministicCollective(TorchDistributedDeterministicCollective):
    """Single-node ROCm fixed-tree collective using HIP IPC and RCCL."""

    backend_id = "rocm_ipc_fixed_tree"

    def __init__(
        self,
        group: dist.ProcessGroup | None = None,
        device: torch.device | str | int | None = None,
        *,
        max_size_bytes: int = _DEFAULT_MAX_SIZE_BYTES,
    ) -> None:
        if getattr(torch.version, "hip", None) is None:
            raise RuntimeError("RCCL deterministic collectives require a ROCm PyTorch build")
        if not torch.cuda.is_available():
            raise RuntimeError("RCCL deterministic collectives require an available ROCm device")
        if device is not None and not isinstance(device, int):
            requested_device = torch.device(device)
            if requested_device.type != "cuda":
                raise ValueError(
                    f"RCCL deterministic collectives require a ROCm device, got {device!r}"
                )
        super().__init__(group=group, device=device, max_size_bytes=max_size_bytes)
        if self.device.type != "cuda":
            raise ValueError(
                f"RCCL deterministic collectives require a ROCm device, got {device!r}"
            )
        if "nccl" not in self._backend:
            raise RuntimeError(
                "RCCL deterministic collectives require PyTorch's NCCL process-group API"
            )
        self._ipc_handle = 0
        self._ipc_staging: torch.Tensor | None = None
        self._direct_staging_views: dict[tuple[tuple[int, ...], torch.dtype], torch.Tensor] = {}
        self._initialize_ipc_transport()

    @property
    def workspace_size_bytes(self) -> int:
        staging = self._ipc_staging
        staging_bytes = 0 if staging is None else int(staging.numel())
        return staging_bytes + super().workspace_size_bytes

    def _initialize_ipc_transport(self) -> None:
        if self.world_size == 1:
            return
        try:
            from rl_engine import _C
        except ImportError:
            return
        required_symbols = (
            "deterministic_collective_rocm_ipc_allocate",
            "deterministic_collective_rocm_ipc_meta",
            "deterministic_collective_rocm_ipc_create",
            "deterministic_collective_rocm_ipc_synchronize",
            "deterministic_collective_rocm_ipc_destroy",
            "deterministic_collective_rocm_ipc_stage",
            "deterministic_collective_rocm_ipc_prepare_staged",
            "deterministic_collective_rocm_ipc_all_reduce_staged",
            "deterministic_collective_rocm_ipc_all_reduce",
            "deterministic_collective_rocm_ipc_all_reduce_input",
            "deterministic_collective_rocm_ipc_reduce_scatter",
            "deterministic_collective_rocm_ipc_reduce_scatter_input",
            "deterministic_collective_rocm_ipc_reduce_scatter_many",
            "deterministic_collective_rocm_ipc_all_gather",
            "deterministic_collective_rocm_ipc_all_gather_input",
        )
        if any(not hasattr(_C, symbol) for symbol in required_symbols):
            return

        staging = _C.deterministic_collective_rocm_ipc_allocate(self.max_size_bytes)
        handle, offset = _C.deterministic_collective_rocm_ipc_meta(staging)
        local_metadata = (socket.gethostname(), handle, int(offset))
        gathered_metadata: list[tuple[str, list[int], int] | None] = [None] * self.world_size
        dist.all_gather_object(gathered_metadata, local_metadata, group=self.group)
        if any(metadata is None for metadata in gathered_metadata):
            raise RuntimeError("failed to exchange ROCm IPC metadata")
        complete_metadata = [metadata for metadata in gathered_metadata if metadata is not None]
        if len({metadata[0] for metadata in complete_metadata}) != 1:
            return
        self._ipc_handle = int(
            _C.deterministic_collective_rocm_ipc_create(
                staging,
                [metadata[1] for metadata in complete_metadata],
                [metadata[2] for metadata in complete_metadata],
                self.rank,
            )
        )
        self._ipc_staging = staging
        self._handle = self._ipc_handle

    def prepare_direct_staging_views(
        self,
        shapes: Iterable[tuple[int, ...]],
        *,
        dtype: torch.dtype,
    ) -> None:
        staging = self._ipc_staging
        if not self._ipc_handle or staging is None:
            return
        if dtype not in _REDUCTION_DTYPES:
            raise TypeError(f"unsupported direct-staging dtype {dtype}")
        element_size = torch.empty((), dtype=dtype).element_size()
        for raw_shape in shapes:
            shape = tuple(int(dim) for dim in raw_shape)
            numel = 1
            for dim in shape:
                if dim < 0:
                    raise ValueError(f"direct-staging dimensions must be non-negative, got {shape}")
                numel *= dim
            size_bytes = numel * element_size
            if size_bytes > min(
                self.max_size_bytes,
                _ROCM_IPC_DIRECT_ALL_REDUCE_MAX_BYTES,
            ):
                continue
            payload = staging.narrow(
                0,
                _ROCM_IPC_CONTROL_BYTES,
                size_bytes,
            )
            self._direct_staging_views[(shape, dtype)] = payload.view(dtype).view(shape)

    def direct_staging_view(
        self,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        return self._direct_staging_views.get((tuple(int(dim) for dim in shape), dtype))

    def _direct_all_reduce(self, input: torch.Tensor, output: torch.Tensor) -> bool:
        handle = self._ipc_handle
        if not handle:
            return False
        from rl_engine import _C

        input_bytes = input.numel() * input.element_size()
        if (
            _ROCM_IPC_DIRECT_ALL_REDUCE_MAX_BYTES
            < input_bytes
            < _ROCM_IPC_SHARDED_ALL_REDUCE_MIN_BYTES
            and input.numel() % self.world_size == 0
        ):
            return False

        if (
            input_bytes <= _ROCM_IPC_DIRECT_ALL_REDUCE_MAX_BYTES
            or input.numel() % self.world_size != 0
        ):
            _C.deterministic_collective_rocm_ipc_all_reduce_input(
                handle,
                input,
                output,
            )
            return True

        shard = self._workspace_for(input, input.numel() // self.world_size)
        _C.deterministic_collective_rocm_ipc_reduce_scatter_input(
            handle,
            input,
            shard,
        )
        dist.all_gather_into_tensor(output.view(-1), shard, group=self.group)
        return True

    def _direct_reduce_scatter(self, input: torch.Tensor, output: torch.Tensor) -> bool:
        handle = self._ipc_handle
        if not handle:
            return False
        from rl_engine import _C

        _C.deterministic_collective_rocm_ipc_reduce_scatter_input(
            handle,
            input,
            output,
        )
        return True

    def _can_direct_reduce_scatter_many(self) -> bool:
        return bool(self._ipc_handle)

    def _direct_reduce_scatter_many(
        self,
        inputs: tuple[torch.Tensor, ...],
        outputs: tuple[torch.Tensor, ...],
    ) -> bool:
        handle = self._ipc_handle
        if not handle:
            return False
        from rl_engine import _C

        _C.deterministic_collective_rocm_ipc_reduce_scatter_many(
            handle,
            inputs,
            outputs,
        )
        return True

    def _direct_all_gather(self, input: torch.Tensor, output: torch.Tensor) -> bool:
        handle = self._ipc_handle
        input_bytes = input.numel() * input.element_size()
        if not handle or input_bytes > _ROCM_IPC_ALL_GATHER_MAX_BYTES:
            return False
        from rl_engine import _C

        _C.deterministic_collective_rocm_ipc_all_gather_input(
            handle,
            input,
            output,
        )
        return True

    def close(self) -> None:
        handle = getattr(self, "_ipc_handle", 0)
        if handle:
            from rl_engine import _C

            _C.deterministic_collective_rocm_ipc_synchronize(handle)
            torch.cuda.synchronize(self.device)
            self._ipc_handle = 0
            self._handle = 0
            _C.deterministic_collective_rocm_ipc_destroy(handle)
            self._ipc_staging = None
            self._direct_staging_views.clear()
        super().close()
