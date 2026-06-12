# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import hashlib
import inspect
import os
import socket
import sys
import time
import uuid
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from typing import Any, Mapping, Optional, Protocol, cast

import torch

from rl_engine.utils.logger import logger

_BRIDGE_METADATA_KEY = "weight_bridge"
_SHARED_MEMORY_FORMAT = "python-multiprocessing-shared-memory-v1"
_CUDA_IPC_FORMAT = "pytorch-cuda-ipc-reduce-tensor-v1"
_CUDA_VMM_FORMAT = "cuda-vmm-posix-fd-v1"
_CUDA_VMM_TENSOR_ALIGNMENT = 256
_SHARED_MEMORY_HAS_TRACK = "track" in inspect.signature(shared_memory.SharedMemory).parameters
_SUPPORTED_LAYOUT_KINDS = {"full-state", "replicated"}


@dataclass(frozen=True)
class WeightLayout:
    """
    Describes how model state is represented in a published update.

    Issue #13 supports complete full-state or replicated layouts. ZeRO-3 is
    allowed only after the training worker exports a gathered full-state view;
    tensor-parallel shards and multi-node/RDMA transfers still pass through an
    explicit blocker until layout-aware transports are implemented and tested.
    """

    kind: str = "full-state"
    world_size: int = 1
    rank: int = 0
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    zero_stage: int = 0
    node_count: int = 1
    rdma_enabled: bool = False

    @classmethod
    def from_metadata(cls, metadata: Optional[Mapping[str, Any]]) -> WeightLayout:
        if metadata is None:
            return cls()
        raw_layout = metadata.get("layout", {})
        if raw_layout is None:
            return cls()
        if not isinstance(raw_layout, Mapping):
            raise WeightManifestValidationError("weight layout metadata must be a mapping")
        return cls(
            kind=str(raw_layout.get("kind", "full-state")),
            world_size=int(raw_layout.get("world_size", 1)),
            rank=int(raw_layout.get("rank", 0)),
            tensor_parallel_size=int(raw_layout.get("tensor_parallel_size", 1)),
            data_parallel_size=int(raw_layout.get("data_parallel_size", 1)),
            zero_stage=int(raw_layout.get("zero_stage", 0)),
            node_count=int(raw_layout.get("node_count", 1)),
            rdma_enabled=bool(raw_layout.get("rdma_enabled", False)),
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "world_size": self.world_size,
            "rank": self.rank,
            "tensor_parallel_size": self.tensor_parallel_size,
            "data_parallel_size": self.data_parallel_size,
            "zero_stage": self.zero_stage,
            "node_count": self.node_count,
            "rdma_enabled": self.rdma_enabled,
        }

    def validate_supported(self) -> None:
        if self.kind not in _SUPPORTED_LAYOUT_KINDS:
            raise WeightBridgeUnavailableError(
                f"weight layout {self.kind!r} is not supported by this bridge. "
                "Publish a gathered full-state update or implement a layout-aware "
                "transport before exposing it to rollout workers."
            )
        if self.tensor_parallel_size != 1:
            raise WeightBridgeUnavailableError(
                "tensor-parallel weight layouts are not supported by this bridge. "
                "Publish a complete per-consumer layout or add a tensor-parallel adapter."
            )
        if self.node_count != 1 or self.rdma_enabled:
            raise WeightBridgeUnavailableError(
                "multi-node/RDMA weight transport is not implemented. Use same-node "
                "local/shared-memory transport or add an RDMA/NCCL transport first."
            )
        if self.world_size < 1 or self.rank < 0 or self.rank >= self.world_size:
            raise WeightManifestValidationError(
                f"invalid weight layout rank/world_size: rank={self.rank}, "
                f"world_size={self.world_size}"
            )
        if self.data_parallel_size < 1:
            raise WeightManifestValidationError("data_parallel_size must be >= 1")


class VLLMWeightInstallAdapter:
    """
    Guarded adapter for installing imported tensors into a rollout engine.

    vLLM does not expose one stable public hot-weight update API across versions,
    so the adapter accepts explicit callable capability hooks instead of poking
    private internals. That lets production integrations opt in when their vLLM
    runtime provides a verified install method, while default construction stays
    explicit and safe.
    """

    def __init__(
        self,
        engine: Any,
        *,
        install_callable: Optional[Any] = None,
        request_builder: Optional[Any] = None,
        release_callable: Optional[Any] = None,
    ):
        self.engine = engine
        self._install_callable = install_callable
        self._request_builder = request_builder
        self._release_callable = release_callable
        self.active_weight_version: Optional[int] = None
        self.active_update_id: Optional[str] = None

    def install(
        self,
        manifest: WeightUpdateManifest,
        tensors: Mapping[str, torch.Tensor],
    ) -> None:
        tensors = dict(tensors)
        if self._install_callable is not None:
            self._install_callable(manifest, tensors)
        else:
            update_weights = getattr(self.engine, "update_weights", None)
            if callable(update_weights):
                if self._request_builder is None:
                    raise WeightBridgeUnavailableError(
                        "vLLM weight install requires a request_builder for engines "
                        "that expose update_weights(request). vLLM does not accept a "
                        "raw manifest/tensor mapping for this public API."
                    )
                try:
                    update_weights(self._request_builder(manifest, tensors))
                except Exception:
                    request_builder_release = getattr(self._request_builder, "release", None)
                    if callable(request_builder_release):
                        request_builder_release(manifest.update_id)
                    raise
            else:
                install = self._resolve_manifest_install_callable()
                install(manifest, tensors)
        self.active_weight_version = manifest.weight_version
        self.active_update_id = manifest.update_id

    def release(self, update_id: str) -> None:
        if self._release_callable is not None:
            self._release_callable(update_id)
        request_builder_release = getattr(self._request_builder, "release", None)
        if callable(request_builder_release):
            request_builder_release(update_id)
        if self.active_update_id == update_id:
            self.active_update_id = None

    def _resolve_manifest_install_callable(self) -> Any:
        for attr in (
            "install_weight_update",
            "update_weights_from_manifest",
            "load_weights_from_manifest",
        ):
            candidate = getattr(self.engine, attr, None)
            if callable(candidate):
                return candidate
        raise WeightBridgeUnavailableError(
            "vLLM weight install is unavailable because this engine does not expose "
            "a verified hot-weight install capability. Pass an explicit "
            "install_callable for the installed vLLM version."
        )


class VLLMIPCWeightUpdateRequestBuilder:
    """
    Build the public vLLM IPC weight-update request shape.

    vLLM 0.18+ expects `LLM.update_weights({"update_info": ...})` for IPC
    backends. The update info carries CUDA IPC handles produced by PyTorch's
    `reduce_tensor`, so the source CUDA tensors must remain alive until the
    vLLM update call completes. This builder keeps those contiguous tensors in
    an update-scoped keepalive registry and releases them when the install
    adapter releases the update.
    """

    def __init__(
        self,
        *,
        is_checkpoint_format: bool = True,
        reduce_tensor_fn: Optional[Any] = None,
        gpu_uuid: Optional[str] = None,
    ):
        self.is_checkpoint_format = bool(is_checkpoint_format)
        self._reduce_tensor_fn = reduce_tensor_fn
        self._gpu_uuid = gpu_uuid
        self._keepalive: dict[str, list[torch.Tensor]] = {}

    def __call__(
        self,
        manifest: WeightUpdateManifest,
        tensors: Mapping[str, torch.Tensor],
    ) -> dict[str, Any]:
        WeightLayout.from_metadata(manifest.metadata).validate_supported()
        tensor_map = dict(tensors)
        if set(tensor_map) != set(manifest.tensors):
            missing = sorted(set(manifest.tensors) - set(tensor_map))
            extra = sorted(set(tensor_map) - set(manifest.tensors))
            raise WeightManifestValidationError(
                f"vLLM IPC tensor set mismatch: missing={missing}, extra={extra}"
            )

        manifest_ipc_handles = self._manifest_ipc_handles(manifest)
        gpu_uuid = self._gpu_uuid or self._manifest_gpu_uuid(manifest) or self._current_gpu_uuid()
        names: list[str] = []
        dtype_names: list[str] = []
        shapes: list[list[int]] = []
        ipc_handles: list[dict[str, Any]] = []
        keepalive: list[torch.Tensor] = []

        for name, descriptor in manifest.tensors.items():
            tensor = tensor_map[name]
            if getattr(tensor.device, "type", None) != "cuda":
                raise WeightBridgeUnavailableError(
                    "vLLM IPC weight update requires CUDA tensors. "
                    f"Tensor {name} is on {tensor.device}."
                )
            if tuple(int(dim) for dim in tensor.shape) != descriptor.shape:
                raise WeightManifestValidationError(
                    f"vLLM IPC tensor shape mismatch for {name}: "
                    f"expected {descriptor.shape}, got {tuple(tensor.shape)}"
                )
            if str(tensor.dtype) != descriptor.dtype:
                raise WeightManifestValidationError(
                    f"vLLM IPC tensor dtype mismatch for {name}: "
                    f"expected {descriptor.dtype}, got {tensor.dtype}"
                )

            weight = tensor.detach().contiguous()
            keepalive.append(weight)
            names.append(name)
            dtype_names.append(str(weight.dtype).split(".")[-1])
            shapes.append([int(dim) for dim in weight.shape])
            if manifest_ipc_handles is not None:
                handle = manifest_ipc_handles[name]
            else:
                handle = self._resolve_reduce_tensor()(weight)
            ipc_handles.append({gpu_uuid: handle})

        self._keepalive[manifest.update_id] = keepalive
        return {
            "update_info": {
                "names": names,
                "dtype_names": dtype_names,
                "shapes": shapes,
                "ipc_handles": ipc_handles,
                "is_checkpoint_format": self.is_checkpoint_format,
            }
        }

    def release(self, update_id: str) -> None:
        self._keepalive.pop(update_id, None)

    def _manifest_ipc_handles(self, manifest: WeightUpdateManifest) -> Optional[dict[str, Any]]:
        if manifest.transport != "cuda-ipc":
            return None
        bridge_metadata = manifest.metadata.get(_BRIDGE_METADATA_KEY)
        if not isinstance(bridge_metadata, Mapping):
            return None
        if bridge_metadata.get("format") != _CUDA_IPC_FORMAT:
            return None
        entries = bridge_metadata.get("tensors")
        if not isinstance(entries, Mapping) or set(entries) != set(manifest.tensors):
            raise WeightManifestValidationError("vLLM IPC manifest handle mismatch")
        handles: dict[str, Any] = {}
        for name, entry in entries.items():
            if not isinstance(entry, Mapping) or "handle" not in entry:
                raise WeightManifestValidationError(
                    f"vLLM IPC manifest is missing handle for tensor {name}"
                )
            handles[str(name)] = entry["handle"]
        return handles

    def _manifest_gpu_uuid(self, manifest: WeightUpdateManifest) -> Optional[str]:
        bridge_metadata = manifest.metadata.get(_BRIDGE_METADATA_KEY)
        if not isinstance(bridge_metadata, Mapping):
            return None
        gpu_uuid = bridge_metadata.get("gpu_uuid")
        return str(gpu_uuid) if gpu_uuid else None

    def _resolve_reduce_tensor(self) -> Any:
        if self._reduce_tensor_fn is not None:
            return self._reduce_tensor_fn
        try:
            from torch.multiprocessing.reductions import reduce_tensor
        except ImportError as exc:
            raise WeightBridgeUnavailableError(
                "PyTorch CUDA IPC reductions are unavailable in this runtime."
            ) from exc
        return reduce_tensor

    def _current_gpu_uuid(self) -> str:
        if not torch.cuda.is_available():
            raise WeightBridgeUnavailableError(
                "vLLM IPC weight update requires CUDA, but torch.cuda.is_available() is false."
            )
        device_index = int(torch.cuda.current_device())
        return _resolve_cuda_device_uuid(device_index)


class VLLMInProcessWeightReloadAdapter:
    """
    Install a manifest through vLLM's in-process `reload_weights` utility path.

    This adapter is for single-process vLLM deployments, for example vLLM V1
    with `VLLM_ENABLE_V1_MULTIPROCESSING=0`. It is a real hot-weight install
    path, but it is not CUDA IPC zero-copy: tensors are passed directly to the
    in-process worker and vLLM performs the model reload/copy into GPU weights.
    Multiprocess vLLM should use the IPC or NCCL public `update_weights` APIs
    once those transports are validated on the target hardware.
    """

    def __init__(
        self,
        engine: Any,
        *,
        target_dtype: Optional[torch.dtype] = None,
        target_device: Optional[torch.device | str] = None,
        is_checkpoint_format: bool = True,
        synchronize_cuda: bool = True,
    ):
        self.engine = engine
        self.target_dtype = target_dtype
        self.target_device = target_device
        self.is_checkpoint_format = bool(is_checkpoint_format)
        self.synchronize_cuda = bool(synchronize_cuda)
        self.active_weight_version: Optional[int] = None
        self.active_update_id: Optional[str] = None

    def install(
        self,
        manifest: WeightUpdateManifest,
        tensors: Mapping[str, torch.Tensor],
    ) -> None:
        WeightLayout.from_metadata(manifest.metadata).validate_supported()
        tensor_map = dict(tensors)
        if set(tensor_map) != set(manifest.tensors):
            missing = sorted(set(manifest.tensors) - set(tensor_map))
            extra = sorted(set(tensor_map) - set(manifest.tensors))
            raise WeightManifestValidationError(
                f"vLLM reload tensor set mismatch: missing={missing}, extra={extra}"
            )

        weights: list[tuple[str, torch.Tensor]] = []
        for name, descriptor in manifest.tensors.items():
            tensor = tensor_map[name]
            if tuple(int(dim) for dim in tensor.shape) != descriptor.shape:
                raise WeightManifestValidationError(
                    f"vLLM reload tensor shape mismatch for {name}: "
                    f"expected {descriptor.shape}, got {tuple(tensor.shape)}"
                )
            if str(tensor.dtype) != descriptor.dtype:
                raise WeightManifestValidationError(
                    f"vLLM reload tensor dtype mismatch for {name}: "
                    f"expected {descriptor.dtype}, got {tensor.dtype}"
                )

            weight = tensor.detach()
            if self.target_dtype is not None and weight.dtype != self.target_dtype:
                weight = weight.to(dtype=self.target_dtype)
            if self.target_device is not None and torch.device(self.target_device) != weight.device:
                weight = weight.to(device=self.target_device)
            if not weight.is_contiguous():
                weight = weight.contiguous()
            weights.append((name, weight))

        try:
            reload_weights = self._resolve_reload_weights()
            reload_weights(weights)
            if self.synchronize_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            self.active_update_id = None
            raise

        self.active_weight_version = manifest.weight_version
        self.active_update_id = manifest.update_id

    def release(self, update_id: str) -> None:
        if self.active_update_id == update_id:
            self.active_update_id = None

    def _resolve_reload_weights(self) -> Any:
        reload_weights = getattr(self.engine, "reload_weights", None)
        if callable(reload_weights):
            return lambda weights: reload_weights(
                weights_iterator=weights,
                is_checkpoint_format=self.is_checkpoint_format,
            )

        collective_rpc = getattr(self.engine, "collective_rpc", None)
        if not callable(collective_rpc):
            llm_engine = getattr(self.engine, "llm_engine", None)
            collective_rpc = getattr(llm_engine, "collective_rpc", None)
        if callable(collective_rpc):
            return lambda weights: collective_rpc(
                "reload_weights",
                kwargs={
                    "weights_iterator": weights,
                    "is_checkpoint_format": self.is_checkpoint_format,
                },
            )

        raise WeightBridgeUnavailableError(
            "vLLM in-process weight reload is unavailable because this engine "
            "does not expose reload_weights(...) or llm_engine.collective_rpc(...)."
        )


class VLLMCheckpointWeightReloadAdapter:
    """
    Install a manifest through vLLM's checkpoint-path reload utility path.

    This path works with vLLM's default EngineCore multiprocessing because the
    worker process reloads weights from `weights_path` itself; no CUDA tensor is
    serialized across the RPC boundary. It is a production-aligned hot reload
    fallback for environments where CUDA IPC/NCCL transport is unavailable, but
    it is not a zero-copy transport.
    """

    def __init__(
        self,
        engine: Any,
        *,
        weights_path: Optional[str] = None,
        weights_path_resolver: Optional[Any] = None,
        metadata_key: str = "vllm_weights_path",
        is_checkpoint_format: bool = True,
        synchronize_cuda: bool = True,
    ):
        self.engine = engine
        self.weights_path = weights_path
        self.weights_path_resolver = weights_path_resolver
        self.metadata_key = metadata_key
        self.is_checkpoint_format = bool(is_checkpoint_format)
        self.synchronize_cuda = bool(synchronize_cuda)
        self.active_weight_version: Optional[int] = None
        self.active_update_id: Optional[str] = None

    def install(
        self,
        manifest: WeightUpdateManifest,
        tensors: Mapping[str, torch.Tensor],
    ) -> None:
        WeightLayout.from_metadata(manifest.metadata).validate_supported()
        tensor_map = dict(tensors)
        if set(tensor_map) != set(manifest.tensors):
            missing = sorted(set(manifest.tensors) - set(tensor_map))
            extra = sorted(set(tensor_map) - set(manifest.tensors))
            raise WeightManifestValidationError(
                f"vLLM checkpoint reload tensor set mismatch: missing={missing}, extra={extra}"
            )

        weights_path = self._resolve_weights_path(manifest, tensor_map)
        try:
            reload_weights = self._resolve_reload_weights()
            reload_weights(weights_path)
            if self.synchronize_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            self.active_update_id = None
            raise

        self.active_weight_version = manifest.weight_version
        self.active_update_id = manifest.update_id

    def release(self, update_id: str) -> None:
        if self.active_update_id == update_id:
            self.active_update_id = None

    def _resolve_weights_path(
        self,
        manifest: WeightUpdateManifest,
        tensors: Mapping[str, torch.Tensor],
    ) -> str:
        if self.weights_path:
            return self.weights_path
        if self.weights_path_resolver is not None:
            resolved = self.weights_path_resolver(manifest, tensors)
            if resolved:
                return str(resolved)
        resolved = manifest.metadata.get(self.metadata_key)
        if resolved:
            return str(resolved)
        raise WeightBridgeUnavailableError(
            "vLLM checkpoint reload requires a weights_path, a weights_path_resolver, "
            f"or manifest.metadata[{self.metadata_key!r}]."
        )

    def _resolve_reload_weights(self) -> Any:
        reload_weights = getattr(self.engine, "reload_weights", None)
        if callable(reload_weights):
            return lambda weights_path: reload_weights(
                weights_path=weights_path,
                is_checkpoint_format=self.is_checkpoint_format,
            )

        collective_rpc = getattr(self.engine, "collective_rpc", None)
        if not callable(collective_rpc):
            llm_engine = getattr(self.engine, "llm_engine", None)
            collective_rpc = getattr(llm_engine, "collective_rpc", None)
        if callable(collective_rpc):
            return lambda weights_path: collective_rpc(
                "reload_weights",
                kwargs={
                    "weights_path": weights_path,
                    "is_checkpoint_format": self.is_checkpoint_format,
                },
            )

        raise WeightBridgeUnavailableError(
            "vLLM checkpoint weight reload is unavailable because this engine "
            "does not expose reload_weights(...) or llm_engine.collective_rpc(...)."
        )


def _install_vllm_cuda_vmm_aliases_on_worker(
    manifest: WeightUpdateManifest,
    *,
    device_index: int,
    source_worker: str,
    source_rank: int,
) -> Callable[[torch.nn.Module], dict[str, Any]]:
    def install(model: torch.nn.Module) -> dict[str, Any]:
        model_handle = cast(Any, model)
        bridge = CUDAVMMTensorBridge(
            source_worker=source_worker,
            source_rank=source_rank,
            device_index=device_index,
        )
        imported = dict(bridge.import_update(manifest))
        rebound: list[str] = []
        original_data = dict(getattr(model_handle, "_kernel_align_cuda_vmm_original_data", {}))
        try:
            named_parameters = dict(model.named_parameters())
            with torch.no_grad():
                for name, tensor in imported.items():
                    parameter = named_parameters.get(name)
                    if parameter is None:
                        continue
                    if tuple(parameter.shape) != tuple(tensor.shape):
                        raise WeightManifestValidationError(
                            f"vLLM CUDA VMM parameter shape mismatch for {name}: "
                            f"expected {tuple(parameter.shape)}, got {tuple(tensor.shape)}"
                        )
                    if parameter.dtype != tensor.dtype:
                        raise WeightManifestValidationError(
                            f"vLLM CUDA VMM parameter dtype mismatch for {name}: "
                            f"expected {parameter.dtype}, got {tensor.dtype}"
                        )
                    original_data.setdefault(name, parameter.data)
                    parameter.data = tensor
                    rebound.append(name)
            bridge.acknowledge(manifest.update_id)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_handle._kernel_align_cuda_vmm_bridge = bridge
            model_handle._kernel_align_cuda_vmm_update_id = manifest.update_id
            model_handle._kernel_align_cuda_vmm_tensors = imported
            model_handle._kernel_align_cuda_vmm_original_data = original_data
            return {
                "update_id": manifest.update_id,
                "weight_version": manifest.weight_version,
                "rebound": rebound,
                "tensor_count": len(imported),
                "zero_copy": True,
            }
        except Exception:
            named_parameters = dict(model.named_parameters())
            with torch.no_grad():
                for name in rebound:
                    parameter = named_parameters.get(name)
                    original = original_data.get(name)
                    if parameter is not None and original is not None:
                        parameter.data = original
            for attr in (
                "_kernel_align_cuda_vmm_bridge",
                "_kernel_align_cuda_vmm_update_id",
                "_kernel_align_cuda_vmm_tensors",
                "_kernel_align_cuda_vmm_original_data",
            ):
                if hasattr(model, attr):
                    delattr(model, attr)
            imported.clear()
            bridge.release(manifest.update_id)
            raise

    return install


class VLLMCUDAVMMExternalStorageAdapter:
    """
    Bind vLLM worker parameters to tensors imported from a CUDA VMM manifest.

    The vLLM worker imports the manifest itself through `apply_model`, then
    replaces matching `torch.nn.Parameter.data` storage with CUDA VMM DLPack
    aliases. This is a same-node zero-copy storage binding path and does not use
    legacy PyTorch CUDA IPC.
    """

    def __init__(
        self,
        engine: Any,
        *,
        device_index: int = 0,
        source_worker: str = "vllm-rollout",
        source_rank: int = 0,
        require_all_parameters: bool = False,
        synchronize_cuda: bool = True,
    ):
        self.engine = engine
        self.device_index = int(device_index)
        self.source_worker = source_worker
        self.source_rank = int(source_rank)
        self.require_all_parameters = bool(require_all_parameters)
        self.synchronize_cuda = bool(synchronize_cuda)
        self.active_weight_version: Optional[int] = None
        self.active_update_id: Optional[str] = None
        self.last_result: list[dict[str, Any]] = []

    def install(
        self,
        manifest: WeightUpdateManifest,
        tensors: Mapping[str, torch.Tensor],
    ) -> None:
        del tensors
        WeightLayout.from_metadata(manifest.metadata).validate_supported()
        if manifest.transport != CUDAVMMTensorBridge.transport:
            raise WeightBridgeUnavailableError(
                "vLLM CUDA VMM external-storage install requires a cuda-vmm manifest."
            )
        previous_update_id = self.active_update_id
        if previous_update_id is not None and previous_update_id != manifest.update_id:
            self.release(previous_update_id)
        apply_model = self._resolve_apply_model()
        install_fn = _install_vllm_cuda_vmm_aliases_on_worker(
            manifest,
            device_index=self.device_index,
            source_worker=self.source_worker,
            source_rank=self.source_rank,
        )
        results = apply_model(install_fn)
        if not isinstance(results, list):
            results = [results]
        normalized = [dict(result or {}) for result in results]
        rebound = set().union(*(set(result.get("rebound", [])) for result in normalized))
        missing = sorted(set(manifest.tensors) - rebound)
        if self.require_all_parameters and missing:
            self.release(manifest.update_id)
            raise WeightManifestValidationError(
                f"vLLM CUDA VMM install did not bind every manifest tensor: missing={missing}"
            )
        if self.synchronize_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.last_result = normalized
        self.active_weight_version = manifest.weight_version
        self.active_update_id = manifest.update_id

    def release(self, update_id: str) -> None:
        release_fn = self._resolve_release_model_fn(update_id)
        if release_fn is not None:
            try:
                release_fn()
            except Exception:
                logger.exception("Failed to release vLLM CUDA VMM update %s", update_id)
        if self.active_update_id == update_id:
            self.active_update_id = None

    def _resolve_apply_model(self) -> Callable[[Any], Any]:
        apply_model = getattr(self.engine, "apply_model", None)
        if callable(apply_model):
            return apply_model
        llm_engine = getattr(self.engine, "llm_engine", None)
        apply_model = getattr(llm_engine, "apply_model", None)
        if callable(apply_model):
            return apply_model
        collective_rpc = getattr(self.engine, "collective_rpc", None)
        if not callable(collective_rpc) and llm_engine is not None:
            collective_rpc = getattr(llm_engine, "collective_rpc", None)
        if callable(collective_rpc):
            return lambda fn: collective_rpc("apply_model", args=(fn,))
        raise WeightBridgeUnavailableError(
            "vLLM CUDA VMM external-storage install requires apply_model(...) "
            "or llm_engine.collective_rpc('apply_model', ...)."
        )

    def _resolve_release_model_fn(self, update_id: str) -> Optional[Callable[[], Any]]:
        try:
            apply_model = self._resolve_apply_model()
        except WeightBridgeUnavailableError:
            return None

        def release_worker(model: torch.nn.Module) -> dict[str, Any]:
            active_update_id = getattr(model, "_kernel_align_cuda_vmm_update_id", None)
            if active_update_id != update_id:
                return {"released": False, "update_id": active_update_id}
            bridge = getattr(model, "_kernel_align_cuda_vmm_bridge", None)
            tensors = getattr(model, "_kernel_align_cuda_vmm_tensors", None)
            original_data = getattr(model, "_kernel_align_cuda_vmm_original_data", None)
            if isinstance(original_data, dict):
                named_parameters = dict(model.named_parameters())
                with torch.no_grad():
                    for name, data in original_data.items():
                        parameter = named_parameters.get(name)
                        if parameter is not None:
                            parameter.data = data
            if isinstance(tensors, dict):
                tensors.clear()
            if bridge is not None:
                bridge.release(update_id)
            for attr in (
                "_kernel_align_cuda_vmm_bridge",
                "_kernel_align_cuda_vmm_update_id",
                "_kernel_align_cuda_vmm_tensors",
                "_kernel_align_cuda_vmm_original_data",
            ):
                if hasattr(model, attr):
                    delattr(model, attr)
            return {"released": True, "update_id": update_id}

        return lambda: apply_model(release_worker)


def _create_shared_memory(size: int) -> shared_memory.SharedMemory:
    return shared_memory.SharedMemory(create=True, size=size)


def _attach_shared_memory(name: str) -> shared_memory.SharedMemory:
    if _SHARED_MEMORY_HAS_TRACK:
        shared_memory_cls = cast(Any, shared_memory.SharedMemory)
        return shared_memory_cls(name=name, track=False)
    return shared_memory.SharedMemory(name=name)


def _dtype_from_name(name: str) -> torch.dtype:
    dtype_name = name.removeprefix("torch.")
    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise WeightManifestValidationError(f"unsupported tensor dtype in manifest: {name}")
    return dtype


def _required_storage_numel(
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    storage_offset: int,
) -> int:
    if any(dim < 0 for dim in shape):
        raise WeightManifestValidationError(f"invalid tensor shape: {shape}")
    if any(item < 0 for item in stride):
        raise WeightManifestValidationError(f"negative strides are not supported: {stride}")
    if len(shape) != len(stride):
        raise WeightManifestValidationError(
            f"shape and stride rank mismatch: shape={shape}, stride={stride}"
        )
    if any(dim == 0 for dim in shape):
        return 0
    if not shape:
        return storage_offset + 1
    max_offset = sum((dim - 1) * item for dim, item in zip(shape, stride, strict=False))
    return storage_offset + max_offset + 1


def _tensor_sha256(tensor: torch.Tensor) -> str:
    snapshot = tensor.detach()
    if snapshot.device.type != "cpu":
        snapshot = snapshot.cpu()
    if not snapshot.is_contiguous():
        snapshot = snapshot.contiguous()
    return hashlib.sha256(bytes(snapshot.untyped_storage())).hexdigest()


def _resolve_cuda_device_uuid(device_index: int) -> str:
    """
    Return a stable identifier for a CUDA device.

    torch.cuda.get_device_properties(...).uuid only exists on PyTorch >= 2.6.
    This project supports torch>=2.4.1, so fall back to pynvml and finally to a
    name:index identifier when the property is unavailable.
    """
    props = torch.cuda.get_device_properties(device_index)
    uuid_attr = getattr(props, "uuid", None)
    if uuid_attr is not None:
        return str(uuid_attr)

    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(device_index))
            raw = pynvml.nvmlDeviceGetUUID(handle)
            return raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw)
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        logger.debug(
            "pynvml GPU UUID lookup failed for device %s; " "falling back to name:index identifier",
            device_index,
            exc_info=True,
        )

    return f"{props.name}:{int(device_index)}"


def _send_fd(sock: socket.socket, fd: int) -> None:
    import array

    socket_module = cast(Any, socket)
    socket_handle = cast(Any, sock)
    fds = array.array("i", [int(fd)])
    socket_handle.sendmsg([b"F"], [(socket_module.SOL_SOCKET, socket_module.SCM_RIGHTS, fds)])


def _recv_fd(sock: socket.socket) -> int:
    import array

    socket_module = cast(Any, socket)
    socket_handle = cast(Any, sock)
    fds = array.array("i")
    message, ancdata, _flags, _address = socket_handle.recvmsg(
        1,
        socket_module.CMSG_LEN(fds.itemsize),
    )
    if message != b"F":
        raise WeightBridgeUnavailableError("CUDA VMM broker returned an invalid fd message")
    for level, control_type, data in ancdata:
        if level == socket_module.SOL_SOCKET and control_type == socket_module.SCM_RIGHTS:
            fds.frombytes(data[: fds.itemsize])
            return int(fds[0])
    raise WeightBridgeUnavailableError("CUDA VMM broker did not send a POSIX fd")


def _unix_stream_socket() -> socket.socket:
    socket_module = cast(Any, socket)
    return socket.socket(socket_module.AF_UNIX, socket.SOCK_STREAM)


def _dtype_to_dlpack(dtype: torch.dtype) -> tuple[int, int, int]:
    if dtype is torch.float16:
        return (2, 16, 1)
    if dtype is torch.float32:
        return (2, 32, 1)
    if dtype is torch.float64:
        return (2, 64, 1)
    if dtype is torch.bfloat16:
        return (4, 16, 1)
    if dtype is torch.int8:
        return (0, 8, 1)
    if dtype is torch.int16:
        return (0, 16, 1)
    if dtype is torch.int32:
        return (0, 32, 1)
    if dtype is torch.int64:
        return (0, 64, 1)
    if dtype is torch.uint8:
        return (1, 8, 1)
    if dtype is torch.bool:
        return (1, 1, 1)
    raise WeightBridgeUnavailableError(f"CUDA VMM DLPack export does not support {dtype}")


def _dlpack_ctypes():
    import ctypes

    if hasattr(_dlpack_ctypes, "_cache"):
        return _dlpack_ctypes._cache

    class _DLDevice(ctypes.Structure):
        _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]

    class _DLDataType(ctypes.Structure):
        _fields_ = [
            ("code", ctypes.c_uint8),
            ("bits", ctypes.c_uint8),
            ("lanes", ctypes.c_uint16),
        ]

    class _DLTensor(ctypes.Structure):
        _fields_ = [
            ("data", ctypes.c_void_p),
            ("device", _DLDevice),
            ("ndim", ctypes.c_int),
            ("dtype", _DLDataType),
            ("shape", ctypes.POINTER(ctypes.c_int64)),
            ("strides", ctypes.POINTER(ctypes.c_int64)),
            ("byte_offset", ctypes.c_uint64),
        ]

    class _DLManagedTensor(ctypes.Structure):
        pass

    deleter_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))

    @deleter_type
    def _noop_deleter(_ptr):
        return None

    _DLManagedTensor._fields_ = [
        ("dl_tensor", _DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", deleter_type),
    ]
    _dlpack_ctypes._cache = (
        ctypes,
        _DLDevice,
        _DLDataType,
        _DLTensor,
        _DLManagedTensor,
        _noop_deleter,
    )
    return _dlpack_ctypes._cache


class _DLPackOwner:
    def __init__(
        self,
        *,
        address: int,
        shape: tuple[int, ...],
        stride: tuple[int, ...],
        dtype: torch.dtype,
        device_index: int,
    ):
        ctypes, _DLDevice, _DLDataType, _DLTensor, _DLManagedTensor, _noop_deleter = (
            _dlpack_ctypes()
        )

        self._ctypes = ctypes
        self._shape = (ctypes.c_int64 * len(shape))(*shape)
        self._stride = (ctypes.c_int64 * len(stride))(*stride)
        code, bits, lanes = _dtype_to_dlpack(dtype)
        self._managed = _DLManagedTensor()
        self._managed.dl_tensor = _DLTensor(
            ctypes.c_void_p(address),
            _DLDevice(2, int(device_index)),
            len(shape),
            _DLDataType(code, bits, lanes),
            self._shape,
            self._stride,
            0,
        )
        self._managed.manager_ctx = None
        self._managed.deleter = _noop_deleter

    def to_tensor(self) -> torch.Tensor:
        ctypes = self._ctypes
        capsule_new = ctypes.pythonapi.PyCapsule_New
        capsule_new.restype = ctypes.py_object
        capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
        capsule = capsule_new(
            ctypes.cast(ctypes.pointer(self._managed), ctypes.c_void_p),
            b"dltensor",
            None,
        )
        return torch.utils.dlpack.from_dlpack(capsule)


class _CUDAVMMDriverBackend:
    def __init__(self, *, device_index: int = 0):
        import ctypes
        import ctypes.util

        self.ctypes = ctypes
        self.device_index = int(device_index)
        self.lib = ctypes.CDLL(self._find_libcuda(ctypes.util.find_library("cuda")))
        self._configure_signatures()
        self._ensure_context()

    @staticmethod
    def _find_libcuda(discovered_path: Optional[str]) -> str:
        candidates = [
            os.environ.get("KERNEL_ALIGN_LIBCUDA_PATH"),
            discovered_path,
            "libcuda.so.1",
            "libcuda.so",
        ]
        if sys.platform.startswith("linux"):
            candidates.extend(
                [
                    "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
                    "/usr/local/cuda/lib64/stubs/libcuda.so",
                    "/usr/lib/wsl/lib/libcuda.so.1",
                ]
            )

        errors: list[str] = []
        for candidate in candidates:
            if not candidate:
                continue
            try:
                import ctypes

                ctypes.CDLL(candidate)
                return candidate
            except OSError as exc:
                errors.append(f"{candidate}: {exc}")
        raise WeightBridgeUnavailableError(
            "CUDA VMM requires libcuda.so, but it could not be loaded. "
            "Set KERNEL_ALIGN_LIBCUDA_PATH to the CUDA driver library path. "
            f"Tried: {'; '.join(errors)}"
        )

    def _configure_signatures(self) -> None:
        c = self.ctypes
        for name, args in {
            "cuInit": [c.c_uint],
            "cuDeviceGet": [c.POINTER(c.c_int), c.c_int],
            "cuDeviceGetAttribute": [c.POINTER(c.c_int), c.c_int, c.c_int],
            "cuDevicePrimaryCtxRetain": [c.POINTER(c.c_void_p), c.c_int],
            "cuCtxSetCurrent": [c.c_void_p],
            "cuGetErrorName": [c.c_int, c.POINTER(c.c_char_p)],
            "cuGetErrorString": [c.c_int, c.POINTER(c.c_char_p)],
            "cuMemGetAllocationGranularity": [
                c.POINTER(c.c_size_t),
                c.POINTER(self.CUmemAllocationProp),
                c.c_int,
            ],
            "cuMemCreate": [
                c.POINTER(c.c_uint64),
                c.c_size_t,
                c.POINTER(self.CUmemAllocationProp),
                c.c_uint64,
            ],
            "cuMemRelease": [c.c_uint64],
            "cuMemAddressReserve": [
                c.POINTER(c.c_uint64),
                c.c_size_t,
                c.c_size_t,
                c.c_uint64,
                c.c_uint64,
            ],
            "cuMemAddressFree": [c.c_uint64, c.c_size_t],
            "cuMemMap": [c.c_uint64, c.c_size_t, c.c_size_t, c.c_uint64, c.c_uint64],
            "cuMemUnmap": [c.c_uint64, c.c_size_t],
            "cuMemSetAccess": [
                c.c_uint64,
                c.c_size_t,
                c.POINTER(self.CUmemAccessDesc),
                c.c_size_t,
            ],
            "cuMemExportToShareableHandle": [
                c.c_void_p,
                c.c_uint64,
                c.c_int,
                c.c_uint64,
            ],
            "cuMemImportFromShareableHandle": [c.POINTER(c.c_uint64), c.c_void_p, c.c_int],
            "cuMemcpyHtoD_v2": [c.c_uint64, c.c_void_p, c.c_size_t],
            "cuMemcpyDtoD_v2": [c.c_uint64, c.c_uint64, c.c_size_t],
            "cuMemcpyDtoDAsync_v2": [c.c_uint64, c.c_uint64, c.c_size_t, c.c_void_p],
        }.items():
            getattr(self.lib, name).argtypes = args

    @property
    def CUmemLocation(self):
        if hasattr(self, "_CUmemLocationType"):
            return self._CUmemLocationType
        c = self.ctypes

        class _CUmemLocation(c.Structure):
            _fields_ = [("type", c.c_int), ("id", c.c_int)]

        self._CUmemLocationType = _CUmemLocation
        return self._CUmemLocationType

    @property
    def CUmemAllocFlags(self):
        if hasattr(self, "_CUmemAllocFlagsType"):
            return self._CUmemAllocFlagsType
        c = self.ctypes

        class _CUmemAllocFlags(c.Structure):
            _fields_ = [
                ("compressionType", c.c_ubyte),
                ("gpuDirectRDMACapable", c.c_ubyte),
                ("usage", c.c_ushort),
                ("reserved", c.c_ubyte * 4),
            ]

        self._CUmemAllocFlagsType = _CUmemAllocFlags
        return self._CUmemAllocFlagsType

    @property
    def CUmemAllocationProp(self):
        if hasattr(self, "_CUmemAllocationPropType"):
            return self._CUmemAllocationPropType
        c = self.ctypes
        location_type = self.CUmemLocation
        flags_type = self.CUmemAllocFlags

        class _CUmemAllocationProp(c.Structure):
            _fields_ = [
                ("type", c.c_int),
                ("requestedHandleTypes", c.c_int),
                ("location", location_type),
                ("win32HandleMetaData", c.c_void_p),
                ("allocFlags", flags_type),
            ]

        self._CUmemAllocationPropType = _CUmemAllocationProp
        return self._CUmemAllocationPropType

    @property
    def CUmemAccessDesc(self):
        if hasattr(self, "_CUmemAccessDescType"):
            return self._CUmemAccessDescType
        c = self.ctypes
        location_type = self.CUmemLocation

        class _CUmemAccessDesc(c.Structure):
            _fields_ = [("location", location_type), ("flags", c.c_int)]

        self._CUmemAccessDescType = _CUmemAccessDesc
        return self._CUmemAccessDescType

    def _ensure_context(self) -> None:
        c = self.ctypes
        self._check(self.lib.cuInit(0), "cuInit")
        device = c.c_int()
        self._check(self.lib.cuDeviceGet(c.byref(device), self.device_index), "cuDeviceGet")
        self.device = device.value
        context = c.c_void_p()
        self._check(
            self.lib.cuDevicePrimaryCtxRetain(c.byref(context), self.device),
            "cuDevicePrimaryCtxRetain",
        )
        self._check(self.lib.cuCtxSetCurrent(context), "cuCtxSetCurrent")
        self.context = context
        torch.cuda.init()

    def _error_detail(self, code: int) -> str:
        c = self.ctypes
        name = c.c_char_p()
        message = c.c_char_p()
        self.lib.cuGetErrorName(code, c.byref(name))
        self.lib.cuGetErrorString(code, c.byref(message))
        return (
            f"code={code} name={(name.value or b'').decode()} "
            f"msg={(message.value or b'').decode()}"
        )

    def _check(self, code: int, operation: str) -> None:
        if code:
            raise WeightBridgeUnavailableError(
                f"CUDA VMM operation {operation} failed: {self._error_detail(code)}"
            )

    def _allocation_prop(self):
        prop = self.CUmemAllocationProp()
        prop.type = 1
        prop.requestedHandleTypes = 1
        prop.location = self.CUmemLocation(1, self.device)
        return prop

    def _access_desc(self):
        return self.CUmemAccessDesc(self.CUmemLocation(1, self.device), 3)

    def supports_posix_fd_vmm(self) -> bool:
        c = self.ctypes
        for attr in (102, 103):
            value = c.c_int()
            self._check(
                self.lib.cuDeviceGetAttribute(c.byref(value), attr, self.device),
                f"cuDeviceGetAttribute({attr})",
            )
            if value.value != 1:
                return False
        return True

    def granularity(self) -> int:
        c = self.ctypes
        value = c.c_size_t()
        prop = self._allocation_prop()
        self._check(
            self.lib.cuMemGetAllocationGranularity(c.byref(value), c.byref(prop), 0),
            "cuMemGetAllocationGranularity",
        )
        return int(value.value)

    def create_allocation(self, requested_nbytes: int) -> tuple[Any, int, int, int]:
        c = self.ctypes
        granularity = self.granularity()
        mapped_nbytes = max(
            granularity,
            ((int(requested_nbytes) + granularity - 1) // granularity) * granularity,
        )
        prop = self._allocation_prop()
        handle = c.c_uint64()
        address = c.c_uint64()
        exported_fd = c.c_int(-1)
        self._check(
            self.lib.cuMemCreate(c.byref(handle), mapped_nbytes, c.byref(prop), 0),
            "cuMemCreate",
        )
        try:
            self._check(
                self.lib.cuMemAddressReserve(c.byref(address), mapped_nbytes, 0, 0, 0),
                "cuMemAddressReserve",
            )
            self._check(
                self.lib.cuMemMap(address.value, mapped_nbytes, 0, handle.value, 0),
                "cuMemMap",
            )
            access = self._access_desc()
            self._check(
                self.lib.cuMemSetAccess(address.value, mapped_nbytes, c.byref(access), 1),
                "cuMemSetAccess",
            )
            self._check(
                self.lib.cuMemExportToShareableHandle(c.byref(exported_fd), handle.value, 1, 0),
                "cuMemExportToShareableHandle",
            )
        except Exception:
            self.release_allocation(handle.value, address.value, mapped_nbytes, exported_fd.value)
            raise
        return handle.value, address.value, mapped_nbytes, exported_fd.value

    def release_allocation(
        self,
        handle: Any,
        address: int,
        mapped_nbytes: int,
        exported_fd: int = -1,
    ) -> None:
        if address:
            self.lib.cuMemUnmap(int(address), int(mapped_nbytes))
            self.lib.cuMemAddressFree(int(address), int(mapped_nbytes))
        if handle:
            self.lib.cuMemRelease(int(handle))
        if exported_fd is not None and int(exported_fd) >= 0:
            try:
                os.close(int(exported_fd))
            except OSError:
                pass

    def import_allocation(self, fd: int, mapped_nbytes: int) -> tuple[Any, int]:
        c = self.ctypes
        handle = c.c_uint64()
        address = c.c_uint64()
        self._check(
            self.lib.cuMemImportFromShareableHandle(c.byref(handle), c.c_void_p(int(fd)), 1),
            "cuMemImportFromShareableHandle",
        )
        try:
            self._check(
                self.lib.cuMemAddressReserve(c.byref(address), int(mapped_nbytes), 0, 0, 0),
                "cuMemAddressReserve(import)",
            )
            self._check(
                self.lib.cuMemMap(address.value, int(mapped_nbytes), 0, handle.value, 0),
                "cuMemMap(import)",
            )
            access = self._access_desc()
            self._check(
                self.lib.cuMemSetAccess(
                    address.value,
                    int(mapped_nbytes),
                    c.byref(access),
                    1,
                ),
                "cuMemSetAccess(import)",
            )
        except Exception:
            self.release_import(handle.value, address.value, mapped_nbytes)
            raise
        return handle.value, address.value

    def release_import(self, handle: Any, address: int, mapped_nbytes: int) -> None:
        if address:
            self.lib.cuMemUnmap(int(address), int(mapped_nbytes))
            self.lib.cuMemAddressFree(int(address), int(mapped_nbytes))
        if handle:
            self.lib.cuMemRelease(int(handle))

    def copy_tensor_to_address(
        self,
        destination_address: int,
        tensor: torch.Tensor,
        *,
        stream: Optional[int] = None,
    ) -> None:
        if tensor.device.type != "cuda":
            raise WeightBridgeUnavailableError("CUDA VMM copy requires a CUDA source tensor")
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        nbytes = int(tensor.numel() * tensor.element_size())
        if stream is None:
            self._check(
                self.lib.cuMemcpyDtoD_v2(
                    int(destination_address),
                    int(tensor.data_ptr()),
                    nbytes,
                ),
                "cuMemcpyDtoD",
            )
            return
        self._check(
            self.lib.cuMemcpyDtoDAsync_v2(
                int(destination_address),
                int(tensor.data_ptr()),
                nbytes,
                self.ctypes.c_void_p(int(stream)),
            ),
            "cuMemcpyDtoDAsync",
        )


class WeightBridgeError(RuntimeError):
    """Base class for weight synchronization bridge failures."""


class WeightBridgeUnavailableError(WeightBridgeError):
    """Raised when a requested transport is not supported by this runtime."""


class WeightManifestValidationError(WeightBridgeError):
    """Raised when a weight update manifest is incomplete or inconsistent."""


class WeightUpdateRejectedError(WeightBridgeError):
    """Raised when a weight update cannot be imported or acknowledged."""


def _validated_manifest_metadata(metadata: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    manifest_metadata = dict(metadata or {})
    layout = WeightLayout.from_metadata(manifest_metadata)
    layout.validate_supported()
    manifest_metadata["layout"] = layout.to_metadata()
    return manifest_metadata


@dataclass(frozen=True)
class TensorDescriptor:
    """Transport-independent metadata for one tensor in a weight update."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    stride: tuple[int, ...]
    device: str
    numel: int
    nbytes: int
    sha256: str

    @classmethod
    def from_tensor(cls, name: str, tensor: torch.Tensor) -> TensorDescriptor:
        return cls(
            name=name,
            shape=tuple(int(dim) for dim in tensor.shape),
            dtype=str(tensor.dtype),
            stride=tuple(int(item) for item in tensor.stride()),
            device=str(tensor.device),
            numel=int(tensor.numel()),
            nbytes=int(tensor.numel() * tensor.element_size()),
            sha256=_tensor_sha256(tensor),
        )


@dataclass(frozen=True)
class WeightUpdateManifest:
    """Immutable public record for a complete published weight update."""

    update_id: str
    source_worker: str
    source_rank: int
    weight_version: int
    transport: str
    tensors: Mapping[str, TensorDescriptor]
    created_at: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def tensor_count(self) -> int:
        return len(self.tensors)

    @property
    def total_nbytes(self) -> int:
        return sum(descriptor.nbytes for descriptor in self.tensors.values())


class WeightPublisher(Protocol):
    def publish(
        self,
        model: torch.nn.Module,
        *,
        weight_version: int,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> WeightUpdateManifest: ...

    def release(self, update_id: str) -> None: ...


class WeightConsumer(Protocol):
    def import_update(self, manifest: WeightUpdateManifest) -> Mapping[str, torch.Tensor]: ...

    def acknowledge(self, update_id: str) -> None: ...

    def reject(self, update_id: str, reason: str) -> None: ...

    def release(self, update_id: str) -> None: ...


class WeightBridge(WeightPublisher, WeightConsumer, Protocol):
    pass


class WeightInstallAdapter(Protocol):
    def install(
        self,
        manifest: WeightUpdateManifest,
        tensors: Mapping[str, torch.Tensor],
    ) -> None: ...

    def release(self, update_id: str) -> None: ...


@dataclass
class _LocalUpdateRecord:
    manifest: WeightUpdateManifest
    tensors: dict[str, torch.Tensor]
    state: str = "published"
    import_count: int = 0
    rejection_reason: Optional[str] = None


@dataclass
class _CUDAVMMAllocationRecord:
    handle: Any
    address: int
    mapped_nbytes: int
    exported_fd: int
    socket_path: str
    thread: Any
    stop_event: Any
    sync_event: Any = None
    copy_sources: list[torch.Tensor] = field(default_factory=list)


@dataclass
class _CUDAVMMImportRecord:
    allocation_handle: Any
    address: int
    mapped_nbytes: int
    socket_fd: int
    posix_fd: int
    dlpack_owners: list[Any] = field(default_factory=list)
    sync_event: Any = None


def _pack_tensors_for_cuda_vmm(
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, int]], int]:
    tensors: dict[str, torch.Tensor] = {}
    entries: dict[str, dict[str, int]] = {}
    offset = 0
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.device.type != "cuda":
            raise WeightBridgeUnavailableError(
                "CUDAVMMTensorBridge requires CUDA tensors. "
                f"Tensor {name} is on {tensor.device}."
            )
        snapshot = tensor.detach()
        if not snapshot.is_contiguous():
            snapshot = snapshot.contiguous()
        tensor_nbytes = int(snapshot.numel() * snapshot.element_size())
        offset = (
            (offset + _CUDA_VMM_TENSOR_ALIGNMENT - 1) // _CUDA_VMM_TENSOR_ALIGNMENT
        ) * _CUDA_VMM_TENSOR_ALIGNMENT
        tensors[name] = snapshot
        entries[name] = {
            "offset": offset,
            "nbytes": tensor_nbytes,
            "storage_offset": 0,
        }
        offset += tensor_nbytes
    return tensors, entries, offset


class LocalTensorCopyBridge:
    """
    Safe local transport for the weight synchronization protocol.

    This transport intentionally copies tensors. It is not the final zero-copy
    transport, but it exercises the same versioned publish/import/ack/release
    lifecycle that CUDA IPC and vLLM adapters must obey later.
    """

    transport = "local-clone"

    def __init__(self, *, source_worker: str = "local-training", source_rank: int = 0):
        self.source_worker = source_worker
        self.source_rank = int(source_rank)
        self._updates: dict[str, _LocalUpdateRecord] = {}
        self._latest_published_weight_version = -1
        self.active_weight_version: Optional[int] = None
        self.active_update_id: Optional[str] = None

    def publish(
        self,
        model: torch.nn.Module,
        *,
        weight_version: int,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> WeightUpdateManifest:
        manifest_metadata = _validated_manifest_metadata(metadata)
        version = int(weight_version)
        if version <= self._latest_published_weight_version:
            raise WeightManifestValidationError(
                "weight_version must increase monotonically "
                f"(got {version}, latest {self._latest_published_weight_version})"
            )

        tensors = self._clone_state_dict(model)
        if not tensors:
            raise WeightManifestValidationError("model state_dict produced no tensors")

        descriptors = {
            name: TensorDescriptor.from_tensor(name, tensor) for name, tensor in tensors.items()
        }
        manifest = WeightUpdateManifest(
            update_id=str(uuid.uuid4()),
            source_worker=self.source_worker,
            source_rank=self.source_rank,
            weight_version=version,
            transport=self.transport,
            tensors=descriptors,
            created_at=time.perf_counter(),
            metadata=manifest_metadata,
        )
        self._updates[manifest.update_id] = _LocalUpdateRecord(
            manifest=manifest,
            tensors=tensors,
        )
        self._latest_published_weight_version = version
        logger.info(
            "Published weight update %s version=%s tensors=%s bytes=%s",
            manifest.update_id,
            manifest.weight_version,
            manifest.tensor_count,
            manifest.total_nbytes,
        )
        return manifest

    def import_update(self, manifest: WeightUpdateManifest) -> Mapping[str, torch.Tensor]:
        record = self._require_record(manifest.update_id)
        if record.state == "rejected":
            raise WeightUpdateRejectedError(
                f"weight update {manifest.update_id} was rejected: {record.rejection_reason}"
            )
        if record.state == "released":
            raise WeightUpdateRejectedError(f"weight update {manifest.update_id} was released")

        self._validate_manifest(record, manifest)
        record.import_count += 1
        if record.state == "published":
            record.state = "imported"

        return {
            name: tensor.detach().clone(memory_format=torch.preserve_format)
            for name, tensor in record.tensors.items()
        }

    def acknowledge(self, update_id: str) -> None:
        record = self._require_record(update_id)
        if record.state == "rejected":
            raise WeightUpdateRejectedError(
                f"cannot acknowledge rejected update {update_id}: {record.rejection_reason}"
            )
        if record.state == "released":
            raise WeightUpdateRejectedError(f"cannot acknowledge released update {update_id}")
        if record.import_count == 0:
            raise WeightUpdateRejectedError(
                f"cannot acknowledge update {update_id} before import_update succeeds"
            )
        self._validate_manifest(record, record.manifest)
        record.state = "acknowledged"
        self.active_weight_version = record.manifest.weight_version
        self.active_update_id = update_id

    def reject(self, update_id: str, reason: str) -> None:
        record = self._require_record(update_id)
        if record.state == "acknowledged":
            raise WeightUpdateRejectedError(f"cannot reject acknowledged update {update_id}")
        record.state = "rejected"
        record.rejection_reason = reason

    def release(self, update_id: str) -> None:
        record = self._updates.pop(update_id, None)
        if record is None:
            return
        record.tensors.clear()
        record.state = "released"
        if self.active_update_id == update_id:
            self.active_update_id = None

    def update_status(self, update_id: str) -> str:
        record = self._updates.get(update_id)
        if record is None:
            return "released"
        return record.state

    def get_manifest(self, update_id: str) -> WeightUpdateManifest:
        return self._require_record(update_id).manifest

    def debug_tensor_data_ptrs(self, update_id: str) -> Mapping[str, int]:
        """Return storage pointers for tests and benchmark transport verification."""

        record = self._require_record(update_id)
        return {name: int(tensor.data_ptr()) for name, tensor in record.tensors.items()}

    def _clone_state_dict(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        cloned: dict[str, torch.Tensor] = {}
        for name, tensor in model.state_dict().items():
            if not isinstance(tensor, torch.Tensor):
                continue
            cloned[name] = tensor.detach().clone(memory_format=torch.preserve_format)
        return cloned

    def _require_record(self, update_id: str) -> _LocalUpdateRecord:
        try:
            return self._updates[update_id]
        except KeyError as exc:
            raise WeightUpdateRejectedError(f"unknown weight update {update_id}") from exc

    def _validate_manifest(
        self,
        record: _LocalUpdateRecord,
        manifest: WeightUpdateManifest,
    ) -> None:
        expected = record.manifest
        WeightLayout.from_metadata(manifest.metadata).validate_supported()
        if manifest.transport != expected.transport:
            raise WeightManifestValidationError(
                f"transport mismatch: expected {expected.transport}, got {manifest.transport}"
            )
        if manifest.weight_version != expected.weight_version:
            raise WeightManifestValidationError(
                "weight_version mismatch: "
                f"expected {expected.weight_version}, got {manifest.weight_version}"
            )
        if set(manifest.tensors) != set(expected.tensors):
            missing = sorted(set(expected.tensors) - set(manifest.tensors))
            extra = sorted(set(manifest.tensors) - set(expected.tensors))
            raise WeightManifestValidationError(
                f"tensor manifest mismatch: missing={missing}, extra={extra}"
            )
        for name, descriptor in expected.tensors.items():
            actual = manifest.tensors[name]
            if actual != descriptor:
                raise WeightManifestValidationError(
                    f"tensor descriptor mismatch for {name}: "
                    f"expected {descriptor}, got {actual}"
                )
            current = TensorDescriptor.from_tensor(name, record.tensors[name])
            if current != descriptor:
                if current.sha256 != descriptor.sha256:
                    raise WeightManifestValidationError(
                        f"tensor checksum mismatch for {name}: "
                        f"expected {descriptor.sha256}, got {current.sha256}"
                    )
                raise WeightManifestValidationError(
                    f"stored tensor descriptor mismatch for {name}: "
                    f"expected {descriptor}, got {current}"
                )


class SharedMemoryTensorBridge(LocalTensorCopyBridge):
    """
    Same-node shared-memory transport with zero-copy import semantics.

    Publishing creates a shared-memory snapshot of model state. Importing that
    manifest returns tensor aliases to the shared storage instead of cloning.
    This proves the bridge lifecycle can expose a complete version through
    shared memory, while keeping the CUDA IPC transport as a separate follow-up.
    """

    transport = "shared-memory"

    def __init__(self, *, source_worker: str = "local-training", source_rank: int = 0):
        super().__init__(source_worker=source_worker, source_rank=source_rank)
        self._shared_memory_segments: dict[str, dict[str, shared_memory.SharedMemory]] = {}
        self._owned_shared_memory_update_ids: set[str] = set()

    def publish(
        self,
        model: torch.nn.Module,
        *,
        weight_version: int,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> WeightUpdateManifest:
        user_metadata = _validated_manifest_metadata(metadata)
        if _BRIDGE_METADATA_KEY in user_metadata:
            raise WeightManifestValidationError(
                f"metadata key {_BRIDGE_METADATA_KEY!r} is reserved for bridge internals"
            )

        version = int(weight_version)
        if version <= self._latest_published_weight_version:
            raise WeightManifestValidationError(
                "weight_version must increase monotonically "
                f"(got {version}, latest {self._latest_published_weight_version})"
            )

        update_id = str(uuid.uuid4())
        tensors, segments, shared_metadata = self._snapshot_to_shared_memory(model)
        if not tensors:
            for segment in segments.values():
                segment.close()
                segment.unlink()
            raise WeightManifestValidationError("model state_dict produced no tensors")

        descriptors = {
            name: TensorDescriptor.from_tensor(name, tensor) for name, tensor in tensors.items()
        }
        manifest = WeightUpdateManifest(
            update_id=update_id,
            source_worker=self.source_worker,
            source_rank=self.source_rank,
            weight_version=version,
            transport=self.transport,
            tensors=descriptors,
            created_at=time.perf_counter(),
            metadata={**user_metadata, _BRIDGE_METADATA_KEY: shared_metadata},
        )
        self._updates[manifest.update_id] = _LocalUpdateRecord(
            manifest=manifest,
            tensors=tensors,
        )
        self._shared_memory_segments[manifest.update_id] = segments
        self._owned_shared_memory_update_ids.add(manifest.update_id)
        self._latest_published_weight_version = version
        logger.info(
            "Published shared-memory weight update %s version=%s tensors=%s bytes=%s",
            manifest.update_id,
            manifest.weight_version,
            manifest.tensor_count,
            manifest.total_nbytes,
        )
        return manifest

    def _snapshot_to_shared_memory(
        self,
        model: torch.nn.Module,
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, shared_memory.SharedMemory],
        dict[str, Any],
    ]:
        shared_tensors: dict[str, torch.Tensor] = {}
        segments: dict[str, shared_memory.SharedMemory] = {}
        tensor_metadata: dict[str, dict[str, Any]] = {}
        try:
            for name, tensor in model.state_dict().items():
                if not isinstance(tensor, torch.Tensor):
                    continue
                if tensor.device.type != "cpu":
                    raise WeightBridgeUnavailableError(
                        "SharedMemoryTensorBridge currently supports CPU tensors only. "
                        f"Tensor {name} is on {tensor.device}."
                    )
                snapshot = tensor.detach().contiguous()
                storage_offset = int(snapshot.storage_offset())
                shape = tuple(int(dim) for dim in snapshot.shape)
                stride = tuple(int(item) for item in snapshot.stride())
                storage_numel = _required_storage_numel(shape, stride, storage_offset)
                storage_nbytes = storage_numel * snapshot.element_size()
                if storage_numel == 0:
                    shared_view = torch.empty_strided(
                        shape,
                        stride,
                        dtype=snapshot.dtype,
                    )
                    shared_tensors[name] = shared_view
                    tensor_metadata[name] = {
                        "name": None,
                        "size": 0,
                        "storage_numel": 0,
                        "storage_nbytes": 0,
                        "storage_offset": 0,
                    }
                    continue

                segment = _create_shared_memory(max(storage_nbytes, 1))
                try:
                    base = torch.frombuffer(
                        segment.buf,
                        dtype=snapshot.dtype,
                        count=storage_numel,
                    )
                    shared_view = base.as_strided(shape, stride, storage_offset)
                    shared_view.copy_(snapshot)
                except Exception:
                    segment.close()
                    segment.unlink()
                    raise

                shared_tensors[name] = shared_view
                segments[name] = segment
                tensor_metadata[name] = {
                    "name": segment.name,
                    "size": int(segment.size),
                    "storage_numel": storage_numel,
                    "storage_nbytes": storage_nbytes,
                    "storage_offset": storage_offset,
                }
        except Exception:
            for segment in segments.values():
                try:
                    segment.close()
                finally:
                    segment.unlink()
            raise

        return (
            shared_tensors,
            segments,
            {
                "format": _SHARED_MEMORY_FORMAT,
                "tensors": tensor_metadata,
            },
        )

    def import_update(self, manifest: WeightUpdateManifest) -> Mapping[str, torch.Tensor]:
        record = self._updates.get(manifest.update_id)
        if record is None:
            record = self._attach_shared_memory_manifest(manifest)
        if record.state == "rejected":
            raise WeightUpdateRejectedError(
                f"weight update {manifest.update_id} was rejected: {record.rejection_reason}"
            )
        if record.state == "released":
            raise WeightUpdateRejectedError(f"weight update {manifest.update_id} was released")

        self._validate_manifest(record, manifest)
        record.import_count += 1
        if record.state == "published":
            record.state = "imported"
        return dict(record.tensors)

    def release(self, update_id: str) -> None:
        super().release(update_id)
        segments = self._shared_memory_segments.pop(update_id, {})
        should_unlink = update_id in self._owned_shared_memory_update_ids
        self._owned_shared_memory_update_ids.discard(update_id)
        for segment in segments.values():
            try:
                segment.close()
            finally:
                if should_unlink:
                    try:
                        segment.unlink()
                    except FileNotFoundError:
                        pass

    def _attach_shared_memory_manifest(self, manifest: WeightUpdateManifest) -> _LocalUpdateRecord:
        if manifest.transport != self.transport:
            raise WeightManifestValidationError(
                f"transport mismatch: expected {self.transport}, got {manifest.transport}"
            )
        bridge_metadata = manifest.metadata.get(_BRIDGE_METADATA_KEY)
        if not isinstance(bridge_metadata, Mapping):
            raise WeightManifestValidationError("shared-memory manifest is missing bridge metadata")
        if bridge_metadata.get("format") != _SHARED_MEMORY_FORMAT:
            raise WeightManifestValidationError(
                "shared-memory manifest has unsupported metadata format: "
                f"{bridge_metadata.get('format')}"
            )
        entries = bridge_metadata.get("tensors")
        if not isinstance(entries, Mapping):
            raise WeightManifestValidationError("shared-memory manifest has no tensor handles")
        if set(entries) != set(manifest.tensors):
            missing = sorted(set(manifest.tensors) - set(entries))
            extra = sorted(set(entries) - set(manifest.tensors))
            raise WeightManifestValidationError(
                f"shared-memory manifest handle mismatch: missing={missing}, extra={extra}"
            )

        tensors: dict[str, torch.Tensor] = {}
        segments: dict[str, shared_memory.SharedMemory] = {}
        try:
            for name, descriptor in manifest.tensors.items():
                entry = entries.get(name)
                if not isinstance(entry, Mapping):
                    raise WeightManifestValidationError(
                        f"shared-memory manifest is missing handle for tensor {name}"
                    )
                if descriptor.device != "cpu":
                    raise WeightManifestValidationError(
                        f"shared-memory manifest tensor {name} is on {descriptor.device}"
                    )
                dtype = _dtype_from_name(descriptor.dtype)
                try:
                    storage_numel = int(entry["storage_numel"])
                    storage_offset = int(entry["storage_offset"])
                    expected_size = int(entry["size"])
                    storage_nbytes = int(entry["storage_nbytes"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise WeightManifestValidationError(
                        f"shared-memory manifest has invalid handle metadata for tensor {name}"
                    ) from exc

                required_storage_numel = _required_storage_numel(
                    descriptor.shape,
                    descriptor.stride,
                    storage_offset,
                )
                if storage_numel < required_storage_numel:
                    raise WeightManifestValidationError(
                        f"shared-memory tensor {name} storage is smaller than descriptor"
                    )
                if storage_nbytes != storage_numel * torch.empty((), dtype=dtype).element_size():
                    raise WeightManifestValidationError(
                        f"shared-memory tensor {name} storage byte count does not match dtype"
                    )
                if storage_numel == 0:
                    tensor = torch.empty_strided(descriptor.shape, descriptor.stride, dtype=dtype)
                    actual_descriptor = TensorDescriptor.from_tensor(name, tensor)
                    if actual_descriptor != descriptor:
                        raise WeightManifestValidationError(
                            f"tensor descriptor mismatch for {name}: "
                            f"expected {descriptor}, got {actual_descriptor}"
                        )
                    tensors[name] = tensor
                    continue

                segment = _attach_shared_memory(str(entry["name"]))
                if expected_size > segment.size or storage_nbytes > segment.size:
                    segment.close()
                    raise WeightManifestValidationError(
                        f"shared-memory segment for tensor {name} is smaller than manifest"
                    )
                base = torch.frombuffer(segment.buf, dtype=dtype, count=storage_numel)
                tensor = base.as_strided(descriptor.shape, descriptor.stride, storage_offset)
                actual_descriptor = TensorDescriptor.from_tensor(name, tensor)
                if actual_descriptor != descriptor:
                    segment.close()
                    if actual_descriptor.sha256 != descriptor.sha256:
                        raise WeightManifestValidationError(
                            f"tensor checksum mismatch for {name}: "
                            f"expected {descriptor.sha256}, got {actual_descriptor.sha256}"
                        )
                    raise WeightManifestValidationError(
                        f"tensor descriptor mismatch for {name}: "
                        f"expected {descriptor}, got {actual_descriptor}"
                    )
                tensors[name] = tensor
                segments[name] = segment
        except Exception:
            for segment in segments.values():
                segment.close()
            raise

        record = _LocalUpdateRecord(
            manifest=manifest,
            tensors=tensors,
            state="published",
            import_count=0,
        )
        self._updates[manifest.update_id] = record
        self._shared_memory_segments[manifest.update_id] = segments
        return record


class CUDAVMMTensorBridge(LocalTensorCopyBridge):
    """
    Same-node CUDA VMM transport with POSIX-fd zero-copy import semantics.

    This is the modern CUDA IPC path for WSL2/native Linux runtimes where legacy
    `cudaIpcOpenMemHandle` is unavailable. Publishing packs a complete CUDA
    model-state snapshot into one exportable CUDA VMM allocation. Consumers
    connect to the publisher broker, receive the allocation fd over SCM_RIGHTS,
    map it into their process, and build PyTorch tensors that alias the mapped
    GPU memory through DLPack.
    """

    transport = "cuda-vmm"

    def __init__(
        self,
        *,
        source_worker: str = "cuda-training",
        source_rank: int = 0,
        device_index: int = 0,
        backend_factory: Optional[Callable[[], Any]] = None,
    ):
        super().__init__(source_worker=source_worker, source_rank=source_rank)
        self.device_index = int(device_index)
        self._backend_factory = backend_factory
        self._backend: Optional[Any] = None
        self._cuda_vmm_allocations: dict[str, _CUDAVMMAllocationRecord] = {}
        self._cuda_vmm_imports: dict[str, _CUDAVMMImportRecord] = {}
        self._owned_cuda_vmm_update_ids: set[str] = set()

    def publish(
        self,
        model: torch.nn.Module,
        *,
        weight_version: int,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> WeightUpdateManifest:
        user_metadata = _validated_manifest_metadata(metadata)
        if _BRIDGE_METADATA_KEY in user_metadata:
            raise WeightManifestValidationError(
                f"metadata key {_BRIDGE_METADATA_KEY!r} is reserved for bridge internals"
            )
        version = int(weight_version)
        if version <= self._latest_published_weight_version:
            raise WeightManifestValidationError(
                "weight_version must increase monotonically "
                f"(got {version}, latest {self._latest_published_weight_version})"
            )

        state_tensors, entries, requested_nbytes = _pack_tensors_for_cuda_vmm(model.state_dict())
        if not state_tensors:
            raise WeightManifestValidationError("model state_dict produced no CUDA tensors")

        descriptors = {
            name: TensorDescriptor.from_tensor(name, tensor)
            for name, tensor in state_tensors.items()
        }
        backend = self._get_backend()
        if not backend.supports_posix_fd_vmm():
            raise WeightBridgeUnavailableError(
                "CUDA VMM POSIX-fd export is not supported by this device/runtime."
            )
        handle, address, mapped_nbytes, exported_fd = backend.create_allocation(requested_nbytes)
        try:
            tensors: dict[str, torch.Tensor] = {}
            stream = torch.cuda.current_stream(self.device_index)
            stream_handle = int(stream.cuda_stream)
            for name, tensor in state_tensors.items():
                entry = entries[name]
                destination = address + int(entry["offset"])
                backend.copy_tensor_to_address(destination, tensor, stream=stream_handle)
                tensors[name] = self._tensor_from_address(
                    destination,
                    tuple(int(dim) for dim in tensor.shape),
                    tuple(int(item) for item in tensor.stride()),
                    tensor.dtype,
                )
            sync_event = torch.cuda.Event(interprocess=True)
            sync_event.record(stream)
            ipc_event_handle = sync_event.ipc_handle()
            update_id = str(uuid.uuid4())
            socket_path, thread, stop_event = self._start_fd_broker(update_id, exported_fd)
            manifest = WeightUpdateManifest(
                update_id=update_id,
                source_worker=self.source_worker,
                source_rank=self.source_rank,
                weight_version=version,
                transport=self.transport,
                tensors=descriptors,
                created_at=time.perf_counter(),
                metadata={
                    **user_metadata,
                    _BRIDGE_METADATA_KEY: {
                        "format": _CUDA_VMM_FORMAT,
                        "socket_path": socket_path,
                        "mapped_nbytes": mapped_nbytes,
                        "requested_nbytes": requested_nbytes,
                        "device_index": self.device_index,
                        "ipc_event_handle": ipc_event_handle,
                        "tensors": entries,
                    },
                },
            )
            self._updates[manifest.update_id] = _LocalUpdateRecord(
                manifest=manifest,
                tensors=tensors,
            )
            self._cuda_vmm_allocations[manifest.update_id] = _CUDAVMMAllocationRecord(
                handle=handle,
                address=address,
                mapped_nbytes=mapped_nbytes,
                exported_fd=exported_fd,
                socket_path=socket_path,
                thread=thread,
                stop_event=stop_event,
                sync_event=sync_event,
                copy_sources=list(state_tensors.values()),
            )
            self._owned_cuda_vmm_update_ids.add(manifest.update_id)
            self._latest_published_weight_version = version
            logger.info(
                "Published CUDA VMM weight update %s version=%s tensors=%s bytes=%s",
                manifest.update_id,
                manifest.weight_version,
                manifest.tensor_count,
                manifest.total_nbytes,
            )
            return manifest
        except Exception:
            backend.release_allocation(handle, address, mapped_nbytes, exported_fd)
            raise

    def import_update(self, manifest: WeightUpdateManifest) -> Mapping[str, torch.Tensor]:
        record = self._updates.get(manifest.update_id)
        if record is None:
            record = self._attach_cuda_vmm_manifest(manifest)
        if record.state == "rejected":
            raise WeightUpdateRejectedError(
                f"weight update {manifest.update_id} was rejected: {record.rejection_reason}"
            )
        if record.state == "released":
            raise WeightUpdateRejectedError(f"weight update {manifest.update_id} was released")

        self._wait_for_cuda_vmm_publish_event(manifest)
        self._validate_manifest(record, manifest)
        record.import_count += 1
        if record.state == "published":
            record.state = "imported"
        return dict(record.tensors)

    def release(self, update_id: str) -> None:
        record = self._updates.pop(update_id, None)
        released_tensors = list(record.tensors.values()) if record is not None else []
        if record is not None:
            record.tensors.clear()
            record.state = "released"
        if self.active_update_id == update_id:
            self.active_update_id = None
        allocation = self._cuda_vmm_allocations.pop(update_id, None)
        if allocation is not None:
            allocation.stop_event.set()
            try:
                with _unix_stream_socket() as sock:
                    sock.settimeout(0.1)
                    sock.connect(allocation.socket_path)
            except OSError:
                pass
            allocation.thread.join(timeout=1)
            try:
                os.unlink(allocation.socket_path)
            except FileNotFoundError:
                pass
            self._get_backend().release_allocation(
                allocation.handle,
                allocation.address,
                allocation.mapped_nbytes,
                allocation.exported_fd,
            )
        imported = self._cuda_vmm_imports.pop(update_id, None)
        if imported is not None:
            imported.dlpack_owners.clear()
            self._get_backend().release_import(
                imported.allocation_handle,
                imported.address,
                imported.mapped_nbytes,
            )
            try:
                os.close(imported.posix_fd)
            except OSError:
                pass
            try:
                os.close(imported.socket_fd)
            except OSError:
                pass
        released_tensors.clear()

    def _get_backend(self) -> Any:
        if self._backend is None:
            if self._backend_factory is not None:
                self._backend = self._backend_factory()
            else:
                self._backend = _CUDAVMMDriverBackend(device_index=self.device_index)
        return self._backend

    def _wait_for_cuda_vmm_publish_event(self, manifest: WeightUpdateManifest) -> Any:
        bridge_metadata = manifest.metadata.get(_BRIDGE_METADATA_KEY)
        if not isinstance(bridge_metadata, Mapping):
            raise WeightManifestValidationError("CUDA VMM manifest is missing bridge metadata")
        ipc_event_handle = bridge_metadata.get("ipc_event_handle")
        if ipc_event_handle is None:
            raise WeightManifestValidationError(
                "CUDA VMM manifest is missing IPC sync event handle"
            )

        stream = torch.cuda.current_stream(self.device_index)
        allocation = self._cuda_vmm_allocations.get(manifest.update_id)
        if allocation is not None and allocation.sync_event is not None:
            allocation.sync_event.wait(stream)
            return allocation.sync_event

        remote_event = torch.cuda.Event.from_ipc_handle(self.device_index, ipc_event_handle)
        remote_event.wait(stream)
        return remote_event

    def _tensor_from_address(
        self,
        address: int,
        shape: tuple[int, ...],
        stride: tuple[int, ...],
        dtype: torch.dtype,
        owners: Optional[list[Any]] = None,
    ) -> torch.Tensor:
        owner = _DLPackOwner(
            address=address,
            shape=shape,
            stride=stride,
            dtype=dtype,
            device_index=self.device_index,
        )
        tensor = owner.to_tensor()
        try:
            cast(Any, tensor)._kernel_align_dlpack_owner = owner
        except AttributeError:
            pass
        if owners is not None:
            owners.append(owner)
        else:
            weakref.finalize(tensor, lambda held_owner: None, owner)
        return tensor

    def _start_fd_broker(self, update_id: str, exported_fd: int):
        import threading

        socket_path = f"/tmp/kernel-align-cuda-vmm-{os.getpid()}-{update_id}.sock"
        try:
            os.unlink(socket_path)
        except FileNotFoundError:
            pass
        stop_event = threading.Event()

        def broker() -> None:
            server = _unix_stream_socket()
            try:
                server.bind(socket_path)
                server.listen(16)
                server.settimeout(0.1)
                while not stop_event.is_set():
                    try:
                        connection, _ = server.accept()
                    except TimeoutError:
                        continue
                    except OSError:
                        break
                    with connection:
                        try:
                            _send_fd(connection, exported_fd)
                        except BrokenPipeError:
                            if not stop_event.is_set():
                                logger.debug(
                                    "CUDA VMM fd broker client disconnected before fd send"
                                )
            finally:
                server.close()

        thread = threading.Thread(target=broker, name=f"cuda-vmm-fd-{update_id}", daemon=True)
        thread.start()
        return socket_path, thread, stop_event

    def _attach_cuda_vmm_manifest(self, manifest: WeightUpdateManifest) -> _LocalUpdateRecord:
        if manifest.transport != self.transport:
            raise WeightManifestValidationError(
                f"transport mismatch: expected {self.transport}, got {manifest.transport}"
            )
        bridge_metadata = manifest.metadata.get(_BRIDGE_METADATA_KEY)
        if not isinstance(bridge_metadata, Mapping):
            raise WeightManifestValidationError("CUDA VMM manifest is missing bridge metadata")
        if bridge_metadata.get("format") != _CUDA_VMM_FORMAT:
            raise WeightManifestValidationError(
                "CUDA VMM manifest has unsupported metadata format: "
                f"{bridge_metadata.get('format')}"
            )
        entries = bridge_metadata.get("tensors")
        if not isinstance(entries, Mapping) or set(entries) != set(manifest.tensors):
            raise WeightManifestValidationError("CUDA VMM manifest handle mismatch")

        socket_path = str(bridge_metadata.get("socket_path"))
        mapped_nbytes = int(bridge_metadata.get("mapped_nbytes", 0))
        if not socket_path or mapped_nbytes <= 0:
            raise WeightManifestValidationError("CUDA VMM manifest has invalid fd metadata")

        backend = self._get_backend()
        socket_fd = -1
        posix_fd = -1
        allocation_handle = None
        address = 0
        owners: list[Any] = []
        try:
            sock = _unix_stream_socket()
            socket_fd = int(sock.fileno())
            sock.connect(socket_path)
            posix_fd = _recv_fd(sock)
            allocation_handle, address = backend.import_allocation(posix_fd, mapped_nbytes)
            sync_event = self._wait_for_cuda_vmm_publish_event(manifest)
            tensors: dict[str, torch.Tensor] = {}
            for name, descriptor in manifest.tensors.items():
                entry = entries[name]
                offset = int(entry["offset"])
                dtype = _dtype_from_name(descriptor.dtype)
                tensor = self._tensor_from_address(
                    address + offset,
                    descriptor.shape,
                    descriptor.stride,
                    dtype,
                    owners,
                )
                actual_descriptor = TensorDescriptor.from_tensor(name, tensor)
                if actual_descriptor != descriptor:
                    if actual_descriptor.sha256 != descriptor.sha256:
                        raise WeightManifestValidationError(
                            f"tensor checksum mismatch for {name}: "
                            f"expected {descriptor.sha256}, got {actual_descriptor.sha256}"
                        )
                    raise WeightManifestValidationError(
                        f"tensor descriptor mismatch for {name}: "
                        f"expected {descriptor}, got {actual_descriptor}"
                    )
                tensors[name] = tensor
            record = _LocalUpdateRecord(
                manifest=manifest,
                tensors=tensors,
                state="published",
                import_count=0,
            )
            self._updates[manifest.update_id] = record
            self._cuda_vmm_imports[manifest.update_id] = _CUDAVMMImportRecord(
                allocation_handle=allocation_handle,
                address=address,
                mapped_nbytes=mapped_nbytes,
                socket_fd=socket_fd,
                posix_fd=posix_fd,
                dlpack_owners=owners,
                sync_event=sync_event,
            )
            sock.detach()
            return record
        except Exception:
            owners.clear()
            if allocation_handle is not None or address:
                backend.release_import(allocation_handle, address, mapped_nbytes)
            for fd in (posix_fd, socket_fd):
                if fd >= 0:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
            raise


class IPCWeightBridge(LocalTensorCopyBridge):
    """
    Same-node legacy PyTorch CUDA IPC transport.

    Publishing creates a complete CUDA snapshot and stores PyTorch
    `reduce_tensor` handles in the manifest. Consumers rebuild CUDA tensor
    aliases from those handles and then use the normal import/ack/reject/release
    lifecycle. Some WSL2 driver paths can still reject the underlying CUDA IPC
    handle with `invalid resource handle`; those failures are surfaced as
    explicit transport blockers instead of synthetic success.
    """

    transport = "cuda-ipc"

    def __init__(
        self,
        *,
        source_worker: str = "cuda-training",
        source_rank: int = 0,
        reduce_tensor_fn: Optional[Any] = None,
    ):
        super().__init__(source_worker=source_worker, source_rank=source_rank)
        self._reduce_tensor_fn = reduce_tensor_fn
        self.handle_registry: dict[str, Any] = {}
        self._ipc_keepalive: dict[str, list[torch.Tensor]] = {}

    def publish(
        self,
        model: torch.nn.Module,
        *,
        weight_version: int,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> WeightUpdateManifest:
        user_metadata = _validated_manifest_metadata(metadata)
        if _BRIDGE_METADATA_KEY in user_metadata:
            raise WeightManifestValidationError(
                f"metadata key {_BRIDGE_METADATA_KEY!r} is reserved for bridge internals"
            )
        if not torch.cuda.is_available():
            raise WeightBridgeUnavailableError(
                "CUDA IPC weight bridge is unavailable because CUDA is not available."
            )

        version = int(weight_version)
        if version <= self._latest_published_weight_version:
            raise WeightManifestValidationError(
                "weight_version must increase monotonically "
                f"(got {version}, latest {self._latest_published_weight_version})"
            )

        tensors: dict[str, torch.Tensor] = {}
        handles: dict[str, Any] = {}
        reduce_tensor = self._resolve_reduce_tensor()
        for name, tensor in model.state_dict().items():
            if not isinstance(tensor, torch.Tensor):
                continue
            if tensor.device.type != "cuda":
                raise WeightBridgeUnavailableError(
                    "CUDA IPC weight bridge requires CUDA tensors. "
                    f"Tensor {name} is on {tensor.device}."
                )
            snapshot = tensor.detach().contiguous()
            tensors[name] = snapshot
            handles[name] = reduce_tensor(snapshot)

        if not tensors:
            raise WeightManifestValidationError("model state_dict produced no CUDA tensors")

        descriptors = {
            name: TensorDescriptor.from_tensor(name, tensor) for name, tensor in tensors.items()
        }
        update_id = str(uuid.uuid4())
        manifest = WeightUpdateManifest(
            update_id=update_id,
            source_worker=self.source_worker,
            source_rank=self.source_rank,
            weight_version=version,
            transport=self.transport,
            tensors=descriptors,
            created_at=time.perf_counter(),
            metadata={
                **user_metadata,
                _BRIDGE_METADATA_KEY: {
                    "format": _CUDA_IPC_FORMAT,
                    "device_index": int(torch.cuda.current_device()),
                    "gpu_uuid": self._current_gpu_uuid(),
                    "tensors": {
                        name: {
                            "handle": handle,
                        }
                        for name, handle in handles.items()
                    },
                },
            },
        )
        self._updates[manifest.update_id] = _LocalUpdateRecord(
            manifest=manifest,
            tensors=tensors,
        )
        self._ipc_keepalive[manifest.update_id] = list(tensors.values())
        self.handle_registry[manifest.update_id] = handles
        self._latest_published_weight_version = version
        logger.info(
            "Published CUDA IPC weight update %s version=%s tensors=%s bytes=%s",
            manifest.update_id,
            manifest.weight_version,
            manifest.tensor_count,
            manifest.total_nbytes,
        )
        return manifest

    def import_update(self, manifest: WeightUpdateManifest) -> Mapping[str, torch.Tensor]:
        record = self._updates.get(manifest.update_id)
        if record is None:
            record = self._attach_cuda_ipc_manifest(manifest)
        if record.state == "rejected":
            raise WeightUpdateRejectedError(
                f"weight update {manifest.update_id} was rejected: {record.rejection_reason}"
            )
        if record.state == "released":
            raise WeightUpdateRejectedError(f"weight update {manifest.update_id} was released")

        self._validate_manifest(record, manifest)
        record.import_count += 1
        if record.state == "published":
            record.state = "imported"
        return dict(record.tensors)

    def export_model_handles(self, model: torch.nn.Module) -> Mapping[str, Any]:
        manifest = self.publish(
            model,
            weight_version=self._latest_published_weight_version + 1,
        )
        bridge_metadata = manifest.metadata.get(_BRIDGE_METADATA_KEY)
        if not isinstance(bridge_metadata, Mapping):
            raise WeightManifestValidationError("CUDA IPC manifest is missing bridge metadata")
        tensor_metadata = bridge_metadata.get("tensors")
        if not isinstance(tensor_metadata, Mapping):
            raise WeightManifestValidationError("CUDA IPC manifest has no tensor handles")
        return {
            name: {
                "handle": entry["handle"],
                "shape": descriptor.shape,
                "dtype": descriptor.dtype,
                "stride": descriptor.stride,
                "update_id": manifest.update_id,
            }
            for name, descriptor in manifest.tensors.items()
            for entry in [tensor_metadata[name]]
        }

    def import_model_weights(self, ipc_handles: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
        tensors: dict[str, torch.Tensor] = {}
        for name, info in ipc_handles.items():
            if not isinstance(info, Mapping) or "handle" not in info:
                raise WeightManifestValidationError(
                    f"CUDA IPC handle entry for {name} must be a mapping with a handle"
                )
            tensors[name] = self._rebuild_cuda_ipc_tensor(info["handle"])
        return tensors

    def release(self, update_id: str) -> None:
        super().release(update_id)
        self._ipc_keepalive.pop(update_id, None)
        self.handle_registry.pop(update_id, None)

    def _attach_cuda_ipc_manifest(self, manifest: WeightUpdateManifest) -> _LocalUpdateRecord:
        if manifest.transport != self.transport:
            raise WeightManifestValidationError(
                f"transport mismatch: expected {self.transport}, got {manifest.transport}"
            )
        if not torch.cuda.is_available():
            raise WeightBridgeUnavailableError(
                "CUDA IPC weight bridge is unavailable because CUDA is not available."
            )
        bridge_metadata = manifest.metadata.get(_BRIDGE_METADATA_KEY)
        if not isinstance(bridge_metadata, Mapping):
            raise WeightManifestValidationError("CUDA IPC manifest is missing bridge metadata")
        if bridge_metadata.get("format") != _CUDA_IPC_FORMAT:
            raise WeightManifestValidationError(
                "CUDA IPC manifest has unsupported metadata format: "
                f"{bridge_metadata.get('format')}"
            )
        entries = bridge_metadata.get("tensors")
        if not isinstance(entries, Mapping) or set(entries) != set(manifest.tensors):
            raise WeightManifestValidationError("CUDA IPC manifest handle mismatch")

        tensors: dict[str, torch.Tensor] = {}
        for name, descriptor in manifest.tensors.items():
            entry = entries[name]
            if not isinstance(entry, Mapping) or "handle" not in entry:
                raise WeightManifestValidationError(
                    f"CUDA IPC manifest is missing handle for tensor {name}"
                )
            tensor = self._rebuild_cuda_ipc_tensor(entry["handle"])
            actual_descriptor = TensorDescriptor.from_tensor(name, tensor)
            if actual_descriptor != descriptor:
                if actual_descriptor.sha256 != descriptor.sha256:
                    raise WeightManifestValidationError(
                        f"tensor checksum mismatch for {name}: "
                        f"expected {descriptor.sha256}, got {actual_descriptor.sha256}"
                    )
                raise WeightManifestValidationError(
                    f"tensor descriptor mismatch for {name}: "
                    f"expected {descriptor}, got {actual_descriptor}"
                )
            tensors[name] = tensor

        record = _LocalUpdateRecord(
            manifest=manifest,
            tensors=tensors,
            state="published",
            import_count=0,
        )
        self._updates[manifest.update_id] = record
        return record

    def _rebuild_cuda_ipc_tensor(self, handle: Any) -> torch.Tensor:
        if not isinstance(handle, tuple) or len(handle) != 2:
            raise WeightManifestValidationError("CUDA IPC handle must be a (callable, args) tuple")
        rebuild, args = handle
        if not callable(rebuild):
            raise WeightManifestValidationError("CUDA IPC rebuild entry is not callable")
        try:
            list_args = list(args)
            if len(list_args) > 6:
                list_args[6] = int(torch.cuda.current_device())
            tensor = rebuild(*list_args)
            if not isinstance(tensor, torch.Tensor):
                raise WeightManifestValidationError(
                    f"CUDA IPC rebuild returned {type(tensor)!r}, not a torch.Tensor"
                )
            return tensor
        except WeightManifestValidationError:
            raise
        except Exception as exc:
            raise WeightBridgeUnavailableError(
                "CUDA IPC handle reconstruction failed in this runtime. "
                "This commonly indicates an unsupported WSL2/driver CUDA IPC path; "
                "use CUDAVMMTensorBridge (`cuda-vmm`) for same-node zero-copy when "
                "legacy CUDA IPC is unavailable. "
                f"Underlying error: {type(exc).__name__}: {exc}"
            ) from exc

    def _resolve_reduce_tensor(self) -> Any:
        if self._reduce_tensor_fn is not None:
            return self._reduce_tensor_fn
        try:
            from torch.multiprocessing.reductions import reduce_tensor
        except ImportError as exc:
            raise WeightBridgeUnavailableError(
                "PyTorch CUDA IPC reductions are unavailable in this runtime."
            ) from exc
        return reduce_tensor

    def _current_gpu_uuid(self) -> str:
        device_index = int(torch.cuda.current_device())
        return _resolve_cuda_device_uuid(device_index)


def make_weight_bridge(
    transport: str = "local-clone",
    *,
    source_worker: str = "local-training",
    source_rank: int = 0,
) -> WeightBridge:
    if transport in {"local", "local-clone"}:
        return LocalTensorCopyBridge(
            source_worker=source_worker,
            source_rank=source_rank,
        )
    if transport in {"shared-memory", "shm"}:
        return SharedMemoryTensorBridge(
            source_worker=source_worker,
            source_rank=source_rank,
        )
    if transport in {"cuda-vmm", "cuda-vmm-fd"}:
        return CUDAVMMTensorBridge(
            source_worker=source_worker,
            source_rank=source_rank,
        )
    if transport in {"cuda-ipc", "ipc"}:
        return IPCWeightBridge()
    if transport in {"multi-node", "rdma", "nccl-rdma"}:
        raise WeightBridgeUnavailableError(
            "multi-node/RDMA weight transport is not implemented. Use local-clone "
            "or shared-memory on one node, or add a production RDMA/NCCL transport."
        )
    raise WeightBridgeUnavailableError(f"unknown weight bridge transport: {transport}")


__all__ = [
    "CUDAVMMTensorBridge",
    "IPCWeightBridge",
    "LocalTensorCopyBridge",
    "SharedMemoryTensorBridge",
    "TensorDescriptor",
    "VLLMCUDAVMMExternalStorageAdapter",
    "VLLMCheckpointWeightReloadAdapter",
    "VLLMInProcessWeightReloadAdapter",
    "VLLMIPCWeightUpdateRequestBuilder",
    "VLLMWeightInstallAdapter",
    "WeightBridge",
    "WeightBridgeError",
    "WeightBridgeUnavailableError",
    "WeightConsumer",
    "WeightInstallAdapter",
    "WeightLayout",
    "WeightManifestValidationError",
    "WeightPublisher",
    "WeightUpdateManifest",
    "WeightUpdateRejectedError",
    "make_weight_bridge",
]
