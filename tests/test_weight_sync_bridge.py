# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import ctypes
from dataclasses import replace
import multiprocessing as mp
from queue import Empty

import pytest
import torch

import rl_engine.executors.bridge as bridge_module
from rl_engine.executors.bridge import (
    CUDAVMMTensorBridge,
    IPCWeightBridge,
    LocalTensorCopyBridge,
    SharedMemoryTensorBridge,
    TensorDescriptor,
    VLLMCUDAVMMExternalStorageAdapter,
    VLLMCheckpointWeightReloadAdapter,
    VLLMInProcessWeightReloadAdapter,
    VLLMIPCWeightUpdateRequestBuilder,
    VLLMWeightInstallAdapter,
    WeightBridgeUnavailableError,
    WeightManifestValidationError,
    WeightUpdateRejectedError,
    make_weight_bridge,
)


def _model() -> torch.nn.Module:
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.LayerNorm(4),
        torch.nn.Linear(4, 2, bias=False),
    )
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.fill_(index + 1)
    return model


def _model_with_empty_buffer() -> torch.nn.Module:
    model = _model()
    model.register_buffer("empty_buffer", torch.empty(0, 3))
    return model


class _FakeCudaDevice:
    type = "cuda"

    def __str__(self):
        return "cuda:0"


class _FakeCudaTensor:
    device = _FakeCudaDevice()
    dtype = torch.float32

    def __init__(self, shape):
        self.shape = torch.Size(shape)

    def detach(self):
        return self

    def contiguous(self):
        return self

    def numel(self):
        total = 1
        for dim in self.shape:
            total *= int(dim)
        return total


def _fake_cuda_tensors_for_manifest(manifest):
    return {
        name: _FakeCudaTensor(descriptor.shape) for name, descriptor in manifest.tensors.items()
    }


def test_local_bridge_publishes_manifest_with_tensor_metadata():
    model = _model()
    bridge = LocalTensorCopyBridge(source_worker="trainer-0", source_rank=0)

    manifest = bridge.publish(model, weight_version=7, metadata={"step": 3})

    assert manifest.source_worker == "trainer-0"
    assert manifest.source_rank == 0
    assert manifest.weight_version == 7
    assert manifest.transport == "local-clone"
    assert manifest.metadata["step"] == 3
    assert manifest.metadata["layout"] == {
        "kind": "full-state",
        "world_size": 1,
        "rank": 0,
        "tensor_parallel_size": 1,
        "data_parallel_size": 1,
        "zero_stage": 0,
        "node_count": 1,
        "rdma_enabled": False,
    }
    assert manifest.tensor_count == len(model.state_dict())
    assert manifest.total_nbytes > 0
    assert set(manifest.tensors) == set(model.state_dict())

    for name, tensor in model.state_dict().items():
        descriptor = manifest.tensors[name]
        assert descriptor == TensorDescriptor.from_tensor(name, tensor)
        assert len(descriptor.sha256) == 64


def test_local_bridge_import_acknowledge_and_release_lifecycle():
    model = _model()
    bridge = LocalTensorCopyBridge()
    manifest = bridge.publish(model, weight_version=1)

    imported = bridge.import_update(manifest)

    assert bridge.update_status(manifest.update_id) == "imported"
    assert set(imported) == set(model.state_dict())
    for name, original in model.state_dict().items():
        assert torch.equal(imported[name], original)
        assert imported[name].data_ptr() != original.data_ptr()

    bridge.acknowledge(manifest.update_id)
    assert bridge.active_weight_version == 1
    assert bridge.active_update_id == manifest.update_id
    assert bridge.update_status(manifest.update_id) == "acknowledged"

    bridge.release(manifest.update_id)
    bridge.release(manifest.update_id)
    assert bridge.update_status(manifest.update_id) == "released"
    assert bridge.active_update_id is None
    assert bridge.active_weight_version == 1


def test_reject_path_does_not_advance_active_version():
    bridge = LocalTensorCopyBridge()
    first = bridge.publish(_model(), weight_version=1)
    bridge.import_update(first)
    bridge.acknowledge(first.update_id)

    second = bridge.publish(_model(), weight_version=2)
    bridge.reject(second.update_id, "consumer validation failed")

    assert bridge.update_status(second.update_id) == "rejected"
    assert bridge.active_weight_version == 1
    assert bridge.active_update_id == first.update_id
    with pytest.raises(WeightUpdateRejectedError, match="rejected"):
        bridge.import_update(second)


def test_acknowledge_requires_successful_import():
    bridge = LocalTensorCopyBridge()
    manifest = bridge.publish(_model(), weight_version=1)

    with pytest.raises(WeightUpdateRejectedError, match="before import_update"):
        bridge.acknowledge(manifest.update_id)


def test_acknowledge_revalidates_imported_content_before_switching_active_version():
    bridge = LocalTensorCopyBridge()
    manifest = bridge.publish(_model(), weight_version=1)
    imported = bridge.import_update(manifest)
    assert imported
    name = next(iter(manifest.tensors))
    bridge._updates[manifest.update_id].tensors[name].add_(1)  # noqa: SLF001

    with pytest.raises(WeightManifestValidationError, match="checksum mismatch"):
        bridge.acknowledge(manifest.update_id)

    assert bridge.active_weight_version is None
    assert bridge.active_update_id is None


def test_metadata_mismatch_rejects_import_without_advancing_active_version():
    bridge = LocalTensorCopyBridge()
    manifest = bridge.publish(_model(), weight_version=1)
    name, descriptor = next(iter(manifest.tensors.items()))
    mutated_descriptor = replace(descriptor, shape=(999,))
    mutated_manifest = replace(
        manifest,
        tensors={**dict(manifest.tensors), name: mutated_descriptor},
    )

    with pytest.raises(WeightManifestValidationError, match="descriptor mismatch"):
        bridge.import_update(mutated_manifest)

    assert bridge.active_weight_version is None
    assert bridge.update_status(manifest.update_id) == "published"


def test_content_checksum_mismatch_rejects_import_without_advancing_active_version():
    bridge = LocalTensorCopyBridge()
    manifest = bridge.publish(_model(), weight_version=1)
    source_ptrs = bridge.debug_tensor_data_ptrs(manifest.update_id)
    assert source_ptrs
    name = next(iter(manifest.tensors))
    bridge._updates[manifest.update_id].tensors[name].add_(1)  # noqa: SLF001

    with pytest.raises(WeightManifestValidationError, match="checksum mismatch"):
        bridge.import_update(manifest)

    assert bridge.active_weight_version is None
    assert bridge.update_status(manifest.update_id) == "published"


def test_publish_requires_monotonic_weight_versions():
    bridge = LocalTensorCopyBridge()
    bridge.publish(_model(), weight_version=3)

    with pytest.raises(WeightManifestValidationError, match="monotonically"):
        bridge.publish(_model(), weight_version=3)


def test_publish_accepts_zero3_after_full_state_export():
    bridge = LocalTensorCopyBridge()

    manifest = bridge.publish(
        _model(),
        weight_version=1,
        metadata={"layout": {"zero_stage": 3}},
    )

    assert manifest.metadata["layout"]["kind"] == "full-state"
    assert manifest.metadata["layout"]["zero_stage"] == 3
    assert manifest.metadata["layout"]["world_size"] == 1
    assert manifest.metadata["layout"]["rank"] == 0


def test_publish_rejects_unsupported_weight_layouts():
    bridge = LocalTensorCopyBridge()

    with pytest.raises(WeightBridgeUnavailableError, match="weight layout"):
        bridge.publish(
            _model(),
            weight_version=1,
            metadata={"layout": {"kind": "zero-shard"}},
        )

    with pytest.raises(WeightBridgeUnavailableError, match="tensor-parallel"):
        bridge.publish(
            _model(),
            weight_version=1,
            metadata={"layout": {"tensor_parallel_size": 2}},
        )

    with pytest.raises(WeightBridgeUnavailableError, match="multi-node/RDMA"):
        bridge.publish(
            _model(),
            weight_version=1,
            metadata={"layout": {"node_count": 2}},
        )


def test_factory_and_cuda_ipc_unavailable_path_are_explicit():
    local = make_weight_bridge("local-clone")
    assert isinstance(local, LocalTensorCopyBridge)
    shared = make_weight_bridge("shared-memory")
    assert isinstance(shared, SharedMemoryTensorBridge)
    cuda_vmm = make_weight_bridge("cuda-vmm")
    assert isinstance(cuda_vmm, CUDAVMMTensorBridge)

    ipc = make_weight_bridge("cuda-ipc")
    assert isinstance(ipc, IPCWeightBridge)
    with pytest.raises(WeightBridgeUnavailableError, match="CUDA IPC"):
        ipc.export_model_handles(_model())

    with pytest.raises(WeightBridgeUnavailableError, match="multi-node/RDMA"):
        make_weight_bridge("rdma")


def test_shared_memory_bridge_imports_without_second_copy():
    model = _model()
    bridge = SharedMemoryTensorBridge()
    manifest = bridge.publish(model, weight_version=1)
    source_ptrs = bridge.debug_tensor_data_ptrs(manifest.update_id)

    imported = bridge.import_update(manifest)

    shared_metadata = manifest.metadata["weight_bridge"]
    assert shared_metadata["format"] == "python-multiprocessing-shared-memory-v1"
    assert set(shared_metadata["tensors"]) == set(manifest.tensors)
    for name, tensor in imported.items():
        assert tensor.data_ptr() == source_ptrs[name]

    first_name = next(iter(imported))
    imported[first_name].add_(10)
    with pytest.raises(WeightManifestValidationError, match="checksum mismatch"):
        bridge.import_update(manifest)

    bridge.release(manifest.update_id)


def _shared_memory_tensor_pickle_child(imported, queue):
    first_name = next(iter(imported))
    imported[first_name].add_(5)
    queue.put(float(imported[first_name].flatten()[0].item()))


def _shared_memory_manifest_child(manifest, queue):
    consumer = SharedMemoryTensorBridge(source_worker="rollout-worker", source_rank=1)
    imported = dict(consumer.import_update(manifest))
    try:
        consumer.acknowledge(manifest.update_id)
        first_name = next(iter(imported))
        value_before = float(imported[first_name].flatten()[0].item())
        queue.put(
            {
                "value_before": value_before,
                "active_weight_version": consumer.active_weight_version,
                "tensor_count": len(imported),
            }
        )
    finally:
        consumer.release(manifest.update_id)


def test_shared_memory_bridge_tensor_alias_is_visible_across_processes():
    bridge = SharedMemoryTensorBridge()
    manifest = bridge.publish(_model(), weight_version=1)
    imported = dict(bridge.import_update(manifest))
    first_name = next(iter(imported))
    before = float(imported[first_name].flatten()[0].item())

    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_shared_memory_tensor_pickle_child, args=(imported, queue))
    process.start()
    process.join(timeout=10)

    assert process.exitcode == 0
    child_value = queue.get(timeout=1)
    assert child_value == before + 5
    assert float(imported[first_name].flatten()[0].item()) == before + 5

    bridge.release(manifest.update_id)


def test_shared_memory_manifest_import_works_from_fresh_process():
    bridge = SharedMemoryTensorBridge()
    manifest = bridge.publish(_model(), weight_version=42)
    imported = dict(bridge.import_update(manifest))
    first_name = next(iter(imported))
    before = float(imported[first_name].flatten()[0].item())

    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_shared_memory_manifest_child, args=(manifest, queue))
    process.start()
    process.join(timeout=10)

    assert process.exitcode == 0
    try:
        child_result = queue.get(timeout=1)
    except Empty as exc:
        raise AssertionError("child did not report shared-memory import result") from exc

    assert child_result["value_before"] == before
    assert child_result["active_weight_version"] == 42
    assert child_result["tensor_count"] == manifest.tensor_count
    assert float(imported[first_name].flatten()[0].item()) == before

    bridge.release(manifest.update_id)
    consumer_after_release = SharedMemoryTensorBridge()
    with pytest.raises(FileNotFoundError):
        consumer_after_release.import_update(manifest)


def _shared_memory_manifest_tamper_child(manifest, queue):
    consumer = SharedMemoryTensorBridge(source_worker="rollout-worker", source_rank=1)
    imported = dict(consumer.import_update(manifest))
    first_name = next(iter(imported))
    imported[first_name].add_(11)
    try:
        consumer.import_update(manifest)
    except WeightManifestValidationError as exc:
        queue.put(
            {
                "error": str(exc),
                "status": consumer.update_status(manifest.update_id),
                "active_weight_version": consumer.active_weight_version,
            }
        )
    finally:
        consumer.release(manifest.update_id)


def test_shared_memory_manifest_checksum_rejects_cross_process_tamper():
    bridge = SharedMemoryTensorBridge()
    manifest = bridge.publish(_model(), weight_version=42)
    imported = dict(bridge.import_update(manifest))
    first_name = next(iter(imported))
    before = float(imported[first_name].flatten()[0].item())

    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_shared_memory_manifest_tamper_child, args=(manifest, queue))
    process.start()
    process.join(timeout=10)

    assert process.exitcode == 0
    child_result = queue.get(timeout=1)

    assert "checksum mismatch" in child_result["error"]
    assert child_result["status"] == "imported"
    assert child_result["active_weight_version"] is None
    assert float(imported[first_name].flatten()[0].item()) == before + 11

    with pytest.raises(WeightManifestValidationError, match="checksum mismatch"):
        bridge.import_update(manifest)

    bridge.release(manifest.update_id)


def test_shared_memory_publish_rejects_reserved_metadata_key():
    bridge = SharedMemoryTensorBridge()

    with pytest.raises(WeightManifestValidationError, match="reserved"):
        bridge.publish(_model(), weight_version=1, metadata={"weight_bridge": {}})


def test_shared_memory_manifest_rejects_handle_mismatch():
    bridge = SharedMemoryTensorBridge()
    manifest = bridge.publish(_model(), weight_version=1)
    shared_metadata = manifest.metadata["weight_bridge"]
    mutated_manifest = replace(
        manifest,
        metadata={
            **manifest.metadata,
            "weight_bridge": {
                **shared_metadata,
                "tensors": {
                    **shared_metadata["tensors"],
                    "unexpected.weight": {
                        "name": None,
                        "size": 0,
                        "storage_numel": 0,
                        "storage_nbytes": 0,
                        "storage_offset": 0,
                    },
                },
            },
        },
    )

    consumer = SharedMemoryTensorBridge()
    with pytest.raises(WeightManifestValidationError, match="handle mismatch"):
        consumer.import_update(mutated_manifest)

    bridge.release(manifest.update_id)


def test_shared_memory_manifest_handles_empty_tensors():
    bridge = SharedMemoryTensorBridge()
    manifest = bridge.publish(_model_with_empty_buffer(), weight_version=1)

    consumer = SharedMemoryTensorBridge()
    imported = consumer.import_update(manifest)

    assert tuple(imported["empty_buffer"].shape) == (0, 3)
    assert imported["empty_buffer"].numel() == 0
    assert imported["empty_buffer"].device.type == "cpu"

    consumer.release(manifest.update_id)
    bridge.release(manifest.update_id)


def test_cuda_ipc_bridge_remains_blocked_until_snapshot_semantics_are_validated():
    bridge = IPCWeightBridge()
    model = _model().to("cuda") if torch.cuda.is_available() else _model()

    with pytest.raises(WeightBridgeUnavailableError, match="not production-ready|unavailable"):
        bridge.publish(model, weight_version=1)


def test_cuda_vmm_bridge_requires_cuda_tensors_before_loading_backend():
    backend_calls = []
    bridge = CUDAVMMTensorBridge(backend_factory=lambda: backend_calls.append("loaded"))

    with pytest.raises(WeightBridgeUnavailableError, match="requires CUDA tensors"):
        bridge.publish(_model(), weight_version=1)

    assert backend_calls == []


def test_cuda_vmm_bridge_rejects_reserved_metadata_key():
    bridge = CUDAVMMTensorBridge(backend_factory=lambda: object())

    with pytest.raises(WeightManifestValidationError, match="reserved"):
        bridge.publish(_model(), weight_version=1, metadata={"weight_bridge": {}})


def test_vllm_weight_install_adapter_uses_explicit_capability():
    bridge = LocalTensorCopyBridge()
    manifest = bridge.publish(_model(), weight_version=5)
    tensors = bridge.import_update(manifest)
    calls = []

    adapter = VLLMWeightInstallAdapter(
        object(),
        install_callable=lambda incoming_manifest, incoming_tensors: calls.append(
            (incoming_manifest.weight_version, set(incoming_tensors))
        ),
    )

    adapter.install(manifest, tensors)

    assert adapter.active_weight_version == 5
    assert adapter.active_update_id == manifest.update_id
    assert calls == [(5, set(manifest.tensors))]


def test_vllm_weight_install_adapter_requires_verified_capability():
    bridge = LocalTensorCopyBridge()
    manifest = bridge.publish(_model(), weight_version=5)
    tensors = bridge.import_update(manifest)
    adapter = VLLMWeightInstallAdapter(object())

    with pytest.raises(WeightBridgeUnavailableError, match="vLLM weight install"):
        adapter.install(manifest, tensors)


def test_vllm_weight_install_adapter_uses_public_update_weights_request_builder():
    bridge = LocalTensorCopyBridge()
    manifest = bridge.publish(_model(), weight_version=6)
    tensors = bridge.import_update(manifest)
    calls = []

    class FakeVLLMEngine:
        def update_weights(self, request):
            calls.append(request)

    adapter = VLLMWeightInstallAdapter(
        FakeVLLMEngine(),
        request_builder=lambda incoming_manifest, incoming_tensors: {
            "update_id": incoming_manifest.update_id,
            "weight_version": incoming_manifest.weight_version,
            "tensor_count": len(incoming_tensors),
        },
    )

    adapter.install(manifest, tensors)

    assert adapter.active_weight_version == 6
    assert calls == [
        {
            "update_id": manifest.update_id,
            "weight_version": 6,
            "tensor_count": manifest.tensor_count,
        }
    ]


def test_vllm_update_weights_api_requires_explicit_request_builder():
    bridge = LocalTensorCopyBridge()
    manifest = bridge.publish(_model(), weight_version=6)
    tensors = bridge.import_update(manifest)

    class FakeVLLMEngine:
        def update_weights(self, request):
            del request

    adapter = VLLMWeightInstallAdapter(FakeVLLMEngine())

    with pytest.raises(WeightBridgeUnavailableError, match="request_builder"):
        adapter.install(manifest, tensors)


def test_vllm_ipc_request_builder_creates_public_update_request_for_cuda_tensors():
    manifest = LocalTensorCopyBridge().publish(_model(), weight_version=1)
    cuda_tensors = _fake_cuda_tensors_for_manifest(manifest)
    fake_handles = []

    def fake_reduce_tensor(tensor):
        fake_handles.append((tuple(tensor.shape), str(tensor.dtype), str(tensor.device)))
        return ("rebuild_cuda_tensor", (tensor.numel(),))

    builder = VLLMIPCWeightUpdateRequestBuilder(
        reduce_tensor_fn=fake_reduce_tensor,
        gpu_uuid="GPU-test",
    )

    request = builder(manifest, cuda_tensors)

    update_info = request["update_info"]
    assert update_info["names"] == list(manifest.tensors)
    assert update_info["dtype_names"] == ["float32"] * manifest.tensor_count
    assert update_info["shapes"] == [
        list(descriptor.shape) for descriptor in manifest.tensors.values()
    ]
    assert update_info["ipc_handles"] == [
        {"GPU-test": ("rebuild_cuda_tensor", (descriptor.numel,))}
        for descriptor in manifest.tensors.values()
    ]
    assert update_info["is_checkpoint_format"] is True
    assert len(fake_handles) == manifest.tensor_count

    builder.release(manifest.update_id)
    assert builder._keepalive == {}  # noqa: SLF001


def test_vllm_ipc_request_builder_requires_cuda_tensors():
    bridge = LocalTensorCopyBridge()
    manifest = bridge.publish(_model(), weight_version=1)
    tensors = bridge.import_update(manifest)
    builder = VLLMIPCWeightUpdateRequestBuilder(
        reduce_tensor_fn=lambda tensor: ("unused", ()),
        gpu_uuid="GPU-test",
    )

    with pytest.raises(WeightBridgeUnavailableError, match="requires CUDA tensors"):
        builder(manifest, tensors)


def test_vllm_install_adapter_releases_ipc_request_builder_keepalive():
    manifest = LocalTensorCopyBridge().publish(_model(), weight_version=1)
    cuda_tensors = _fake_cuda_tensors_for_manifest(manifest)
    builder = VLLMIPCWeightUpdateRequestBuilder(
        reduce_tensor_fn=lambda tensor: ("rebuild_cuda_tensor", (tensor.numel(),)),
        gpu_uuid="GPU-test",
    )
    calls = []

    class FakeVLLMEngine:
        def update_weights(self, request):
            calls.append(request)

    adapter = VLLMWeightInstallAdapter(FakeVLLMEngine(), request_builder=builder)
    adapter.install(manifest, cuda_tensors)

    assert manifest.update_id in builder._keepalive  # noqa: SLF001
    adapter.release(manifest.update_id)

    assert calls
    assert builder._keepalive == {}  # noqa: SLF001
    assert adapter.active_update_id is None


def test_vllm_install_adapter_releases_ipc_keepalive_when_update_fails():
    manifest = LocalTensorCopyBridge().publish(_model(), weight_version=1)
    builder = VLLMIPCWeightUpdateRequestBuilder(
        reduce_tensor_fn=lambda tensor: ("rebuild_cuda_tensor", (tensor.numel(),)),
        gpu_uuid="GPU-test",
    )

    class FailingVLLMEngine:
        def update_weights(self, request):
            assert request["update_info"]["ipc_handles"]
            raise RuntimeError("vLLM update failed")

    adapter = VLLMWeightInstallAdapter(FailingVLLMEngine(), request_builder=builder)

    with pytest.raises(RuntimeError, match="vLLM update failed"):
        adapter.install(manifest, _fake_cuda_tensors_for_manifest(manifest))

    assert builder._keepalive == {}  # noqa: SLF001
    assert adapter.active_update_id is None


def test_vllm_in_process_reload_adapter_uses_collective_rpc_reload_weights():
    bridge = LocalTensorCopyBridge()
    manifest = bridge.publish(_model(), weight_version=8)
    tensors = bridge.import_update(manifest)
    calls = []

    class FakeLLMEngine:
        def collective_rpc(self, method, kwargs):
            calls.append((method, kwargs))

    class FakeLLM:
        llm_engine = FakeLLMEngine()

    adapter = VLLMInProcessWeightReloadAdapter(
        FakeLLM(),
        target_dtype=torch.float16,
        synchronize_cuda=False,
    )

    adapter.install(manifest, tensors)

    assert adapter.active_weight_version == 8
    assert adapter.active_update_id == manifest.update_id
    assert len(calls) == 1
    method, kwargs = calls[0]
    assert method == "reload_weights"
    assert kwargs["is_checkpoint_format"] is True
    weights = kwargs["weights_iterator"]
    assert [name for name, _ in weights] == list(manifest.tensors)
    assert all(tensor.dtype == torch.float16 for _, tensor in weights)
    assert all(tensor.is_contiguous() for _, tensor in weights)

    adapter.release(manifest.update_id)
    assert adapter.active_update_id is None


def test_vllm_in_process_reload_adapter_uses_direct_reload_weights_capability():
    bridge = LocalTensorCopyBridge()
    manifest = bridge.publish(_model(), weight_version=9)
    tensors = bridge.import_update(manifest)
    calls = []

    class FakeReloadEngine:
        def reload_weights(self, *, weights_iterator, is_checkpoint_format):
            calls.append((weights_iterator, is_checkpoint_format))

    adapter = VLLMInProcessWeightReloadAdapter(
        FakeReloadEngine(),
        target_device="cpu",
        is_checkpoint_format=False,
        synchronize_cuda=False,
    )

    adapter.install(manifest, tensors)

    assert adapter.active_weight_version == 9
    assert len(calls) == 1
    weights, is_checkpoint_format = calls[0]
    assert is_checkpoint_format is False
    assert [name for name, _ in weights] == list(manifest.tensors)
    assert all(tensor.device.type == "cpu" for _, tensor in weights)


def test_vllm_in_process_reload_adapter_requires_complete_manifest():
    bridge = LocalTensorCopyBridge()
    manifest = bridge.publish(_model(), weight_version=10)
    tensors = dict(bridge.import_update(manifest))
    tensors.pop(next(iter(tensors)))
    adapter = VLLMInProcessWeightReloadAdapter(object(), synchronize_cuda=False)

    with pytest.raises(WeightManifestValidationError, match="tensor set mismatch"):
        adapter.install(manifest, tensors)


def test_vllm_in_process_reload_adapter_does_not_activate_failed_reload():
    bridge = LocalTensorCopyBridge()
    manifest = bridge.publish(_model(), weight_version=11)
    tensors = bridge.import_update(manifest)

    class FailingReloadEngine:
        def reload_weights(self, *, weights_iterator, is_checkpoint_format):
            assert weights_iterator
            assert is_checkpoint_format is True
            raise RuntimeError("reload failed")

    adapter = VLLMInProcessWeightReloadAdapter(
        FailingReloadEngine(),
        synchronize_cuda=False,
    )

    with pytest.raises(RuntimeError, match="reload failed"):
        adapter.install(manifest, tensors)

    assert adapter.active_weight_version is None
    assert adapter.active_update_id is None


def test_vllm_checkpoint_reload_adapter_uses_manifest_weights_path():
    bridge = LocalTensorCopyBridge()
    manifest = bridge.publish(
        _model(),
        weight_version=12,
        metadata={"vllm_weights_path": "/models/checkpoint-v12"},
    )
    tensors = bridge.import_update(manifest)
    calls = []

    class FakeLLMEngine:
        def collective_rpc(self, method, kwargs):
            calls.append((method, kwargs))

    class FakeLLM:
        llm_engine = FakeLLMEngine()

    adapter = VLLMCheckpointWeightReloadAdapter(
        FakeLLM(),
        synchronize_cuda=False,
    )

    adapter.install(manifest, tensors)

    assert adapter.active_weight_version == 12
    assert adapter.active_update_id == manifest.update_id
    assert calls == [
        (
            "reload_weights",
            {
                "weights_path": "/models/checkpoint-v12",
                "is_checkpoint_format": True,
            },
        )
    ]


def test_vllm_checkpoint_reload_adapter_uses_resolver():
    bridge = LocalTensorCopyBridge()
    manifest = bridge.publish(_model(), weight_version=13)
    tensors = bridge.import_update(manifest)
    calls = []

    class FakeReloadEngine:
        def reload_weights(self, *, weights_path, is_checkpoint_format):
            calls.append((weights_path, is_checkpoint_format))

    adapter = VLLMCheckpointWeightReloadAdapter(
        FakeReloadEngine(),
        weights_path_resolver=lambda incoming_manifest, incoming_tensors: (
            f"/models/v{incoming_manifest.weight_version}-{len(incoming_tensors)}"
        ),
        is_checkpoint_format=False,
        synchronize_cuda=False,
    )

    adapter.install(manifest, tensors)

    assert calls == [("/models/v13-5", False)]
    adapter.release(manifest.update_id)
    assert adapter.active_update_id is None


def test_vllm_checkpoint_reload_adapter_requires_weights_path():
    bridge = LocalTensorCopyBridge()
    manifest = bridge.publish(_model(), weight_version=14)
    tensors = bridge.import_update(manifest)
    adapter = VLLMCheckpointWeightReloadAdapter(object(), synchronize_cuda=False)

    with pytest.raises(WeightBridgeUnavailableError, match="weights_path"):
        adapter.install(manifest, tensors)


def test_vllm_cuda_vmm_external_storage_adapter_requires_cuda_vmm_manifest():
    bridge = LocalTensorCopyBridge()
    manifest = bridge.publish(_model(), weight_version=15)
    adapter = VLLMCUDAVMMExternalStorageAdapter(object(), synchronize_cuda=False)

    with pytest.raises(WeightBridgeUnavailableError, match="cuda-vmm"):
        adapter.install(manifest, bridge.import_update(manifest))


def test_vllm_cuda_vmm_external_storage_adapter_rebinds_matching_parameters(monkeypatch):
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.LayerNorm(4))
    replacement = _model()
    bridge = LocalTensorCopyBridge(source_worker="trainer", source_rank=0)
    local_manifest = bridge.publish(replacement, weight_version=16)
    tensors = bridge.import_update(local_manifest)
    manifest = replace(local_manifest, transport="cuda-vmm")
    calls = []

    class FakeCUDAVMMTensorBridge:
        transport = "cuda-vmm"

        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def import_update(self, incoming_manifest):
            calls.append(("import", incoming_manifest.update_id))
            return tensors

        def acknowledge(self, update_id):
            calls.append(("ack", update_id))

        def release(self, update_id):
            calls.append(("release", update_id))

    monkeypatch.setattr(bridge_module, "CUDAVMMTensorBridge", FakeCUDAVMMTensorBridge)

    class FakeLLMEngine:
        def apply_model(self, func):
            return [func(model)]

    adapter = VLLMCUDAVMMExternalStorageAdapter(
        FakeLLMEngine(),
        require_all_parameters=False,
        synchronize_cuda=False,
    )

    adapter.install(manifest, {})

    assert adapter.active_weight_version == 16
    assert adapter.last_result[0]["zero_copy"] is True
    rebound = set(adapter.last_result[0]["rebound"])
    for name in rebound:
        assert model.get_parameter(name).data_ptr() == tensors[name].data_ptr()

    adapter.release(manifest.update_id)

    assert ("ack", manifest.update_id) in calls
    assert ("release", manifest.update_id) in calls
    assert adapter.active_update_id is None


def test_vllm_cuda_vmm_external_storage_adapter_rolls_back_partial_rebind(monkeypatch):
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.LayerNorm(4))
    bridge = LocalTensorCopyBridge(source_worker="trainer", source_rank=0)
    local_manifest = bridge.publish(model, weight_version=17)
    tensors = dict(bridge.import_update(local_manifest))
    manifest = replace(local_manifest, transport="cuda-vmm")
    original_ptrs = {name: parameter.data_ptr() for name, parameter in model.named_parameters()}
    calls = []

    bad_tensors = dict(tensors)
    bad_tensors["1.weight"] = torch.ones(5)

    class FakeCUDAVMMTensorBridge:
        transport = "cuda-vmm"

        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def import_update(self, incoming_manifest):
            calls.append(("import", incoming_manifest.update_id))
            return bad_tensors

        def acknowledge(self, update_id):
            calls.append(("ack", update_id))

        def release(self, update_id):
            calls.append(("release", update_id))

    monkeypatch.setattr(bridge_module, "CUDAVMMTensorBridge", FakeCUDAVMMTensorBridge)

    class FakeLLMEngine:
        def apply_model(self, func):
            return [func(model)]

    adapter = VLLMCUDAVMMExternalStorageAdapter(FakeLLMEngine(), synchronize_cuda=False)

    with pytest.raises(WeightManifestValidationError, match="shape mismatch"):
        adapter.install(manifest, {})

    assert {
        name: parameter.data_ptr() for name, parameter in model.named_parameters()
    } == original_ptrs
    assert ("release", manifest.update_id) in calls
    assert ("ack", manifest.update_id) not in calls
    assert not hasattr(model, "_kernel_align_cuda_vmm_bridge")


def test_cuda_vmm_driver_backend_prefers_env_libcuda(monkeypatch):
    calls = []

    def fake_cdll(path):
        calls.append(path)
        if path == "/custom/libcuda.so.1":
            return object()
        raise OSError(f"missing {path}")

    monkeypatch.setenv("KERNEL_ALIGN_LIBCUDA_PATH", "/custom/libcuda.so.1")
    monkeypatch.setattr(ctypes, "CDLL", fake_cdll)

    assert bridge_module._CUDAVMMDriverBackend._find_libcuda(None) == "/custom/libcuda.so.1"
    assert calls == ["/custom/libcuda.so.1"]
