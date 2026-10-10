# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import multiprocessing as mp
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_engine.integrations.common.weights import (  # noqa: E402
    CUDAVMMTensorBridge,
    IPCWeightBridge,
    LocalTensorCopyBridge,
    SharedMemoryTensorBridge,
    VLLMCUDAVMMExternalStorageAdapter,
    VLLMIPCWeightUpdateRequestBuilder,
    VLLMWeightInstallAdapter,
    WeightBridgeUnavailableError,
    WeightManifestValidationError,
)
from rl_engine.integrations.engines.rollout.interface import RolloutExecutor  # noqa: E402

CSV_COLUMNS = [
    "timestamp",
    "candidate_id",
    "mode",
    "status",
    "environment",
    "tensor_count",
    "total_nbytes",
    "publish_ms",
    "import_ms",
    "ack_ms",
    "release_ms",
    "active_weight_version",
    "notes",
]


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _environment() -> str:
    cuda = torch.version.cuda or ""
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return f"torch={torch.__version__};cuda={cuda};gpu={device};gpu_mem_gb={memory_gb:.2f}"
    return f"torch={torch.__version__};cuda={cuda};gpu=none"


def _write_csv_row(row: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    exists = output.exists() and output.stat().st_size > 0
    if exists:
        first_line = output.read_text(encoding="utf-8").splitlines()[0]
        if first_line.split(",") != CSV_COLUMNS:
            exists = False
    with output.open("a" if exists else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _model(hidden_dim: int, layers: int) -> torch.nn.Module:
    modules: list[torch.nn.Module] = []
    for _ in range(layers):
        modules.append(torch.nn.Linear(hidden_dim, hidden_dim))
        modules.append(torch.nn.LayerNorm(hidden_dim))
    modules.append(torch.nn.Linear(hidden_dim, hidden_dim, bias=False))
    return torch.nn.Sequential(*modules)


def _run_local(args: argparse.Namespace) -> dict[str, Any]:
    model = _model(args.hidden_dim, args.layers)
    bridge = LocalTensorCopyBridge(source_worker="benchmark-training", source_rank=0)

    publish_started = time.perf_counter()
    manifest = bridge.publish(
        model,
        weight_version=args.weight_version,
        metadata={"benchmark": "weight_sync_bridge"},
    )
    publish_finished = time.perf_counter()

    import_started = time.perf_counter()
    tensors = bridge.import_update(manifest)
    import_finished = time.perf_counter()

    ack_started = time.perf_counter()
    bridge.acknowledge(manifest.update_id)
    ack_finished = time.perf_counter()

    release_started = time.perf_counter()
    bridge.release(manifest.update_id)
    release_finished = time.perf_counter()

    if len(tensors) != manifest.tensor_count:
        raise RuntimeError("imported tensor count did not match manifest")

    return {
        "timestamp": _timestamp(),
        "candidate_id": "issue-13-candidate-1",
        "mode": "local",
        "status": "pass",
        "environment": _environment(),
        "tensor_count": manifest.tensor_count,
        "total_nbytes": manifest.total_nbytes,
        "publish_ms": (publish_finished - publish_started) * 1000,
        "import_ms": (import_finished - import_started) * 1000,
        "ack_ms": (ack_finished - ack_started) * 1000,
        "release_ms": (release_finished - release_started) * 1000,
        "active_weight_version": bridge.active_weight_version,
        "notes": "Local clone transport validates manifest lifecycle; not zero-copy evidence.",
    }


def _shared_memory_manifest_child(manifest: Any, queue: Any) -> None:
    consumer = SharedMemoryTensorBridge(source_worker="benchmark-rollout", source_rank=1)
    imported = dict(consumer.import_update(manifest))
    try:
        consumer.acknowledge(manifest.update_id)
        first_name = next(iter(imported))
        before = float(imported[first_name].flatten()[0].item())
        queue.put(
            {
                "before": before,
                "active_weight_version": consumer.active_weight_version,
                "tensor_count": len(imported),
            }
        )
    finally:
        consumer.release(manifest.update_id)


def _shared_memory_manifest_tamper_child(manifest: Any, queue: Any) -> None:
    consumer = SharedMemoryTensorBridge(source_worker="benchmark-rollout", source_rank=1)
    imported = dict(consumer.import_update(manifest))
    first_name = next(iter(imported))
    imported[first_name].add_(13)
    try:
        consumer.acknowledge(manifest.update_id)
    except WeightManifestValidationError as exc:
        queue.put({"blocked": True, "error": str(exc)})
    finally:
        consumer.release(manifest.update_id)


def _run_shared_memory(args: argparse.Namespace) -> dict[str, Any]:
    model = _model(args.hidden_dim, args.layers)
    bridge = SharedMemoryTensorBridge(source_worker="benchmark-training", source_rank=0)

    publish_started = time.perf_counter()
    manifest = bridge.publish(
        model,
        weight_version=args.weight_version,
        metadata={"benchmark": "weight_sync_bridge"},
    )
    publish_finished = time.perf_counter()

    import_started = time.perf_counter()
    tensors = dict(bridge.import_update(manifest))
    import_finished = time.perf_counter()

    source_ptrs = bridge.debug_tensor_data_ptrs(manifest.update_id)
    alias_count = sum(
        1 for name, tensor in tensors.items() if int(tensor.data_ptr()) == source_ptrs[name]
    )
    shared_handle_count = len(manifest.metadata["weight_bridge"]["tensors"])

    first_name = next(iter(tensors))
    before = float(tensors[first_name].flatten()[0].item())
    context = mp.get_context("spawn")

    manifest_queue = context.Queue()
    manifest_process_started = time.perf_counter()
    manifest_process = context.Process(
        target=_shared_memory_manifest_child,
        args=(manifest, manifest_queue),
    )
    manifest_process.start()
    manifest_process.join(timeout=10)
    if manifest_process.exitcode != 0:
        raise RuntimeError(f"shared-memory manifest child exited with {manifest_process.exitcode}")
    manifest_child_result = manifest_queue.get(timeout=1)
    manifest_process_finished = time.perf_counter()
    parent_after_manifest_child = float(tensors[first_name].flatten()[0].item())
    if (
        abs(manifest_child_result["before"] - before) > 1e-5
        or abs(parent_after_manifest_child - before) > 1e-5
        or manifest_child_result["active_weight_version"] != manifest.weight_version
        or manifest_child_result["tensor_count"] != manifest.tensor_count
    ):
        raise RuntimeError("shared-memory manifest import was not visible across processes")

    ack_started = time.perf_counter()
    bridge.acknowledge(manifest.update_id)
    ack_finished = time.perf_counter()

    release_started = time.perf_counter()
    bridge.release(manifest.update_id)
    release_finished = time.perf_counter()

    tamper_bridge = SharedMemoryTensorBridge(source_worker="benchmark-training", source_rank=0)
    tamper_manifest = tamper_bridge.publish(
        model,
        weight_version=args.weight_version + 1,
        metadata={"benchmark": "weight_sync_bridge_tamper"},
    )
    tamper_queue = context.Queue()
    tamper_process = context.Process(
        target=_shared_memory_manifest_tamper_child,
        args=(tamper_manifest, tamper_queue),
    )
    tamper_process.start()
    tamper_process.join(timeout=10)
    if tamper_process.exitcode != 0:
        raise RuntimeError(f"shared-memory tamper child exited with {tamper_process.exitcode}")
    tamper_result = tamper_queue.get(timeout=1)
    tamper_bridge.release(tamper_manifest.update_id)
    if not tamper_result.get("blocked"):
        raise RuntimeError("shared-memory checksum did not block tampered update")

    return {
        "timestamp": _timestamp(),
        "candidate_id": "issue-13-candidate-2",
        "mode": "shared-memory",
        "status": "pass",
        "environment": _environment(),
        "tensor_count": manifest.tensor_count,
        "total_nbytes": manifest.total_nbytes,
        "publish_ms": (publish_finished - publish_started) * 1000,
        "import_ms": (import_finished - import_started) * 1000,
        "ack_ms": (ack_finished - ack_started) * 1000,
        "release_ms": (release_finished - release_started) * 1000,
        "active_weight_version": bridge.active_weight_version,
        "notes": (
            "Same-node shared-memory zero-copy smoke passed; "
            f"alias_count={alias_count}; shared_handle_count={shared_handle_count}; "
            "checksum_tamper_blocked=True; "
            "manifest_attach_process_visible_ms="
            f"{(manifest_process_finished - manifest_process_started) * 1000:.4f}"
        ),
    }


def _run_cuda_ipc() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise WeightBridgeUnavailableError("CUDA IPC smoke requires torch CUDA availability")

    bridge = IPCWeightBridge(source_worker="benchmark-cuda-training", source_rank=0)
    model = _model(8, 1).to("cuda")
    manifest = None
    imported: dict[str, torch.Tensor] = {}
    publish_started = time.perf_counter()
    result: dict[str, Any] | None = None
    try:
        manifest = bridge.publish(
            model,
            weight_version=1,
            metadata={"benchmark": "weight_sync_bridge_cuda_ipc"},
        )
        publish_finished = time.perf_counter()

        import_started = time.perf_counter()
        imported = dict(bridge.import_update(manifest))
        import_finished = time.perf_counter()
        first_name = next(iter(imported))
        parent_before = imported[first_name].detach().cpu().flatten()[:4].tolist()

        ack_started = time.perf_counter()
        bridge.acknowledge(manifest.update_id)
        ack_finished = time.perf_counter()

        context = mp.get_context("spawn")
        queue = context.Queue()
        process_started = time.perf_counter()
        process = context.Process(target=_cuda_ipc_manifest_child, args=(manifest, queue))
        process.start()
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            raise WeightBridgeUnavailableError("CUDA IPC child process timed out")
        if process.exitcode != 0 and queue.empty():
            raise WeightBridgeUnavailableError(
                f"CUDA IPC child process exited with {process.exitcode}"
            )
        child_result = queue.get(timeout=1)
        process_finished = time.perf_counter()
        if child_result.get("status") != "pass":
            raise WeightBridgeUnavailableError(
                "CUDA IPC child import failed: "
                f"{child_result.get('error', child_result.get('type', 'unknown error'))}"
            )

        torch.cuda.synchronize()
        parent_after = imported[first_name].detach().cpu().flatten()[:4].tolist()
        if parent_after == parent_before:
            raise WeightBridgeUnavailableError(
                "CUDA IPC child mutation was not visible to the publisher snapshot"
            )

        release_started = time.perf_counter()
        bridge.release(manifest.update_id)
        release_finished = time.perf_counter()

        return {
            "timestamp": _timestamp(),
            "candidate_id": "issue-13-candidate-3",
            "mode": "cuda-ipc",
            "status": "pass",
            "environment": _environment(),
            "tensor_count": manifest.tensor_count,
            "total_nbytes": manifest.total_nbytes,
            "publish_ms": (publish_finished - publish_started) * 1000,
            "import_ms": (import_finished - import_started) * 1000,
            "ack_ms": (ack_finished - ack_started) * 1000,
            "release_ms": (release_finished - release_started) * 1000,
            "active_weight_version": bridge.active_weight_version,
            "notes": (
                "Legacy CUDA IPC smoke passed; child process rebuilt manifest handles, "
                f"mutated {first_name} from {child_result['before']} to "
                f"{child_result['after']}, and parent observed {parent_before} -> "
                f"{parent_after}; manifest_attach_process_visible_ms="
                f"{(process_finished - process_started) * 1000:.4f}"
            ),
        }
    except WeightBridgeUnavailableError as exc:
        notes = str(exc)
    finally:
        imported.clear()
        if manifest is not None:
            bridge.release(manifest.update_id)

    return {
        "timestamp": _timestamp(),
        "candidate_id": "issue-13-candidate-3",
        "mode": "cuda-ipc",
        "status": "blocked",
        "environment": _environment(),
        "tensor_count": manifest.tensor_count if manifest is not None else 0,
        "total_nbytes": manifest.total_nbytes if manifest is not None else 0,
        "publish_ms": (
            (publish_finished - publish_started) * 1000 if "publish_finished" in locals() else ""
        ),
        "import_ms": "",
        "ack_ms": "",
        "release_ms": "",
        "active_weight_version": "",
        "notes": notes,
    }


def _cuda_ipc_manifest_child(manifest: Any, queue: Any) -> None:
    consumer = IPCWeightBridge(source_worker="benchmark-rollout", source_rank=1)
    imported: dict[str, torch.Tensor] = {}
    tensor = None
    try:
        imported = dict(consumer.import_update(manifest))
        consumer.acknowledge(manifest.update_id)
        first_name = next(iter(imported))
        tensor = imported[first_name]
        before = tensor.detach().cpu().flatten()[:4].tolist()
        tensor.add_(5)
        torch.cuda.synchronize()
        after = tensor.detach().cpu().flatten()[:4].tolist()
        queue.put(
            {
                "status": "pass",
                "before": before,
                "after": after,
                "data_ptr": int(tensor.data_ptr()),
                "active_weight_version": consumer.active_weight_version,
            }
        )
    except Exception as exc:
        queue.put(
            {
                "status": "blocked",
                "type": type(exc).__name__,
                "error": str(exc),
            }
        )
    finally:
        del tensor
        imported.clear()
        consumer.release(manifest.update_id)


def _cuda_vmm_manifest_child(manifest: Any, queue: Any) -> None:
    consumer = CUDAVMMTensorBridge(source_worker="benchmark-rollout", source_rank=1)
    imported = dict(consumer.import_update(manifest))
    tensor = None
    try:
        consumer.acknowledge(manifest.update_id)
        first_name = next(iter(imported))
        tensor = imported[first_name]
        before = tensor.detach().cpu().flatten()[:4].tolist()
        tensor.add_(10)
        torch.cuda.synchronize()
        after = tensor.detach().cpu().flatten()[:4].tolist()
        queue.put(
            {
                "before": before,
                "after": after,
                "data_ptr": int(tensor.data_ptr()),
                "active_weight_version": consumer.active_weight_version,
            }
        )
    finally:
        del tensor
        imported.clear()
        consumer.release(manifest.update_id)


def _run_cuda_vmm(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise WeightBridgeUnavailableError("CUDA VMM smoke requires torch CUDA availability")

    model = _model(args.hidden_dim, args.layers).to("cuda")
    publisher = CUDAVMMTensorBridge(source_worker="benchmark-training", source_rank=0)

    manifest = None
    parent_imported: dict[str, torch.Tensor] = {}
    try:
        publish_started = time.perf_counter()
        manifest = publisher.publish(
            model,
            weight_version=args.weight_version,
            metadata={"benchmark": "weight_sync_bridge_cuda_vmm"},
        )
        publish_finished = time.perf_counter()

        import_started = time.perf_counter()
        parent_imported = dict(publisher.import_update(manifest))
        import_finished = time.perf_counter()
        first_name = next(iter(parent_imported))
        parent_before = parent_imported[first_name].detach().cpu().flatten()[:4].tolist()
        parent_ptr = int(parent_imported[first_name].data_ptr())

        ack_started = time.perf_counter()
        publisher.acknowledge(manifest.update_id)
        ack_finished = time.perf_counter()

        context = mp.get_context("spawn")
        queue = context.Queue()
        process_started = time.perf_counter()
        process = context.Process(target=_cuda_vmm_manifest_child, args=(manifest, queue))
        process.start()
        process.join(timeout=30)
        if process.exitcode != 0:
            raise RuntimeError(f"CUDA VMM manifest child exited with {process.exitcode}")
        child_result = queue.get(timeout=1)
        process_finished = time.perf_counter()
        torch.cuda.synchronize()
        parent_after = parent_imported[first_name].detach().cpu().flatten()[:4].tolist()
        if child_result["before"] != parent_before:
            raise RuntimeError("CUDA VMM child did not observe the published tensor values")
        if child_result["after"] != parent_after:
            raise RuntimeError("CUDA VMM parent did not observe the child mutation")
        same_virtual_address = child_result["data_ptr"] == parent_ptr
    finally:
        release_started = time.perf_counter()
        parent_imported.clear()
        if manifest is not None:
            publisher.release(manifest.update_id)
        release_finished = time.perf_counter()

    metadata = manifest.metadata["weight_bridge"]
    return {
        "timestamp": _timestamp(),
        "candidate_id": "issue-13-candidate-8",
        "mode": "cuda-vmm",
        "status": "pass",
        "environment": _environment(),
        "tensor_count": manifest.tensor_count,
        "total_nbytes": manifest.total_nbytes,
        "publish_ms": (publish_finished - publish_started) * 1000,
        "import_ms": (import_finished - import_started) * 1000,
        "ack_ms": (ack_finished - ack_started) * 1000,
        "release_ms": (release_finished - release_started) * 1000,
        "active_weight_version": publisher.active_weight_version,
        "notes": (
            "CUDA VMM POSIX-fd zero-copy smoke passed; "
            f"mapped_nbytes={metadata['mapped_nbytes']}; "
            f"parent_data_ptr={parent_ptr}; "
            f"child_data_ptr={child_result['data_ptr']}; "
            f"same_virtual_address={same_virtual_address}; "
            f"child_attach_process_visible_ms={(process_finished - process_started) * 1000:.4f}; "
            f"parent_before={parent_before}; parent_after={parent_after}"
        ),
    }


def _run_rollout_update(args: argparse.Namespace) -> dict[str, Any]:
    model = _model(args.hidden_dim, args.layers)
    publisher = SharedMemoryTensorBridge(source_worker="benchmark-training", source_rank=0)
    consumer = SharedMemoryTensorBridge(source_worker="benchmark-rollout", source_rank=1)
    install_calls: list[tuple[str, int]] = []
    adapter = VLLMWeightInstallAdapter(
        object(),
        install_callable=lambda manifest, tensors: install_calls.append(
            (manifest.update_id, len(tensors))
        ),
    )
    rollout = RolloutExecutor(
        {"weight_transport": "shared-memory"},
        weight_bridge=consumer,
        weight_install_adapter=adapter,
    )

    publish_started = time.perf_counter()
    manifest = publisher.publish(
        model,
        weight_version=args.weight_version,
        metadata={"benchmark": "weight_sync_bridge_rollout_update"},
    )
    publish_finished = time.perf_counter()

    update_started = time.perf_counter()
    weights = rollout.update_weights(manifest)
    update_finished = time.perf_counter()

    if rollout.active_weight_version != manifest.weight_version:
        raise RuntimeError("rollout active weight version did not advance")
    if install_calls != [(manifest.update_id, manifest.tensor_count)]:
        raise RuntimeError("vLLM install adapter did not receive the complete manifest")
    if set(weights) != set(model.state_dict()):
        raise RuntimeError("rollout imported tensor names did not match model state")

    release_started = time.perf_counter()
    rollout.release_weights()
    publisher.release(manifest.update_id)
    release_finished = time.perf_counter()

    return {
        "timestamp": _timestamp(),
        "candidate_id": "issue-13-candidate-5",
        "mode": "rollout-update",
        "status": "pass",
        "environment": _environment(),
        "tensor_count": manifest.tensor_count,
        "total_nbytes": manifest.total_nbytes,
        "publish_ms": (publish_finished - publish_started) * 1000,
        "import_ms": (update_finished - update_started) * 1000,
        "ack_ms": "",
        "release_ms": (release_finished - release_started) * 1000,
        "active_weight_version": manifest.weight_version,
        "notes": (
            "RolloutExecutor.update_weights imported shared-memory manifest, "
            "called vLLM install adapter, acknowledged, activated, and released."
        ),
    }


def _run_cuda_vmm_rollout_update(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise WeightBridgeUnavailableError(
            "CUDA VMM rollout update smoke requires torch CUDA availability"
        )

    model = _model(args.hidden_dim, args.layers).to("cuda")
    publisher = CUDAVMMTensorBridge(source_worker="benchmark-training", source_rank=0)
    consumer = CUDAVMMTensorBridge(source_worker="benchmark-rollout", source_rank=1)
    install_calls: list[tuple[str, int, str]] = []
    adapter = VLLMWeightInstallAdapter(
        object(),
        install_callable=lambda manifest, tensors: install_calls.append(
            (manifest.update_id, len(tensors), str(next(iter(tensors.values())).device))
        ),
    )
    rollout = RolloutExecutor(
        {"weight_transport": "cuda-vmm"},
        weight_bridge=consumer,
        weight_install_adapter=adapter,
    )
    manifest = None
    try:
        publish_started = time.perf_counter()
        manifest = publisher.publish(
            model,
            weight_version=args.weight_version,
            metadata={"benchmark": "cuda_vmm_rollout_update"},
        )
        publish_finished = time.perf_counter()

        update_started = time.perf_counter()
        weights = rollout.update_weights(manifest)
        update_finished = time.perf_counter()

        if rollout.active_weight_version != manifest.weight_version:
            raise RuntimeError("rollout active weight version did not advance")
        if install_calls != [(manifest.update_id, manifest.tensor_count, "cuda:0")]:
            raise RuntimeError("vLLM install adapter did not receive CUDA VMM tensors")
        if set(weights) != set(model.state_dict()):
            raise RuntimeError("rollout imported tensor names did not match model state")

        first_name = next(iter(weights))
        before = weights[first_name].detach().cpu().flatten()[:4].tolist()
        weights[first_name].add_(7)
        torch.cuda.synchronize()
        after = weights[first_name].detach().cpu().flatten()[:4].tolist()
        if not all(abs((b + 7) - a) < 1e-2 for b, a in zip(before, after, strict=False)):
            raise RuntimeError("CUDA VMM rollout tensor mutation was not visible")
    finally:
        release_started = time.perf_counter()
        rollout.release_weights()
        if manifest is not None:
            publisher.release(manifest.update_id)
        release_finished = time.perf_counter()

    return {
        "timestamp": _timestamp(),
        "candidate_id": "issue-13-candidate-9",
        "mode": "cuda-vmm-rollout-update",
        "status": "pass",
        "environment": _environment(),
        "tensor_count": manifest.tensor_count,
        "total_nbytes": manifest.total_nbytes,
        "publish_ms": (publish_finished - publish_started) * 1000,
        "import_ms": (update_finished - update_started) * 1000,
        "ack_ms": "",
        "release_ms": (release_finished - release_started) * 1000,
        "active_weight_version": manifest.weight_version,
        "notes": (
            "RolloutExecutor.update_weights consumed a CUDA VMM manifest, "
            "called the vLLM install adapter with CUDA tensors, acknowledged, "
            f"activated, and observed in-place CUDA alias mutation from {before} to {after}."
        ),
    }


class _StateDictModule(torch.nn.Module):
    def __init__(self, state_dict: Mapping[str, torch.Tensor]):
        super().__init__()
        self._state_dict = dict(state_dict)

    def state_dict(self, *args: Any, **kwargs: Any) -> Mapping[str, torch.Tensor]:
        del args, kwargs
        return self._state_dict


def _run_vllm_cuda_vmm_external_storage(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise WeightBridgeUnavailableError(
            "vLLM CUDA VMM external-storage smoke requires torch CUDA availability"
        )
    model_path = getattr(args, "model", None)
    if not model_path:
        raise WeightBridgeUnavailableError(
            "vLLM CUDA VMM external-storage smoke requires --model pointing to a local model"
        )

    from vllm import LLM, SamplingParams

    def generate(llm: Any) -> dict[str, Any]:
        output = llm.generate(["Hello"], SamplingParams(max_tokens=4, temperature=0.0))[0]
        return {
            "text": output.outputs[0].text,
            "token_ids": list(output.outputs[0].token_ids),
        }

    def parameter_specs(model: torch.nn.Module) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "shape": tuple(int(dim) for dim in parameter.shape),
                "dtype": str(parameter.dtype),
            }
            for name, parameter in model.named_parameters()
        ]

    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        dtype="float16",
        max_model_len=64,
        gpu_memory_utilization=0.25,
        trust_remote_code=False,
        enforce_eager=True,
    )
    before = generate(llm)
    specs = llm.llm_engine.apply_model(parameter_specs)[0]
    state = {
        spec["name"]: torch.zeros(
            tuple(spec["shape"]),
            dtype=_dtype_from_string(spec["dtype"]),
            device="cuda",
        ).contiguous()
        for spec in specs
    }

    publisher = CUDAVMMTensorBridge(source_worker="benchmark-training", source_rank=0)
    manifest = publisher.publish(
        _StateDictModule(state),
        weight_version=args.weight_version,
        metadata={"benchmark": "vllm_cuda_vmm_external_storage"},
    )
    adapter = VLLMCUDAVMMExternalStorageAdapter(
        llm.llm_engine,
        require_all_parameters=True,
        synchronize_cuda=True,
    )
    try:
        install_started = time.perf_counter()
        adapter.install(manifest, {})
        install_finished = time.perf_counter()
        after = generate(llm)
        if before == after:
            raise RuntimeError("vLLM generation did not change after CUDA VMM storage binding")
        rebound = adapter.last_result[0]["rebound"] if adapter.last_result else []
        if len(rebound) != manifest.tensor_count:
            raise RuntimeError("vLLM CUDA VMM adapter did not bind every manifest tensor")
    finally:
        release_started = time.perf_counter()
        adapter.release(manifest.update_id)
        publisher.release(manifest.update_id)
        release_finished = time.perf_counter()

    return {
        "timestamp": _timestamp(),
        "candidate_id": "issue-13-candidate-10",
        "mode": "vllm-cuda-vmm-external-storage",
        "status": "pass",
        "environment": _environment() + f";vllm_model={model_path}",
        "tensor_count": manifest.tensor_count,
        "total_nbytes": manifest.total_nbytes,
        "publish_ms": "",
        "import_ms": (install_finished - install_started) * 1000,
        "ack_ms": "",
        "release_ms": (release_finished - release_started) * 1000,
        "active_weight_version": manifest.weight_version,
        "notes": (
            "Real vLLM worker imported CUDA VMM manifest via apply_model, rebound "
            f"{len(rebound)} kernel-format parameters to external storage, and generation "
            f"changed from token_ids={before['token_ids']} to token_ids={after['token_ids']}."
        ),
    }


def _run_vllm_cuda_ipc_hot_update(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise WeightBridgeUnavailableError(
            "vLLM CUDA IPC hot-update smoke requires torch CUDA availability"
        )
    model_path = getattr(args, "model", None)
    if not model_path:
        raise WeightBridgeUnavailableError(
            "vLLM CUDA IPC hot-update smoke requires --model pointing to a local model"
        )

    from vllm import LLM, SamplingParams
    from vllm.config.weight_transfer import WeightTransferConfig

    def generate(llm: Any) -> dict[str, Any]:
        output = llm.generate(["Hello"], SamplingParams(max_tokens=4, temperature=0.0))[0]
        return {
            "text": output.outputs[0].text,
            "token_ids": list(output.outputs[0].token_ids),
        }

    def parameter_specs(model: torch.nn.Module) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "shape": tuple(int(dim) for dim in parameter.shape),
                "dtype": str(parameter.dtype),
            }
            for name, parameter in model.named_parameters()
        ]

    manifest = None
    publisher = None
    builder = None
    publish_ms: float | str = ""
    request_ms: float | str = ""
    update_ms: float | str = ""
    release_ms: float | str = ""
    active_weight_version: int | str = ""
    before: dict[str, Any] | None = None
    try:
        llm = LLM(
            model=model_path,
            tokenizer=model_path,
            dtype="float16",
            max_model_len=64,
            gpu_memory_utilization=0.25,
            trust_remote_code=False,
            enforce_eager=True,
            weight_transfer_config=WeightTransferConfig(backend="ipc"),
        )
        before = generate(llm)
        specs = llm.llm_engine.apply_model(parameter_specs)[0]
        state = {
            spec["name"]: torch.zeros(
                tuple(spec["shape"]),
                dtype=_dtype_from_string(spec["dtype"]),
                device="cuda",
            ).contiguous()
            for spec in specs
        }

        publisher = IPCWeightBridge(source_worker="benchmark-training", source_rank=0)
        publish_started = time.perf_counter()
        manifest = publisher.publish(
            _StateDictModule(state),
            weight_version=args.weight_version,
            metadata={"benchmark": "vllm_cuda_ipc_hot_update"},
        )
        publish_finished = time.perf_counter()
        publish_ms = (publish_finished - publish_started) * 1000

        imported = dict(publisher.import_update(manifest))
        builder = VLLMIPCWeightUpdateRequestBuilder(is_checkpoint_format=False)
        request_started = time.perf_counter()
        request = builder(manifest, imported)
        request_finished = time.perf_counter()
        request_ms = (request_finished - request_started) * 1000

        llm.init_weight_transfer_engine({"init_info": {}})
        update_started = time.perf_counter()
        llm.update_weights(request)
        update_finished = time.perf_counter()
        update_ms = (update_finished - update_started) * 1000
        active_weight_version = manifest.weight_version

        after = generate(llm)
        if before == after:
            raise RuntimeError("vLLM generation did not change after CUDA IPC hot update")

        release_started = time.perf_counter()
        if builder is not None and manifest is not None:
            builder.release(manifest.update_id)
        if publisher is not None and manifest is not None:
            publisher.release(manifest.update_id)
        release_finished = time.perf_counter()
        release_ms = (release_finished - release_started) * 1000

        result = {
            "timestamp": _timestamp(),
            "candidate_id": "issue-13-candidate-11",
            "mode": "vllm-cuda-ipc-hot-update",
            "status": "pass",
            "environment": _environment() + f";vllm_model={model_path}",
            "tensor_count": manifest.tensor_count,
            "total_nbytes": manifest.total_nbytes,
            "publish_ms": publish_ms,
            "import_ms": update_ms,
            "ack_ms": "",
            "release_ms": release_ms,
            "active_weight_version": active_weight_version,
            "notes": (
                "Real vLLM IPC public update_weights path accepted CUDA IPC handles "
                f"for {manifest.tensor_count} kernel-format parameters and generation "
                f"changed from token_ids={before['token_ids']} to token_ids={after['token_ids']}."
            ),
        }
        return result
    except Exception as exc:
        release_started = time.perf_counter()
        if builder is not None and manifest is not None:
            builder.release(manifest.update_id)
        if publisher is not None and manifest is not None:
            publisher.release(manifest.update_id)
        release_finished = time.perf_counter()
        release_ms = (release_finished - release_started) * 1000

        result = {
            "timestamp": _timestamp(),
            "candidate_id": "issue-13-candidate-11",
            "mode": "vllm-cuda-ipc-hot-update",
            "status": "blocked",
            "environment": _environment() + f";vllm_model={model_path}",
            "tensor_count": manifest.tensor_count if manifest is not None else 0,
            "total_nbytes": manifest.total_nbytes if manifest is not None else 0,
            "publish_ms": publish_ms,
            "import_ms": update_ms,
            "ack_ms": request_ms,
            "release_ms": release_ms,
            "active_weight_version": active_weight_version,
            "notes": (
                "Real vLLM IPC hot-update path did not complete. "
                f"before_token_ids={before['token_ids'] if before else 'unavailable'}; "
                f"error={type(exc).__name__}: {exc}"
            ),
        }
        return result
    finally:
        if result is None:
            if builder is not None and manifest is not None:
                builder.release(manifest.update_id)
            if publisher is not None and manifest is not None:
                publisher.release(manifest.update_id)


def _dtype_from_string(value: str) -> torch.dtype:
    dtype = getattr(torch, value.removeprefix("torch."), None)
    if not isinstance(dtype, torch.dtype):
        raise WeightManifestValidationError(f"unsupported dtype in vLLM spec: {value}")
    return dtype


def _run_runtime_blockers() -> dict[str, Any]:
    optional_runtimes = ("deepspeed", "ray", "vllm")
    missing = [
        runtime for runtime in optional_runtimes if importlib.util.find_spec(runtime) is None
    ]
    if missing:
        status = "blocked"
        notes = (
            "Real runtime validation is blocked because these optional packages "
            f"are not installed in this environment: {', '.join(missing)}. "
            "Contract tests use fakes and guarded adapters; no production runtime "
            "success is claimed for the missing packages."
        )
    else:
        status = "pass"
        notes = (
            "Optional runtime packages are importable. This row is only a capability "
            "probe; run dedicated DeepSpeed/Ray/vLLM integration tests before claiming "
            "production hot-weight update support."
        )
    return {
        "timestamp": _timestamp(),
        "candidate_id": "issue-13-runtime-probe",
        "mode": "runtime-blockers",
        "status": status,
        "environment": _environment(),
        "tensor_count": 0,
        "total_nbytes": 0,
        "publish_ms": "",
        "import_ms": "",
        "ack_ms": "",
        "release_ms": "",
        "active_weight_version": "",
        "notes": notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--mode",
        choices=[
            "local",
            "shared-memory",
            "cuda-ipc",
            "cuda-vmm",
            "cuda-vmm-rollout-update",
            "rollout-update",
            "runtime-blockers",
            "vllm-cuda-ipc-hot-update",
            "vllm-cuda-vmm-external-storage",
        ],
        default="local",
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--weight-version", type=int, default=1)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output path. JSON is always printed to stdout.",
    )
    args = parser.parse_args()

    if args.smoke:
        args.hidden_dim = min(args.hidden_dim, 64)
        args.layers = min(args.layers, 4)

    if args.mode == "local":
        row = _run_local(args)
    elif args.mode == "shared-memory":
        row = _run_shared_memory(args)
    elif args.mode == "cuda-ipc":
        row = _run_cuda_ipc()
    elif args.mode == "cuda-vmm":
        row = _run_cuda_vmm(args)
    elif args.mode == "cuda-vmm-rollout-update":
        row = _run_cuda_vmm_rollout_update(args)
    elif args.mode == "rollout-update":
        row = _run_rollout_update(args)
    elif args.mode == "vllm-cuda-ipc-hot-update":
        row = _run_vllm_cuda_ipc_hot_update(args)
    elif args.mode == "vllm-cuda-vmm-external-storage":
        row = _run_vllm_cuda_vmm_external_storage(args)
    else:
        row = _run_runtime_blockers()

    if args.output is not None:
        _write_csv_row(row, args.output)
    print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
