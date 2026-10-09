# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Serialize the real native VIME/Megatron training state, one TP rank at a time.

DistributedOptimizer.state_dict() deliberately omits master weights and Adam
moments. The bounded native parameter getter/setter adapter preserves all of
these states without allocating whole-bucket CPU packing copies. Publication
and recovery use the all-rank atomic protocol.
"""

import gc
import json
import os
import uuid
from pathlib import Path

import torch
import torch.distributed as dist
from qwen3_next_checkpoint import (
    newest_complete_checkpoint,
    publish_checkpoint,
    require_checkpoint_disk_space,
    validate_checkpoint,
    write_rank_manifest,
)
from qwen3_next_contract import atomic_json
from qwen3_next_optimizer_checkpoint import (
    load_bounded_parameter_state,
    native_parameter_layout,
    native_parameter_states,
    parameter_state_files,
)
from qwen3_next_optimizer_checkpoint import parameter_state_records as _optimizer_parameter_records
from qwen3_next_optimizer_checkpoint import (
    save_bounded_parameter_state,
    verify_hdo_optimizer_master_aliases,
)
from qwen3_next_runtime import create_staging, owned_root

from rl_engine.integrations.qwen3_next_training import (
    capture_rng_state,
    named_trainable_parameters,
    restore_rng_state,
    state_digest,
)
from rl_engine.testing.qwen3_next_rl_fixture import PolicyFixture, require_policy_configuration

SCHEMA_VERSION = 1
CLASS_TAG = "__qwen3_next_native_optimizer_class__"


def _native_leaves(args, model, optimizer, tracker):
    """Reject an alternate optimizer/topology before any checkpoint work starts."""
    require_policy_configuration(args)
    if tracker is None or not tracker.get_states():
        raise ValueError("Native checkpoint must include the initialized Megatron RNG tracker")
    if type(getattr(args, "consumed_train_samples", None)) is not int:
        raise ValueError("Native training sample progress must be explicit")
    from megatron.core import parallel_state
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from megatron.core.optimizer.optimizer import ChainedOptimizer

    if not dist.is_initialized() or dist.get_world_size() != 4:
        raise ValueError("Native checkpoint requires an initialized four-rank process group")
    topology = (
        parallel_state.get_tensor_model_parallel_world_size(),
        parallel_state.get_pipeline_model_parallel_world_size(),
        parallel_state.get_context_parallel_world_size(),
        parallel_state.get_expert_model_parallel_world_size(),
        parallel_state.get_data_parallel_world_size(),
    )
    if topology != (4, 1, 1, 1, 1):
        raise ValueError("Native checkpoint requires TP4 and PP/CP/EP/DP1")
    if parallel_state.get_tensor_model_parallel_rank() != dist.get_rank():
        raise ValueError("Checkpoint rank order must equal tensor-parallel rank order")
    leaves = (
        list(optimizer.chained_optimizers)
        if isinstance(optimizer, ChainedOptimizer)
        else [optimizer]
    )
    if not leaves or any(not isinstance(leaf, DistributedOptimizer) for leaf in leaves):
        raise TypeError("Checkpoint requires actual native DistributedOptimizer leaves")
    for leaf in leaves:
        if getattr(leaf, "is_stub_optimizer", False):
            raise ValueError("Stub optimizers cannot provide complete training state")
        if leaf.ddp_config.use_megatron_fsdp:
            raise ValueError("The fixed TP4 checkpoint path does not use FSDP")
        for group in (leaf.data_parallel_group, leaf.data_parallel_group_gloo):
            if group is None or dist.get_world_size(group) != 1 or dist.get_rank(group) != 0:
                raise ValueError("Serial native optimizer saving requires singleton DP groups")
        native = leaf.optimizer
        if not isinstance(native, HybridDeviceOptimizer):
            raise TypeError("Native checkpoint requires Megatron CPU optimizer offload")
        if (
            native.offload_fraction != 1.0
            or not native.param_update_in_fp32
            or native.overlap_cpu_optimizer_d2h_h2d
            or native.gpu_optimizer is not None
        ):
            raise ValueError("Native optimizer must fully offload FP32 masters without overlap")
    parameters = named_trainable_parameters(model)
    if any(value.dtype != torch.bfloat16 or not value.is_cuda for value in parameters.values()):
        raise ValueError("The complete native model must retain BF16 CUDA parameters")
    return leaves


def _all_rank_call(action, label):
    """Propagate a Python failure before another rank enters a later collective."""
    value, error = None, None
    try:
        value = action()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = [None] * 4
    dist.all_gather_object(errors, error)
    if any(errors):
        raise RuntimeError(f"{label} failed across ranks: {errors}")
    return value


def _rank_zero_call(action, label):
    value = _all_rank_call(lambda: action() if dist.get_rank() == 0 else None, label)
    values = [value]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def _serial_rank_call(action, label):
    result = None
    for owner in range(4):
        dist.barrier()
        current = _all_rank_call(
            lambda: action() if dist.get_rank() == owner else None, f"{label} rank {owner}"
        )
        if dist.get_rank() == owner:
            result = current
            gc.collect()
    return result


def _native_class_allowlist(leaves):
    allowed = {}
    for leaf in leaves:
        for name in ("cpu_optimizer_cls", "gpu_optimizer_cls"):
            cls = getattr(leaf.optimizer, name)
            if cls is not None:
                key = f"{cls.__module__}.{cls.__qualname__}"
                if not isinstance(cls, type) or (key in allowed and allowed[key] is not cls):
                    raise ValueError("Ambiguous native optimizer class identity")
                allowed[key] = cls
    return allowed


def _encode_header(value, allowed):
    """Preserve native class-valued param-group defaults without unsafe unpickling."""
    if isinstance(value, type):
        key = f"{value.__module__}.{value.__qualname__}"
        if allowed.get(key) is not value:
            raise TypeError(f"Unpinned class in native optimizer header: {key}")
        return {CLASS_TAG: key}
    if isinstance(value, dict):
        if CLASS_TAG in value:
            raise ValueError("Reserved native class tag in optimizer header")
        return {key: _encode_header(item, allowed) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_encode_header(item, allowed) for item in value)
    if value is None or isinstance(value, torch.Tensor) or type(value) in (str, bool, int, float):
        return value
    raise TypeError(f"Unsupported native optimizer header value: {type(value)!r}")


def _decode_header(value, allowed):
    if isinstance(value, dict):
        if CLASS_TAG in value:
            if set(value) != {CLASS_TAG} or value[CLASS_TAG] not in allowed:
                raise ValueError(
                    "Saved optimizer class is absent from the running native optimizer"
                )
            return allowed[value[CLASS_TAG]]
        return {key: _decode_header(item, allowed) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_decode_header(item, allowed) for item in value)
    return value


def _model_states(model):
    if not model:
        raise ValueError("Cannot save an empty native model")
    return [chunk.state_dict_for_save_checkpoint() for chunk in model]


def _state_signature(value):
    if isinstance(value, torch.Tensor):
        return {"dtype": str(value.dtype), "shape": list(value.shape)}
    if isinstance(value, dict):
        return {key: _state_signature(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_state_signature(item) for item in value)
    if value is None or type(value) in (str, bool, int, float):
        return type(value).__name__
    raise TypeError(f"Unsupported native model state type: {type(value)!r}")


def _tensor_bytes(value, seen=None):
    if seen is None:
        seen = set()
    if isinstance(value, torch.Tensor):
        # A serialized view retains its underlying storage, including TP padding.
        storage = value.untyped_storage()
        key = (value.device, storage.data_ptr(), storage.nbytes())
        if key in seen:
            return 0
        seen.add(key)
        return storage.nbytes()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item, seen) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item, seen) for item in value)
    return 0


def _encode_rng(state):
    numpy_state = state["numpy"]
    return {
        **state,
        "numpy": (
            numpy_state[0],
            torch.from_numpy(numpy_state[1].copy()),
            *numpy_state[2:],
        ),
    }


def _decode_rng(state):
    numpy_state = state["numpy"]
    if numpy_state[1].dtype != torch.uint32 or numpy_state[1].device.type != "cpu":
        raise ValueError("Invalid saved NumPy RNG state")
    return {**state, "numpy": (numpy_state[0], numpy_state[1].numpy().copy(), *numpy_state[2:])}


def _save_tensor_file(path, payload):
    with Path(path).open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())


def _rank_files(rank, chunk_count, rank_dir):
    return {
        "model": [f"rank-{rank}/model-{index}.pt" for index in range(chunk_count)],
        "optimizer": [f"rank-{rank}/{name}" for name in parameter_state_files(rank_dir)],
        "rng": [f"rank-{rank}/rng.pt"],
        "progress": [
            f"rank-{rank}/{name}" for name in ("progress.json", "scheduler.pt", "fixture.pt")
        ],
    }


def _validate_native_checkpoint(checkpoint, identity, chunk_count):
    # Sizes and manifests only: _load_rank compares every loaded tensor's digest.
    complete = validate_checkpoint(checkpoint, identity, verify_files=False)
    for rank in range(4):
        manifest = json.loads((checkpoint / f"rank-{rank}.manifest.json").read_text())
        files = {
            role: sorted(record["path"] for record in records)
            for role, records in manifest["files"].items()
        }
        expected = {
            role: sorted(paths)
            for role, paths in _rank_files(rank, chunk_count, checkpoint / f"rank-{rank}").items()
        }
        if files != expected:
            raise ValueError(
                "Native checkpoint manifest omits or substitutes a required state file"
            )
    return complete


def _save_rank(
    model,
    optimizer,
    scheduler,
    fixture,
    staging,
    *,
    leaves,
    identity,
    step,
    sample_position,
    tracker,
    native_args,
    update_evidence,
):
    rank = dist.get_rank()
    rank_dir = staging / f"rank-{rank}"
    rank_dir.mkdir()
    torch.cuda.synchronize()
    rng = capture_rng_state(tracker=tracker)
    states = _model_states(model)
    model_hashes = [state_digest(state) for state in states]
    native_records, _ = _optimizer_parameter_records(model, leaves)
    header = _encode_header(optimizer.state_dict(), _native_class_allowlist(leaves))
    schedule = scheduler.state_dict()
    for index, state in enumerate(states):
        _save_tensor_file(rank_dir / f"model-{index}.pt", state)
    del states
    _save_tensor_file(rank_dir / "optimizer-header.pt", header)
    save_bounded_parameter_state(model, leaves, rank_dir, _save_tensor_file)
    verify_hdo_optimizer_master_aliases(model, leaves)
    _save_tensor_file(rank_dir / "rng.pt", _encode_rng(rng))
    _save_tensor_file(rank_dir / "scheduler.pt", schedule)
    _save_tensor_file(
        rank_dir / "fixture.pt",
        {
            "tokens": fixture.tokens,
            "old_logprobs": fixture.old_logprobs,
            "baseline_identity": fixture.baseline_identity,
            "content_sha256": fixture.content_sha256,
        },
    )
    progress = {
        "schema_version": SCHEMA_VERSION,
        "scope": "native_vime_training_checkpoint",
        "rank": rank,
        "world_size": 4,
        "identity": identity,
        "step": step,
        "sample_position": sample_position,
        "consumed_train_samples": native_args.consumed_train_samples,
        "update_evidence": update_evidence,
        "fixture_sha256": fixture.content_sha256,
        "model_sha256": model_hashes,
        "optimizer_parameter_sha256": native_records,
        "optimizer_header_sha256": state_digest(header),
        "scheduler_sha256": state_digest(schedule),
        "rng_sha256": state_digest(rng),
        "serialization": "native_parameter_views_bounded_serial_tp_ranks",
    }
    atomic_json(rank_dir / "progress.json", progress)
    if state_digest(capture_rng_state(tracker=tracker)) != progress["rng_sha256"]:
        raise ValueError("Checkpoint serialization unexpectedly changed training RNG state")
    write_rank_manifest(
        staging,
        rank=rank,
        world_size=4,
        identity=identity,
        step=step,
        sample_position=sample_position,
        files=_rank_files(rank, len(model), rank_dir),
    )
    return progress


def _load_rank(
    model,
    optimizer,
    scheduler,
    fixture,
    checkpoint,
    *,
    leaves,
    identity,
    complete,
    tracker,
    native_args,
):
    rank = dist.get_rank()
    rank_dir = checkpoint / f"rank-{rank}"
    progress = json.loads((rank_dir / "progress.json").read_text())
    expected = {
        "schema_version": SCHEMA_VERSION,
        "scope": "native_vime_training_checkpoint",
        "rank": rank,
        "world_size": 4,
        "identity": identity,
        "step": complete["step"],
        "sample_position": complete["sample_position"],
        "fixture_sha256": fixture.content_sha256,
        "serialization": "native_parameter_views_bounded_serial_tp_ranks",
    }
    if any(progress.get(key) != value for key, value in expected.items()):
        raise ValueError("Saved native training progress or fixture identity differs")
    saved_fixture = PolicyFixture(**torch.load(rank_dir / "fixture.pt", weights_only=True))
    saved_fixture.validate()
    if saved_fixture.content_sha256 != fixture.content_sha256:
        raise ValueError("Recovery must reuse the frozen baseline policy samples/logprobs")
    header = torch.load(rank_dir / "optimizer-header.pt", map_location="cpu", weights_only=True)
    if state_digest(header) != progress["optimizer_header_sha256"]:
        raise ValueError("Native optimizer header content mismatch")
    # Native load_state_dict may initialize state with a dummy step. Load the
    # saved BF16 model afterwards and restore RNG only after every other load.
    optimizer.load_state_dict(_decode_header(header, _native_class_allowlist(leaves)))
    load_bounded_parameter_state(model, leaves, rank_dir)
    verify_hdo_optimizer_master_aliases(model, leaves)
    if len(progress["model_sha256"]) != len(model):
        raise ValueError("Saved model chunk count differs")
    for index, chunk in enumerate(model):
        saved = torch.load(
            rank_dir / f"model-{index}.pt", map_location="cpu", weights_only=True, mmap=True
        )
        current = chunk.state_dict_for_save_checkpoint()
        if _state_signature(saved) != _state_signature(current):
            raise ValueError("Model checkpoint would change tensor dtypes, shapes or state keys")
        if state_digest(saved) != progress["model_sha256"][index]:
            raise ValueError("Saved native model content mismatch")
        del current
        chunk.load_state_dict(saved, strict=True)
        del saved
    schedule = torch.load(rank_dir / "scheduler.pt", map_location="cpu", weights_only=True)
    scheduler.load_state_dict(schedule)
    torch.cuda.synchronize()
    if [state_digest(value) for value in _model_states(model)] != progress["model_sha256"]:
        raise ValueError("Native model reload changed original tensor bits")
    restored, _ = _optimizer_parameter_records(model, leaves)
    if restored != progress["optimizer_parameter_sha256"]:
        raise ValueError("Native optimizer reload changed FP32 master weights or Adam moments")
    current_header = _encode_header(optimizer.state_dict(), _native_class_allowlist(leaves))
    if state_digest(current_header) != progress["optimizer_header_sha256"]:
        raise ValueError("Native optimizer reload changed optimizer metadata or step")
    if state_digest(scheduler.state_dict()) != progress["scheduler_sha256"]:
        raise ValueError("Native scheduler reload changed schedule progress")
    if type(progress.get("consumed_train_samples")) is not int:
        raise ValueError("Saved native consumed sample progress is absent")
    native_args.consumed_train_samples = progress["consumed_train_samples"]
    rng = _decode_rng(torch.load(rank_dir / "rng.pt", map_location="cpu", weights_only=True))
    if state_digest(rng) != progress["rng_sha256"]:
        raise ValueError("Saved RNG content mismatch")
    restore_rng_state(rng, tracker=tracker)
    if state_digest(capture_rng_state(tracker=tracker)) != progress["rng_sha256"]:
        raise ValueError("Native RNG restore changed original state")
    return progress


def save_training_checkpoint(
    args,
    model,
    optimizer,
    scheduler,
    fixture,
    root,
    *,
    step,
    sample_position,
    identity,
    tracker,
    export_bytes,
    update_evidence=None,
):
    """Collectively save a complete native checkpoint; call on all four TP ranks."""
    if (
        type(step) is not int
        or step not in (1, 2, 3)
        or type(sample_position) is not int
        or sample_position != step * 4
    ):
        raise ValueError("Checkpoint progress must match complete four-sample native updates")
    fixture.validate()
    leaves = _all_rank_call(
        lambda: _native_leaves(args, model, optimizer, tracker), "native topology"
    )
    root = Path(root).absolute()

    def estimate():
        optimizer_bytes = sum(
            value.numel() * value.element_size()
            for record in native_parameter_layout(model, leaves).values()
            for states in [native_parameter_states(record)]
            for value in states.values()
            if isinstance(value, torch.Tensor)
        )
        # Four-rank allgather carries only byte counts, never parameter tensors.
        model_bytes = sum(_tensor_bytes(state) for state in _model_states(model))
        # Native DP packing retains buffer padding, rather than only model numels.
        return model_bytes + (optimizer_bytes * 105 // 100) + 64 * 1024**2

    local_bytes = _all_rank_call(estimate, "checkpoint disk estimate")
    estimates = [None] * 4
    dist.all_gather_object(estimates, local_bytes)

    def prepare():
        owned_root(root, identity)
        recovery = newest_complete_checkpoint(root, identity, verify_files=False)
        retained = 0
        if recovery["checkpoint"]:
            retained = sum(
                path.stat().st_size
                for path in Path(recovery["checkpoint"]).rglob("*")
                if path.is_file()
            )
        require_checkpoint_disk_space(
            root,
            checkpoint_bytes=sum(estimates),
            export_bytes=export_bytes,
            retained_complete_bytes=retained,
        )
        staging, destination = create_staging(
            root,
            f"step-{step:08d}-{uuid.uuid4().hex}",
            {
                "schema_version": 1,
                "scope": "native_vime_training_checkpoint",
                "status": "staging",
                "identity": identity,
                "step": step,
                "sample_position": sample_position,
            },
        )
        return {"staging": str(staging), "destination": str(destination), "recovery": recovery}

    paths = _rank_zero_call(prepare, "checkpoint disk budget and staging")
    staging, destination = Path(paths["staging"]), Path(paths["destination"])
    _serial_rank_call(
        lambda: _save_rank(
            model,
            optimizer,
            scheduler,
            fixture,
            staging,
            leaves=leaves,
            identity=identity,
            step=step,
            sample_position=sample_position,
            tracker=tracker,
            native_args=args,
            update_evidence=update_evidence,
        ),
        "native checkpoint save",
    )
    complete = _rank_zero_call(
        lambda: publish_checkpoint(staging, destination, world_size=4, identity=identity),
        "checkpoint all-file validation and publication",
    )
    return {"checkpoint": str(destination), "complete": complete, "recovery": paths["recovery"]}


def load_training_checkpoint(
    args, model, optimizer, scheduler, fixture, checkpoint, *, identity, tracker
):
    """Verify every rank's file before any load, then restore native state serially."""
    fixture.validate()
    leaves = _all_rank_call(
        lambda: _native_leaves(args, model, optimizer, tracker), "native topology"
    )
    checkpoint = Path(checkpoint).resolve()
    complete = _rank_zero_call(
        lambda: _validate_native_checkpoint(checkpoint, identity, len(model)),
        "checkpoint pre-load all-file validation",
    )
    if complete["step"] not in (1, 2, 3) or complete["sample_position"] != complete["step"] * 4:
        raise ValueError("Checkpoint progress differs from complete fixed native updates")
    progress = _serial_rank_call(
        lambda: _load_rank(
            model,
            optimizer,
            scheduler,
            fixture,
            checkpoint,
            leaves=leaves,
            identity=identity,
            complete=complete,
            tracker=tracker,
            native_args=args,
        ),
        "native checkpoint load",
    )
    return {
        "checkpoint": str(checkpoint),
        "step": complete["step"],
        "sample_position": complete["sample_position"],
        "rank_evidence": progress,
    }


def training_state_identity(
    args, model, optimizer, scheduler, fixture, *, step, sample_position, identity, tracker
):
    """Exact per-rank control/recovery state identity, with no PID or path fields."""
    fixture.validate()
    if type(step) is not int or step not in (1, 2, 3) or sample_position != step * 4:
        raise ValueError("State audit requires complete fixed native updates")
    leaves = _native_leaves(args, model, optimizer, tracker)
    torch.cuda.synchronize()
    records, _ = _optimizer_parameter_records(model, leaves)
    header = _encode_header(optimizer.state_dict(), _native_class_allowlist(leaves))
    result = {
        "schema_version": SCHEMA_VERSION,
        "scope": "native_vime_training_state",
        "rank": dist.get_rank(),
        "identity": identity,
        "step": step,
        "sample_position": sample_position,
        "consumed_train_samples": args.consumed_train_samples,
        "fixture_sha256": fixture.content_sha256,
        "model_sha256": [state_digest(value) for value in _model_states(model)],
        "optimizer_parameter_sha256": records,
        "optimizer_header_sha256": state_digest(header),
        "scheduler_sha256": state_digest(scheduler.state_dict()),
        "rng_sha256": state_digest(capture_rng_state(tracker=tracker)),
    }
    return {**result, "state_sha256": state_digest(result)}
