# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Publish fresh BF16 HF weights through the installed native VIME exporter.

An export completion record preserves provenance after historical weights are
retired. Only validate_training_export, which rehashes every current file, can
authorize loading an export. A retained record alone cannot authorize a load.
"""

import hashlib
import json
import math
import os
import re
import uuid
from pathlib import Path

import torch
import torch.distributed as dist
from qwen3_next_checkpoint import require_checkpoint_disk_space
from qwen3_next_contract import atomic_json, checkpoint_identity
from qwen3_next_runtime import create_staging, file_digest, owned_root
from qwen3_next_training_checkpoint import _all_rank_call, _rank_zero_call

from rl_engine.integrations.orchestrators.vime.providers.qwen3_next_training import (
    capture_rng_state,
    named_trainable_parameters,
    state_digest,
    tensor_digest,
)

COMPLETE_FILE = "EXPORT_COMPLETE.json"
INDEX_FILE = "model.safetensors.index.json"
KIND = "qwen3_next_native_hf_export"
# Official tensors the native trainer neither loads nor updates: the multi-token-prediction head.
# VIME's HF saver writes only Megatron-owned parameters, so these are carried from the pinned base.
PASSTHROUGH_PREFIXES = ("mtp.",)
PASSTHROUGH_SHARD = "model-base-passthrough-{idx:05d}-of-{total:05d}.safetensors"
IDENTITY_KEYS = (
    "source_commit",
    "source_tree",
    "environment_sha256",
    "provider_config",
    "topology",
)


def _json_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _reference():
    path = (
        Path(__file__).resolve().parents[3]
        / "rl_engine/validation/models/qwen3_next_norm_manifest.json"
    )
    return json.loads(path.read_text())["model_identity"]


def _schema_reference():
    path = (
        Path(__file__).resolve().parents[3]
        / "rl_engine/validation/models/qwen3_next_export_schema.json"
    )
    record = json.loads(path.read_text())
    if (
        record.get("schema_version") != 1
        or record.get("checkpoint_revision") != _reference()["revision"]
    ):
        raise ValueError("Pinned export schema must identify the official checkpoint revision")
    return record


def _safe_file(root, name):
    path = Path(name)
    if len(path.parts) != 1 or path.is_absolute() or path.name in (".", ".."):
        raise ValueError("HF export file must be directly inside its export directory")
    candidate = root / path
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"Missing or external HF export file: {name}")
    return candidate


def _weight_schema(root, *, check_finite):
    from safetensors import safe_open

    root = Path(root)
    index = json.loads((root / INDEX_FILE).read_text())
    weight_map = index["weight_map"]
    if not weight_map or any(not isinstance(key, str) for key in weight_map):
        raise ValueError("An HF export must contain the complete named model")
    by_shard = {}
    for name, shard in weight_map.items():
        if Path(shard).name != shard or not shard.endswith(".safetensors"):
            raise ValueError("HF tensor index points outside safetensors shards")
        by_shard.setdefault(shard, set()).add(name)
    schema = {}
    for shard, names in sorted(by_shard.items()):
        with safe_open(root / shard, framework="pt", device="cpu") as reader:
            if set(reader.keys()) != names:
                raise ValueError("HF shard tensor inventory differs from its weight index")
            for name in sorted(names):
                view = reader.get_slice(name)
                shape, dtype = view.get_shape(), view.get_dtype()
                if dtype != "BF16" or not shape or any(value <= 0 for value in shape):
                    raise ValueError(f"HF export must preserve nonempty BF16 weights: {name}")
                schema[name] = {"dtype": dtype, "shape": shape}
                if check_finite:
                    # Keep verification memory bounded, including embedding/LM head.
                    row_size = max(1, math.prod(shape[1:]))
                    rows = max(1, (32 * 1024**2) // (row_size * 2))
                    for start in range(0, shape[0], rows):
                        if not torch.isfinite(view[start : start + rows]).all():
                            raise ValueError(f"Nonfinite exported model weight: {name}")
    layers = {
        int(match.group(1))
        for name in schema
        if (match := re.match(r"model\.layers\.(\d+)\.", name))
    }
    if layers != set(range(48)):
        raise ValueError("HF export must retain all 48 real decoder layers")
    if index.get("metadata", {}).get("total_size") != sum(
        math.prod(record["shape"]) * 2 for record in schema.values()
    ):
        raise ValueError("HF tensor index total byte size differs from its tensors")
    return schema


def validate_training_export_record(record, expected_identity):
    """Validate a durable history record; this deliberately does not authorize loading."""
    if record != expected_identity:
        raise ValueError("Published HF export identity differs from the requested weights")
    if (record.get("schema_version"), record.get("kind"), record.get("status")) != (
        1,
        KIND,
        "complete",
    ):
        raise ValueError("HF export has no supported completion record")
    if (record.get("architecture_layers"), record.get("weights_dtype")) != (48, "bfloat16"):
        raise ValueError("HF export architecture or precision differs")
    expected_versions = {"update-1": 1, "update-2": 2, "reload-2": 2}
    if (
        record.get("weight_version") not in expected_versions
        or type(record.get("step")) is not int
        or record.get("step") != expected_versions[record["weight_version"]]
    ):
        raise ValueError("HF export does not identify an accepted native update/reload phase")
    training, base = record["training_identity"], record["base_checkpoint_identity"]
    if (
        training.get("checkpoint") != record["base_checkpoint"]
        or training.get("checkpoint_identity") != base
        or any(key not in training for key in IDENTITY_KEYS)
    ):
        raise ValueError(
            "Updated weights require explicit fixed baseline/source/environment provenance"
        )
    if base.get("revision") != _reference()["revision"]:
        raise ValueError("HF export base revision differs from the official pinned checkpoint")
    hashes = record.get("model_rank_sha256")
    if record.get("model_identity_method") != "native_parameter_hashes_after":
        raise ValueError("HF export must bind directly to the native update parameter audit")
    if not isinstance(hashes, list) or len(hashes) != 4:
        raise ValueError("HF export requires four actual native model state hashes")
    passthrough = record.get("base_passthrough")
    if (
        not isinstance(passthrough, dict)
        or passthrough.get("prefixes") != list(PASSTHROUGH_PREFIXES)
        or type(passthrough.get("tensor_count")) is not int
        or passthrough["tensor_count"] < 0
        or not isinstance(passthrough.get("shards"), dict)
    ):
        raise ValueError("HF export must state which untrained official tensors came from the base")
    if any(
        not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value) for value in hashes
    ):
        raise ValueError("HF export has an invalid native model state hash")
    files = record.get("files", {})
    shards = {name: value for name, value in files.items() if name.endswith(".safetensors")}
    if not shards or INDEX_FILE not in files or "config.json" not in files:
        raise ValueError("HF export is missing model shards, index or configuration")
    for name, item in files.items():
        if (
            Path(name).name != name
            or type(item.get("size")) is not int
            or item["size"] <= 0
            or not isinstance(item.get("sha256"), str)
            or not re.fullmatch("[0-9a-f]{64}", item["sha256"])
        ):
            raise ValueError("Invalid file identity in HF export record")
    if record.get("weights_content_sha256") != _json_digest(shards):
        raise ValueError("HF export shard content proof differs")
    if record.get("metadata") != {
        name: value["sha256"]
        for name, value in files.items()
        if not name.endswith(".safetensors") and name != "STAGING.json"
    }:
        raise ValueError("HF export metadata proof differs")
    reference = _schema_reference()
    if (
        record.get("weight_schema_sha256") != reference["weight_schema_sha256"]
        or record.get("tensor_count") != reference["tensor_count"]
    ):
        raise ValueError("HF export omits the pinned complete official tensor schema proof")
    return record


def validate_training_export(checkpoint, expected_identity):
    """Rehash every current export file and exact full-model schema before loading."""
    checkpoint = Path(checkpoint)
    if checkpoint.suffix == ".partial" or checkpoint.name.startswith("."):
        raise ValueError("An unpublished HF export cannot be loaded")
    record = json.loads((checkpoint / COMPLETE_FILE).read_text())
    validate_training_export_record(record, expected_identity)
    present = {path.name for path in checkpoint.iterdir() if path.name != COMPLETE_FILE}
    if present != record["files"].keys():
        raise ValueError("HF export file inventory changed after publication")
    for name, expected in record["files"].items():
        path = _safe_file(checkpoint, name)
        if path.stat().st_size != expected["size"] or file_digest(path) != expected["sha256"]:
            raise ValueError(f"HF export file checksum mismatch: {name}")
    base = Path(record["base_checkpoint"])
    if checkpoint_identity(base) != record["base_checkpoint_identity"]:
        raise ValueError("HF export official base metadata or shard identity changed")
    schema = _weight_schema(checkpoint, check_finite=True)
    if (
        schema != _weight_schema(base, check_finite=False)
        or _json_digest(schema) != record["weight_schema_sha256"]
        or len(schema) != record["tensor_count"]
    ):
        raise ValueError("HF export differs from the complete official model tensor schema")
    return record


def _carry_base_passthrough(staging, base_checkpoint):
    """Copy the official tensors the native model does not own from the pinned base, byte for byte.

    Returns the passthrough record for publication. Any missing tensor outside
    ``PASSTHROUGH_PREFIXES`` is a real omission by the saver and is rejected; the saver must
    also not have written part of the passthrough head itself.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    staging, base = Path(staging), Path(base_checkpoint)
    index = json.loads((staging / INDEX_FILE).read_text())
    staged = index["weight_map"]
    official = json.loads((base / INDEX_FILE).read_text())["weight_map"]
    extra = sorted(set(staged) - set(official))
    if extra:
        raise ValueError(f"Native HF saver wrote tensors outside the official schema: {extra[:5]}")
    missing = sorted(set(official) - set(staged))
    foreign = [name for name in missing if not name.startswith(PASSTHROUGH_PREFIXES)]
    if foreign:
        raise ValueError(f"Native HF saver omitted trained official tensors: {foreign[:5]}")
    if any(name.startswith(PASSTHROUGH_PREFIXES) for name in staged):
        raise ValueError("Native HF saver wrote part of the base passthrough head itself")
    record = {"prefixes": list(PASSTHROUGH_PREFIXES), "tensor_count": len(missing), "shards": {}}
    if not missing:
        return record
    by_shard = {}
    for name in missing:
        by_shard.setdefault(official[name], []).append(name)
    added = 0
    for idx, source in enumerate(sorted(by_shard), start=1):
        if Path(source).name != source or not source.endswith(".safetensors"):
            raise ValueError("Official tensor index points outside safetensors shards")
        filename = PASSTHROUGH_SHARD.format(idx=idx, total=len(by_shard))
        if (staging / filename).exists():
            raise ValueError(f"Passthrough shard already present in the staging export: {filename}")
        with safe_open(base / source, framework="pt", device="cpu") as reader:
            tensors = {name: reader.get_tensor(name) for name in by_shard[source]}
        for name, value in tensors.items():
            if (
                value.dtype != torch.bfloat16
                or value.numel() == 0
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"Official passthrough tensor is not finite BF16: {name}")
        save_file(tensors, staging / filename, metadata={"format": "pt"})
        with safe_open(staging / filename, framework="pt", device="cpu") as reader:
            if set(reader.keys()) != set(tensors) or any(
                not torch.equal(reader.get_tensor(name), value) for name, value in tensors.items()
            ):
                raise ValueError(
                    f"Passthrough shard does not reproduce the base tensors: {filename}"
                )
        for name, value in tensors.items():
            staged[name] = filename
            added += value.numel() * value.element_size()
        record["shards"][filename] = {"source": source, "tensor_count": len(tensors)}
    metadata = index.setdefault("metadata", {})
    metadata["total_size"] = int(metadata.get("total_size", 0)) + added
    atomic_json(staging / INDEX_FILE, index)
    record["names_sha256"] = _json_digest(missing)
    return record


def _publish_export(
    staging,
    destination,
    *,
    step,
    weight_version,
    identity,
    base_checkpoint,
    base_checkpoint_identity,
    model_hashes,
):
    staging, destination, base_checkpoint = Path(staging), Path(destination), Path(base_checkpoint)
    if staging.suffix != ".partial" or staging.parent != destination.parent or destination.exists():
        raise ValueError("HF export publication requires a new sibling destination")
    if checkpoint_identity(base_checkpoint) != base_checkpoint_identity:
        raise ValueError("HF export official base identity changed")
    passthrough = _carry_base_passthrough(staging, base_checkpoint)
    schema = _weight_schema(staging, check_finite=True)
    if schema != _weight_schema(base_checkpoint, check_finite=False):
        raise ValueError("Native HF saver did not preserve the complete official tensor schema")
    expected_assets = {
        name: digest
        for name, digest in base_checkpoint_identity["metadata"].items()
        if name != INDEX_FILE
    }
    actual_assets = {
        path.name: file_digest(path)
        for path in staging.iterdir()
        if path.name not in (INDEX_FILE, "STAGING.json") and path.suffix != ".safetensors"
    }
    if expected_assets != actual_assets:
        raise ValueError("Native HF export changed tokenizer/configuration assets")
    files = {}
    for path in sorted(staging.iterdir()):
        _safe_file(staging, path.name)
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        files[path.name] = {"size": path.stat().st_size, "sha256": file_digest(path)}
    record = {
        "schema_version": 1,
        "kind": KIND,
        "status": "complete",
        "weight_version": weight_version,
        "step": step,
        "export_id": destination.name,
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_identity": base_checkpoint_identity,
        "training_identity": identity,
        "architecture_layers": 48,
        "weights_dtype": "bfloat16",
        "model_rank_sha256": model_hashes,
        "model_identity_method": "native_parameter_hashes_after",
        "base_passthrough": passthrough,
        "files": files,
        "metadata": {
            name: value["sha256"]
            for name, value in files.items()
            if not name.endswith(".safetensors") and name != "STAGING.json"
        },
        "weight_schema_sha256": _json_digest(schema),
        "tensor_count": len(schema),
        "weights_content_sha256": _json_digest(
            {name: value for name, value in files.items() if name.endswith(".safetensors")}
        ),
    }
    validate_training_export_record(record, record)
    atomic_json(staging / COMPLETE_FILE, record)
    staging.rename(destination)
    fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return record


def export_training_weights(
    args,
    model,
    root,
    *,
    step,
    weight_version,
    identity,
    base_checkpoint,
    base_checkpoint_identity,
    checkpoint_bytes,
    retained_complete_bytes=0,
):
    """Collectively export the current native model; call on all four TP ranks."""
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
    from vime.backends.megatron_utils.hf_checkpoint_saver import save_hf_model_to_path

    from rl_engine.integrations.orchestrators.vime.providers.qwen3_next_vime import (
        install_vime_gdn_weight_hooks,
    )

    if (
        type(step) is not int
        or {"update-1": 1, "update-2": 2, "reload-2": 2}.get(weight_version) != step
    ):
        raise ValueError("Native HF export requires an explicit accepted update/reload phase")
    if (
        identity.get("checkpoint") != str(base_checkpoint)
        or identity.get("checkpoint_identity") != base_checkpoint_identity
    ):
        raise ValueError("Native HF export provenance must retain the fixed official baseline")
    if not dist.is_initialized() or dist.get_world_size() != 4:
        raise ValueError("Native HF export requires four actual distributed ranks")
    if (
        parallel_state.get_tensor_model_parallel_world_size(),
        parallel_state.get_expert_tensor_parallel_world_size(),
        parallel_state.get_pipeline_model_parallel_world_size(),
        parallel_state.get_context_parallel_world_size(),
        parallel_state.get_expert_model_parallel_world_size(),
    ) != (4, 4, 1, 1, 1):
        raise ValueError("Native HF export requires TP/ETP4 and PP/CP/EP1")
    if Path(args.hf_checkpoint).resolve() != Path(base_checkpoint).resolve():
        raise ValueError("Native HF conversion metadata must come from the pinned official base")
    if (
        getattr(args, "actor_num_nodes", None) != 1
        or getattr(args, "actor_num_gpus_per_node", None) != 4
    ):
        raise ValueError("Native HF export requires one node writer for its four GPUs")
    if not 0 < args.update_weight_buffer_size <= 4 * 1024**3:
        raise ValueError("Native HF export must use a pinned bounded parameter gather size")
    if save_hf_model_to_path.__module__ != "vime.backends.megatron_utils.hf_checkpoint_saver":
        raise ValueError("Native VIME HF exporter was replaced")
    install_vime_gdn_weight_hooks()
    tracker = get_cuda_rng_tracker()
    parameters = named_trainable_parameters(model)
    if any(value.dtype != torch.bfloat16 for value in parameters.values()):
        raise ValueError("Native HF export must retain BF16 model parameters")
    torch.cuda.synchronize()
    rng_before = state_digest(capture_rng_state(tracker=tracker))

    def parameter_identity():
        return state_digest({name: tensor_digest(value) for name, value in parameters.items()})

    model_before = _all_rank_call(parameter_identity, "native export model identity")
    model_hashes = [None] * 4
    dist.all_gather_object(model_hashes, model_before)
    root = Path(root).absolute()

    def prepare():
        owned_root(root, identity)
        if checkpoint_identity(base_checkpoint) != base_checkpoint_identity:
            raise ValueError("Native HF export official base identity changed before conversion")
        export_bytes = sum(item["size"] for item in base_checkpoint_identity["shards"].values())
        require_checkpoint_disk_space(
            root,
            checkpoint_bytes=checkpoint_bytes,
            export_bytes=export_bytes,
            retained_complete_bytes=retained_complete_bytes,
        )
        staging, destination = create_staging(
            root,
            f"{weight_version}-{uuid.uuid4().hex}",
            {
                "schema_version": 1,
                "scope": KIND,
                "status": "staging",
                "identity": identity,
                "step": step,
                "weight_version": weight_version,
                "base_checkpoint": str(base_checkpoint),
                "base_checkpoint_identity": base_checkpoint_identity,
            },
        )
        return {"staging": str(staging), "destination": str(destination)}

    paths = _rank_zero_call(prepare, "native HF export disk budget and staging")
    save_hf_model_to_path(
        args,
        paths["staging"],
        model,
        model_name="qwen3_next",
        quantization_config=None,
        progress_desc=f"Native HF export {weight_version}",
    )
    torch.cuda.synchronize()

    def verify_immutable():
        if (
            parameter_identity() != model_before
            or state_digest(capture_rng_state(tracker=tracker)) != rng_before
        ):
            raise ValueError("Native HF export unexpectedly changed model parameters or RNG")

    _all_rank_call(verify_immutable, "native export model/RNG immutability")
    record = _rank_zero_call(
        lambda: _publish_export(
            paths["staging"],
            paths["destination"],
            step=step,
            weight_version=weight_version,
            identity=identity,
            base_checkpoint=base_checkpoint,
            base_checkpoint_identity=base_checkpoint_identity,
            model_hashes=model_hashes,
        ),
        "native HF all-file publication",
    )
    return {
        "checkpoint": paths["destination"],
        "checkpoint_identity": record,
        "weight_version": weight_version,
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_identity": base_checkpoint_identity,
    }
