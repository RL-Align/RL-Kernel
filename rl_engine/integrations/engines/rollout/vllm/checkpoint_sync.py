# SPDX-License-Identifier: Apache-2.0
"""Read back every original nonquantized tensor after the actual vLLM loader."""

import hashlib

import torch


def digest(tensor):
    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def pipeline_stage_payload(wrapper, payload):
    """Select only state owned by this actual rollout stage, without changing dtype."""
    from rl_engine.integrations.common.dsv4_checkpoint_partition import DSV4CheckpointPartition

    weights, manifest = payload["weights"], payload["manifest"]
    if not weights or set(weights) != set(manifest):
        raise ValueError("nonempty actor update and manifest must have identical keys")
    model = wrapper.model
    stage = DSV4CheckpointPartition(
        tuple(range(model.start_layer, model.end_layer)),
        hasattr(model.embed_tokens, "weight"),
        hasattr(model.norm, "weight") and hasattr(wrapper.lm_head, "weight"),
        wrapper.config.num_hidden_layers,
    )
    owned = {name: value for name, value in weights.items() if stage.owns(name)}
    if not owned:
        raise ValueError("actor update has no state for this rollout pipeline stage")
    return dict(payload, weights=owned, manifest={name: manifest[name] for name in owned})


def readback_original_tensor(wrapper, logical_name, value):
    """Resolve the exact local parameter slice used by the actual vLLM loader."""
    actual_name = wrapper.hf_to_vllm_mapper.map_name(logical_name)
    if actual_name is None:
        raise ValueError(f"actor update would be ignored: {logical_name}")
    shard = None
    for stem, index in ((".compressor.wkv.", 0), (".compressor.wgate.", 1)):
        if stem in actual_name:
            actual_name = actual_name.replace(stem, ".compressor.fused_wkv_wgate.")
            shard = index
            break
    params = dict(wrapper.named_parameters())
    if actual_name not in params:
        raise ValueError(f"missing actual rollout parameter for {logical_name}: {actual_name}")
    actual = params[actual_name]
    if shard is not None:
        if actual.shape[0] != 2 * value.shape[0]:
            raise ValueError(f"unsupported compressor merge shape: {actual_name}")
        actual = actual.narrow(0, shard * value.shape[0], value.shape[0])
    if logical_name in ("embed.weight", "head.weight"):
        owner = wrapper.model.embed_tokens if logical_name == "embed.weight" else wrapper.lm_head
        indices = getattr(owner, "shard_indices", None)
        if indices is None:
            if actual.shape != value.shape:
                raise ValueError("sharded vocabulary requires native shard metadata")
            return actual_name, shard, actual
        if indices.num_added_elements:
            raise ValueError("original checkpoint readback does not support adapter vocabulary")
        actual = actual.narrow(0, 0, indices.num_org_elements)
    return actual_name, shard, actual


def local_original_value(wrapper, logical_name, value, actual):
    """Use real loader ownership for original floating-state TP readback."""
    if logical_name in ("embed.weight", "head.weight") and hasattr(
        wrapper.model.embed_tokens if logical_name == "embed.weight" else wrapper.lm_head,
        "shard_indices",
    ):
        owner = wrapper.model.embed_tokens if logical_name == "embed.weight" else wrapper.lm_head
        indices = owner.shard_indices
        start, stop = indices.org_vocab_start_index, indices.org_vocab_end_index
        expected = value.narrow(0, start, stop - start)
        partition = dict(axis=0, start=start, stop=stop, logical_shape=list(value.shape))
    elif tuple(actual.shape) == tuple(value.shape):
        expected, partition = value, None
    else:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        tp, rank = get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank()
        if (
            logical_name.endswith(".attn.attn_sink")
            and value.ndim == actual.ndim == 1
            and value.shape[0] == actual.shape[0] * tp
        ):
            start, stop = rank * actual.shape[0], (rank + 1) * actual.shape[0]
            expected = value.narrow(0, start, stop - start)
            partition = dict(axis=0, start=start, stop=stop, logical_shape=list(value.shape))
        else:
            raise ValueError(
                f"unqualified original tensor TP ownership: {logical_name}: "
                f"{value.shape} -> {actual.shape}"
            )
    if expected.shape != actual.shape:
        raise ValueError(f"actual loader shard shape differs: {logical_name}")
    return expected, partition


def verify_loaded_update(wrapper, payload, loaded):
    weights, manifest = payload["weights"], payload["manifest"]
    if set(weights) != set(manifest) or not weights:
        raise ValueError("nonempty actor update and manifest must have identical keys")
    expected_loaded = set()
    checked = {}
    for logical_name, value in weights.items():
        declared = manifest[logical_name]
        if (
            list(value.shape) != declared["shape"]
            or str(value.dtype) != declared["dtype"]
            or digest(value) != declared["sha256"]
        ):
            raise ValueError(f"actor update payload differs from its manifest: {logical_name}")
        actual_name, shard, actual = readback_original_tensor(wrapper, logical_name, value)
        expected, partition = local_original_value(wrapper, logical_name, value, actual)
        expected_digest = digest(expected)
        if actual.dtype != expected.dtype or digest(actual) != expected_digest:
            raise RuntimeError(
                f"rollout parameter readback differs from actor bytes: {logical_name}"
            )
        checked[logical_name] = dict(
            parameter=actual_name,
            shard=shard,
            sha256=expected_digest,
            dtype=str(actual.dtype),
            shape=list(actual.shape),
        )
        if partition is not None and list(actual.shape) != list(value.shape):
            checked[logical_name].update(
                tp_partition=partition,
                logical_sha256=declared["sha256"],
                logical_shape=declared["shape"],
            )
        expected_loaded.add(actual_name)
    if set(loaded) != expected_loaded:
        raise RuntimeError(
            f"actual loader names disagree: missing={expected_loaded - set(loaded)}, "
            f"extra={set(loaded) - expected_loaded}"
        )
    return checked
