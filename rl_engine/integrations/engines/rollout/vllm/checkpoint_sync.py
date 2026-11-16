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
    return actual_name, shard, actual


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
        if (
            actual.shape != value.shape
            or actual.dtype != value.dtype
            or digest(actual) != declared["sha256"]
        ):
            raise RuntimeError(
                f"rollout parameter readback differs from actor bytes: {logical_name}"
            )
        checked[logical_name] = dict(parameter=actual_name, shard=shard, sha256=declared["sha256"])
        expected_loaded.add(actual_name)
    if set(loaded) != expected_loaded:
        raise RuntimeError(
            f"actual loader names disagree: missing={expected_loaded - set(loaded)}, "
            f"extra={set(loaded) - expected_loaded}"
        )
    return checked
