# SPDX-License-Identifier: Apache-2.0
"""Read back every original nonquantized tensor after the actual vLLM loader."""

import hashlib

import torch


def digest(tensor):
    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def verify_loaded_update(wrapper, payload, loaded):
    weights, manifest = payload["weights"], payload["manifest"]
    if set(weights) != set(manifest) or not weights:
        raise ValueError("nonempty actor update and manifest must have identical keys")
    params = dict(wrapper.named_parameters())
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
        actual_name = wrapper.hf_to_vllm_mapper.map_name(logical_name)
        if actual_name is None:
            raise ValueError(f"actor update would be ignored: {logical_name}")
        shard = None
        for stem, index in ((".compressor.wkv.", 0), (".compressor.wgate.", 1)):
            if stem in actual_name:
                actual_name = actual_name.replace(stem, ".compressor.fused_wkv_wgate.")
                shard = index
                break
        if actual_name not in params:
            raise ValueError(f"missing actual rollout parameter for {logical_name}: {actual_name}")
        actual = params[actual_name]
        if shard is not None:
            if actual.shape[0] != 2 * value.shape[0]:
                raise ValueError(f"unsupported compressor merge shape: {actual_name}")
            actual = actual.narrow(0, shard * value.shape[0], value.shape[0])
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
