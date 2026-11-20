# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Native Megatron construction/checkpoint loading, with no model-name registry."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Callable, Mapping

import torch

from .shards import ShardSet, TensorShard


def _exact_checkpoint_shapes(value):
    """Disable native padding/truncation without mutating provider metadata.

    strict='raise_all' checks checkpoint keys, not allow_shape_mismatch.
    Factories must retain their native merge function while tightening the
    descriptors produced by their build function.
    """
    from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory

    if isinstance(value, ShardedTensor):
        return replace(value, allow_shape_mismatch=False)
    if isinstance(value, ShardedTensorFactory):
        original_build = value.build_fn

        def build(*args, **kwargs):
            return _exact_checkpoint_shapes(original_build(*args, **kwargs))

        return replace(value, build_fn=build)
    if isinstance(value, Mapping):
        return {key: _exact_checkpoint_shapes(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_exact_checkpoint_shapes(child) for child in value)
    return value


def load_megatron_model(
    provider: Callable[..., torch.nn.Module],
    checkpoint: str | Path,
    *,
    provider_kwargs: Mapping | None = None,
    model_key: str | None = None,
) -> torch.nn.Module:
    """Construct an unfamiliar model using its *native* provider and load strictly.

    Initialize Megatron's process groups/configuration before calling this on
    every participating worker. ``checkpoint`` is a distributed model checkpoint
    directory, not an HF checkpoint or a pickle containing executable providers.
    ``model_key`` selects a native top-level model envelope, when present.

    The provider owns architecture, device, dtype and layer specs. This function
    does not guess them from a model name, convert weights, initialize process
    groups, or change train/eval mode. A failed load never returns a usable model.
    """
    if not callable(provider):
        raise TypeError("a trusted native Megatron model provider is required")
    checkpoint = Path(checkpoint)
    if not checkpoint.is_dir():
        raise ValueError("Megatron distributed checkpoint directory does not exist")
    if model_key is not None and (not isinstance(model_key, str) or not model_key):
        raise ValueError("model_key must be a nonempty native checkpoint key")
    from megatron.core import dist_checkpointing

    model = provider(**dict(provider_kwargs or {}))
    if not isinstance(model, torch.nn.Module) or not callable(
        getattr(model, "sharded_state_dict", None)
    ):
        raise TypeError("provider must return a native module with sharded_state_dict()")
    state = _exact_checkpoint_shapes(model.sharded_state_dict())
    requested = {model_key: state} if model_key is not None else state
    loaded = dist_checkpointing.load(
        requested,
        str(checkpoint),
        validate_access_integrity=True,
        strict="raise_all",
    )
    model.load_state_dict(loaded[model_key] if model_key is not None else loaded, strict=True)
    return model


def megatron_weight_shards(model: torch.nn.Module) -> dict[str, ShardSet]:
    """Derive global ownership from actual native distributed-checkpoint metadata.

    Handles stacked layer axes and native factories (e.g. SwiGLU splits). The
    complete declared global shape is observed, including any stored padding.
    The checkpoint loader's allow_shape_mismatch flag is not evidence of actual
    padding and does not authorize dropping rows here. Flattened optimizer/DP
    fragments and logical unpadding need dedicated maps. Every tensor is read
    from the actual model, not checkpoint hashes or another worker's evidence.
    """
    from megatron.core.dist_checkpointing.mapping import (
        ShardedObject,
        ShardedTensor,
        ShardedTensorFactory,
    )

    fragments = defaultdict(list)

    def visit(value, path):
        if isinstance(value, ShardedTensorFactory):
            visit(value.build(), path)
        elif isinstance(value, ShardedObject):
            # Native linear modules include an explicit no-extra-state marker.
            # TE additionally uses a zero-length uint8 serialization buffer
            # when quantizer state is absent. Never unpickle nonempty payloads.
            empty_bytes = (
                isinstance(value.data, torch.Tensor)
                and value.data.dtype == torch.uint8
                and value.data.ndim == 1
                and value.data.numel() == 0
            )
            if value.data is not None and not empty_bytes:
                raise ValueError(f"{path}: nonempty native extra state needs an explicit map")
        elif isinstance(value, ShardedTensor):
            if value.flattened_range is not None:
                raise ValueError(f"{value.key}: flattened weights need an explicit map")
            if not isinstance(value.data, torch.Tensor):
                raise ValueError(f"{value.key}: missing native weight data")
            local_shape = (1,) * value.prepend_axis_num + tuple(value.local_shape)
            tensor = value.data.reshape(local_shape)
            if len(local_shape) != len(value.global_shape):
                raise ValueError(f"{value.key}: invalid native shard dimensions")
            indices = [
                range(offset, offset + size)
                for offset, size in zip(value.global_offset, local_shape, strict=True)
            ]
            fragments[value.key].append(TensorShard(tensor, value.global_shape, indices))
        elif isinstance(value, torch.Tensor):
            fragments[path].append(TensorShard.replicated(value))
        elif isinstance(value, Mapping):
            for key, child in value.items():
                visit(child, f"{path}.{key}" if path else str(key))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{path}.{index}")
        elif value is not None:
            # Opaque extra state may affect execution; don't silently certify it.
            raise ValueError(f"{path}: unsupported native checkpoint state {type(value).__name__}")

    visit(model.sharded_state_dict(), "")
    if not fragments:
        raise ValueError("native model exposes no tensor shards")
    return {key: ShardSet(parts) for key, parts in fragments.items()}
