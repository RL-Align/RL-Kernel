# SPDX-License-Identifier: Apache-2.0
"""Resolve exported buffers after model placement or precision wrapping replaces them."""

import torch


def checkpoint_export_value(model, tensor):
    if isinstance(tensor, torch.nn.Parameter):
        return tensor
    names = [
        name for name, buffer in model.named_buffers(remove_duplicate=False) if buffer is tensor
    ]
    if len(names) != 1:
        raise ValueError("checkpoint export requires one registered buffer or a Parameter")
    parent, _, name = names[0].rpartition(".")

    def current_buffer():
        value = getattr(model.get_submodule(parent), name)
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"checkpoint buffer is no longer available: {names[0]}")
        return value

    return current_buffer


def copy_original_tensor(destination, source, name):
    """Load original state without silently narrowing frozen routing metadata.

    Preserve the destination object registered by the framework. Native hash
    kernels accept I64 or form an explicitly derived I32 ABI view themselves.
    """
    if destination.shape != source.shape:
        raise ValueError(f"shape mismatch: {name}: {source.shape} -> {destination.shape}")
    with torch.no_grad():
        if destination.dtype != source.dtype:
            if not (
                name.endswith(".ffn.gate.tid2eid")
                and source.dtype == torch.int64
                and destination.dtype == torch.int32
                and not destination.requires_grad
            ):
                raise TypeError(
                    f"original dtype mismatch: {name}: {source.dtype} -> {destination.dtype}"
                )
            destination.data = source.detach().clone()
        else:
            destination.copy_(source)
