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
