# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU double of the Megatron DDP gradient plumbing (Megatron-LM 1dcf0da) used by T05 tests.

Mirrors only what the gradient observer depends on:
* ``distributed_data_parallel.py``: a hook on each trainable parameter's
  AccumulateGrad node adds ``.grad`` to ``main_grad`` unless
  ``grad_added_to_main_grad`` is set, then clears ``.grad``;
  ``zero_grad_buffer`` resets the flags (not under TE CUDA graphs) and zeros
  the buffer; ``finish_grad_sync`` closes the window.
* ``tensor_parallel/layers.py``: the fused-wgrad backward writes ``main_grad``
  directly, sets the flag, and returns a dummy weight gradient.
* ``param_and_grad_buffer.py``: ``param_index_map`` and ``bucket_indices``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

DOUBLE_OF = "megatron.core.distributed.DistributedDataParallel (gradient plumbing only)"


class FusedWgradLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, weight: torch.nn.Parameter) -> torch.Tensor:
        ctx.save_for_backward(inputs)
        ctx.weight = weight
        return inputs @ weight.t()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        (inputs,) = ctx.saved_tensors
        weight = ctx.weight
        weight.main_grad.add_(grad_output.t() @ inputs)
        weight.grad_added_to_main_grad = True
        return grad_output @ weight, torch.zeros_like(weight)  # dummy wgrad, as in Megatron


class TinyModel(torch.nn.Module):
    """base (frozen) + adapter (autograd wgrad) + head (fused wgrad) + unused (never reached)."""

    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Parameter(torch.eye(4), requires_grad=False)
        self.adapter = torch.nn.Parameter(torch.full((4, 4), 0.25))
        self.head = torch.nn.Parameter(torch.full((3, 4), 0.5))
        self.unused = torch.nn.Parameter(torch.ones(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x @ self.base.t() + x @ self.adapter.t()
        return FusedWgradLinear.apply(hidden, self.head)


class FakeMegatronDDP:
    def __init__(self, module: torch.nn.Module, *, cuda_graph_impl: str = "none") -> None:
        self.module = module
        self.config = SimpleNamespace(cuda_graph_impl=cuda_graph_impl)
        self.params_with_grad = [p for p in module.parameters() if p.requires_grad]
        offset = 0
        param_index_map = {}
        for param in self.params_with_grad:
            param.grad_added_to_main_grad = False
            param.main_grad = torch.zeros_like(param)
            param_index_map[param] = (offset, offset + param.numel(), 0)
            offset += param.numel()
        self.buffers = [
            SimpleNamespace(
                param_index_map=param_index_map,
                bucket_indices=[(0, offset)],
                grad_data=torch.zeros(offset),
            )
        ]
        self.grad_accs = []
        for param in self.params_with_grad:
            grad_acc = param.expand_as(param).grad_fn.next_functions[0][0]
            grad_acc.register_hook(self._make_backward_post_hook(param))
            self.grad_accs.append(grad_acc)
        self.sync_calls = 0

    def _make_backward_post_hook(self, param: torch.nn.Parameter):
        def hook(*unused):
            if param.grad is not None and not param.grad_added_to_main_grad:
                param.main_grad.add_(param.grad.data)
            param.grad = None

        return hook

    def zero_grad_buffer(self) -> None:
        if self.config.cuda_graph_impl != "transformer_engine":
            for param in self.params_with_grad:
                param.grad_added_to_main_grad = False
        for param in self.params_with_grad:
            param.main_grad.zero_()

    def finish_grad_sync(self) -> None:
        self.sync_calls += 1
        data = self.buffers[0].grad_data
        for param, (start, end, _bucket) in self.buffers[0].param_index_map.items():
            data[start:end] = param.main_grad.reshape(-1)
