# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""SGLang diffusion plugin: the strict H3 rollout adapter inside SGLang's own workers.

SGLang (``sglang.multimodal_gen``) loads its plugins in every worker before it loads a model
(``worker_bootstrap``: ``load_plugins()`` then ``apply_plugin_hooks()``). RL-Kernel registers
``register_sglang_plugin`` under the entry-point group ``sglang.multimodal_gen.plugins``
(``pyproject.toml``). It does nothing unless ``RL_KERNEL_SGLANG_H3_STRICT=1``; then it hooks
``TransformerLoader.load_customized``, and every ``MiniMaxH3DiTModel`` a worker loads gets:

* ``check_sglang_server_args`` on the worker's server settings;
* ``install_h3_sglang_rollout`` in strict mode at SGLang's TP size, with a
  ``DeterministicCollective`` over SGLang's TP group;
* a forward guard. Before each DiT forward it refuses the request's step-skipping settings
  (``check_sglang_sampling_params``); after it, it asserts the readback: every bound forward ran
  exactly once per DiT forward so far (``handle.assert_ran``). A bypassed or skipped bound
  forward stops the worker instead of serving a non-strict result.

With ``RL_KERNEL_SGLANG_H3_READBACK_DIR`` each worker writes its report (``report``, with the call
counts and the kernels) to ``<dir>/rank<global rank>.json`` at install and after every forward.
``RL_KERNEL_SGLANG_H3_COLLECTIVE_BYTES`` sizes the collective's staging buffer (default 512 MiB;
a row-parallel reduction exchanges FP32 partials of ``rows x 5376``). The handle is
``model.rl_kernel_h3_handle``; ``handle.remove()`` restores SGLang's forwards and removes it.
A forward that raises is not counted: neither the model forward nor the bound forwards it ran.
"""

from __future__ import annotations

import inspect
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from rl_engine.integrations.engines.rollout.sglang import h3 as adapter

ENABLE = "RL_KERNEL_SGLANG_H3_STRICT"
READBACK_DIR = "RL_KERNEL_SGLANG_H3_READBACK_DIR"
COLLECTIVE_BYTES = "RL_KERNEL_SGLANG_H3_COLLECTIVE_BYTES"
DEFAULT_COLLECTIVE_BYTES = 512 << 20
TARGET = (
    "sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader."
    "TransformerLoader.load_customized"
)
HANDLE = "rl_kernel_h3_handle"


def enabled() -> bool:
    return os.getenv(ENABLE, "").strip() in {"1", "true", "True"}


def register_sglang_plugin() -> None:
    """SGLang diffusion-plugin entry point; inactive unless ``RL_KERNEL_SGLANG_H3_STRICT=1``."""

    if not enabled():
        return
    from sglang.multimodal_gen.runtime.platforms.plugins import HookType, plugin_hook

    plugin_hook(TARGET, HookType.AFTER)(after_transformer_load)


def after_transformer_load(model, *args, **kwargs):
    """AFTER hook of ``TransformerLoader.load_customized(self, path, server_args, name, ...)``."""

    if type(model).__name__ != "MiniMaxH3DiTModel":
        return model
    from sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader import (
        TransformerLoader,
    )

    bound = inspect.signature(TransformerLoader.load_customized).bind(*args, **kwargs)
    install_in_worker(model, bound.arguments["server_args"])
    return model


def _collective(tp: int):
    if tp == 1:
        return None
    from sglang.multimodal_gen.runtime.distributed.parallel_state import get_tp_group

    from rl_engine.distributed.algorithms.collectives import DeterministicCollective

    size = int(os.getenv(COLLECTIVE_BYTES, DEFAULT_COLLECTIVE_BYTES))
    return DeterministicCollective(group=get_tp_group().device_group, max_size_bytes=size)


def _forward_batch():
    """The request of the current forward, or ``None`` for a direct call outside a request."""

    from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context_or_none

    context = get_forward_context_or_none()
    return None if context is None else context.forward_batch


def install_in_worker(
    model: torch.nn.Module, server_args: Any, *, readback_dir: str | None = None
) -> adapter.H3SGLangRolloutHandle:
    """Install the strict adapter on a worker's ``MiniMaxH3DiTModel`` and guard its forward."""

    from sglang.multimodal_gen.runtime.distributed.parallel_state import get_tp_world_size

    adapter.check_sglang_server_args(server_args)
    tp = get_tp_world_size()
    handle = adapter.install_h3_sglang_rollout(model, collective=_collective(tp), tp=tp)
    directory = readback_dir if readback_dir is not None else os.getenv(READBACK_DIR)
    path = None
    if directory:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        path = Path(directory) / f"rank{rank}.json"
        path.parent.mkdir(parents=True, exist_ok=True)

    def write() -> None:
        if path is not None:
            path.write_text(json.dumps(adapter.report(model, handle), indent=2) + "\n")

    original = model.forward

    def forward(*args, **kwargs):
        batch = _forward_batch()
        if batch is not None:
            adapter.check_sglang_sampling_params(batch)
        before = Counter(handle.calls)
        try:
            out = original(*args, **kwargs)
        except BaseException:
            # A failed forward serves nothing, and SGLang's worker goes on to the next request:
            # uncount the bound forwards it ran, or every later forward would fail the readback.
            handle.calls.clear()
            handle.calls.update(before)
            raise
        handle.forwards += 1
        write()
        handle.assert_ran(handle.forwards)
        return out

    handle._restore.append((model, model.__dict__.get("forward", adapter._ABSENT)))
    model.forward = forward
    handle.overrides.append("forward: guarded (sampling settings, readback after every forward)")
    handle._attributes.append((model, HANDLE, model.__dict__.get(HANDLE, adapter._ABSENT)))
    setattr(model, HANDLE, handle)
    write()
    return handle
