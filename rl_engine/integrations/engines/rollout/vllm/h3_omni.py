# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RFC #420 ``vllm_runtime_adapter_h3``: strict H3 projections inside a vLLM-Omni DiT.

vLLM-Omni's ``MiniMaxH3Attention`` / MLP keep fused linears: ``attn.qkv_proj`` (``[Q;K;V]`` rows),
``attn.out_proj`` (row parallel), ``mlp.fc1`` (gate-first), ``mlp.fc2`` (row parallel); each
returns ``(output, bias)``. ``install_h3_rollout(model)`` replaces those four forwards with the
fixed-order kernels so rollout reads the same bits as the strict training graph:

* ``qkv_proj`` / ``fc1``: ``h3-projection-gemm-v1`` over the fused rows (a column of a fused
  projection equals the same column computed alone, so fused == separate bitwise);
* ``out_proj`` / ``fc2``: the canonical 8-unit row-parallel reduction of ``h3-parallel-v1``
  (``tp`` ranks with a deterministic collective, ``tp=1`` alone), one BF16 rounding.

H3's four projections have no bias, and the patched forward computes ``x @ W.T`` and returns
``(output, None)``. A module that carries a bias is therefore refused in strict mode (and left
on vLLM-Omni's own forward otherwise) instead of silently losing its bias.

Attention, rotary and norm stay vLLM-Omni's (their rows are separate); the report lists them as
``external``. Everything is restored by ``handle.remove()``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch

from rl_engine.backends.cuda.model_specific.minimax_h3 import parallel as par
from rl_engine.backends.shared.triton.gemm import h3_projection as g

LINEAR_NAMES = {"qkv_proj": "column", "fc1": "column", "out_proj": "row", "fc2": "row"}
# The four projections of a transformer block or a token-refiner block, by module path (an outer
# wrapper prefix is allowed). A module elsewhere that happens to share a leaf name is not touched.
_PROJECTION_PATH = re.compile(
    r"(?:^|\.)(?:token_refiner\.)?blocks\.\d+\.(?:attn\.(?:qkv_proj|out_proj)|mlp\.(?:fc1|fc2))$"
)
_ABSENT = object()


@dataclass
class H3RolloutHandle:
    patched: list[str] = field(default_factory=list)
    _restore: list = field(default_factory=list)  # (module, its own forward or _ABSENT)

    def remove(self) -> None:
        """Put back exactly what each module had: its own instance ``forward`` or none."""

        for module, previous in reversed(self._restore):
            if previous is _ABSENT:
                module.__dict__.pop("forward", None)
            else:
                module.forward = previous
        self._restore.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()

    def report(self) -> dict:
        return {
            "contract": par.CONTRACT,
            "patched": list(self.patched),
            "external": ["attention", "rotary", "qk_norm", "swiglu", "adaln"],
        }


def _wrap(module, kind, collective, tp):
    def forward(x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).contiguous()
        if kind == "column":
            out = g.gemm(flat, module.weight, b_t=True)
        else:
            out = par.row_parallel_linear(flat, module.weight, collective, tp)
        return out.view(*shape[:-1], out.shape[-1]), None

    return forward


def install_h3_rollout(
    model: torch.nn.Module, *, collective=None, tp: int = 1, strict: bool = True
) -> H3RolloutHandle:
    """Patch every ``qkv_proj`` / ``out_proj`` / ``fc1`` / ``fc2`` of ``blocks`` and the refiner.

    Only modules at a block projection path are considered. Every candidate is checked before
    any forward changes, so a strict refusal leaves the model untouched.
    """

    if tp > 1 and collective is None:
        raise ValueError("tp > 1 needs a deterministic collective")
    selected = []
    for name, module in model.named_modules():
        if not _PROJECTION_PATH.search(name) or not hasattr(module, "weight"):
            continue
        weight = module.weight
        if not weight.is_cuda or weight.dtype != torch.bfloat16:
            if strict:
                raise RuntimeError(
                    f"{name}: strict H3 rollout needs CUDA BF16 weights, got "
                    f"{weight.dtype} on {weight.device}"
                )
            continue
        if getattr(module, "bias", None) is not None:
            if strict:
                raise RuntimeError(
                    f"{name}: has a bias; the strict H3 rollout kernels compute the bias-free H3 "
                    "projections only"
                )
            continue
        selected.append((name, module))
    if strict and not selected:
        raise RuntimeError("no H3 projection modules found (qkv_proj/out_proj/fc1/fc2)")
    handle = H3RolloutHandle()
    for name, module in selected:
        kind = LINEAR_NAMES[name.rsplit(".", 1)[-1]]
        handle._restore.append((module, module.__dict__.get("forward", _ABSENT)))
        module.forward = _wrap(module, kind, collective, tp)
        handle.patched.append(name)
    return handle
