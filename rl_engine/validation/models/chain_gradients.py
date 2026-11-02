# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""C10 required trainable parameter names (#266 / #270 / #276).

C10 compares ``tensor.grad`` after a real ``loss.backward()``. The required
set is every official Qwen3-8B Dense trainable leaf: embedding, final norm,
LM head, and every decoder-layer Q/K/V/O, QK-norm, MLP, and RMSNorm weight.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class GradParamSpec:
    name: str
    kind: str
    op_class: str


def required_grad_names(*, num_hidden_layers: int = 36) -> tuple[str, ...]:
    names = ["embed_tokens.weight", "norm.weight", "lm_head.weight"]
    for index in range(int(num_hidden_layers)):
        prefix = f"layers.{index}"
        names.extend(
            [
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.self_attn.q_norm.weight",
                f"{prefix}.self_attn.k_norm.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
            ]
        )
    return tuple(names)


def _specs_for(names: Sequence[str]) -> tuple[GradParamSpec, ...]:
    specs: list[GradParamSpec] = []
    for name in names:
        if name.endswith("lm_head.weight"):
            kind = "lm_head"
        elif name.endswith("embed_tokens.weight"):
            kind = "embedding"
        elif (
            "layernorm" in name
            or name.endswith("norm.weight")
            or ".q_norm." in name
            or ".k_norm." in name
        ):
            kind = "rms_norm"
        else:
            kind = "linear"
        specs.append(GradParamSpec(name=name, kind=kind, op_class="reduction"))
    return tuple(specs)


REQUIRED_GRAD_NAMES = required_grad_names()
GRAD_PARAM_SPECS = _specs_for(REQUIRED_GRAD_NAMES)
GRADIENT_SCOPE = "all_required_trainable_parameters"
REQUIRED_GRAD_KINDS = frozenset({"embedding", "rms_norm", "linear", "lm_head"})


__all__ = [
    "GRAD_PARAM_SPECS",
    "GRADIENT_SCOPE",
    "REQUIRED_GRAD_KINDS",
    "REQUIRED_GRAD_NAMES",
    "GradParamSpec",
    "required_grad_names",
]
