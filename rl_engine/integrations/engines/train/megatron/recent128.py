# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Keep Megatron's native visible-K arithmetic and VJP; never mutate rollout KV."""

from __future__ import annotations

import importlib

import torch

from rl_engine.contracts.operators.recent128 import Recent128Contract, Recent128ContractError

_RUNTIME = "megatron.lite.model.deepseek_v4.vllm.primitive.attention.runtime"


def recent128_visible_k(
    functional_k: torch.Tensor,
    *,
    contract: Recent128Contract,
) -> torch.Tensor:
    """Delegate to mLite's quantized_main_k_visible, including its native gradient.

    This is a training/recompute tensor boundary, not a paged inference update.
    It shares the OCP/64 storage contract with the rollout writer and does not
    claim to implement the surrounding sparse attention or CP backward.
    """
    contract.require_megatron_visible_compatible()
    if (
        functional_k.ndim != 2
        or functional_k.shape[1] != 512
        or functional_k.dtype != torch.bfloat16
    ):
        raise Recent128ContractError("Megatron visible-K expects post-RoPE BF16 [tokens,512]")
    try:
        runtime = importlib.import_module(_RUNTIME)
        native = runtime.quantized_main_k_visible
    except (ImportError, AttributeError) as exc:
        raise Recent128ContractError(
            "the documented Megatron mLite DSV4 runtime is required; "
            "no replacement quantizer or surrogate backward is selected"
        ) from exc
    return native(functional_k)
