# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Reviewed framework/model adapters, separate from plan selection."""

from dataclasses import dataclass
from typing import Any, Callable

from .catalog import RuntimeContext


@dataclass(frozen=True)
class PlanAdapter:
    validate: Callable[[RuntimeContext], None]
    initialize_training: Callable[[Any], Any]
    initialize_rollout: Callable[[], None]


def _validate_qwen3_vime(context: RuntimeContext) -> None:
    import json

    if context.model_id != "qwen3-8b" or context.framework != "vime":
        raise ValueError("the vime.qwen3.v1 adapter requires Qwen3-8B and Vime")
    config = json.loads(context.model_config)
    if config.get("quantization_config") or config.get("rope_scaling"):
        raise ValueError("the Qwen3 reference BI recipe does not support quantization/rope scaling")
    if config.get("use_sliding_window", False):
        raise ValueError("the Qwen3 reference BI recipe does not support sliding-window attention")
    expected_gpu = {"cuda": "H100", "rocm": "MI300X"}.get(context.platform)
    if (
        expected_gpu is None
        or context.topology != (8, 4, 2, 4, 1)
        or len(context.hardware) != 8
        or any(expected_gpu not in gpu for gpu in context.hardware)
    ):
        raise ValueError(
            "this BI adapter currently covers 8x H100/MI300X, train TP4/CP2, rollout TP4"
        )


def _initialize_qwen3_training(args: Any) -> Any:
    from rl_engine.integrations.megatron_runtime import _initialize_qwen3_from_environment

    return _initialize_qwen3_from_environment(args)


def _initialize_qwen3_rollout() -> None:
    from rl_engine.integrations.vllm_runtime import install_vllm_integration, plan_from_environment

    install_vllm_integration(plan_from_environment())


def builtin_adapters() -> dict[str, PlanAdapter]:
    # Model contributors add their actual paired installers here. No plugin is
    # imported from a user environment variable or downloaded at launch time.
    return {
        "vime.qwen3.v1": PlanAdapter(
            _validate_qwen3_vime, _initialize_qwen3_training, _initialize_qwen3_rollout
        )
    }


def get_adapter(adapter_id: str) -> PlanAdapter:
    try:
        return builtin_adapters()[adapter_id]
    except KeyError as exc:
        raise ValueError(f"BI plan has no shipped adapter: {adapter_id}") from exc
