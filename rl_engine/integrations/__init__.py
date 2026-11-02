# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Framework integration entry points owned by RL-Kernel."""

from rl_engine.integrations.engines.rollout.vllm.adapter import VllmIntegration
from rl_engine.integrations.engines.rollout.vllm.runtime import configure_vllm_environment
from rl_engine.integrations.engines.train.megatron.adapter import MegatronIntegration
from rl_engine.integrations.engines.train.megatron.runtime import install_megatron_integration
from rl_engine.runtime.plan import (
    Implementation,
    IntegrationPlan,
    OperatorAblationCase,
    configure_integration_environment,
    integration_plan_from_environment,
    operator_ablation_case,
    operator_ablation_cases,
)
from rl_engine.validation.ablation.rocm import (
    ROCM_ATTENTION_CASE_IDS,
    RocmAblationCaseResult,
    RocmAttentionAblationCase,
    rocm_attention_ablation_matrix,
    run_rocm_attention_ablation,
)

__all__ = [
    "Implementation",
    "IntegrationPlan",
    "MegatronIntegration",
    "OperatorAblationCase",
    "ROCM_ATTENTION_CASE_IDS",
    "RocmAblationCaseResult",
    "RocmAttentionAblationCase",
    "VllmIntegration",
    "configure_integration_environment",
    "configure_vllm_environment",
    "install_megatron_integration",
    "integration_plan_from_environment",
    "operator_ablation_case",
    "operator_ablation_cases",
    "rocm_attention_ablation_matrix",
    "run_rocm_attention_ablation",
]
