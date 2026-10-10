# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Runtime adapters for the WS2 Qwen3-8B Megatron + vLLM cross-config target."""

from rl_engine.integrations.common.cross_config import (
    QWEN3_8B,
    AttentionRuntimeReadback,
    Qwen3ModelSpec,
)
from rl_engine.integrations.common.knobs import (
    MEGATRON_ATTENTION_BACKENDS,
    WS2_ATTENTION_KNOB_DESCRIPTORS,
    WS2_ATTENTION_KNOBS,
    WS2_ATTENTION_NORMALIZERS,
)
from rl_engine.integrations.engines.rollout.vllm.cross_config import (
    VllmProvenanceAdapter,
    VllmRolloutMaterializer,
)
from rl_engine.integrations.engines.train.megatron.cross_config import (
    MegatronAttentionMaterializer,
    MegatronProvenanceAdapter,
)

__all__ = [
    "MEGATRON_ATTENTION_BACKENDS",
    "AttentionRuntimeReadback",
    "MegatronAttentionMaterializer",
    "MegatronProvenanceAdapter",
    "QWEN3_8B",
    "Qwen3ModelSpec",
    "VllmProvenanceAdapter",
    "VllmRolloutMaterializer",
    "WS2_ATTENTION_KNOBS",
    "WS2_ATTENTION_KNOB_DESCRIPTORS",
    "WS2_ATTENTION_NORMALIZERS",
]
