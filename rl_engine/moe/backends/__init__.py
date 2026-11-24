# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""MoE kernel backends of the DSv4 MoE block (``rl_engine.models.deepseek_v4.moe``).

``rl_engine.moe.provider`` holds only the generic pieces: the ``ExpertProvider``
protocol, the oracle-backed ``ReferenceProvider`` and the fail-closed ``StubProvider``.

- ``sm90_fused_mlp.Sm90FusedMoeMlp``: the routed experts (MXFP8 x MXFP4), one fused
  kernel for fc1 -> clamp-SwiGLU -> MX re-quantization -> fc3
  (``csrc/cuda/moe/sm90_fused_moe_mlp.cu``) and its dX backward. It fuses the whole MLP
  behind one call, so it is not an ``ExpertProvider``; ``tests/test_sm90_fused_moe_mlp.py``
  validates it.
- ``deepgemm_shared_expert.DeepGemmSharedExpertProvider``: the shared expert on DeepGEMM
  FP8 with the clamped SwiGLU (profile ``dsv4-deepgemm-fp8-v1``).
- ``routed_checks``: the per-launch checks of the routed batch.
"""

from rl_engine.moe.backends.deepgemm_shared_expert import DeepGemmSharedExpertProvider
from rl_engine.moe.backends.sm90_fused_mlp import Sm90FusedMoeMlp

__all__ = ["DeepGemmSharedExpertProvider", "Sm90FusedMoeMlp"]
