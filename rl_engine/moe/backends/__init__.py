# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""MoE kernel backends.

``rl_engine.moe.provider`` holds only the generic pieces -- the
``ExpertProvider`` protocol, the oracle-backed ``ReferenceProvider``, and the
fail-closed ``StubProvider``. Everything that binds to a real kernel lives
here, one module per operator family.

Production path
---------------
``sm90_fused_mlp.Sm90FusedMoeMlp`` is the routed expert's execution path: one
fused kernel for fc1 -> clamp-SwiGLU -> MX re-quantization -> fc3
(``csrc/cuda/moe/sm90_fused_moe_mlp.cu``). It fuses the whole MLP behind a
single call instead of implementing the nine per-operator hooks, so it is not
an ``ExpertProvider`` and ``scripts/check_p5.py`` cannot drive it; it is
validated by ``tests/test_sm90_fused_moe_mlp.py`` instead.

Reference backends
------------------
The remaining modules are the per-operator P5 artifacts. Each overrides only
the operators its kernel delivers and leaves the rest on the FP32 oracle, so
``check_p5.py`` runs end to end against any one of them. They exist to pin the
numerics operator by operator; they are not the path production takes.

==================== ====================================================
``mxfp8_act_quant``  P5-1 activation quantization (CUDA + Triton)
``grouped_gemm``     P5-4 base GEMM + P5-3 LoRA delta (CUDA)
``lora_delta``       P5-3 LoRA delta (torch-native / CUDA / Triton)
``clamp_swiglu``     P5-2 clamp-SwiGLU with route weight (CUDA)
``shared_expert``    P5-5 shared expert MLP (CUDA + Triton)
==================== ====================================================
"""

from rl_engine.moe.backends.clamp_swiglu import ClampSwiGLUWeightedCudaProvider
from rl_engine.moe.backends.grouped_gemm import CudaP5GemmProvider
from rl_engine.moe.backends.lora_delta import (
    LoRADeltaCudaProvider,
    LoRADeltaProvider,
    LoRADeltaTritonProvider,
)
from rl_engine.moe.backends.mxfp8_act_quant import (
    CudaMXFP8ActQuantProvider,
    TritonMXFP8ActQuantProvider,
)
from rl_engine.moe.backends.shared_expert import (
    CudaDetSharedExpertProvider,
    CudaSharedExpertProvider,
    TritonDetSharedExpertProvider,
    TritonSharedExpertProvider,
)
from rl_engine.moe.backends.sm90_fused_mlp import Sm90FusedMoeMlp

__all__ = [
    # Production path.
    "Sm90FusedMoeMlp",
    # Per-operator reference backends.
    "ClampSwiGLUWeightedCudaProvider",
    "CudaMXFP8ActQuantProvider",
    "CudaDetSharedExpertProvider",
    "CudaP5GemmProvider",
    "CudaSharedExpertProvider",
    "LoRADeltaCudaProvider",
    "LoRADeltaProvider",
    "LoRADeltaTritonProvider",
    "TritonDetSharedExpertProvider",
    "TritonMXFP8ActQuantProvider",
    "TritonSharedExpertProvider",
]
