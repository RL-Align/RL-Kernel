# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""P5-1 providers: MXFP8 activation quantization backends.

Each provider subclasses :class:`~rl_engine.moe.provider.ReferenceProvider` and
overrides only the operators its backend delivers, per the start-kit protocol
(``docs/design/dsv4_p5_expert_start_kit.md``); everything else stays on the
FP32 oracle so ``scripts/check_p5.py`` runs end to end from day one.

Both P5-1 backends are byte-equal with the oracle, so they inherit the oracle's
numeric profile instead of registering a relaxed one:

    python scripts/check_p5.py --provider \
        rl_engine.moe.backends.mxfp8_act_quant:TritonMXFP8ActQuantProvider --device cuda
    python scripts/check_p5.py --provider \
        rl_engine.moe.backends.mxfp8_act_quant:CudaMXFP8ActQuantProvider --device cuda

Fail-closed: the backend is resolved in ``__init__`` and raises
``NotImplementedError`` when unavailable (what ``check_p5.py`` renders as a
FAIL row), a CPU tensor or a non-finite input raises instead of silently
falling back to the oracle, and the non-finite read-back is always on — the
P5-1 spec makes the raise part of the contract.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from rl_engine.moe.mx_format import MXTensor
from rl_engine.moe.provider import ReferenceProvider


class _MXFP8ActQuantProvider(ReferenceProvider):
    """Shared plumbing: the subclass supplies ``_resolve()`` -> (fwd, bwd, linkage)."""

    backend: str

    def __init__(self) -> None:
        try:
            self._fwd, self._bwd, self._linkage = self._resolve()
        except (ImportError, RuntimeError) as exc:
            raise NotImplementedError(
                f"{self.name} backend unavailable (fail-closed, no oracle fallback): {exc}"
            ) from exc

    def _resolve(self) -> tuple[Callable[..., MXTensor], Callable[..., torch.Tensor], str]:
        raise NotImplementedError

    def capabilities(self) -> dict[str, Any]:
        return {
            **super().capabilities(),
            "backend": self.backend,
            "operators": ["mxfp8_act_quant"],
            "devices": ["cuda"],
            "dtypes": ["bfloat16", "float16", "float32"],
            "byte_equal_with_oracle": True,
        }

    def provenance(self) -> dict[str, Any]:
        return {
            **super().provenance(),
            "operators_overridden": ["mxfp8_act_quant_fwd", "mxfp8_act_quant_bwd"],
            "linkage": self._linkage,
            "device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        }

    def mxfp8_act_quant_fwd(self, x: torch.Tensor) -> MXTensor:
        return self._fwd(x)

    def mxfp8_act_quant_bwd(self, dy: torch.Tensor) -> torch.Tensor:
        return self._bwd(dy)


class TritonMXFP8ActQuantProvider(_MXFP8ActQuantProvider):
    """Triton MXFP8 activation quantization; every other operator is the oracle."""

    name = "mxfp8-act-quant-triton"
    backend = "triton"

    def _resolve(self):
        from rl_engine.kernels.ops.triton import moe

        return moe.mxfp8_act_quant_fwd_triton, moe.mxfp8_act_quant_bwd_triton, "triton-jit"


class CudaMXFP8ActQuantProvider(_MXFP8ActQuantProvider):
    """CUDA MXFP8 activation quantization; every other operator is the oracle."""

    name = "mxfp8-act-quant-cuda"
    backend = "cuda"

    def _resolve(self):
        from rl_engine.kernels.ops.cuda import moe

        moe.mxfp8_act_quant.backend()  # AOT symbols or the JIT build, or raise now
        return moe.mxfp8_act_quant_fwd_cuda, moe.mxfp8_act_quant_bwd_cuda, moe.backend_name()
