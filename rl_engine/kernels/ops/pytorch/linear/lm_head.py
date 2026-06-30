# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import torch


class NativeLMHeadOp:
    """
    Pure PyTorch native language-model-head reference.
    out = hidden @ weight.t() (+ bias)

    Projects hidden states back to vocabulary logits -- the final layer of the
    Qwen3/Llama stack. For Qwen3-8B the weight is the output projection
    ``[vocab=151936, hidden=4096]`` in the HF ``nn.Linear`` ``[out, in]``
    convention, so it is transposed internally (``weight.t()``). This is the
    one difference from the bare ``matmul`` op (which computes ``a @ b`` with no
    transpose) -- do not use them interchangeably. The lm_head weight is
    *independent* from the embedding table (``tie_word_embeddings=false``), and
    Qwen3 has no bias (pass ``bias=None``).

    Unlike embedding (a lossless row gather), this is a reduction over the
    ``hidden`` dimension: the low-precision ``forward`` path accumulates in the
    input dtype and therefore drifts from the fp32 ``forward_fp32`` ground
    truth. Axis-B accuracy uses a tolerance (``torch.allclose``), not bitwise
    equality. Axis-A batch invariance still holds bitwise within a single dtype
    (each output row reduces over ``hidden`` independently of the batch).
    """

    def __init__(self) -> None:
        """No state; the op is a pure function over (hidden, weight, bias)."""

    def __call__(
        self,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Alias for ``forward`` so the op is callable like a module."""
        return self.forward(hidden, weight, bias=bias)

    def forward(
        self,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Canonical entry: project in the input dtype, output the input dtype.
        This is the dtype-behavior path used as the Axis-B accuracy candidate.
        """
        return self._lm_head(
            hidden, weight, bias, compute_dtype=hidden.dtype, output_dtype=hidden.dtype
        )

    def forward_fp32(
        self,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Ground truth: upcast to fp32, accumulate in fp32, force fp32 output.

        The matmul is wrapped to disable autocast and TF32 so this stays a true
        fp32 reference regardless of the caller's ambient precision context.
        """
        return self._lm_head(
            hidden,
            weight,
            bias,
            compute_dtype=torch.float32,
            output_dtype=torch.float32,
            strict_fp32=True,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _lm_head(
        hidden: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        *,
        compute_dtype: torch.dtype,
        output_dtype: torch.dtype,
        strict_fp32: bool = False,
    ) -> torch.Tensor:
        """Core matmul: cast to ``compute_dtype``, project, optionally add bias, cast out.

        When ``strict_fp32`` is set, the matmul runs with autocast disabled and
        CUDA TF32 turned off so the fp32 reference path is not silently downcast
        by the caller's ambient autocast/TF32 settings.
        """
        h = hidden.to(compute_dtype)
        w = weight.to(compute_dtype)
        # [..., hidden] @ [hidden, vocab] -> [..., vocab]; weight is [vocab, hidden] (HF [out, in]).
        if strict_fp32:
            with NativeLMHeadOp._strict_fp32_matmul(h.device.type):
                out = h @ w.t()
        else:
            out = h @ w.t()
        if bias is not None:
            out = out + bias.to(compute_dtype)
        return out.to(output_dtype)

    @staticmethod
    @contextmanager
    def _strict_fp32_matmul(device_type: str):
        """Disable autocast and TF32 for a true fp32 matmul, restoring state after."""
        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            with torch.autocast(device_type=device_type, enabled=False):
                yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32
