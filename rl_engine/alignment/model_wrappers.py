# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import torch

from rl_engine.testing import selected_logprobs_reference


def _require_tensor(value: Any, source: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    raise TypeError(f"{source} must be a torch.Tensor, got {type(value)!r}")


def extract_logits(model_output: Any) -> torch.Tensor:
    """Extract logits from common model output shapes."""

    if isinstance(model_output, torch.Tensor):
        return model_output

    if isinstance(model_output, Mapping):
        if "logits" not in model_output:
            raise TypeError("model output mapping does not contain a 'logits' entry")
        return _require_tensor(model_output["logits"], "model output['logits']")

    logits = getattr(model_output, "logits", None)
    if logits is not None:
        return _require_tensor(logits, "model output.logits")

    if isinstance(model_output, (tuple, list)) and model_output:
        return extract_logits(model_output[0])

    raise TypeError(f"model output does not expose logits: {type(model_output)!r}")


class PolicyModelWrapper(torch.nn.Module):
    """Standard adapter for the trainable policy model used by RL losses."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, **model_kwargs: Any) -> Any:
        return self.model(input_ids, **model_kwargs)

    def forward_logits(self, input_ids: torch.Tensor, **model_kwargs: Any) -> torch.Tensor:
        return extract_logits(self.forward(input_ids, **model_kwargs))

    def selected_logprobs(
        self,
        input_ids: torch.Tensor,
        token_ids: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        output_dtype: torch.dtype = torch.float32,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        logits = self.forward_logits(input_ids, **model_kwargs)
        return selected_logprobs_reference(
            logits,
            token_ids,
            mask=mask,
            temperature=temperature,
            output_dtype=output_dtype,
        )


class ReferenceModelWrapper(PolicyModelWrapper):
    """Standard adapter for the frozen reference model used by KL penalties."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        freeze: bool = True,
        eval_mode: bool = True,
    ):
        super().__init__(model)
        if freeze:
            self.freeze()
        if eval_mode:
            self.eval()

    def freeze(self) -> "ReferenceModelWrapper":
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        return self

    def forward_logits(self, input_ids: torch.Tensor, **model_kwargs: Any) -> torch.Tensor:
        with torch.no_grad():
            return super().forward_logits(input_ids, **model_kwargs)

    def selected_logprobs(
        self,
        input_ids: torch.Tensor,
        token_ids: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        output_dtype: torch.dtype = torch.float32,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        with torch.no_grad():
            return super().selected_logprobs(
                input_ids,
                token_ids,
                mask=mask,
                temperature=temperature,
                output_dtype=output_dtype,
                **model_kwargs,
            )
