# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Plain selected-logprob API backed by the deterministic Triton kernel."""

import torch

from rl_engine.backends.shared.triton.logprob.batch_invariant_logp import TritonBatchInvariantLogpOp


class TritonLogpOp(TritonBatchInvariantLogpOp):
    def __call__(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        ignore_index: int = -100,
        *,
        validate: bool = True,
    ) -> torch.Tensor:
        return super().__call__(logits, token_ids, ignore_index=ignore_index, validate=validate)

    def forward(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        return self.__call__(logits, token_ids)

    def forward_fp32(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        return self.__call__(logits, token_ids)
