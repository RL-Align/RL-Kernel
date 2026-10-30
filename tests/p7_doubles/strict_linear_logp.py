# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU stand-in for the strict CUDA ``LinearLogpWrapper`` (numerics are NOT the contract).

Only the call surface and the provenance keys mirror the CUDA branch of
``rl_engine.integrations.linear_logp.LinearLogpWrapper``; the mismatch block is
taken from the real ``_mismatch_provenance``. The arithmetic is a plain fp32
reference and certifies nothing; GPU tests run the real wrapper through the
same gate.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from rl_engine.integrations.linear_logp import LinearLogpWrapper

DOUBLE_OF = "rl_engine.integrations.linear_logp.LinearLogpWrapper (CUDA branch provenance)"


class CpuStrictLinearLogpDouble:
    backend_id = LinearLogpWrapper.backend_id

    def __init__(
        self,
        *,
        entrypoint: Optional[str] = None,
        out_dtype: torch.dtype = torch.float32,
        poison_rows: tuple[int, ...] = (),
        extra_provenance: Optional[dict[str, Any]] = None,
        freeze_after_calls: Optional[int] = None,
    ) -> None:
        self._entrypoint = entrypoint
        self._out_dtype = out_dtype
        self._poison_rows = poison_rows
        self._extra = dict(extra_provenance or {})
        self._freeze_after = freeze_after_calls
        self._calls = 0
        self._last_provenance: dict[str, Any] = {}

    @property
    def provenance(self) -> dict[str, Any]:
        return dict(self._last_provenance)

    def _publish(self, provenance: dict[str, Any]) -> None:
        self._calls += 1
        if self._freeze_after is not None and self._calls > self._freeze_after:
            return  # simulate stale provenance left from an earlier call
        provenance.update(self._extra)
        self._last_provenance = provenance

    def _finish(self, logits: torch.Tensor, target_ids: torch.Tensor, start: int) -> torch.Tensor:
        local = target_ids.to(torch.long) - start
        owned = (local >= 0) & (local < logits.size(1))
        values = (
            logits.float().log_softmax(-1).gather(1, local.clamp(0, logits.size(1) - 1).view(-1, 1))
        )
        result = torch.where(owned.view(-1, 1), values, torch.zeros_like(values)).view(-1)
        for row in self._poison_rows:
            result[row] = float("nan")
        return result.to(self._out_dtype)

    def __call__(
        self,
        hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_ids: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        *,
        tp_group: Any = None,
        vocab_start_index: int = 0,
        global_vocab_size: Optional[int] = None,
        real_vocab_size: Optional[int] = None,
        target: str = "training",
        temperature: Any = None,
    ) -> torch.Tensor:
        logits = hidden.float() @ lm_head_weight.float().t()
        if bias is not None:
            logits = logits + bias.float()
        tensor_parallel = tp_group is not None or int(vocab_start_index) != 0
        entry = self._entrypoint or (
            "sm90_deterministic_linear_logp_tp"
            if tensor_parallel
            else "sm90_deterministic_linear_logp"
        )
        global_vocab = (
            int(global_vocab_size) if global_vocab_size is not None else lm_head_weight.size(0)
        )
        self._publish(
            {
                **LinearLogpWrapper._mismatch_provenance(),
                "target": target,
                "runtime_platform": "cuda",
                "triton_used": False,
                "actual_backend": self.backend_id,
                "logprob_kernel_backend": self.backend_id,
                "deterministic_linear_logp": True,
                "strict_entrypoint": entry,
                "hidden_shape": list(hidden.shape),
                "hidden_dtype": str(hidden.dtype).replace("torch.", ""),
                "lm_head_weight_shape": list(lm_head_weight.shape),
                "target_shape": list(target_ids.shape),
                "tp_group_present": tp_group is not None,
                "vocab_start_index": int(vocab_start_index),
                "global_vocab_size": global_vocab,
                "requested_global_vocab_size": global_vocab,
                "real_vocab_size": (
                    int(real_vocab_size) if real_vocab_size is not None else global_vocab
                ),
                "requested_real_vocab_size": real_vocab_size,
                "temperature": None if temperature is None else "provided",
                "contract_version": "cuda-det-gemm-linear-logp-sm90-contract-v2",
                "logits_materialized": True,
            }
        )
        return self._finish(logits, target_ids, int(vocab_start_index))

    def from_local_logits(
        self,
        local_logits: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        tp_group: Any,
        vocab_start_index: int,
        global_vocab_size: int,
        real_vocab_size: int,
        target: str = "rollout",
        temperature: Any = None,
        return_lse: bool = False,
        diagnostics_hidden: Optional[torch.Tensor] = None,
        diagnostics_lm_head_weight: Optional[torch.Tensor] = None,
    ) -> Any:
        self._publish(
            {
                **LinearLogpWrapper._mismatch_provenance(),
                "target": target,
                "runtime_platform": "cuda",
                "triton_used": False,
                "actual_backend": self.backend_id,
                "logprob_kernel_backend": self.backend_id,
                "deterministic_linear_logp": True,
                "strict_entrypoint": self._entrypoint
                or "sm90_deterministic_logp_from_local_logits_tp",
                "local_logits_shape": list(local_logits.shape),
                "target_shape": list(target_ids.shape),
                "tp_group_present": True,
                "vocab_start_index": int(vocab_start_index),
                "global_vocab_size": int(global_vocab_size),
                "real_vocab_size": int(real_vocab_size),
                "temperature": None if temperature is None else "provided",
                "contract_version": "cuda-det-gemm-linear-logp-sm90-contract-v2",
                "logits_materialized": True,
                "lm_head_result_reused": True,
            }
        )
        result = self._finish(local_logits, target_ids, int(vocab_start_index))
        if return_lse:
            return result, local_logits.float().logsumexp(-1)
        return result
