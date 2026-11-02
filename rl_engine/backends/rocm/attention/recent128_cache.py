# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Reuse upstream vLLM kernels verbatim: no RLK quantizer, gather, or index kernel."""

from __future__ import annotations

import hashlib
import importlib
import inspect
from pathlib import Path

import torch

from rl_engine.contracts.operators.recent128 import (
    RECENT128_BACKEND_ID,
    Recent128Contract,
    Recent128ContractError,
)

_MODULE = "vllm.models.deepseek_v4.common.ops.cache_utils"


class VllmRecent128Backend:
    """Explicit ROCm Triton path; never dispatch through the CuTe/NVIDIA fallback."""

    backend_id = RECENT128_BACKEND_ID

    def __init__(self) -> None:
        if torch.version.hip is None:
            raise Recent128ContractError("VllmRecent128Backend requires ROCm PyTorch")
        try:
            module = importlib.import_module(_MODULE)
            self._write = module.quantize_and_insert_k_cache
            self._gather = module.dequantize_and_gather_k_cache_triton
            self._indices = module.combine_topk_swa_indices
        except (ImportError, AttributeError) as exc:
            raise Recent128ContractError(
                "installed vLLM lacks the DSV4 cache ABI; install the documented "
                "DSV4/BI revision, not an unrelated vLLM release"
            ) from exc
        for function in (self._write, self._gather):
            if "use_fnuz" not in inspect.signature(function).parameters:
                raise Recent128ContractError("vLLM writer/gather must expose explicit use_fnuz")
        if "out" not in inspect.signature(self._indices).parameters:
            raise Recent128ContractError("vLLM index builder must accept caller-owned outputs")
        source = Path(module.__file__)
        self.provenance = {
            "backend": self.backend_id,
            "module": _MODULE,
            "source": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "torch": torch.__version__,
            "hip": torch.version.hip,
        }

    def write(self, kv, cache, batch, contract: Recent128Contract) -> None:
        if batch.num_tokens:
            self._write(
                kv,
                cache,
                batch.slot_mapping,
                block_size=contract.block_size,
                is_ue8m0=True,
                use_fnuz=contract.use_fnuz,
            )

    def gather(self, cache, batch, workspace, contract: Recent128Contract) -> None:
        if batch.num_actual_tokens:
            self._gather(
                workspace.kv,
                cache,
                batch.seq_lens,
                batch.gather_lens,
                batch.block_table,
                contract.block_size,
                0,
                use_fnuz=contract.use_fnuz,
            )

    def indices(self, batch, workspace, contract: Recent128Contract) -> None:
        # Padded tokens are not visited by the upstream request-local kernel.
        workspace.indices.fill_(-1)
        workspace.lengths.zero_()
        if batch.num_actual_tokens:
            self._indices(
                workspace.empty_topk,
                batch.query_start_loc,
                batch.seq_lens,
                batch.gather_lens,
                contract.window_size,
                1,
                0,
                contract.workspace_width,
                0,
                out=(workspace.indices, workspace.lengths),
            )
