# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Contract for DSV4 recent-128 cache storage, not an attention arithmetic contract."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from rl_engine.contracts.operators.attention import AttentionContractError, RoPEState

RECENT128_BACKEND_ID = "vllm.dsv4.triton.paged_ue8m0.recent128.v1"
RECENT128_LAYOUT_ID = "dsv4.fp8_448.bf16_64.ue8m0_group64.v1"


class Recent128ContractError(AttentionContractError):
    """Reject an unsupported invocation before modifying cache storage."""


class CacheFP8Encoding(str, Enum):
    OCP = "e4m3fn"
    FNUZ = "e4m3fnuz"


@dataclass(frozen=True)
class CacheVersion:
    """Framework-owned version barrier; a LoRA or scale change invalidates KV."""

    model: str
    adapter: str
    scales: str

    def __post_init__(self) -> None:
        for name in ("model", "adapter", "scales"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise Recent128ContractError(f"{name} version must be a non-empty string")


@dataclass(frozen=True)
class CacheRequest:
    """An incarnation, not the current packed-batch row number."""

    request_id: str
    generation: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise Recent128ContractError("request_id must be a non-empty string")
        if type(self.generation) is not int or self.generation < 0:
            raise Recent128ContractError("generation must be a non-negative integer")


@dataclass(frozen=True)
class Recent128Contract:
    """Both sides must agree on the actual cache encoding, not the GPU default.

    The fixed workspace width is independent of neighboring request lengths.
    The scheduler owns allocation/reclamation and must retain old-prefix plus
    chunk pages until the queued gather completes. CP/Graph are explicit gaps.
    """

    max_query_tokens: int
    block_size: int = 64
    encoding: CacheFP8Encoding = CacheFP8Encoding.OCP
    window_size: int = 128
    head_dim: int = 512
    rope_state: RoPEState = RoPEState.POST_ROPE
    cp_size: int = 1
    graph_capture: bool = False
    backend_id: str = RECENT128_BACKEND_ID
    layout_id: str = RECENT128_LAYOUT_ID

    def __post_init__(self) -> None:
        if type(self.max_query_tokens) is not int or self.max_query_tokens <= 0:
            raise Recent128ContractError("max_query_tokens must be a positive integer")
        if self.max_query_tokens > 2**31 - 128:
            raise Recent128ContractError("workspace indices must fit int32")
        if type(self.block_size) is not int or self.block_size not in (64, 128):
            raise Recent128ContractError("validated page sizes are 64 and 128")
        if type(self.window_size) is not int or self.window_size != 128:
            raise Recent128ContractError("recent128 requires window_size=128")
        if type(self.head_dim) is not int or self.head_dim != 512:
            raise Recent128ContractError("DSV4 cache requires head_dim=512 (448+64)")
        try:
            object.__setattr__(self, "encoding", CacheFP8Encoding(self.encoding))
            object.__setattr__(self, "rope_state", RoPEState(self.rope_state))
        except ValueError as exc:
            raise Recent128ContractError("unsupported encoding or RoPE state") from exc
        if self.rope_state != RoPEState.POST_ROPE:
            raise Recent128ContractError("cache writer consumes post-RoPE KV; it never rotates")
        if type(self.cp_size) is not int or self.cp_size != 1:
            raise Recent128ContractError("CP ownership adapter is not validated; require CP1")
        if self.graph_capture is not False:
            raise Recent128ContractError("stateful Graph replay is not validated")
        if self.backend_id != RECENT128_BACKEND_ID or self.layout_id != RECENT128_LAYOUT_ID:
            raise Recent128ContractError("unknown backend/layout; no implicit fallback")

    @property
    def page_bytes(self) -> int:
        # Same format as vLLM and Megatron _packed_cache: token data, scales,
        # then page alignment. 576 is NOT the complete per-token allocation.
        return ((self.block_size * 584 + 575) // 576) * 576

    @property
    def workspace_width(self) -> int:
        return self.window_size - 1 + self.max_query_tokens

    @property
    def use_fnuz(self) -> bool:
        return self.encoding == CacheFP8Encoding.FNUZ

    def require_megatron_visible_compatible(self) -> None:
        # The referenced native helper explicitly allocates 64-token OCP pages.
        if self.block_size != 64 or self.encoding != CacheFP8Encoding.OCP:
            raise Recent128ContractError("Megatron native visible-K helper requires OCP/64")
