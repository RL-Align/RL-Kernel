# File: rl_engine/backends/cuda/attention/__init__.py

from rl_engine.backends.cuda.attention.deterministic_attn import (
    DeterministicAttentionCoreResult,
    DeterministicAttentionOp,
    RLKernelDeterministicAttentionCore,
)
from rl_engine.backends.cuda.attention.flash_attn import (
    FlashAttentionOp,
    StrictFlashAttention4Core,
    StrictFlashAttentionUnavailable,
)
from rl_engine.backends.cuda.attention.prefix_shared_attn import PrefixSharedAttentionOp

__all__ = [
    "DeterministicAttentionCoreResult",
    "DeterministicAttentionOp",
    "RLKernelDeterministicAttentionCore",
    "FlashAttentionOp",
    "PrefixSharedAttentionOp",
    "StrictFlashAttention4Core",
    "StrictFlashAttentionUnavailable",
]


# CP communication and FlashInfer are optional layers owned by later WS2 PRs.
# Keep the base Attention package importable while those PRs are developed or
# tested independently, then expose their symbols automatically when present.
try:
    from rl_engine.backends.cuda.attention.cp_comm import (
        AttentionCPBlockMetadata,
        AttentionCPCommunication,
        AttentionCPCommunicationPlan,
        AttentionCPCommunicationUnavailable,
        AttentionCPMergedState,
        AttentionCPOutputShard,
        AttentionCPPartialState,
        AttentionParallelSpec,
        CPCommunicationBackend,
        CPCommunicationStatus,
        CUDAAGRSAttentionCPCommunication,
        P2PNCCLAttentionCPCommunication,
        RCCLAGRSAttentionCPCommunication,
        sort_attention_cp_partial_states,
    )
except ModuleNotFoundError as exc:
    if exc.name != f"{__package__}.cp_comm":
        raise
else:
    __all__ += [
        "AttentionCPBlockMetadata",
        "AttentionCPCommunication",
        "AttentionCPCommunicationPlan",
        "AttentionCPCommunicationUnavailable",
        "AttentionCPMergedState",
        "AttentionCPOutputShard",
        "AttentionCPPartialState",
        "AttentionParallelSpec",
        "CPCommunicationBackend",
        "CPCommunicationStatus",
        "CUDAAGRSAttentionCPCommunication",
        "P2PNCCLAttentionCPCommunication",
        "RCCLAGRSAttentionCPCommunication",
        "sort_attention_cp_partial_states",
    ]

try:
    from rl_engine.backends.cuda.attention.flashinfer_paged_attention import (
        FlashInferPagedAttentionConfig,
        FlashInferQwen3PagedAttentionOp,
        FlashInferRoPEFusionConfig,
        FlashInferSplitKVPolicy,
        FlashInferUnavailable,
    )
except ModuleNotFoundError as exc:
    if exc.name != f"{__package__}.flashinfer_paged_attention":
        raise
else:
    __all__ += [
        "FlashInferPagedAttentionConfig",
        "FlashInferQwen3PagedAttentionOp",
        "FlashInferRoPEFusionConfig",
        "FlashInferSplitKVPolicy",
        "FlashInferUnavailable",
    ]
