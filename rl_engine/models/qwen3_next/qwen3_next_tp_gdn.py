# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""TP4 Qwen3-Next GDN projections, HF weight mapping and explicit recurrent state.

The module is shared by engine adapters. It does not install a vLLM backend or
claim model-level acceptance. The process group must contain exactly four ranks.
"""

from collections.abc import Mapping, Sequence

import torch
import torch.distributed as dist
from torch import nn

from rl_engine.integrations.engines.train.vllm.qwen3_next_provider import (
    GDNProviderConfig,
    GDNState,
    shared_gdn_core,
)
from rl_engine.models.qwen3_next.qwen3_next_tp import _CopyToTP, _rank, _ReduceFromTP
from rl_engine.validation.common.tensor_identity import tensor_bitwise_equal

_GLOBAL_SHAPES = {
    "in_proj_qkvz.weight": (12288, 2048),
    "in_proj_ba.weight": (64, 2048),
    "conv1d.weight": (8192, 1, 4),
    "A_log": (32,),
    "dt_bias": (32,),
    "norm.weight": (128,),
    "out_proj.weight": (2048, 4096),
}
_LOCAL_SHAPES = {
    "in_proj_qkvz.weight": (3072, 2048),
    "in_proj_ba.weight": (16, 2048),
    "conv1d.weight": (2048, 1, 4),
    "A_log": (8,),
    "dt_bias": (8,),
    "norm.weight": (128,),
    "out_proj.weight": (2048, 1024),
}


def _validate_weights(weights, shapes):
    if set(weights) != set(shapes):
        raise ValueError("GDN weight names do not match the explicit HF contract")
    device = weights["A_log"].device
    for name, shape in shapes.items():
        value = weights[name]
        if value.shape != shape or value.dtype != torch.bfloat16 or value.device != device:
            raise ValueError(f"{name} must be BF16 {shape} on {device}")


def shard_gdn_weights(weights: Mapping[str, torch.Tensor], rank: int) -> dict[str, torch.Tensor]:
    """Shard official HF GQA-interleaved projections and Q/K/V-segmented conv."""
    _rank(rank)
    _validate_weights(weights, _GLOBAL_SHAPES)
    result = {}
    for name, value in weights.items():
        if name == "norm.weight":
            shard = value
        elif name == "out_proj.weight":
            shard = value[:, rank * 1024 : (rank + 1) * 1024]
        elif name == "conv1d.weight":
            q, k, v = value.split((2048, 2048, 4096), dim=0)
            shard = torch.cat([part.chunk(4, dim=0)[rank] for part in (q, k, v)])
        else:
            shard = value.chunk(4, dim=0)[rank]
        result[name] = shard.contiguous()
    return result


def assemble_gdn_weights(shards: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Inverse mapping; reject divergent replicated norm weights."""
    if len(shards) != 4:
        raise ValueError("Exactly four ordered TP shards are required")
    for shard in shards:
        _validate_weights(shard, _LOCAL_SHAPES)
    result = {}
    for name in _GLOBAL_SHAPES:
        values = [shard[name] for shard in shards]
        if name == "norm.weight":
            if any(not tensor_bitwise_equal(value, values[0]) for value in values[1:]):
                raise ValueError("Replicated GDN norm weights differ across TP ranks")
            result[name] = values[0]
        elif name == "conv1d.weight":
            segments = [value.split((512, 512, 1024), dim=0) for value in values]
            result[name] = torch.cat(
                [segments[rank][part] for part in range(3) for rank in range(4)]
            )
        else:
            result[name] = torch.cat(values, dim=1 if name == "out_proj.weight" else 0)
    return result


class TP4GDN(nn.Module):
    """Complete GDN projection/core/output block with differentiable TP semantics."""

    def __init__(self, *, group, device, config: GDNProviderConfig, eps: float = 1e-6):
        super().__init__()
        if not dist.is_initialized() or dist.get_world_size(group) != 4:
            raise ValueError("TP4GDN requires an initialized four-rank process group")
        config.validate()
        self.group = dist.group.WORLD if group is None else group
        self.provider_config, self.eps = config, eps
        self.rank = dist.get_rank(group)
        factory = {"device": device, "dtype": torch.bfloat16}
        self.in_proj_qkvz = nn.Linear(2048, 3072, bias=False, **factory)
        self.in_proj_ba = nn.Linear(2048, 16, bias=False, **factory)
        self.conv1d = nn.Conv1d(2048, 2048, 4, groups=2048, bias=False, **factory)
        self.A_log = nn.Parameter(torch.zeros(8, **factory))
        self.dt_bias = nn.Parameter(torch.zeros(8, **factory))
        self.norm = nn.Module()
        self.norm.register_parameter("weight", nn.Parameter(torch.ones(128, **factory)))
        self.out_proj = nn.Linear(1024, 2048, bias=False, **factory)

    def load_hf_weights(self, weights: Mapping[str, torch.Tensor]):
        self.load_state_dict(shard_gdn_weights(weights, self.rank), strict=True)

    def initial_state(self, blocks: int) -> GDNState:
        if blocks < 2:
            raise ValueError("State must include a reserved slot and at least one active slot")
        return GDNState(
            self.A_log.new_zeros((blocks, 2048, 3)),
            torch.zeros(blocks, 8, 128, 128, dtype=torch.float32, device=self.A_log.device),
        )

    def forward(self, hidden, state, indices, cu_seqlens):
        from rl_engine.models.qwen3_next.qwen3_next_forward import shared_linear

        if hidden.ndim != 2 or hidden.shape[1] != 2048 or hidden.dtype != torch.bfloat16:
            raise ValueError("GDN hidden states must be BF16 [tokens, 2048]")
        copied = _CopyToTP.apply(hidden, self.group)
        qkvz = shared_linear(copied, self.in_proj_qkvz.weight).reshape(-1, 4, 768)
        ba = shared_linear(copied, self.in_proj_ba.weight).reshape(-1, 4, 4)
        q, k, v, z = qkvz.split((128, 128, 256, 256), dim=-1)
        b, a = ba.split(2, dim=-1)
        tokens = hidden.shape[0]
        packed = torch.cat(
            [value.reshape(tokens, width) for value, width in ((q, 512), (k, 512), (v, 1024))],
            dim=-1,
        )
        # The norm weight is shared by heads on all four ranks; its gradient
        # sums contributions from all local head partitions exactly once.
        norm_weight = _CopyToTP.apply(self.norm.weight, self.group)
        output, next_state = shared_gdn_core(
            packed,
            a.reshape(tokens, 8),
            b.reshape(tokens, 8),
            z.reshape(tokens, 8, 128),
            self.A_log.float(),
            self.dt_bias.float(),
            self.conv1d.weight[:, 0],
            norm_weight,
            state,
            indices,
            cu_seqlens,
            config=self.provider_config,
            num_k_heads=4,
            eps=self.eps,
        )
        local = shared_linear(output.reshape(tokens, 1024), self.out_proj.weight)
        return _ReduceFromTP.apply(local, self.group), next_state
