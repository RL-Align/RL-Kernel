# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""PPO adapters for Megatron Core models: scalar Reward/Critic heads and policy logprobs.

Framework imports are lazy. Validated configuration: dense Megatron TP, no
sequence/pipeline parallelism.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from rl_engine.integrations.common.tp_scalar import TPModelInputs


def megatron_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Convert bool [B,S] token validity to Megatron's True-means-blocked mask.

    Padding queries attend their own diagonal so a left-padded query never has
    a fully masked softmax row; they remain excluded from all PPO objectives.
    """
    length = attention_mask.shape[1]
    causal = torch.ones((length, length), dtype=torch.bool, device=attention_mask.device).tril()
    allowed = causal[None] & attention_mask[:, None, :]
    diagonal = torch.eye(length, dtype=torch.bool, device=attention_mask.device)
    allowed |= (~attention_mask)[:, :, None] & diagonal[None]
    return ~allowed[:, None]


class MegatronScalarAdapter(torch.nn.Module):
    """GPTModel(post_process=False) plus an additive feature-sharded scalar head.

    head is a Linear(H/TP, 1, bias=False); bias is held only on the TP leader.
    The head's full input is scattered using Megatron's autograd mapping, so
    its backward gathers all feature-gradient shards before the backbone.
    Use TPScalarExecutor(output_layout='local_sum') around this adapter.
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        head: torch.nn.Linear,
        *,
        group: dist.ProcessGroup,
        leader: int,
    ) -> None:
        super().__init__()
        if group is None or dist.get_rank(group) < 0:
            raise ValueError("scalar adapter requires an explicit TP group containing this rank")
        if leader not in dist.get_process_group_ranks(group):
            raise ValueError("scalar head leader must belong to the TP group")
        if backbone.pg_collection.tp is not group:
            raise ValueError("backbone and scalar head must use the same TP group")
        if backbone.post_process:
            raise ValueError("scalar backbone must have post_process=False")
        if backbone.config.sequence_parallel or backbone.config.pipeline_model_parallel_size != 1:
            raise ValueError("scalar adapter requires no SP/PP")
        if (
            head.bias is not None
            or head.out_features != 1
            or head.in_features * dist.get_world_size(group) != backbone.config.hidden_size
        ):
            raise ValueError("head must be Linear(hidden_size/TP, 1, bias=False)")
        self.backbone, self.head, self.group = backbone, head, group
        head.weight.tensor_model_parallel = True
        head.weight.partition_dim = 1
        head.weight.partition_stride = 1
        if dist.get_rank() == leader:
            self.bias = torch.nn.Parameter(
                torch.zeros(1, device=head.weight.device, dtype=head.weight.dtype)
            )
        else:
            self.register_buffer(
                "bias", torch.zeros(1, device=head.weight.device, dtype=head.weight.dtype)
            )

    def forward(self, input_ids, attention_mask, position_ids, use_cache=False):
        from megatron.core.tensor_parallel.mappings import scatter_to_tensor_model_parallel_region

        if use_cache:
            raise ValueError("PPO scalar forward does not use KV cache")
        hidden = self.backbone(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=megatron_attention_mask(attention_mask),
        )
        local = scatter_to_tensor_model_parallel_region(hidden, group=self.group)
        return (self.head(local).squeeze(-1) + self.bias).transpose(0, 1).contiguous()


class MegatronPolicyAdapter(torch.nn.Module):
    """Selected-token logprobs from GPTModel's vocabulary-parallel output."""

    def __init__(self, model: torch.nn.Module) -> None:
        from megatron.core import parallel_state

        super().__init__()
        if not model.post_process or not model.parallel_output:
            raise ValueError("policy requires post_process=True and parallel_output=True")
        if model.config.sequence_parallel or model.config.pipeline_model_parallel_size != 1:
            raise ValueError("policy adapter requires no SP/PP")
        if model.pg_collection.tp is not parallel_state.get_tensor_model_parallel_group():
            raise ValueError("policy CE and model must use the same Megatron TP group")
        self.model = model

    def forward(self, inputs: TPModelInputs) -> torch.Tensor:
        from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy

        logits = self.model(
            input_ids=inputs.input_ids,
            position_ids=inputs.position_ids,
            attention_mask=megatron_attention_mask(inputs.attention_mask),
        )
        # Megatron CE owns the vocabulary collectives and their backward semantics.
        return -vocab_parallel_cross_entropy(
            logits[:, :-1].float().contiguous(), inputs.input_ids[:, 1:].contiguous()
        )


def megatron_parameter_layouts(model: torch.nn.Module) -> dict[str, str]:
    """Use Megatron's parameter annotations; a leader-only scalar bias is sharded."""
    layouts = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            layouts[name] = (
                "sharded" if getattr(parameter, "tensor_model_parallel", False) else "replicated"
            )
    for module_name, module in model.named_modules():
        if (
            isinstance(module, MegatronScalarAdapter)
            and isinstance(module.bias, torch.nn.Parameter)
            and module.bias.requires_grad
        ):
            layouts[f"{module_name}.bias" if module_name else "bias"] = "sharded"
    return layouts
