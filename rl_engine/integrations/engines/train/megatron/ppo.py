# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""PPO adapters for Megatron Core models: scalar Reward/Critic heads and policy logprobs.

Framework imports are lazy. Validated configuration: dense Megatron TP, no
sequence/pipeline parallelism.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from rl_engine.integrations.common.tp_scalar import (
    TPModelInputs,
    canonical_feature_dot,
    sum_tp_contributions,
)
from rl_engine.reference.logprob.vocab_parallel_logp import DEFAULT_NUM_VOCAB_TILES


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
    """GPTModel(post_process=False) plus a feature-sharded scalar head.

    head is a Linear(H/TP, 1, bias=False); bias is held only on the TP leader.
    The head's full input is scattered using Megatron's autograd mapping, so
    its backward gathers all feature-gradient shards before the backbone.
    The head output follows the v0.1.0 reduction tree: FP32 leaves over
    ``H / canonical_tp`` features, a local subtree, the fixed-tree TP
    all-reduce, then the bias. Outputs are replicated; use
    TPScalarExecutor(output_layout='replicated') around this adapter.
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        head: torch.nn.Linear,
        *,
        group: dist.ProcessGroup,
        leader: int,
        canonical_tp: int = 8,
    ) -> None:
        """Validate the TP layout, mark ``head.weight`` TP-sharded on dim 1 and create the bias.

        ``head.weight`` is annotated ``partition_dim=1`` (sharded over input features).
        ``leader`` is a global rank in ``group``; only it holds a trainable zero bias, while the
        other ranks hold a zero buffer. ``canonical_tp`` fixes the head's leaf count and
        must be the same for every TP size whose results are compared. Raises ``ValueError``
        for a missing group or leader, a backbone on another TP group, with ``post_process``
        or with SP/PP, or a head that is not ``Linear(hidden_size / TP, 1, bias=False)``.
        """
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
        self.canonical_tp = canonical_tp
        canonical_feature_dot(
            head.weight.new_zeros(head.in_features),
            head.weight[0],
            tp_world=dist.get_world_size(group),
            canonical_tp=canonical_tp,
        )
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
        """Return the replicated FP32 per-token scalar as a ``[B, S]`` tensor.

        ``attention_mask`` is the bool token-validity mask; ``use_cache=True`` is rejected.
        The leader's bias is added after the TP reduction, outside the tree.
        """
        from megatron.core.tensor_parallel.mappings import scatter_to_tensor_model_parallel_region

        if use_cache:
            raise ValueError("PPO scalar forward does not use KV cache")
        hidden = self.backbone(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=megatron_attention_mask(attention_mask),
        )
        local = scatter_to_tensor_model_parallel_region(hidden, group=self.group)
        partial = canonical_feature_dot(
            local,
            self.head.weight[0],
            tp_world=dist.get_world_size(self.group),
            canonical_tp=self.canonical_tp,
        )
        values = sum_tp_contributions(partial, self.group)
        bias = sum_tp_contributions(self.bias.float(), self.group)
        return (values + bias).transpose(0, 1).contiguous()


class MegatronPolicyAdapter(torch.nn.Module):
    """Selected-token logprobs from GPTModel's vocabulary-parallel output.

    Logprobs use RL-Kernel's deterministic vocab-parallel logp: FP32 per-tile
    ``(max, sumexp)`` partials over ``num_vocab_tiles`` fixed tiles move by
    all-gather and are merged in global tile order, so the result does not
    depend on the TP size when the padded vocabulary and tile count match.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        real_vocab_size: int | None = None,
        num_vocab_tiles: int = DEFAULT_NUM_VOCAB_TILES,
    ) -> None:
        """Wrap a vocabulary-parallel GPTModel (``post_process`` and ``parallel_output`` set).

        Raises ``ValueError`` if the model uses SP/PP or its TP group is not
        ``parallel_state``'s tensor-model-parallel group.
        ``real_vocab_size``, when given, excludes padded vocabulary entries from the logprobs.
        ``num_vocab_tiles`` must divide the padded vocabulary with tile-aligned TP shards.
        """
        from megatron.core import parallel_state

        super().__init__()
        if not model.post_process or not model.parallel_output:
            raise ValueError("policy requires post_process=True and parallel_output=True")
        if model.config.sequence_parallel or model.config.pipeline_model_parallel_size != 1:
            raise ValueError("policy adapter requires no SP/PP")
        if model.pg_collection.tp is not parallel_state.get_tensor_model_parallel_group():
            raise ValueError("policy logp and model must use the same Megatron TP group")
        self.model = model
        self.real_vocab_size = real_vocab_size
        self.num_vocab_tiles = num_vocab_tiles

    def forward(self, inputs: TPModelInputs) -> torch.Tensor:
        """Return FP32 logprobs of each next input token as a ``[B, S-1]`` tensor.

        Vocabulary entries at or beyond ``real_vocab_size`` are excluded. Positions whose next
        token is not attended are zero.
        """
        from rl_engine.contracts.operators.logprob import (
            LogprobContract,
            LogprobDType,
            LogprobRole,
            MaskSpec,
            ReductionSpec,
            ShardingSpec,
        )
        from rl_engine.reference.logprob.vocab_parallel_logp import VocabParallelLogprobOp

        logits = self.model(
            input_ids=inputs.input_ids,
            position_ids=inputs.position_ids,
            attention_mask=megatron_attention_mask(inputs.attention_mask),
        )
        group = self.model.pg_collection.tp
        rank, world = dist.get_rank(group), dist.get_world_size(group)
        batch, width, local_vocab = logits.shape[0], logits.shape[1] - 1, logits.shape[-1]
        padded = local_vocab * world
        local_logits = logits[:, :-1].float().reshape(batch * width, local_vocab)
        # A row is active where its target token is attended, so a padding ID outside the real
        # vocabulary at a masked position is not scored; inactive rows return zero.
        active = tuple(bool(v) for v in inputs.attention_mask[:, 1:].reshape(-1).tolist())
        contract = LogprobContract(
            role=LogprobRole.TRAIN,
            dtype=LogprobDType.FP32,
            mask=MaskSpec(num_tokens=batch * width, active_mask=active),
            sharding=ShardingSpec(
                tp_rank=rank,
                tp_world_size=world,
                vocab_shard_bounds=tuple(
                    (r * local_vocab, (r + 1) * local_vocab) for r in range(world)
                ),
                real_vocab_size=self.real_vocab_size or padded,
                padded_vocab_size=padded,
            ),
            reduction=ReductionSpec(),
        )
        logps, _ = VocabParallelLogprobOp()(
            local_logits,
            inputs.input_ids[:, 1:].reshape(-1),
            contract=contract,
            tp_group=group,
            num_vocab_tiles=self.num_vocab_tiles,
        )
        return logps.reshape(batch, width)


_PARTIAL_GRADIENT_MODULES = ("q_layernorm", "k_layernorm")


def megatron_parameter_layouts(model: torch.nn.Module) -> dict[str, str]:
    """Use Megatron's parameter annotations; a leader-only scalar bias is sharded.

    Q/K RMSNorm weights are replicated but normalize only this rank's heads,
    so their gradients are TP partial sums: they get the ``partial`` layout.
    """
    layouts = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if getattr(parameter, "tensor_model_parallel", False):
                layouts[name] = "sharded"
            elif set(name.split(".")).intersection(_PARTIAL_GRADIENT_MODULES):
                layouts[name] = "partial"
            else:
                layouts[name] = "replicated"
    for module_name, module in model.named_modules():
        if (
            isinstance(module, MegatronScalarAdapter)
            and isinstance(module.bias, torch.nn.Parameter)
            and module.bias.requires_grad
        ):
            layouts[f"{module_name}.bias" if module_name else "bias"] = "sharded"
    return layouts
