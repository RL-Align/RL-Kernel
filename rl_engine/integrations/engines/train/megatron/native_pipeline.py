# SPDX-License-Identifier: Apache-2.0
"""Original checkpoint ownership for native DSV4 pipeline layouts.

The split defaults to the qualified 3+1 (four layers) and 4+3 (seven layers)
layouts; RLK_DSV4_PP_SPLIT selects any other split. This describes model
shards, not independent model replicas. MCore retains its native pipeline
scheduler and multi-stream activation/gradient transport.
"""

import re

from rl_engine.integrations.common.dsv4_checkpoint_partition import DSV4CheckpointPartition

_EXPERT_KEY = re.compile(r"^layers\.\d+\.ffn\.experts\.(\d+)\.")

# Qualified default ownership per original layer count (hash layers 0-2 on PP0).
_DEFAULT_SPLITS = {4: "3,1", 7: "4,3"}


def validate_native_parallel_layout(cfg, world_size=None):
    """Admit EP1, or EP4 only as PP2 x DP4 with TP=ETP=CP=1 and expert DP1.

    ``cfg`` needs the MCore parallel-size attributes. ``world_size`` (optional)
    additionally checks DP == EP, i.e. expert data parallel size 1.
    """
    tp = getattr(cfg, "tensor_model_parallel_size", 1)
    cp = getattr(cfg, "context_parallel_size", 1)
    ep = getattr(cfg, "expert_model_parallel_size", 1)
    etp = getattr(cfg, "expert_tensor_parallel_size", None)
    pp = getattr(cfg, "pipeline_model_parallel_size", 1)
    if tp != 1 or cp != 1 or etp not in (None, 1) or ep not in (1, 4):
        raise ValueError(
            "native packed pipeline requires TP=EP=CP=1 (ETP=1); "
            f"only EP=4 is also admitted, got TP={tp} EP={ep} ETP={etp} CP={cp}"
        )
    if ep == 4:
        if pp != 2:
            raise ValueError(f"native packed EP=4 requires PP=2, got PP={pp}")
        if world_size is not None:
            dp = world_size // (tp * pp * cp)
            if world_size % (tp * pp * cp) or dp != ep:
                raise ValueError(
                    f"native packed EP=4 requires DP == EP (expert DP 1); "
                    f"world_size={world_size} gives DP={dp}"
                )
    return ep


def pp2_split(total):
    """Original layers owned by each PP stage: RLK_DSV4_PP_SPLIT="n0,n1,...".

    Defaults to 3,1 for four layers and 4,3 for seven layers. Two or more stages.
    The first stage must own every hash-routed layer (the first three original layers).
    """
    import os

    value = os.environ.get("RLK_DSV4_PP_SPLIT") or _DEFAULT_SPLITS.get(total)
    if value is None:
        raise ValueError(f"set RLK_DSV4_PP_SPLIT for a {total}-layer pipeline")
    split = tuple(int(x) for x in value.split(","))
    if len(split) < 2 or sum(split) != total or split[0] < 3 or min(split) < 1:
        raise ValueError(
            f"RLK_DSV4_PP_SPLIT={split} does not partition {total} original layers "
            "with the three hash layers on PP0"
        )
    return split


def native_checkpoint_stage(model):
    """Validate native global layer numbers and keep every hash layer on PP0."""
    cfg = model.config
    validate_native_parallel_layout(cfg)
    if getattr(cfg, "virtual_pipeline_model_parallel_size", None) not in (None, 1):
        raise ValueError("native packed pipeline does not support virtual stages")
    pp = cfg.pipeline_model_parallel_size
    if cfg.num_layers % 2:
        raise ValueError("native checkpoint requires whole attention/MoE pairs")
    layers = list(model.decoder.layers)
    numbers = [layer.layer_number for layer in layers]
    if not numbers or len(numbers) % 2 or numbers[0] % 2 != 1:
        raise ValueError("pipeline stage must own whole attention/MoE pairs")
    if numbers != list(range(numbers[0], numbers[0] + len(numbers))):
        raise ValueError("native pipeline layer numbers must be contiguous and global")
    total = cfg.num_layers // 2
    ids = tuple((number - 1) // 2 for number in numbers[::2])
    stage = DSV4CheckpointPartition(ids, model.pre_process, model.post_process, total)
    if pp == 1:
        if ids != tuple(range(total)) or not stage.pre_process or not stage.post_process:
            raise ValueError("PP1 must own the complete checkpoint and both endpoints")
    else:
        split = pp2_split(total)
        if len(split) != pp:
            raise ValueError(f"RLK_DSV4_PP_SPLIT={split} does not match PP={pp}")
        starts = [sum(split[:i]) for i in range(pp)]
        if ids[0] not in starts:
            raise ValueError(
                f"stage layers {ids} do not start on a RLK_DSV4_PP_SPLIT={split} boundary"
            )
        index = starts.index(ids[0])
        expected = (
            tuple(range(starts[index], starts[index] + split[index])),
            index == 0,
            index == pp - 1,
        )
        if (ids, stage.pre_process, stage.post_process) != expected:
            raise ValueError(
                f"PP stage {index} must own original layers {expected[0]} (split {split})"
            )
    return stage


def local_checkpoint_keys(model, keys, ep_rank=None):
    """Keys owned by this PP stage; with EP>1 only locally owned routed experts."""
    stage = native_checkpoint_stage(model)
    owned = {key for key in keys if stage.owns(key)}
    ep = model.config.expert_model_parallel_size
    if ep == 1:
        return stage, owned
    if ep_rank is None:
        from megatron.core import parallel_state

        ep_rank = parallel_state.get_expert_model_parallel_rank()
    return stage, filter_local_expert_keys(owned, model.config.num_moe_experts, ep, ep_rank)


def filter_local_expert_keys(keys, total_experts, ep_size, ep_rank):
    from rl_engine.integrations.engines.train.megatron.expert_partition import (
        OriginalExpertPartition,
    )

    local = set(OriginalExpertPartition(total_experts, ep_size, ep_rank).expert_ids)
    kept = set()
    for key in keys:
        match = _EXPERT_KEY.match(key)
        if match is None or int(match[1]) in local:
            kept.add(key)
    return kept


def original_pipeline_layers(total):
    """PP2 ownership: original layers of stage 0 and stage 1 (hash layers on stage 0)."""
    split = pp2_split(total)
    if len(split) != 2:
        raise ValueError(f"RLK_DSV4_PP_SPLIT={split} is not a two-stage split")
    return tuple(range(split[0])), tuple(range(split[0], total))


def configure_native_pipeline(provider):
    """Use the pinned Bridge layout API and native MCore PP scheduler for the selected split."""
    if provider.pipeline_model_parallel_size == 1:
        return
    if (
        provider.num_layers % 2
        or provider.moe_num_hash_layers != 3
        or getattr(provider, "mtp_num_layers", 0)
        or getattr(provider, "virtual_pipeline_model_parallel_size", None) not in (None, 1)
    ):
        raise ValueError(
            "native packed PP requires whole original layers, three hash layers, no VPP/MTP"
        )
    from megatron.bridge.models.deepseek.deepseek_v4_bridge import (
        set_deepseek_v4_pipeline_model_parallel_layout,
    )
    from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout

    split = pp2_split(provider.num_layers // 2)
    if len(split) != provider.pipeline_model_parallel_size:
        raise ValueError(
            f"RLK_DSV4_PP_SPLIT={split} does not match PP={provider.pipeline_model_parallel_size}"
        )
    set_deepseek_v4_pipeline_model_parallel_layout(provider, logical_layers_per_stage=split)
    provider.num_layers_in_first_pipeline_stage = None
    provider.num_layers_in_last_pipeline_stage = None
    provider.pipeline_model_parallel_layout = PipelineParallelLayerLayout(
        layout=provider.pipeline_model_parallel_layout,
        pipeline_model_parallel_size=provider.pipeline_model_parallel_size,
    )
