# SPDX-License-Identifier: Apache-2.0
"""Original checkpoint ownership for the native four-layer PP2 (3+1) layout.

This describes model shards, not independent model replicas. MCore retains its
native pipeline scheduler and multi-stream activation/gradient transport.
"""

from rl_engine.integrations.common.dsv4_checkpoint_partition import DSV4CheckpointPartition


def pp2_split(total):
    """Original layers owned by each PP stage: RLK_DSV4_PP_SPLIT="n0,n1,..." (default 3,1).

    Two or more stages. The first stage must own every hash-routed layer (the first three
    original layers).
    """
    import os

    split = tuple(int(x) for x in os.environ.get("RLK_DSV4_PP_SPLIT", "3,1").split(","))
    if len(split) < 2 or sum(split) != total or split[0] < 3 or min(split) < 1:
        raise ValueError(
            f"RLK_DSV4_PP_SPLIT={split} does not partition {total} original layers "
            "with the three hash layers on PP0"
        )
    return split


def native_checkpoint_stage(model):
    """Validate native global layer numbers and keep every hash layer on PP0."""
    cfg = model.config
    if any(
        getattr(cfg, key) != 1
        for key in (
            "tensor_model_parallel_size",
            "expert_model_parallel_size",
            "context_parallel_size",
        )
    ):
        raise ValueError("native packed pipeline requires TP=EP=CP=1")
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


def local_checkpoint_keys(model, keys):
    stage = native_checkpoint_stage(model)
    return stage, {key for key in keys if stage.owns(key)}


def configure_native_pipeline(provider):
    """Use the pinned Bridge layout API and native MCore PP scheduler for 3+1."""
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

    set_deepseek_v4_pipeline_model_parallel_layout(
        provider, logical_layers_per_stage=pp2_split(provider.num_layers // 2)
    )
    provider.num_layers_in_first_pipeline_stage = None
    provider.num_layers_in_last_pipeline_stage = None
    provider.pipeline_model_parallel_layout = PipelineParallelLayerLayout(
        layout=provider.pipeline_model_parallel_layout,
        pipeline_model_parallel_size=provider.pipeline_model_parallel_size,
    )
