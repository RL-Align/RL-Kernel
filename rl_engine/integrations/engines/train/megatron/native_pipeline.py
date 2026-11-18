# SPDX-License-Identifier: Apache-2.0
"""Original checkpoint ownership for the native four-layer PP2 (3+1) layout.

This describes model shards, not independent model replicas. MCore retains its
native pipeline scheduler and multi-stream activation/gradient transport.
"""

from rl_engine.integrations.common.dsv4_checkpoint_partition import DSV4CheckpointPartition


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
    if pp not in (1, 2) or cfg.num_layers % 2:
        raise ValueError("native checkpoint requires whole attention/MoE pairs and PP=1 or 2")
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
        expected = ((0, 1, 2), True, False) if stage.pre_process else ((3,), False, True)
        if total != 4 or (ids, stage.pre_process, stage.post_process) != expected:
            raise ValueError("four-layer PP2 requires original layers 0-2 / 3 (hash layers on PP0)")
    return stage


def local_checkpoint_keys(model, keys):
    stage = native_checkpoint_stage(model)
    return stage, {key for key in keys if stage.owns(key)}


def configure_native_pipeline(provider):
    """Use the pinned Bridge layout API and native MCore PP scheduler for 3+1."""
    if provider.pipeline_model_parallel_size == 1:
        return
    if (
        provider.pipeline_model_parallel_size != 2
        or provider.num_layers != 8
        or provider.moe_num_hash_layers != 3
        or getattr(provider, "mtp_num_layers", 0)
        or getattr(provider, "virtual_pipeline_model_parallel_size", None) not in (None, 1)
    ):
        raise ValueError(
            "native packed PP2 requires four original layers, three hash layers, no VPP/MTP"
        )
    from megatron.bridge.models.deepseek.deepseek_v4_bridge import (
        set_deepseek_v4_pipeline_model_parallel_layout,
    )
    from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout

    set_deepseek_v4_pipeline_model_parallel_layout(provider, logical_layers_per_stage=(3, 1))
    provider.num_layers_in_first_pipeline_stage = None
    provider.num_layers_in_last_pipeline_stage = None
    provider.pipeline_model_parallel_layout = PipelineParallelLayerLayout(
        layout=provider.pipeline_model_parallel_layout, pipeline_model_parallel_size=2
    )
