# SPDX-License-Identifier: Apache-2.0
"""CPU checks on the real pristine pinned dependencies, never model parity."""

import torch


def test_pristine_native_engine_and_instance_forward_abi():
    from rl_engine.integrations.orchestrators.verl.native_engine import (
        RLKNativeMcoreEngineConfig,
        RLKNativeMegatronEngine,
    )
    from megatron.core.transformer.moe import moe_layer
    from megatron.core.transformer.experimental_attention_variant import (
        deepseek_v4_hybrid_attention as dsv4,
    )

    DSv4HybridAttention = dsv4.DSv4HybridAttention
    from rl_engine.integrations.engines.train.megatron.native_group_projection import (
        grouped_forward,
    )
    from megatron.core.distributed import DistributedDataParallelConfig

    assert RLKNativeMcoreEngineConfig().strategy == "rlk_megatron"
    assert RLKNativeMegatronEngine.__module__.startswith("rl_engine.")
    assert not hasattr(moe_layer, "te_checkpoint")
    assert not hasattr(DSv4HybridAttention, "_apply_grouped_output_projection")
    original = DSv4HybridAttention.forward
    candidate = grouped_forward(original)
    assert DSv4HybridAttention.forward is original
    assert candidate.__code__.co_filename.endswith("native_group_projection.py")
    ddp = DistributedDataParallelConfig(use_distributed_optimizer=False, grad_reduce_in_fp32=True)
    ddp.finalize()
    assert ddp.grad_reduce_in_fp32


def test_pristine_verl_unwrap_real_module_and_float16_wrapper():
    from rl_engine.integrations.orchestrators.verl import native_engine  # noqa: F401
    from megatron.core.transformer.module import Float16Module
    from megatron.core.transformer.transformer_config import TransformerConfig
    from verl.utils.megatron_utils import unwrap_model

    leaf = torch.nn.Linear(4, 4, bias=False)
    parameter = leaf.weight
    cfg = TransformerConfig(num_layers=1, hidden_size=4, num_attention_heads=1, bf16=True)
    wrapped = Float16Module(cfg, leaf)
    assert unwrap_model(leaf) is leaf
    assert unwrap_model(wrapped) is leaf
    assert unwrap_model([wrapped, leaf]) == [leaf, leaf]
    assert leaf.weight is parameter
