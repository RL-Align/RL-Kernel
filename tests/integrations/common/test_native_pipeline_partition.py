# SPDX-License-Identifier: Apache-2.0
"""Checkpoint ownership contracts only; these do not qualify multi-GPU execution."""

from types import SimpleNamespace as NS

import pytest

from rl_engine.integrations.engines.train.megatron.native_pipeline import local_checkpoint_keys


def model(numbers, pre, post, pp=2, **kwargs):
    cfg = dict(
        num_layers=8,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=pp,
        expert_model_parallel_size=1,
        context_parallel_size=1,
    )
    cfg.update(kwargs)
    return NS(
        config=NS(**cfg),
        pre_process=pre,
        post_process=post,
        decoder=NS(layers=[NS(layer_number=n) for n in numbers]),
    )


def test_pp2_loads_disjoint_original_layers_and_owns_endpoints_exactly_once():
    keys = {
        "embed.weight",
        "head.weight",
        "norm.weight",
        "hc_head_fn",
        "hc_head_base",
        "hc_head_scale",
    } | {
        f"layers.{i}.{name}"
        for i in range(4)
        for name in [
            "hc_attn_fn",
            "ffn.experts.0.w1.weight",
            "ffn.experts.0.w1.scale",
            "attn.wq_a.scale",
        ]
    }
    first, a = local_checkpoint_keys(model(range(1, 7), True, False), keys)
    last, b = local_checkpoint_keys(model([7, 8], False, True), keys)
    assert first.layer_ids == (0, 1, 2) and last.layer_ids == (3,)
    assert a | b == keys and not a & b
    assert "embed.weight" in a and "head.weight" in b and "hc_head_fn" in b
    assert "layers.3.ffn.experts.0.w1.scale" in b


def test_pp1_preserves_the_complete_checkpoint():
    keys = {"embed.weight", "norm.weight", "layers.3.hc_ffn_fn"}
    stage, actual = local_checkpoint_keys(model(range(1, 9), True, True, pp=1), keys)
    assert actual == keys and stage.layer_ids == (0, 1, 2, 3)


@pytest.mark.parametrize(
    "numbers,pre,post",
    [
        ([1, 2, 3, 4], True, False),
        ([5, 6, 7, 8], False, True),
        ([1, 2], False, True),
        ([7, 8], True, False),
        ([6, 7, 8], False, True),
        ([7, 9], False, True),
    ],
)
def test_invalid_partial_pair_or_hash_layer_placement_is_rejected(numbers, pre, post):
    with pytest.raises(ValueError):
        local_checkpoint_keys(model(numbers, pre, post), set())


@pytest.mark.parametrize("key", ["layers.4.ffn.gate.weight", "mtp.0.weight", "mystery.weight"])
def test_unknown_or_outside_checkpoint_tensors_are_rejected(key):
    with pytest.raises(ValueError):
        local_checkpoint_keys(model([7, 8], False, True), {key})


@pytest.mark.parametrize(
    "key", ["tensor_model_parallel_size", "expert_model_parallel_size", "context_parallel_size"]
)
def test_unimplemented_parallel_axes_still_fail_closed(key):
    with pytest.raises(ValueError, match="TP=EP=CP=1"):
        local_checkpoint_keys(model([7, 8], False, True, **{key: 2}), set())


def test_pinned_bridge_builds_the_actual_uneven_native_pipeline_layout():
    pytest.importorskip("megatron.core")
    from rl_engine.integrations.orchestrators.verl.native_imports import (
        install_native_import_adapters,
    )

    install_native_import_adapters()
    from megatron.core.models.hybrid.hybrid_model import _validate_hash_moe_pipeline_placement
    from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout

    from rl_engine.integrations.engines.train.megatron.native_pipeline import (
        configure_native_pipeline,
    )

    provider = NS(
        pipeline_model_parallel_size=2,
        num_layers=8,
        moe_num_hash_layers=3,
        mtp_num_layers=0,
        hybrid_layer_pattern="WEWECEHE",
    )
    configure_native_pipeline(provider)
    assert provider.hybrid_layer_pattern == "WEWECE|HE"
    assert isinstance(provider.pipeline_model_parallel_layout, PipelineParallelLayerLayout)
    assert provider.num_layers_in_first_pipeline_stage is None
    assert provider.num_layers_in_last_pipeline_stage is None
    _validate_hash_moe_pipeline_placement(["H", "E"], 6, 6, False)
    with pytest.raises(ValueError, match="all hash MoE layers"):
        _validate_hash_moe_pipeline_placement(["C", "E", "H", "E"], 4, 6, False)


def test_mhc_parameter_names_remain_global_and_only_the_last_stage_owns_the_head():
    import torch

    from rl_engine.integrations.engines.train.megatron.native_mhc import original_mhc_parameters

    def parameter():
        p = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
        p.keep_in_fp32 = True
        return p

    def layer(number):
        return NS(
            layer_number=number,
            hyper_connection=NS(
                mapping_proj=NS(weight=parameter()),
                bias=parameter(),
                alpha_pre=parameter(),
                alpha_post=parameter(),
                alpha_res=parameter(),
            ),
        )

    stage = NS(
        decoder=NS(
            layers=[layer(7), layer(8)],
            post_process=True,
            hc_head_fn=parameter(),
            hc_head_base=parameter(),
            hc_head_scale=parameter(),
        )
    )
    actual = original_mhc_parameters(stage)
    assert len(actual) == 13 and "layers.3.hc_attn.fn" in actual and "layers.3.hc_ffn.fn" in actual
    assert not any(name.startswith("layers.0.") for name in actual)
    first = NS(decoder=NS(layers=[layer(n) for n in range(1, 7)], post_process=False))
    actual = original_mhc_parameters(first)
    assert len(actual) == 30 and not any(name.startswith("hc_head") for name in actual)
