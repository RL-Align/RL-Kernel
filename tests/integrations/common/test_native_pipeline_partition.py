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


@pytest.mark.parametrize("pp", [1, 2])
def test_seven_original_layers_keep_global_ids_and_exhaustive_endpoint_ownership(pp):
    keys = {
        "embed.weight",
        "head.weight",
        "norm.weight",
        "hc_head_fn",
        "hc_head_base",
        "hc_head_scale",
    }
    keys |= {
        f"layers.{i}.{part}"
        for i in range(7)
        for part in (
            "attn.compressor.ape",
            "attn.indexer.compressor.wkv.weight",
            "ffn.gate.tid2eid",
            "ffn.experts.255.w2.scale",
            "hc_ffn_fn",
        )
    }
    if pp == 1:
        stage, actual = local_checkpoint_keys(
            model(range(1, 15), True, True, pp=1, num_layers=14), keys
        )
        assert stage.layer_ids == tuple(range(7)) and actual == keys
    else:
        first, a = local_checkpoint_keys(model(range(1, 9), True, False, num_layers=14), keys)
        last, b = local_checkpoint_keys(model(range(9, 15), False, True, num_layers=14), keys)
        assert first.layer_ids == (0, 1, 2, 3) and last.layer_ids == (4, 5, 6)
        assert a | b == keys and not a & b
        assert "embed.weight" in a and "head.weight" in b and "hc_head_scale" in b
        assert not any(f"layers.{i}." in key for i in range(3) for key in b)


@pytest.mark.parametrize(
    "numbers,pre,post",
    [
        (range(1, 7), True, False),
        (range(7, 15), False, True),
        (range(2, 10), True, False),
        (range(9, 15), True, False),
    ],
)
def test_seven_layer_pp2_rejects_wrong_layout_and_partial_pairs(numbers, pre, post):
    with pytest.raises(ValueError):
        local_checkpoint_keys(model(numbers, pre, post, num_layers=14), set())


def test_pinned_bridge_builds_the_seven_layer_four_three_layout():
    pytest.importorskip("megatron.core")
    from rl_engine.integrations.orchestrators.verl.native_imports import (
        install_native_import_adapters,
    )

    install_native_import_adapters()
    from rl_engine.integrations.engines.train.megatron.native_pipeline import (
        configure_native_pipeline,
    )
    from megatron.core.models.hybrid.hybrid_model import _validate_hash_moe_pipeline_placement

    provider = NS(
        pipeline_model_parallel_size=2,
        num_layers=14,
        moe_num_hash_layers=3,
        mtp_num_layers=0,
        hybrid_layer_pattern="WEWECEHECEHECE",
    )
    configure_native_pipeline(provider)
    assert provider.hybrid_layer_pattern == "WEWECEHE|CEHECE"
    _validate_hash_moe_pipeline_placement(list("CEHECE"), 8, 6, False)
    layout = provider.pipeline_model_parallel_layout
    assert layout is not None


def _ep_cfg(**kwargs):
    cfg = dict(
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=4,
        expert_tensor_parallel_size=1,
        pipeline_model_parallel_size=2,
    )
    cfg.update(kwargs)
    return NS(**cfg)


def test_ep4_layout_guard_accepts_only_pp2_dp4_tp_etp_cp1():
    from rl_engine.integrations.engines.train.megatron.native_pipeline import (
        validate_native_parallel_layout,
    )

    assert validate_native_parallel_layout(_ep_cfg(), world_size=8) == 4
    assert validate_native_parallel_layout(_ep_cfg(expert_model_parallel_size=1), 8) == 1
    for bad in (
        dict(expert_model_parallel_size=2),
        dict(expert_model_parallel_size=8),
        dict(pipeline_model_parallel_size=1),
        dict(tensor_model_parallel_size=2),
        dict(context_parallel_size=2),
        dict(expert_tensor_parallel_size=2),
    ):
        with pytest.raises(ValueError):
            validate_native_parallel_layout(_ep_cfg(**bad), world_size=8)
    # DP != EP (expert DP > 1, or non-divisible world) is rejected.
    for world in (16, 4, 6):
        with pytest.raises(ValueError, match="DP == EP"):
            validate_native_parallel_layout(_ep_cfg(), world_size=world)


@pytest.mark.parametrize("rank", range(4))
def test_ep4_expected_keys_keep_only_local_experts(rank):
    keys = {"embed.weight", "layers.0.ffn.gate.weight", "layers.0.ffn.shared_experts.w1.weight"}
    keys |= {
        f"layers.{layer}.ffn.experts.{e}.{n}.{s}"
        for layer in range(4)
        for e in range(256)
        for n in ("w1", "w2", "w3")
        for s in ("weight", "scale")
    }
    m = model(range(1, 7), True, False, num_moe_experts=256, expert_model_parallel_size=4)
    m.config.pipeline_model_parallel_size = 2
    m.config.expert_tensor_parallel_size = 1
    stage, actual = local_checkpoint_keys(m, keys, ep_rank=rank)
    assert stage.layer_ids == (0, 1, 2)
    experts = {int(k.split(".")[4]) for k in actual if ".ffn.experts." in k}
    assert experts == set(range(64 * rank, 64 * rank + 64))
    assert len([k for k in actual if ".ffn.experts." in k]) == 3 * 64 * 3 * 2
    assert "layers.0.ffn.shared_experts.w1.weight" in actual
    assert "layers.0.ffn.gate.weight" in actual and "embed.weight" in actual


def test_ep1_expected_keys_unchanged_by_expert_filtering():
    keys = {f"layers.3.ffn.experts.{e}.w1.weight" for e in range(256)} | {"head.weight"}
    m = model([7, 8], False, True, num_moe_experts=256)
    _, actual = local_checkpoint_keys(m, keys)
    assert actual == keys
