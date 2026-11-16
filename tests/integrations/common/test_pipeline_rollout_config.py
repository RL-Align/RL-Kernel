# SPDX-License-Identifier: Apache-2.0
"""Pinned real verl config/engine-argument checks; not GPU worker acceptance."""

import pytest

pytest.importorskip("verl")

from rl_engine.integrations.orchestrators.verl.pipeline_rollout import (  # noqa: E402
    RLKPipelineRolloutConfig,
    RLKPipelineVLLMHttpServer,
    RolloutConfig,
    RolloutMode,
    validate_pipeline_server,
)


def config(**overrides):
    return RLKPipelineRolloutConfig(
        **dict(
            dict(
                name="vllm",
                pipeline_model_parallel_size=2,
                tensor_model_parallel_size=1,
                enforce_eager=True,
                enable_sleep_mode=False,
            ),
            **overrides,
        )
    )


def test_upstream_remains_restricted_and_rlk_supplies_actual_native_engine_pp_size():
    with pytest.raises(NotImplementedError, match="pipeline_model_parallel_size"):
        RolloutConfig(name="vllm", pipeline_model_parallel_size=2)
    cfg = config(disaggregation={"enabled": False})
    assert cfg.pipeline_model_parallel_size == 2 and cfg.disaggregation.enabled is False
    server = object.__new__(RLKPipelineVLLMHttpServer)
    server.config = cfg
    args = {"distributed_executor_backend": "mp"}
    server._preprocess_engine_kwargs(args)
    assert args["pipeline_parallel_size"] == 2 and args["distributed_executor_backend"] == "mp"
    for bad in [{"pipeline_parallel_size": 1}, {"distributed_executor_backend": "uni"}]:
        with pytest.raises(ValueError):
            server._preprocess_engine_kwargs(bad)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(mode="sync"),
        dict(pipeline_model_parallel_size=3),
        dict(tensor_model_parallel_size=2),
        dict(data_parallel_size=2),
        dict(expert_parallel_size=2),
        dict(enforce_eager=False),
        dict(enable_sleep_mode=True),
        dict(mtp={"enable": True}),
        dict(disaggregation={"enabled": True}),
    ],
)
def test_unimplemented_modes_and_upstream_validations_still_reject(overrides):
    with pytest.raises(ValueError):
        config(**overrides)


def test_only_two_gpu_single_node_standalone_uses_the_opt_in_server():
    kwargs = dict(
        rollout_mode=RolloutMode.STANDALONE,
        workers=[],
        node_rank=0,
        nnodes=1,
        gpus_per_node=2,
        cuda_visible_devices="0,1",
    )
    validate_pipeline_server(config(), **kwargs)
    for update in [
        dict(workers=[object()]),
        dict(nnodes=2),
        dict(node_rank=1),
        dict(gpus_per_node=1),
        dict(cuda_visible_devices="0,0"),
        dict(rollout_mode=RolloutMode.HYBRID),
    ]:
        with pytest.raises(ValueError, match="standalone native workers"):
            validate_pipeline_server(config(), **dict(kwargs, **update))
