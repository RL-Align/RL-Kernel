# SPDX-License-Identifier: Apache-2.0
"""Opt-in single-node PP2 standalone server using real verl/vLLM workers.

Generic verl PP rollout/IPC remains unsupported. This selected path uses native
vLLM pipeline workers and RLK's stage-owned loading, bindings and checked sync.
"""

from dataclasses import dataclass

from verl.workers.config import RolloutConfig
from verl.workers.rollout.replica import RolloutMode
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer


@dataclass
class RLKPipelineRolloutConfig(RolloutConfig):
    def __post_init__(self):
        mtp_enabled = (
            self.mtp.get("enable", False)
            if isinstance(self.mtp, dict)
            else getattr(self.mtp, "enable", False)
        )
        if (
            self.name != "vllm"
            or self.pipeline_model_parallel_size != 2
            or self.tensor_model_parallel_size != 1
            or self.expert_parallel_size != 1
            or self.data_parallel_size != 1
            or not self.enforce_eager
            or self.enable_sleep_mode
            or mtp_enabled
        ):
            raise ValueError(
                "RLK rollout requires eager standalone TP1 PP2 EP1 DP1 without sleep/MTP"
            )
        # Preserve every upstream validation/coercion other than its generic PP
        # rejection. The dedicated server below supplies actual PP2 engine args.
        object.__setattr__(self, "pipeline_model_parallel_size", 1)
        try:
            super().__post_init__()
        finally:
            object.__setattr__(self, "pipeline_model_parallel_size", 2)
        if self.disaggregation.enabled:
            raise ValueError("RLK PP2 rollout does not support disaggregation")


def validate_pipeline_server(
    config, *, rollout_mode, workers, node_rank, nnodes, gpus_per_node, cuda_visible_devices
):
    ids = cuda_visible_devices.split(",")
    if (
        not isinstance(config, RLKPipelineRolloutConfig)
        or rollout_mode != RolloutMode.STANDALONE
        or workers
        or node_rank != 0
        or nnodes != 1
        or gpus_per_node != 2
        or len(ids) != 2
        or len(set(ids)) != 2
        or any(not x.isdecimal() for x in ids)
    ):
        raise ValueError("RLK PP2 server requires two local GPUs and standalone native workers")


class RLKPipelineVLLMHttpServer(vLLMHttpServer):
    def __init__(
        self,
        config,
        model_config,
        rollout_mode,
        workers,
        replica_rank,
        node_rank,
        gpus_per_node,
        nnodes,
        cuda_visible_devices,
    ):
        validate_pipeline_server(
            config,
            rollout_mode=rollout_mode,
            workers=workers,
            node_rank=node_rank,
            nnodes=nnodes,
            gpus_per_node=gpus_per_node,
            cuda_visible_devices=cuda_visible_devices,
        )
        super().__init__(
            config=config,
            model_config=model_config,
            rollout_mode=rollout_mode,
            workers=workers,
            replica_rank=replica_rank,
            node_rank=node_rank,
            gpus_per_node=gpus_per_node,
            nnodes=nnodes,
            cuda_visible_devices=cuda_visible_devices,
        )

    def _preprocess_engine_kwargs(self, engine_kwargs):
        super()._preprocess_engine_kwargs(engine_kwargs)
        if engine_kwargs.get("distributed_executor_backend", "mp") != "mp":
            raise ValueError("RLK PP2 requires the native vLLM multiprocess executor")
        if engine_kwargs.get("pipeline_parallel_size", 2) != 2:
            raise ValueError("vLLM pipeline size must match the selected real PP2 topology")
        engine_kwargs["pipeline_parallel_size"] = self.config.pipeline_model_parallel_size
        engine_kwargs["distributed_executor_backend"] = "mp"
