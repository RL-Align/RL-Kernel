# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""vLLM spawned-worker registration for the explicit shared Qwen architecture."""

from vllm.v1.worker.gpu_worker import Worker

from rl_engine.integrations.common.qwen3_next_binary import imported_provider_identity
from rl_engine.integrations.engines.rollout.vllm.qwen3_next_vllm_model import register_shared_model


class SharedQwenWorker(Worker):
    """Select through LLM(worker_cls=...) so spawn never relies on parent globals."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("vllm_config", args[0] if args else None)
        identity = (
            config.additional_config.get("qwen3_next_shared_identity")
            if config is not None
            else None
        )
        mode = None
        if identity is not None:
            from rl_engine.integrations.orchestrators.vime.experiments.qwen3_next.profile import (
                activate_for_identity,
            )

            mode = activate_for_identity(identity)
        register_shared_model()
        self.shared_provider_binary = imported_provider_identity()
        super().__init__(*args, **kwargs)
        self.shared_l2_trace = None
        self.shared_provider_execution_mode = mode

    def configure_l2_trace(self, request, output_directory):
        from rl_engine.integrations.engines.rollout.vllm.qwen3_next_native_rollout import (
            RolloutTrace,
        )

        if self.shared_l2_trace is not None:
            raise RuntimeError("The previous rollout trace has not been collected")
        self.shared_l2_trace = RolloutTrace(self, request, output_directory)
        return {"rank": self.rank, "request_sha256": request["request_sha256"]}

    def collect_l2_trace(self):
        if self.shared_l2_trace is None:
            raise RuntimeError("No active rollout trace")
        result = self.shared_l2_trace.finish()
        self.shared_l2_trace = None
        return result

    def execute_model(self, scheduler_output):
        if self.shared_l2_trace is not None:
            self.shared_l2_trace.before_execute(scheduler_output)
        return super().execute_model(scheduler_output)
