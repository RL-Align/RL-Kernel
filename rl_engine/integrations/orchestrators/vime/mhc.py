# SPDX-License-Identifier: Apache-2.0
"""Opt-in DSV4 P1 worker initialization for vime; call after loading local weights."""
from rl_engine.integrations.common.mhc_worker import P1Worker
from rl_engine.integrations.engines.rollout.vllm.mhc import bind_rollout_stage
from rl_engine.integrations.engines.train.megatron.mhc import bind_training_stage


def bind_mhc_worker(model, role, *, backend, mode="split", version=0, observer=None):
    worker = P1Worker("vime", role, backend=backend, mode=mode, version=version, observer=observer)
    bind = bind_training_stage if role == "training" else bind_rollout_stage
    return bind(model, worker)
