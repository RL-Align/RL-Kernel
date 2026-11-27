# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""vLLM worker and verl worker extension for the DSv4 rollout plugin: adapter sync and sleep.

Selected through verl's ``actor_rollout_ref.rollout.engine_kwargs.vllm``:
``worker_cls=rl_engine.integrations.dsv4.vllm_worker.RlkWorker`` and
``worker_extension_cls=rl_engine.integrations.dsv4.vllm_worker.RlkWorkerExtension`` (vLLM
mixes the extension into the worker; verl's own extension is its base). The model plugin
itself loads through RL-Kernel's ``vllm.general_plugins`` entry point (``pyproject.toml``)
with ``RL_KERNEL_DSV4_ROLLOUT=1``; the processes vLLM spawns import it first through the
pickled quantization config (``vllm_model.RlkDeepseekV4FP8Config``).

- Adapter sync: verl's colocated weight update streams what the ``rlkernel`` engine's
  ``get_per_tensor_param`` yields, the LoRA adapters only (BF16, ``<name>.lora_A`` /
  ``<name>.lora_B``, ``verl_engine.named_adapters``). They are installed on the plugin's
  layers (``rlk_set_lora``), not through vLLM's LoRA manager: the rollout runs the
  adapters on RL-Kernel's kernels, as the actor does. verl's ``model.lora_rank`` stays 0
  (it would turn on vLLM's own LoRA); the rank is the engine's (``rlkernel.lora_rank``).
- Sleep (``RlkWorker.sleep``): the base weights are loaded once, from the checkpoint, and
  never re-sent, so a sleep keeps them (level 1: offloaded to host memory and restored)
  even when verl asks for level 2 (discard), as it does for a model it does not see as LoRA.
"""

from __future__ import annotations

import os

os.environ.setdefault("RL_KERNEL_DSV4_ROLLOUT", "1")

from rl_engine.integrations.dsv4 import vllm_model  # noqa: E402

vllm_model.register()

import torch  # noqa: E402
from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import (  # noqa: E402
    BucketedWeightReceiver,
)
from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension  # noqa: E402
from vllm.logger import init_logger  # noqa: E402
from vllm.v1.worker.gpu_worker import Worker  # noqa: E402

from rl_engine.models.deepseek_v4.lora import adapters_from_tensors  # noqa: E402

logger = init_logger(__name__)
# Negative control for the bitwise metric: receive the adapters but keep the old ones.
ENV_SKIP_SYNC = "RL_KERNEL_DSV4_SKIP_ADAPTER_SYNC"
__all__ = ["RlkWorker", "RlkWorkerExtension", "adapters_from_tensors"]


class RlkWorker(Worker):
    def sleep(self, level: int = 1) -> None:
        """Always level 1: keep the base weights (see the module docstring)."""
        super().sleep(1)


class RlkWorkerExtension(vLLMColocateWorkerExtension):
    def update_weights_from_ipc(
        self, peft_config: dict | None = None, base_sync_done=False, use_shm=False
    ):
        if not (peft_config and base_sync_done):
            raise NotImplementedError(
                "DSv4 rollout plugin: the base weights come from the checkpoint; "
                "only LoRA adapters are synced (rlkernel engine)"
            )
        if getattr(self, "device", None) is None:
            self.device = torch.device("cuda", self.local_rank)
        received: dict[str, torch.Tensor] = {}

        def on_bucket_received(weights, is_last: bool) -> None:
            # Copy out of the reused IPC buffer.
            received.update((name, t.to(self.device, copy=True)) for name, t in weights)
            if is_last:
                self.rlk_install(received, peft_config)

        receiver = BucketedWeightReceiver(
            zmq_handle=self._get_zmq_handle(), device=self.device, use_shm=use_shm
        )
        receiver.receive_weights(on_bucket_received=on_bucket_received)

    def rlk_install(self, tensors: dict, peft_config: dict) -> None:
        if os.environ.get(ENV_SKIP_SYNC) == "1":
            logger.warning("rlkernel: %s=1, adapters NOT installed", ENV_SKIP_SYNC)
            return
        model = self.model_runner.get_model().model
        layers = [layer.ffn.layer for layer in model.rlk_layers()]
        model.rlk_set_lora(adapters_from_tensors(tensors, layers, float(peft_config["lora_alpha"])))
        logger.info("rlkernel: installed %d adapter tensors", len(tensors))
