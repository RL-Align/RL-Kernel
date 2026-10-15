# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The shared vLLM model on four real ranks: real config, real model, real expert placement.

Each test launches four processes (``torch.distributed.run``) that initialise
vLLM's own tensor-parallel state and build vLLM's actual ``Qwen3NextForCausalLM``
(``native-bind``) or RL-Kernel's ``RLKernelQwen3NextForCausalLM`` (``wrapper``) from a
real ``VllmConfig``, at expert parallel size 1 (all 512 experts at expert-TP4
width on every rank) and 4 (experts ``[128r, 128r + 128)`` at full width on rank
r, vLLM's linear placement). The weights are allocated but not loaded
(about 40 GB per rank); nothing is mocked.

* ``native-bind``: the real binding function replaces no parameter and no HF
  weight loader; every one of the 48 layers has the shared classes, the local
  expert count, the shapes, and -- at EP4 -- vLLM's own ``expert_map`` equals
  ``[128r, 128r + 128)`` with every expert held by exactly one rank.
* ``wrapper``: real configuration drift (an expert-parallel flag that does not
  match the identity, another placement, a wrong dtype) is rejected before any
  model memory is allocated; the certified configuration constructs.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("mode", ["native-bind", "wrapper"])
@pytest.mark.parametrize("ep", [1, 4])
def test_real_vllm_model_on_four_ranks(ep, mode):
    if torch.cuda.device_count() < 4:
        raise AssertionError("Four allocated GPUs are required")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node",
            "4",
            str(Path(__file__).resolve()),
            "--worker",
            str(ep),
            mode,
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT), "TRITON_F32_DEFAULT": "ieee"},
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-4000:]
    assert result.stdout.count("REAL-MODEL OK") == 4, result.stdout[-2000:]


def _worker(ep: int, mode: str) -> None:
    import torch.distributed as dist
    from vllm.config import set_current_vllm_config
    from vllm.distributed import parallel_state
    from vllm.utils.torch_utils import set_default_torch_dtype

    from rl_engine.integrations.engines.rollout.vllm import qwen3_next_vllm_model as adapter
    from rl_engine.integrations.engines.rollout.vllm.qwen3_next_vllm_attention import (
        register_shared_attention_backend,
    )
    from rl_engine.models.qwen3_next.qwen3_next_ep import expert_map
    from tests.models.qwen3_next.vllm_real import real_vllm_config

    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    config = real_vllm_config(ep)
    with set_current_vllm_config(config):
        parallel_state.init_distributed_environment(4, rank, "env://", rank, "nccl")
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=4)
    group = parallel_state.get_tp_group().device_group

    def check_layers(model):
        assert len(model.layers) == 48
        owned = torch.zeros(512, dtype=torch.int32, device="cuda")
        for index, layer in enumerate(model.layers):
            moe = layer.mlp
            assert type(moe) is adapter.SharedVLLMMoE, index
            assert moe.shared_ep == ep, index
            routed = moe.experts.routed_experts
            if ep == 1:
                assert routed.expert_map is None and routed.local_num_experts == 512, index
                assert tuple(routed.w13_weight.shape) == (512, 256, 2048), index
                assert tuple(routed.w2_weight.shape) == (512, 2048, 128), index
            else:
                assert (routed.global_num_experts, routed.local_num_experts) == (512, 128), index
                assert torch.equal(routed.expert_map.int(), expert_map(rank, "cuda")), index
                assert tuple(routed.w13_weight.shape) == (128, 1024, 2048), index
                assert tuple(routed.w2_weight.shape) == (128, 2048, 512), index
                owned += (routed.expert_map >= 0).int()
        if ep == 4:
            peers = [torch.empty_like(owned) for _ in range(4)]
            dist.all_gather(peers, owned, group=group)
            total = torch.stack(peers).sum(dim=0)
            assert torch.equal(total, torch.full_like(total, 48)), "an expert is not held once"

    # What vLLM's model loader does around construction: the model dtype is the
    # default dtype (the attention layers read it) and the device is the GPU.
    with (
        set_current_vllm_config(config),
        set_default_torch_dtype(config.model_config.dtype),
        torch.device("cuda"),
    ):
        if mode == "native-bind":
            from vllm.model_executor.models.qwen3_next import Qwen3NextForCausalLM

            # vLLM's attention refuses the CUSTOM backend until it is registered;
            # the wrapper does this itself before it constructs the native model.
            register_shared_attention_backend()
            model = Qwen3NextForCausalLM(vllm_config=config, prefix="")
            before = {
                name: (id(value), id(getattr(value, "weight_loader", None)))
                for name, value in model.named_parameters()
            }
            adapter._bind_native_model(model.model, group, ep)
            after = {
                name: (id(value), id(getattr(value, "weight_loader", None)))
                for name, value in model.named_parameters()
            }
            assert after == before, "binding replaced a parameter or a weight loader"
            assert isinstance(model.model, adapter.SharedVLLMModel)
            check_layers(model.model)
        else:
            allocated = torch.cuda.memory_allocated()
            drifts = [
                ("parallel_config", "enable_expert_parallel", ep != 4),
                ("parallel_config", "expert_placement_strategy", "round_robin"),
                ("model_config", "dtype", torch.float32),
            ]
            for scope, name, value in drifts:
                drifted = real_vllm_config(ep)
                setattr(getattr(drifted, scope), name, value)
                with pytest.raises(ValueError):
                    adapter.RLKernelQwen3NextForCausalLM(vllm_config=drifted, prefix="")
                assert torch.cuda.memory_allocated() == allocated, "drift allocated a model"
            model = adapter.RLKernelQwen3NextForCausalLM(vllm_config=config, prefix="")
            check_layers(model.model)
    dist.barrier()
    print(f"REAL-MODEL OK rank {rank} ep {ep} {mode}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        _worker(int(sys.argv[2]), sys.argv[3])
    else:
        raise SystemExit(pytest.main([__file__, "-q"]))
