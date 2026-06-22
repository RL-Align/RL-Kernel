# Deterministic All-Reduce Audit

This audit records the distributed communication points relevant to
[RL-Align/RL-Kernel#112](https://github.com/RL-Align/RL-Kernel/issues/112).

## Search

```bash
rg -n "all_reduce|allreduce|reduce_scatter|all_gather|DistributedDataParallel|FSDP|deepspeed|gradient" \
  rl_engine csrc tests examples benchmarks scripts docs .github
rg -n "torch\.distributed|distributed|dist\.|process_group|ProcessGroup|nccl|NCCL|reduce|all_reduce|reduce_scatter|all_gather|gradient" \
  rl_engine csrc tests examples benchmarks scripts docs .github
```

## Summary

No direct `torch.distributed` all-reduce, reduce-scatter, all-gather, DDP, or
FSDP call sites were found in RL-Kernel source code. The current DP-gradient
communication risk is indirect: `DeepSpeedTrainingWorker` delegates backward and
optimizer behavior to the optional DeepSpeed engine.

CUDA IPC uses of `torch.multiprocessing.reductions.reduce_tensor` are not
collective reductions. They serialize CUDA IPC handles for same-node weight
handoff.

## Inventory

| Location | Kind | In scope for #112 | Handling |
| --- | --- | --- | --- |
| `rl_engine/executors/deepspeed_trainer.py` `DeepSpeedTrainingWorker.train` | Backward / optimizer delegation to DeepSpeed | Yes, indirectly | Do not claim control over DeepSpeed communication order until a tested integration point exists. |
| `rl_engine/executors/deepspeed_trainer.py` `deepspeed.initialize(...)` | Optional distributed runtime setup | Yes, indirectly | Keep missing-DeepSpeed behavior explicit. Any future integration must document the DeepSpeed hook used for gradient reduction. |
| `tests/test_deepspeed_training_worker.py` fake engine tests | Unit tests for worker delegation | Adjacent | These tests prove delegation only; they do not validate distributed gradient ordering. |
| `rl_engine/executors/bridge.py` CUDA IPC `reduce_tensor` use | CUDA IPC handle serialization | No | Keep out of all-reduce scope. |
| `rl_engine/executors/bridge.py` multi-node/RDMA/NCCL transport blockers | Unsupported weight transport guards | Adjacent | Preserve explicit blockers until a tested transport exists. |
| `rl_engine/utils/logger.py` `info_on_rank` | Rank-filtered logging | No | No numeric reduction behavior. |

## Entry Point

New distributed code should route through `rl_engine.distributed` so the
all-reduce contract and fallback/reference behavior stay testable in one place.

## Not Covered

- NVLS / NVLink-Sharp.
- Multi-node or RDMA collectives.
- DeepSpeed internal gradient synchronization order.
