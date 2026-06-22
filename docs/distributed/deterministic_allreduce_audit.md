# Deterministic All-Reduce Audit

This document audits the current RL-Kernel repository for issue
[RL-Align/RL-Kernel#112](https://github.com/RL-Align/RL-Kernel/issues/112):
deterministic NCCL all-reduce, including DP gradient all-reduce.

Issue #112 is part of the WS2 train-inference consistency roadmap in
[RL-Align/RL-Kernel#83](https://github.com/RL-Align/RL-Kernel/issues/83). The
current repository does not expose a direct deterministic all-reduce API yet, so
the first implementation should add one as the intended entry point rather than
claiming that existing distributed paths are already deterministic.

## Audit Commands

The initial audit used these repository-local searches:

```bash
rg -n "all_reduce|allreduce|reduce_scatter|all_gather|DistributedDataParallel|FSDP|deepspeed|gradient" \
  rl_engine csrc tests examples benchmarks scripts docs .github
rg -n "torch\.distributed|distributed|dist\.|process_group|ProcessGroup|nccl|NCCL|reduce|all_reduce|reduce_scatter|all_gather|gradient" \
  rl_engine csrc tests examples benchmarks scripts docs .github
rg --files rl_engine csrc tests examples benchmarks scripts docs .github
```

## Summary

No direct `torch.distributed` collective call sites were found in the audited
paths. In particular, no direct calls to `all_reduce`, `reduce_scatter`,
`all_gather`, `DistributedDataParallel`, or `FSDP` were found.

The only training-side path that can imply DP gradient synchronization today is
the optional DeepSpeed worker. That synchronization, when present, is owned by
the DeepSpeed engine called from RL-Kernel, not by an RL-Kernel collective helper.

The repository also contains CUDA IPC uses of PyTorch's `reduce_tensor`, but
those calls serialize CUDA IPC handles for same-node weight handoff. They are
not collective reductions and do not implement NCCL all-reduce.

## Call Site Inventory

| Location | Kind | Train-inference consistency impact | In scope for #112 | Proposed deterministic handling |
| --- | --- | --- | --- | --- |
| `rl_engine/executors/deepspeed_trainer.py` `DeepSpeedTrainingWorker.train` calls `self.engine.backward(loss)` and `self.engine.step()` | Backward / optimizer path; possible DP gradient synchronization inside DeepSpeed | Yes, when the DeepSpeed runtime is configured for DP > 1 | Yes, as the current DP-gradient-adjacent training path | Add an RL-Kernel deterministic all-reduce helper first. For DeepSpeed, do not claim ordering control until a tested integration point exists. A focused DP gradient smoke test should compare DP=1 gradients against DP=N reduced gradients under a fixed global batch and fixed seed. |
| `rl_engine/executors/deepspeed_trainer.py` `deepspeed.initialize(...)` with `zero_optimization` and `gradient_accumulation_steps` config | Optional distributed training runtime setup | Indirect | Yes, because DeepSpeed may perform gradient communication after initialization | Document that DeepSpeed communication order is not currently controlled by RL-Kernel. Keep fallback behavior explicit when DeepSpeed is missing. |
| `tests/test_deepspeed_training_worker.py` fake DeepSpeed engine tests | Unit tests for the DeepSpeed worker contract | No direct collective behavior | Adjacent only | Existing tests prove the worker delegates to `engine.backward` and `engine.step`, but they do not validate distributed gradient ordering. Add new distributed smoke tests separately instead of extending these fake-engine tests into false coverage. |
| `rl_engine/executors/bridge.py` `VLLMIPCWeightUpdateRequestBuilder._resolve_reduce_tensor` and `IPCWeightBridge._resolve_reduce_tensor` | CUDA IPC handle serialization via `torch.multiprocessing.reductions.reduce_tensor` | No collective reduction; affects same-node weight handoff only | No | Keep out of deterministic all-reduce scope. Do not rename or classify this as all-reduce. |
| `rl_engine/executors/bridge.py` `WeightLayout.validate_supported` and `make_weight_bridge` reject multi-node/RDMA/NCCL transports | Explicit unsupported transport guards | Prevents unsupported distributed weight transport from silently succeeding | Adjacent only | Preserve the explicit blockers. If a future NCCL/RDMA transport is added, it needs its own deterministic contract and validation. |
| `docs/usage/weight-sync-bridge.md` mentions vLLM IPC or NCCL public update APIs and RDMA/NCCL as not implemented | User-facing documentation | Adjacent; weight sync rather than gradient all-reduce | Adjacent only | Keep the documentation clear that current weight bridge transports are same-node or local fallbacks. Do not cite this as NCCL all-reduce validation. |
| `rl_engine/utils/logger.py` `info_on_rank` uses `device_ctx.rank` | Rank-filtered logging utility | No numeric impact | No | No deterministic collective handling needed. |

## Direct Collectives

None found.

Because the repository currently has no direct collective entry point, the next
implementation should introduce a small, explicit API for deterministic
all-reduce. That API can become the reference path for future WS2 distributed
ops and the test oracle for DP gradient synchronization experiments.

## Determinism Contract To Implement

A deterministic all-reduce mode should define stability only under explicit
conditions:

- same world size;
- same global rank order;
- same input tensors;
- same dtype;
- same backend mode;
- same environment and process-group configuration.

For a slow reference path, the concrete operation order should be:

1. gather tensors to rank 0 in ascending global rank order;
2. accumulate on rank 0 in rank order `0, 1, ..., world_size - 1`;
3. use FP32 accumulation when configured;
4. apply `sum` or `mean` at a fixed point;
5. cast back to the original dtype when needed;
6. broadcast the final tensor from rank 0 to all ranks.

The NCCL fast path should be opt-in and should not promise cross-version or
cross-hardware bitwise determinism unless the validation proves it. A helper may
set these environment variables before process-group initialization:

```bash
NCCL_ALGO=Ring
NCCL_PROTO=Simple
NCCL_MIN_NCHANNELS=1
NCCL_MAX_NCHANNELS=1
```

## Unsupported Or Not Yet Covered

- NVLink-Sharp / NVLS deterministic reduce is not implemented or validated in
  the repository.
- Multi-node/RDMA weight transport is explicitly unsupported by the current
  weight bridge.
- DeepSpeed DP gradient communication order is not currently controlled by
  RL-Kernel.
- The current audit environment has CUDA driver visibility and an isolated
  PyTorch/NCCL venv under `.codex-nightly/envs/issue112-py312`, but distributed
  GPU validation has not been run yet.
- No claim is made for multi-node NCCL, RDMA, FSDP, DDP, or DeepSpeed internals.

## Next Reviewable Step

Add `rl_engine.distributed.deterministic_allreduce` with two modes:

- `ordered_rank_fallback`: explicit gather, rank-ordered accumulation on rank 0,
  and broadcast;
- `nccl_ring`: ordinary `torch.distributed.all_reduce` after the caller has
  opted into deterministic NCCL environment configuration before process-group
  initialization.

The first tests should be small distributed smoke tests that can run with
`torchrun` on CPU/Gloo for fallback behavior and on CUDA/NCCL when the runtime is
available.
