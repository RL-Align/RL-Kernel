# Configurable H100 consistency runs

From the configured RL-Kernel checkout, use the local `rlk` entry point. The
machine's Python, framework, checkpoint and dataset paths are read from
`.rlk-profile.json`; they do not need to appear in each command.

```bash
# Two training steps, including real weight updates and automatic validation.
./rlk verify --tp 2 --rollout-tp 4 --temperature 0.7 --top-p 0.95

# Top-k is independent of top-p and temperature.
./rlk verify --tp 4 --rollout-tp 2 --temperature 1.3 --top-p 0.99 --top-k 128

# A longer run with explicit training CP and output length.
./rlk run --tp 2 --cp 4 --rollout-tp 4 --temperature 0.7 --top-p 0.95 \
    --steps 200 --max-response-len 6912

# Same topology with native operators for performance comparison.
./rlk verify --mode native --tp 2 --rollout-tp 4 --temperature 0.7 --top-p 0.95
```

Training TP can be 1, 2, 4 or 8 on this eight-GPU Qwen3-8B profile. If `--cp`
is omitted it is inferred as `8 / TP`; explicit `TP * CP` must equal eight.
Rollout TP can be chosen independently and must divide the GPU count. Engine
counts and CUDA Graph batch sizes are derived from the chosen topology.
Rollout CP is prefill context parallelism and is passed to vLLM's
`ParallelConfig.prefill_context_parallel_size`; decode remains TP-only. The
per-engine GPU count is `rollout TP * rollout CP`, so that product must divide
the eight rollout GPUs. **Rollout CP > 1 currently fails on the pinned vLLM
0.16.0 runtime**: an actual TP4/CP2 rollout launch stopped with
`RlKernelAttentionImpl does not support PCP`. Parameter forwarding does not
implement token partitioning, distributed KV cache, or attention merging.
Use rollout CP1 for the validated path. The launcher also rejects
submission while GPUs have existing compute processes; it cannot reserve
the GPUs against other users starting a process later.
When both training TP and rollout TP are 1, the launcher offloads the training
state during rollout to fit the model, optimizer and KV cache on 80 GB GPUs.
This memory tradeoff is recorded in the manifest and affects throughput.
Training TP1 also uses a smaller microbatch token budget (128 per CP rank),
so activation memory leaves room for Adam states after the first update.
Override it with `--max-tokens-per-gpu`; global batch size remains unchanged.

`--temperature` accepts finite nonnegative values; zero selects greedy
sampling. `--top-p` accepts `(0, 1]`. `--top-k` accepts `-1` (disabled) or a
positive integer up to the real vocabulary size. Full top-p/top-k support is
computed from the real vocabulary, so API top-logprob limits do not truncate
the distribution. Both sides use float32 temperature scaling after the same
BF16 model logits. The rollout adapter also handles per-request temperatures.

The default mode is `consistency`, with no `--use-rollout-logprobs`. `verify`
defaults to two steps, 512 response tokens, learning rate `5e-7`, and KL
coefficient `0.01`; all these values can be overridden. The KL reference is
temperature-scaled over the full vocabulary, rather than independently
truncated: a different reference nucleus can otherwise assign zero
probability to a sampled policy token. The policy logprob retains its actual
top-p/top-k sampling support. This choice is recorded in the run manifest.

Both `run` and `verify` wait by default and automatically run acceptance
checks. `--detach` submits without waiting. `--steps`, `--lr`, `--kl-coef`,
`--weight-decay`, `--max-response-len`, and `--run-id` are ordinary parameters;
the longer existing flag names remain accepted.

Validation checks include the complete expected step sequence, successful Ray
termination, finite training metrics, exact policy/rollout logprobs, and
sidecar sample/token coverage. `verify` additionally requires a nonzero
gradient, positive learning rates and changed exported weights across versions.
The audit fingerprints all exported parameter tensors, including norms and
embeddings. A one-step run cannot satisfy that update check.

Results are stored under the configured output root in an append-only run
directory, with `manifest.json`, `run.log`, `ray-status.txt`,
`run-validation.json`, mismatch sidecars and weight fingerprints. The manifest
records effective sampling, topology, revisions, patches and runtime paths.
Use `./rlk plan` to inspect the fully expanded command before running.

Each configuration's train/rollout equality is distinct from requiring
identical training trajectories across different configurations: engine
scheduling and random sampling can change the data even when logprob kernels
are bitwise invariant. Even with the same first-step tokens, changing training
TP/CP can change backward reductions and the updated weight bytes. This launcher
validates train/rollout logprob bytes within each run; it does not certify
cross-configuration optimizer trajectories. In the two-step H100 checks, all
responses were truncated and the nonzero gradients came from the KL term;
longer training with nonzero reward advantages remains a separate validation.

The [canonical CP backward follow-up](h100-cp-gradient-validation.md) closes
the observed TP2/CP4 versus TP4/CP2 discrepancy for two real update steps:
both gradient norms, all 399 exported parameters after each update, and
sampled tokens are identical. This is a measured pair with rollout TP4/CP1,
temperature 0.7 and top-p 0.95, not an exhaustive topology/sampling guarantee.

Train offload keeps IPC arenas in independent resident allocation pools. This
also applies when the offload hook is currently disabled: the default allocator
can still contain cached VMM blocks that cannot be exported as CUDA IPC handles.
