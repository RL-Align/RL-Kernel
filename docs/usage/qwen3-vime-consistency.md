# Qwen3-8B VIME train–rollout consistency

This guide is the supported path for a new user who wants to compare native
VIME operators with RL-Kernel operators on Qwen3-8B. The standard experiment
does **not** require edits to `run_arm.py`, VIME model scripts, or shell
launchers. Machine-specific paths belong in CLI options or `RLK_REPRO_*`
environment variables; stable experiment changes belong in a copied profile.

## CUDA quick path

On one 8×H100 node, a new user only needs to provide the local Qwen3-8B
checkpoint and a workspace. No VIME script or RL-Kernel example file needs to
be edited:

```bash
python3 -m pip install -e .

export RLK_REPRO_WORKSPACE=/data/rlk-repro
export RLK_REPRO_MODEL_ROOT=/models/Qwen3-8B

rlk-repro prepare \
  --workspace "$RLK_REPRO_WORKSPACE" \
  --download-data \
  --convert-checkpoint

ray start --head \
  --include-dashboard=true \
  --dashboard-host=127.0.0.1 \
  --num-gpus=8 \
  --object-store-memory=200000000000

rlk-repro doctor --workspace "$RLK_REPRO_WORKSPACE"
rlk-repro run \
  --workspace "$RLK_REPRO_WORKSPACE" \
  --mode native \
  --rollouts 8 \
  --wait
rlk-repro run \
  --workspace "$RLK_REPRO_WORKSPACE" \
  --mode consistency \
  --rollouts 8 \
  --wait
```

The default CUDA command uses the reference training topology TP4/CP2 and two
TP4/CP1 rollout engines. Use `rlk-repro plan` before every non-default
topology. Formal evidence runs must use clean source checkouts and must not pass
`--allow-dirty`.

## Supported reference configuration

The bundled CUDA profile targets one Linux node with eight 80 GB NVIDIA H100 GPUs.
Megatron runs TP4/CP2 and two colocated vLLM engines run TP4. The frozen runtime
contract is Python 3.11.15, PyTorch 2.9.1, vLLM 0.16.0, Ray 2.57.0, and
Transformer Engine 2.18. The current topology evidence and limitations are listed below.

Before starting, provide:

- a VIME-compatible Python environment containing the frozen dependencies;
- a local Hugging Face Qwen3-8B checkpoint;
- enough disk for source checkouts, the converted Megatron checkpoint, run
  artifacts, and per-step debug tensors;
- a large `/dev/shm` or a deliberately sized Ray object store and spill path.

The launcher uses the active Python environment by default. Set
`RLK_REPRO_RUNTIME_ROOT` only when the experiment must run in a different
environment.

## Topology support and evidence

The [current CUDA/ROCm audit](cuda-rocm-consistency-audit.md) records the common
short command, exact tested configurations, source dependencies and remaining gaps.
Both platforms accept training TP/CP `(1,8),(2,4),(4,2),(8,1)` and independent
rollout TP1/2/4/8. The [H100 matrix](h100-matrix-validation.md) records 14
two-update configurations, including rollout CP2/4/8, with zero selected-logprob,
token, gradient-norm and exported-parameter differences. ROCm has nine historical
one-step cases with rollout CP1; ROCm PCP remains unvalidated. These results
are not an exhaustive topology or sampling matrix.

After configuring `.rlk-profile.json` once, use the same command on either backend:

```bash
./rlk run --tp 2 --rollout-tp 4 --temperature 0.7 --top-p 0.95 --steps 200
```

CP is inferred from TP unless explicitly supplied. See
[H100 verification](h100-configurable.md) for the CUDA-only `verify` contract.
Per-run train/rollout equality does not certify cross-topology optimizer equality.

### One-round CUDA smoke performance

These numbers are single-run, 256-response-token smoke measurements from one
8×H100 node on September 19, 2026. They verify that the strict paths remain
usable; they are not a statistically controlled benchmark or a long-run
throughput claim.

| Training topology | Rollout topology | Train tokens/s | Rollout tokens/GPU/s | Step time |
|---|---|---:|---:|---:|
| TP4/CP2 | TP4/CP1 | 1292 | 71.7 | 28.0 s |
| TP2/CP4 | TP4/CP1 | 1195 | 71.6 | 29.3 s |
| TP1/CP8 | TP4/CP1 | 882 | 71.9 | 32.2 s |
| TP8/CP1 | TP4/CP1 | 1328 | 52.7 | 29.4 s |
| TP4/CP2 | TP2/CP1 | 1292 | 45.1 | 33.3 s |
| TP4/CP2 | TP8/CP1 | — | 51.8 | — |

These timings predate the fixed canonical TP8 and canonical backward follow-up.
They are historical measurements, not timings of the current source. Physical
TP changes the number of canonical-shard GEMM launches; rollout TP/CP also
changes engine count and request concurrency. See the
[current H100 validation report](h100-matrix-validation.md) for measured results.

## Canonical topology strategy

The CUDA implementation reuses the existing TP4/CP2 operators and IPC
collectives. On this eight-GPU profile, the launcher fixes virtual
`canonical_tp=8` and padded vocabulary 152576 for every physical topology.
A physical TP rank executes its adjacent virtual shards separately and
combines them in the same fixed order. This prevents a topology change from
changing vocabulary or GEMM reduction boundaries.

Attention output projection, packed FFN and selected-token logp use the same
virtual partition. CP changes token ownership; canonical backward restores
unpadded logical sample/token order before parameter reductions. The loss
backward seed is normalized before model derivatives. One CP replica
contributes each complete parameter gradient to the framework's final average,
avoiding CP8 ring rounding from repeatedly adding identical FP32 values.
Shared Q/K norm gradients similarly have one TP contributor.

Rollout CP uses the same `collective_for_group` / `all_gather_many` IPC
transport as training TP4/CP2, followed by the existing strict paged attention
runtime. KV storage is token-sharded; the adapter restores vLLM's interleaved
pages to logical order, divides queries across CP ranks, and gathers outputs
without floating-point attention merging. This is not communication-free:
logical KV pages and query outputs are gathered through the existing transport.
CUDA Graph capture uses preallocated IPC storage. Full decode graphs currently
materialize KV to the model context bound, with measurable short-sequence cost.

See [H100 verification](h100-configurable.md) and the
[full-model matrix](h100-matrix-validation.md) for the exact measured scope.
ROCm PCP, PP/EP, multi-node and arbitrary long optimizer trajectories remain
outside the completed H100 validation.

## Prepare once

From a clean RL-Kernel checkout, activate the experiment environment and run:

```bash
python3 -m pip install -e .

export RLK_REPRO_WORKSPACE=/data/rlk-repro
export RLK_REPRO_MODEL_ROOT=/models/Qwen3-8B

rlk-repro prepare \
  --workspace "$RLK_REPRO_WORKSPACE" \
  --download-data \
  --convert-checkpoint
```

`prepare` pins VIME and Megatron-LM to the profile revisions, downloads and
converts DAPO-Math-17k, and creates the Megatron torch-dist checkpoint from the
local Qwen3-8B model. It does not download model weights or create a Python
environment. Existing source directories must be clean because selecting a
profile revision changes their checked-out commit.

If Transformer Engine or CUDA Python lives outside the active environment,
set these before `doctor` and `run`:

```bash
export RLK_REPRO_TE_ROOT=/opt/transformer-engine/site-packages
export RLK_REPRO_CUDA_PYTHON_SITE=/opt/cuda-python/site-packages
export RLK_REPRO_CUDA_RUNTIME_ROOT=/opt/python/site-packages/nvidia/cuda_runtime
```

Use path flags with the same names (`--te-root`, `--cuda-python-site`, and
`--cuda-runtime-root`) when per-command configuration is clearer.

## Start Ray and check the host

Start one Ray head node. Size the object store for the host; the reference run
used 200 GB:

```bash
ray start --head \
  --include-dashboard=true \
  --dashboard-host=127.0.0.1 \
  --dashboard-port=8265 \
  --num-gpus=8 \
  --object-store-memory=200000000000 \
  --temp-dir=/tmp/rlk-repro

rlk-repro doctor --workspace "$RLK_REPRO_WORKSPACE" --mode native
rlk-repro plan --workspace "$RLK_REPRO_WORKSPACE" --mode consistency
```

Do not continue past a failed `doctor`. Its path report is also the fastest way
to see which model, checkpoint, data, runtime, or source location the launcher
resolved.

## Run the paired comparison

Use an eight-step pair first. `--wait` streams the Ray job and saves `run.log`
and `ray-status.txt` in the append-only run directory:

```bash
rlk-repro run \
  --workspace "$RLK_REPRO_WORKSPACE" \
  --mode native \
  --rollouts 8 \
  --wait

rlk-repro run \
  --workspace "$RLK_REPRO_WORKSPACE" \
  --mode consistency \
  --rollouts 8 \
  --wait
```

`native` uses VIME's native Attention, FFN, and logp operators.
`consistency` uses RL-Kernel for all three. Neither mode reuses rollout
log-probabilities. After the short pair passes, repeat both commands with
`--rollouts 200` for the primary evidence run.

Each invocation prints its `run_dir`. Validate and seal successful runs:

```bash
rlk-repro validate --run-dir /data/rlk-repro/data/runs/convergence/<run-id> --seal
rlk-repro report --workspace "$RLK_REPRO_WORKSPACE"
```

The validator checks the topology, operator readbacks, CUDA Graph settings,
fallback markers, step count, and train–rollout log-probability evidence. A
passing sealed run contains `COMPLETE`.

## ROCm MI300X and gfx942

ROCm shares the `rlk-repro` interface and uses Triton chunked Attention forward,
AITER/CK deterministic backward, RCCL and HIP Graph in consistency mode.
PR #432 already contains #430; the ROCm changes build on #432's canonical
shards and configurable sampling.

Activate the existing ROCm environment and select the machine profile once.
The supplied profile describes the isolated MI300X experiment checkouts; copy
it and edit `paths` for another installation. All four companion checkouts
(RL-Kernel, VIME, Megatron and vLLM) must include the ROCm integration patches.
The [companion patch bundle](../../examples/vime_rocm_attention_ablation/companion_patches/README.md)
records exact bases, patch hashes and validation scope.

```bash
cd /path/to/RL-Kernel
export PATH="$PWD/bin:/opt/venv/bin:$PATH"
export RLK_REPRO_PROFILE="$PWD/examples/vime_rocm_attention_ablation/profiles/mi300x-qwen3-8b.json"

# One complete round: eight samples, maximum response length 7168.
./rlk run --tp 2 --rollout-tp 4 --temperature 0.7 --top-p 0.95 --steps 1

# Change topology and sampling without editing a script.
./rlk run --tp 8 --rollout-tp 4 --temperature 1.3 --top-p 0.8 --steps 1

# Inspect the resolved command; the shared run default is 200 steps.
./rlk plan --tp 4 --rollout-tp 2 --temperature 1 --top-p 1
```

`--profile FILE` overrides `RLK_REPRO_PROFILE`. Explicit CLI values override
the profile's `defaults`; these defaults include backend, mode, topology,
sampling, round count, workload sizes, Ray ports and memory fraction. The short
flags are aliases for `--tp-size`, `--cp-size`, `--rollout-tp-size`,
`--rollout-temperature` and `--rollout-top-p`. The module spelling
`python -m rl_engine.repro` is equivalent to `rlk-repro`.

The ROCm runner waits for completion, records the expanded command and source
fingerprints, and validates operator readbacks, eight compared samples and
strict train/rollout logprob equality before returning success. It explicitly
disables rollout-logprob reuse. Each run gets a unique directory unless
`--run-id NAME` is supplied; existing output directories are never overwritten.
The launcher creates its own Ray cluster and refuses to stop an unrelated one.
The CUDA-only `verify`, standalone utility commands and `--detach` are not ROCm capabilities;
see the audit for the explicit command differences.

On this eight-GPU Qwen3-8B setup, training `(TP, CP)` can be `(1,8)`, `(2,4)`,
`(4,2)` or `(8,1)`; rollout TP can be 1, 2, 4 or 8. The ROCm route retains
its existing canonical-partition and padded-vocabulary contract. The CUDA
follow-up uses fixed virtual TP8 and vocabulary 152576.
Rollout CP is forwarded through `ParallelConfig.prefill_context_parallel_size`.
The former H100 PCP initialization failure has been fixed using the shared IPC
adapter; see the [CUDA PCP results](h100-pcp-validation.md). ROCm PCP remains
unvalidated. The separate decode-context-parallel setting is 1. The
per-engine GPU count is rollout TP × rollout CP and must divide the available
rollout GPUs. PP/EP and sequence parallelism are outside this runner's scope.

Temperature must be finite and positive; top-p must be in `(0,1]`. For top-p
below 1, the companion vLLM sampler returns its complete finite retained-token
set through VIME to the existing deterministic logp kernel. There is no fixed
64/128-token nucleus limit. The dynamic payload adds synchronization and
transport cost; one-round timings include warmup and are not steady-state
performance estimates. Top-k filtering and temperature zero are not supported
by the strict replay contract and are rejected rather than ignored.

Consistency mode now uses the HIP sparse logp/monitoring-entropy fusion for
retained top-p supports, while top-p=1 uses the full-vocabulary strict scorer.
Rebuild the extension and apply the updated VIME companion patch. See the
[performance reproduction command](rocm-sparse-performance.md) for the TP4/CP2
configuration, explicit sampling/LR options, and limits of the historical
4.44% end-to-end throughput comparison. Rollout-logprob reuse remains off.

The direct `examples.vime_rocm_attention_ablation.run_qwen3_8b` module remains
available for older scripts. `--num-rollout` is its compatibility alias.
Numerical evidence certifies train/rollout logprobs for the tested rounds;
it does not certify identical trajectories or gradients across all topologies.

## Configuration without script edits

Use the following order of precedence:

1. CLI path options for one-off runs.
2. `RLK_REPRO_*` environment variables for one host.
3. A copied JSON profile for a shared, reviewed experiment definition.

To create a custom profile, copy
`examples/vime_qwen3_8b_tp4_cp2_200/profiles/qwen3-8b-tp4-cp2.json`, update
repository revisions, paths, or `runner_args`, and pass `--profile FILE` to
every command. Run `rlk-repro plan` and review the expanded command before
submission.

Only change Python or shell runners when changing experiment semantics that the
profile cannot express, such as GPU topology, model architecture, operator
routing rules, validation gates, or Ray runtime construction. Supported
topology sizes are now CLI options and do not require script edits. Other such
changes must update the profile, manifest schema or validator when applicable,
and the launcher tests. Host paths, model locations, output directories, and
Ray API addresses are not reasons to edit a script.

## Common failure modes

- **`doctor` reports a missing Python or package path:** activate the intended
  environment, or set `RLK_REPRO_RUNTIME_ROOT` and the specific path override.
- **The model or reference checkpoint is missing:** set
  `RLK_REPRO_MODEL_ROOT`, then rerun `prepare --convert-checkpoint`.
- **Ray submission cannot connect:** start Ray and, for a non-default dashboard,
  pass `--ray-address http://host:port` to `plan` and `run`.
- **A source checkout is dirty:** commit or clean it. Use `--allow-dirty` only
  for development runs and retain source fingerprints; label those checks
  separately from clean-checkout release validation.
- **A run directory already exists:** choose a new `--run-id`. Run directories
  are append-only and are never overwritten.

The long-form [reproduction runbook](../../examples/vime_qwen3_8b_tp4_cp2_200/REPRODUCTION.md)
remains the audit reference for historical experiments and manual recovery.
