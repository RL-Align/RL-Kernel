# Persistent HAL auto-tuning

Auto-tuning is disabled by default. Enable it explicitly in performance mode:

```bash
export RL_KERNEL_MODE=auto
export RL_KERNEL_AUTOTUNE=1
# Optional; default: ~/.cache/rl_kernel/autotune/
export RL_KERNEL_AUTOTUNE_CACHE_DIR=/path/to/cache
```

Both settings are required. `strict`, `audit`, and `off` retain fixed launches.

The cache directory must be private. The tuner creates it with mode `0700`. If it
already exists and is owned by another user, or is writable by group or others,
selections stay in memory and a warning is logged. Anyone who can write cache
records can choose among the qualified candidates, so do not share a writable
cache across users.
Autograd retains the forward policy for backward, even if environment variables
subsequently change. Configure the environment before starting work.

## Coverage

| Operator | Tuned work |
| --- | --- |
| SiLU / SwiGLU | Forward and backward elementwise blocks and launch parameters |
| RoPE | Forward and backward launch parameters; rotation layout unchanged |
| Embedding | Forward launch parameters; sorted backward remains fixed |
| RMSNorm | Forward and input-gradient kernels; weight-gradient row fold fixed |
| ratio/KL | Forward and backward launch parameters |
| GRPO | Reward normalization; PyTorch loss reductions unchanged |
| Explicit tuned GEMM | FP32 GEMM and BF16 tree-leaf launch parameters; K-tree fixed |
| Explicit tuned attention | Forward and backward kernels; arithmetic and softmax lanes fixed |
| Explicit tuned logprob | Forward and backward; vocabulary block and traversal fixed |

CUDA and ROCm use the same adapter. Other devices keep their existing launches.
Reduction layouts, arithmetic precision, K partitioning and Split-K policy are
not search dimensions. Reduction candidates vary pipeline stage count only;
warp layouts are fixed because changing them can reorder a reduction and produce
differences on later inputs despite matching the initial validation input.
Elementwise candidates also vary warp counts; SiLU and SwiGLU vary block sizes.

## Independent core backends

Existing deterministic and batch-invariant APIs always retain fixed launches,
including indirect GEMM callers such as LM head. The strict attention and
logprob contract registries do not register the tuned implementations.
Use explicit legacy registry selection:

```python
from rl_engine.runtime.registry import kernel_registry

op = kernel_registry.get_op(
    "logp", device="cuda:0", backend="rlkernel.triton.tuned_logp"
)
logp = op(logits, token_ids)
logp.sum().backward()
print(op.autotune_provenance)  # backend, enabled flag, per-launch config/cache source
```

The other backend IDs are `rlkernel.triton.tuned_gemm` (`det_gemm`) and
`rlkernel.triton.tuned_attention` (`attn`). Ordinary automatic routing is unchanged;
explicit selection is required for these core backends. Their input/output APIs
match the existing Triton implementations, including GEMM's existing BF16-only
backward restriction for FP32-input forward calls. Tuned attention compares against the
fixed **Triton** implementation; CUDA native-libm parity is not claimed.

Each candidate is compared byte-for-byte against the fixed launch, including
auxiliary outputs and mutated workspace. Mismatches are rejected before timing;
ties select the original launch. Forward and backward select independently.
There is no new cross-device or cross-batch invariance guarantee. Validation on
one input does not establish equivalence for every possible input: hardware
regression tests remain necessary before deploying on a new architecture.

## Cache and execution behavior

The first call pays compilation, validation and timing costs. Trials copy every
mutated tensor to private storage and validate all candidates before timing.
Timing uses interleaved CUDA/HIP graph replays of 32 launches to exclude Python
submission gaps. Read/write buffers are restored inside each replay; their
identical reset cost is included for every candidate. Graphs are captured on a
private stream and replayed/timed on the caller's device and current stream. Large GEMM workspaces therefore increase
cold-call memory usage. Only known Triton resource-limit failures reject a
candidate; other execution errors propagate.

Cache keys include device UUID/PCI information, architecture, driver and runtime
versions, PyTorch/Triton versions, module source revision, candidate set, tensor
shape/dtype/stride/device, launch grid and scalar arguments. Separate kernel
identities distinguish forward and backward. No tensor contents are written.
Missing stable hardware identity or driver metadata disables disk persistence,
but still permits process-local reuse.

Records use versioned JSON with a per-key process lock and atomic replacement.
Invalid records warn and trigger retuning. Unavailable storage warns and falls
back to memory. Stop processes using the cache before deleting its directory;
restart them to also discard memory selections. Software, kernel or candidate
changes invalidate matching keys automatically.

Graph capture uses an already populated memory selection, or the fixed launch.
It never profiles or accesses disk. For graphs containing backward, warm up and
capture on the same non-default stream. Warm up each relevant shape before capture;
Triton still needs its normal compilation warmup even when tuning is disabled.

## Verification and benchmark

```bash
python -m pytest -q tests/platforms/test_persistent_autotuner.py tests/ops/test_autotuning_gpu.py
python -m benchmarks.tuning.benchmark_autotuning --cache-dir /tmp/autotune-bench
```

The benchmark creates an isolated cache directory and reports cold first-call
latency and a second process's disk cache reuse. It separates two measurements:

- `eager_wall_us`: end-to-end time per call, including Python dispatch, allocations,
  and completion of queued GPU work; compare this for ordinary eager execution.
- `graph_device_us`: GPU event time per captured launch, with 64 launches per
  replay to remove Python submission gaps. It includes both the captured operator
  and the raw kernel with a preallocated output. Compare this for graph execution.

Both use repeated inputs and alternating measurement order. First-call latency
also includes Triton initialization and compilation; it is not pure tuning time.
`--shape 1 4096` changes the input shape. `--scan-candidates` additionally measures
all SiLU candidates independently. A CUDA event around a single eager Python call
can include GPU idle time while Python submits the kernel, so that measurement
must not be labeled kernel execution time or used to rank very short kernels.

A faster GPU kernel does not guarantee an eager speedup: cache lookup and dispatch
can outweigh a sub-microsecond kernel improvement. Auto-tuning therefore remains
opt-in. Run the GPU suite separately on CUDA and ROCm. CPU cache tests are not
hardware numerical acceptance.

For test results, skips, measured eager versus graph results and remaining
hardware gaps, see the
[B200 issue #50 evidence](https://github.com/fusheng-ji/RL-Kernel/blob/6d6193d94a83266d5f5f62bb9d90fb3215716ee3/pr536/reports/experiments/issue50_b200/README.md).
Full CUDA/ROCm acceptance remains incomplete.

## Candidate qualification and expanded experiments

Run the fixed GEMM contracts with both policies before evaluating performance:

```bash
RL_KERNEL_MODE=auto RL_KERNEL_AUTOTUNE=0 python -m pytest -q -rs tests/ops/gemm/test_det_gemm.py
RL_KERNEL_MODE=auto RL_KERNEL_AUTOTUNE=1 python -m pytest -q -rs tests/ops/gemm/test_det_gemm.py
python -m pytest -q -rs tests/ops/test_autotuning_candidates.py tests/ops/test_autotuning_gpu.py
python -m pytest -q -rs tests/benchmarks/test_autotuning_benchmark.py
```

The candidate suite enumerates the same configurations as production and launches
all of them directly. A mismatching candidate fails the test, rather than being
hidden by runtime rejection or fixed-configuration fallback. Comparisons include
all declared auxiliary outputs/workspaces and backward launches. JUnit properties
record the qualified kernel/configuration pairs and existing unsupported-input
errors. Seeds are 0, 1 and 2026. The matrix includes supported floating dtypes,
block boundaries, strided inputs, special values and empty inputs. Non-power-of-two
RoPE half dimensions retain the fixed implementation's compilation error.
Resource-limit skips and unsupported dtypes are not numerical passes.

The expanded benchmark adds all ten operator families and separate phases.
Forward uses autograd-enabled inputs (except GRPO), matching the training-phase
measurements; its absolute latency should not be compared directly with the
legacy SiLU benchmark, whose inputs do not require gradients:

```bash
python -m benchmarks.tuning.benchmark_autotuning --operator all --dtype bfloat16 \
  --preset all --phase all --trials 5 --output /tmp/autotuning-matrix.json
```

`--operator` also accepts one operator from the coverage table (`ratio_kl`, `grpo`,
`gemm`, `attention`, `logp` use these CLI names). `--preset` accepts `small`,
`throughput`, or `all`; `--phase` accepts `forward`, `backward`,
`forward_backward`, or `all`. Embedding and GRPO report forward only. Unsupported
dtype/phase combinations receive explicit records. Existing SiLU `--shape` and
`--scan-candidates` commands remain available for isolated raw-kernel timing;
the expanded matrix uses preset shapes and reports full phase execution.

| Preset | Shape parameters |
| --- | --- |
| Small | rows=3, width=129; RoPE/attention width=128 |
| Throughput SiLU/SwiGLU/RMSNorm/ratio-KL/logprob | rows=1024, width=4096 |
| Throughput embedding | 4096 indices, embedding width=1024 |
| Throughput GRPO | 1024 groups of 8 rewards |
| Throughput GEMM | (64,1024) × (1024,65) |
| Throughput attention | GQA heads 2:1, query length=129, KV length=131, head dimension=128 |
| Throughput RoPE | (2,1024,128) |

Each trial launches cold, restarted and precompiled worker processes. The last
worker compiles all candidates using a disposable selection cache, then measures
fresh validation/selection with a new cache. First-call times exclude input and
operator construction; backward-only times also exclude its fixed forward setup.
`profile_ms` separately records time inside validation/timing (including JIT in
the ordinary cold worker). A worker runs cases sequentially, so later cases can
reuse normal Triton compilation caches; these are cold *selection* experiments,
not claims of a pristine compiler cache for every case.

Core performance baselines use the existing fixed public Triton backends;
ordinary operators use their tuning-disabled path. Steady-state measurements
use 11 alternating samples, 50 eager calls per batch,
and 32 phase executions per graph replay. Backward uses independent
`autograd.grad` outputs instead of accumulating into user gradients. Forward
setup, backward warmup and capture use the same private stream. GPU occupancy is
checked before and after each case; other processes cause the formal experiment
to fail rather than publishing contaminated results. ROCm occupancy checking
requires working AMD SMI through PyTorch.

The report retains per-trial samples, selected configurations, cache sources and
between-process ranges. A benefit is confirmed only when fixed and tuned ranges
of process medians do not overlap in the favorable direction. Otherwise the
result is `unconfirmed`, and no finite break-even estimate is reported. Confirmed
break-even uses the complete precompiled first-call cost, a conservative estimate
that includes the selected launch. Validation-only benchmark tests exercise the
same cold/disk/hot/precompiled and graph paths without collecting steady-state
timings; their results are functional evidence, not performance evidence.
