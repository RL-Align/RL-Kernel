# One-switch BI execution plans

After setting up the existing [Qwen3 Vime environment](qwen3-vime-consistency.md), enable
model-level plan selection with:

```bash
export RL_KERNEL_BI=1
./rlk run --steps 8 --wait
```

Keep your existing machine profile, model/checkpoint paths, and workload arguments.
For ROCm, use the existing MI300X profile:

```bash
./rlk run --profile examples/vime_rocm_attention_ablation/profiles/mi300x-qwen3-8b.json \
  --steps 8 --wait
```

The flag enables startup selection of a **complete paired training/rollout recipe**.
It reads the local model's `config.json`, resolves a reviewed plan, and installs the
plan's framework adapters. It does not identify models from directory names. The
selected route remains fixed for the lifetime of the job, including weight updates
and CUDA/HIP graph replay. It does not add per-token dispatch or benchmarking.

The current catalog contains Qwen3-8B BF16 with Vime + Megatron + vLLM on one
8×H100 or 8×MI300X node, training TP4/CP2 and rollout TP4/CP1. These recipes reuse
the existing strict Attention, FFN, LogP, projection and collective paths. CUDA
pins `cublaslt_nosplitk`; ROCm pins `triton_mfma` and the existing `triton` attention
route. Quantization, modified RoPE, sliding-window attention, other GPU families,
and other topologies are rejected by this initial adapter.

DSV4, Gemma, Qwen3-Next and H3 contributors can add profiles, paired adapters and
validated plans through the same interfaces. They are **not enabled by registering
a model name alone**. This change does not finish those model implementations.
Direct, unprepared `vllm serve` or arbitrary framework entry points fail with an
actionable error; they must call the preparation API before starting workers.

## What selects the fastest path

| Component | Responsibility |
| --- | --- |
| `ModelProfile` | Match architecture and structural configuration from model metadata. |
| `ExecutionPlan` | Name a paired adapter, exact routing settings, and reviewable evidence. |
| `PlanAdapter` | Validate the supported configuration and install both framework sides. |
| `RuntimeContext` | Fingerprint model config, GPU inventory, topology, dependencies/sources, workload and routing settings. |
| `Benchmark` | Supply a single comparable E2E experiment covering the reference and candidate plans, with WS1/WS2 and nonempty zero-difference E2E evidence. |
| `Catalog.select` | Choose the lowest E2E time in that exact comparison, retaining the reference on a tie. |

Selection uses **the fastest measured, qualified plan in the shipped comparison for
this configuration**. It makes no claim about untested upstream kernels, different
hardware, or all possible workloads. A comparison cannot combine independently
fastest Attention/GEMM/collective results. Mixed plans need their own WS1, WS2 and
end-to-end validation, including forward, backward and weight refresh as applicable.

Without an exact matching comparison, selection keeps the existing reference
recipe and logs `shipped_reference; no comparable candidate benchmark`. The
initial catalog deliberately contains no synthetic performance scores. Historical
reference evidence documents the existing numerical path; it is not fresh GPU
validation of this launcher change or every possible dependency/workload.

Registering a `Benchmark` requires a report, passing WS1/WS2, positive compared
element count, zero mismatch count and zero maximum absolute difference, and
finite positive E2E timings. These metadata checks do not prove correctness by
themselves: reviewers must inspect the referenced test artifacts. Compare the
same workload, inputs, measurement method, warmup, graph settings and runtime.
Only one reviewed comparison is active per context; replace it when a new
candidate arrives rather than merging numbers from unrelated experiments.

## Startup and failure behavior

1. The launcher builds its usual workload and strict runtime settings. The BI
   layer reads the actual model and devices, validates the paired adapter, and
   resolves the plan before Ray submission.
2. The launcher exports the selected settings and a versioned `RL_KERNEL_BI_PLAN`
   envelope to all workers. Do not set this internal variable yourself.
3. Megatron initialization and the vLLM plugin verify the catalog plan, model
   configuration, dependency/source identity and exported settings. GPU checks
   run in model workers; frontend plugin discovery does not initialize CUDA.
4. Readbacks include the plan/context digests. Run validators check that training
   and rollout agree with the launch manifest. ROCm graph cache namespaces also
   include the BI plan/context identity.

Model mismatch, unsupported scope, missing adapter, conflicting native/P/R/R/P
settings, changed worker dependencies, a missing plan envelope, or changed plan
identity stop initialization. Existing strict runtime fallback/mismatch checks
remain enabled. A numerical failure never silently selects an unvalidated fast
library. A replacement path must be qualified as another complete plan first.

`RL_KERNEL_BI=1` requires `RL_KERNEL_MODE=strict` and consistency mode. Unset the
flag before native or mixed-arm experiments. Leaving the flag unset (or setting it
to `0`) preserves the previous launcher routing. Changing it after initialization
is unsupported; restart the job to select a new plan.

The existing launchers still own checkpoint conversion, weight synchronization,
sampling semantics, graph settings, mismatch sidecars and their preflight checks.
Vime, Megatron and editable dependency source checkouts must be clean for BI
identity verification. Package versions/build records, driver information and
RL-Kernel implementation content participate in the runtime fingerprint.
The envelope is a reproducibility record, not a cryptographic trust boundary.

## Adding a reviewed upstream implementation

Contributors can add reviewed implementations through these interfaces. Plans
and comparisons ship with the installed RL-Kernel version.

1. Add the upstream implementation behind an exact backend route, preserving the
   numerical and backward contracts. Reuse the existing semantic registry where
   appropriate. If framework hooks differ, add a paired `PlanAdapter` in
   `rl_engine/bi/adapters.py` instead of modifying model selection conditionals.
2. Register a distinct immutable `ExecutionPlan` in `builtin_catalog()`. Include
   evidence locations and exact environment selectors supported by the adapter.
   A candidate without a matching benchmark stays inactive.
3. Capture the reference run's `bi_plan.context` and `context_digest`. Run the
   complete candidate against that reference with the same runtime and workload.
   Preserve raw WS1, WS2, E2E equality and timing artifacts. Record source/build
   versions for reused upstream libraries in those artifacts.
4. Register a `Benchmark` in `rl_engine/bi/builtin.py` with that context digest, both content-addressed plan
   digests, and the measured E2E seconds. Submit implementation and evidence for
   human review. The report must cover every included plan. Keep benchmark
   registration in that catalog module: its content is excluded from the runtime
   source hash to avoid a self-referential context digest. Adapters and numerical
   implementations remain included in the hash.
5. Once merged and installed, subsequent jobs with the same user flag and a
   matching context select the winner. Dependency, model, topology or workload
   changes invalidate the comparison match. Existing running jobs keep their plan.

For another framework's launcher, call
`rl_engine.bi.runtime.prepare_environment(environment, framework=..., ...)` after
constructing its baseline settings and before starting worker processes. Supply
the actual model path, topology, workload and source repository identities.
Forward the entire returned environment; workers call `verify_worker_plan()` and
dispatch through the selected adapter. Model contributors must define any extra
MoE, recurrent-state, multimodal or communication contracts in their own adapter
and evidence suite. This initial implementation has no such model-specific logic.

## Validation of this change

CPU tests cover selection, invalid or stale evidence, model identification,
settings conflicts, CUDA launch manifest/Ray propagation, ROCm shell forwarding,
framework bootstrap guards, worker identity changes and cross-worker readbacks.
CPU tests do not certify GPU arithmetic or speed.

Before merging, run the following on the existing reference environments:

```bash
# 8x H100, existing prepared CUDA profile
export RL_KERNEL_BI=1
./rlk run --mode consistency --steps 2 --require-updates --wait

# 8x MI300X, existing prepared ROCm profile
./rlk run --profile examples/vime_rocm_attention_ablation/profiles/mi300x-qwen3-8b.json \
  --mode consistency --steps 8 --wait
```

For both backends, require identical plan/context identities in the launch
manifest and every training/rollout readback; nonempty comparisons with
`mismatch_count=0` and `max_abs_diff=0`; no fallback; successful graph capture/replay;
and successful post-update weight refresh. Repeat the existing 200-step workload
with BI enabled and the existing strict route with BI unset, using the same
seeds/inputs and measurement protocol, to check that startup selection adds no
steady-state regression. Attach measured results to the PR before claiming a
new performance result. Benchmark candidate selection additionally needs a real
upstream candidate comparison; the unit tests use explicitly synthetic timings.
