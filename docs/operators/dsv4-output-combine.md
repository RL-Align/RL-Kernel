# DSV4 ROCm fixed-order output combine

Tasks 53–55 in [#522](https://github.com/RL-Align/RL-Kernel/issues/522) implement
the arithmetic after expert outputs return from P4. They build on the plan and
mapping contracts in [#611](https://github.com/RL-Align/RL-Kernel/pull/611) and
[#612](https://github.com/RL-Align/RL-Kernel/pull/612).

## Arithmetic

P5 has already applied each route weight before FC2. P6 consumes those expert
outputs without weighting them again. For every token and hidden column:

1. Start with FP32 positive zero.
2. Add slots 0, 1, 2, 3, 4 and 5, in that order. Convert each input to FP32
   before its addition. Keep the accumulator in FP32.
3. If a separate shared-expert output is supplied, add it once in FP32.
4. Cast once to the declared output dtype. BF16 uses round-to-nearest, ties-to-even.

The numerical profile is `dsv4-output-combine-fp32-slot012345-shared-last-v1`.
Neither a pairwise tree nor a parallel slot reduction is equivalent to this
profile. Starting from positive zero also fixes how negative-zero inputs are
handled. FP32 accumulation can overflow; the implementation does not saturate
or rearrange additions to avoid it.

There is no residual input to this execution path. P1's `mhc_post` owns the
four-stream residual. A plan requesting `residual_mode="external"` is rejected;
supporting a different residual requires an explicit P1/P6 contract first.

## Task boundaries

- Task 53: `fixed_output_combine_reference` restores canonical slot order and
  exposes all six FP32 accumulation steps and the value before the final cast.
- Task 54: the shared output is added after slot 5 and before that cast, once
  per token. The shared MLP itself remains task 47.
- Task 55: one Triton kernel gathers each slot through a prepared row map,
  performs the same additions, optionally adds shared output and stores the
  result. Hidden columns run in parallel; slot additions remain sequential.

The task 52 mapping validation is factored into `canonical_expert_rows` and
shared by its original CPU inverse-map reference and this backend. The original
reference still returns `[T,6,H]`. Normal fused execution instead uses only an
integer `[T,6]` row map and the `[T,H]` output. Debug mode additionally allocates
FP32 step traces; it preserves the arithmetic.

This is an explicitly selected MI300X/gfx942 and MI355X/gfx950 candidate. It uses
fixed 256-column tiles, four warps, 64-bit address offsets and disabled floating
point fusion. It has no atomics, autotuning, shared global workspace or automatic
replacement of AITER/other combine paths. Native semantic-registry eligibility
is not claimed before hardware acceptance.

## Preparation and execution

Use task 51's `MoEOutputCombinePlan` and `MoEOutputCombineInputs`. The plan must
declare Top-6, `expert_weight_applied=True`, and excluded residuals. Expert and
shared outputs are both BF16, or both FP32 for an explicitly declared FP32 path.
Token/slot/expert indices may be int32 or int64. Strided value buffers are allowed.

```python
from rl_engine.contracts.operators.fixed_output_combine import prepare_fixed_output_combine
from rl_engine.ops.moe.output_combine import output_combine

prepared = prepare_fixed_output_combine(
    plan, inputs,
    expected_mapping_run_id=run_id,
    num_experts=global_expert_count,
)
y = output_combine(
    prepared, inputs.expert_output, inputs.shared_output,
    mapping_run_id=run_id,
)
trace = output_combine(
    prepared, inputs.expert_output, inputs.shared_output,
    mapping_run_id=run_id, debug=True,
)
# trace.output, trace.slot_steps[T,6,H], trace.before_cast[T,H]
```

Preparation checks the full token-slot domain, missing/duplicate/out-of-range
entries, declared expert range, plan/schema and expected run identity. It also
validates the initial finite expert/shared values. Invalid slots are rejected;
they are not silently replaced with zeros. The current task 51/52 contract
requires a positive token count, so T=0 is rejected explicitly.

Preparation may synchronize and must run outside Graph capture. It owns a copy
of the mapping; later mutations of the producer's metadata do not change that
snapshot. The public diagnostic accessor returns a copy as well. A different
routing plan or run requires new preparation and a new graph capture.

Execution checks run identity, shape, dtype, device and mapping version counters
without reading tensor values on the host. It permits value-buffer updates for
the same routing plan. Callers must validate new values outside capture; this
execution API does not rescan for NaN/Inf on each call or graph replay. Replay
uses the captured pointers and mapping. Private mapping buffers must not be
modified; Python guards cannot execute during device graph replay.

## Autograd

The output VJP copies each token's incoming gradient to its six original expert
rows and returns the same incoming gradient to the shared branch, if present.
Every source row is unique in the validated map, so this copy needs no atomic
addition. It does not multiply by route weights or reduce expert input gradients.

This is first-order autograd with the usual straight-through cast derivative.
Debug traces are non-differentiable. Saved mapping tensors participate in
PyTorch mutation checks. Task 56 still owns the ordered merge of expert, shared
and gate dX before P1 normalization backward; tasks 58/59 own distributed
validation and measured performance.

## Validation

```bash
OMP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
tests/reference/test_moe_output_combine.py tests/ops/moe/test_output_combine.py

TRITON_INTERPRET=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
tests/backends/rocm/test_output_combine_interpreter.py

RLK_REQUIRE_ROCM=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
tests/backends/rocm/test_dsv4_output_combine.py
```

CPU checks cover every accumulation step, cancellation, one-ULP differences,
signed zero, final-cast placement, shared-once handling, invalid/stale mappings,
inference-mode preparation and local autograd wiring.

Interpreter checks cover the actual Triton gather/add/VJP bodies, FP32 stages
for BF16 and FP32 inputs, and debug-mode invariance. Triton 3.7's CPU interpreter
truncates FP32-to-BF16 conversion, so it is not an oracle for the final BF16 cast.
Exact BF16 output checks remain in the native GPU suite and are not relaxed.

Device tests compare fused output and step traces against the reference, sweep
token counts and positions, and cover shuffled arrivals, tails, strided storage,
gradients, concurrent streams and Graph value updates. Run them on both target
GPUs. CPU checks and offline compilation do not establish device correctness,
performance, real EP transport or whole-model train/rollout consistency.
