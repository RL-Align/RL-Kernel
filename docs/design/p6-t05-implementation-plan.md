# P6 T05 implementation plan

Goal: implement the forward fused kernel against #448's proposed profile.
Spec: `docs/design/p6-t05-fused-combine.md`.

Execution: implement inline; preserve the original worktree; leave changes
uncommitted for review. The user explicitly requested direct implementation.

Status: the final-audit A100 suite passed 91 tests (43 CPU + 48 actual CUDA),
with 12 optional offline compile tests skipped, as reported by the user.
The immutable attempt-002 readback also passed artifact integrity/CPU replay
verification (9 cases, 45 operator recordings; no GPU reexecution).
Six shared-A100 benchmark runs were BYTE_EQUAL, but are not stable performance
certification. Local host checks/offline compilation and independent review passed. See
`docs/validation/p6-t05-2026-10-02.json` for executed checks. Review fixed lazy
negative storage flags and grid.y bounds with host RED -> GREEN regressions.
Triton 3.2's offline compiler uses `constants`; 3.8 uses `constexprs`; only the
optional compile test adapts that tooling API. Production kernel math is unchanged.

## Files and interfaces

- `rl_engine/p6/combine.py`: lazy CUDA binding, validated preparation,
  prepared graph launch, explicit status checking and debug serialization.
- `rl_engine/p6/combine_provider.py`: input-bound actual candidate registration
  through T01's seam, not a production/autograd/Foundation adapter.
- `rl_engine/kernels/ops/triton/moe/combine.py`: one fused CUDA kernel,
  with `add.rn.f32` instructions and optional boundary stores.
- `tests/p6/test_fused_combine.py`: CPU metadata/errors and opt-in GPU bytes.
- `benchmarks/p6_combine.py`: prepared-kernel versus eager-reference timing.
- `docs/operators/p6-fused-combine.md`: remote A100 preflight and test commands.

## Task 1: preparation and checked binding

- [x] Write CPU tests: lookup uses opaque token IDs, missing/duplicate slot and
  wrong context fail; CPU backend and unsupported tensor inputs fail closed.
- [x] Run tests and observe the missing implementation failure.
- [x] Implement `prepare_combine(plan, context, device)` and immutable prepared
  metadata. Support NVIDIA CUDA capability >= 8.0; never silently copy inputs.
- [x] Run CPU tests and the inherited P6 tests.

## Task 2: fusion and debug

- [x] Write GPU tests against the independently frozen start-kit bytes.
  Cover every stage, debug on/off, tails, empty tokens/routes, invalid/padding,
  cancellation, ties and signed zero. Add layout/dtype/autograd negative tests.
- [x] Implement FP32 ordered gather/fold/merge in one Triton kernel; upload only
  `[T,6]` lookup. Debug-off must not allocate `[T,6,H]` or slot partials.
- [x] Status check active inputs, intermediate additions and BF16 output;
  reject nonfinite/subnormal/overflow. No atomics for computation or status.
- [x] Add changed-input CUDA Graph replay and explicit post-replay status check.
- [x] Add cases for physical permutation, batch partitioning, poison padding and
  two hidden block widths without changing expected token bytes.
- [x] Run host tests and compile GPU kernels offline if tooling allows;
  GPU execution is delegated to the user's A100 with explicit instructions.

## Task 3: handoff and performance

- [x] Write a benchmark with correctness before timing, debug disabled,
  allocation-free prepared launch and the T02-T04 arithmetic reference.
- [x] Document GPU idle checks, environment setup, opt-in tests, compute-sanitizer
  and benchmark commands; explain synthetic versus live certification.
- [x] Ruff, whole pytest collection/run and independent review. Full collection is
  blocked by an inherited ROCm import error; see the validation record.
- [x] Initial 19 actual kernel cases on A100, user-reported pass; fix integer
  overflow in the test fixture with an independent CPU RED -> GREEN regression.
- [x] Audit official contribution requirements: navigation, document contract,
  CI trigger/formatter/compile gates, T02-T04 serial reference, P4 mock scheduling,
  exact candidate/recorded comparison and immutable GPU-byte evidence test.
- [x] Fix write-buffer/input alias races; host regressions RED -> GREEN for
  output and debug stores, including nonzero storage offsets.
- [x] Execute the updated 48 CUDA + 43 CPU cases on A100; user-reported pass.
- [x] Persist attempt-002 and verify its seal; retain remote JUnit/readback files.
- [x] Collect six shared-A100 benchmark runs, including checked-API costs;
  keep raw evidence in `docs/validation/p6-t05-a100-2026-10-03`.
- [ ] Compute Sanitizer if available (otherwise report unavailable) and exclusive/idle
  A100 performance repeat before making a stable speedup claim.
- [ ] T02/T03/T04 and T01 owner approval of provisional interface/profile.

Review focus: strict signed-zero initialization; inactive NaN padding;
zero-token launch; stale saved identity; graph state reset on every replay;
BF16 overflow after a finite FP32 merge; fail-closed numeric status under capture.
