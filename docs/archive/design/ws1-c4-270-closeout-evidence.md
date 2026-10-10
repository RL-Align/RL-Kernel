# WS1 C4 (#270) closeout evidence

> Historical snapshot from the original C4 landing. The current adapter and
> accumulator protocol supersedes the runtime tally below: CUDA/Triton
> RMSNorm and deterministic GEMM parameter VJPs now use logical-row FP32
> contributions, CUDA generic logp has a row-local VJP, and Triton embedding /
> LM-head / plain logp candidates are declared in C2. Re-run the current C8
> sweep for authoritative evidence.

**Parent:** #266 · **Depends on:** #267 / #268 · **Branch:** `feat/ws1-c4-gradient-invariance-270`
**Scope:** shared gradient harness + enumerable adapters only

## Acceptance map

| #270 / #266 criterion | Evidence |
| --- | --- |
| Cross-config API | `assert_gradient_batch_invariant(...) -> GradientInvarianceReport` |
| Accuracy vs invariance | `accuracy_reports` (`gradient_accuracy`) and `invariance_reports` / `singleton_aggregate_reports` (`gradient_invariance`) |
| Batch/Chunk bitwise after logical aggregation | C1 `gradient_invariance` resolver; adapters run on `config.physical_layout` and token grads restore through C2's map |
| Shared upstream / reduction / denom | Harness injects `active_token_denominator`, `loss_reduction`, `aggregation_order`; adapters seed `autograd.grad` with an upstream that is a pure function of logical identity |
| Stable grad names | `GRADIENT_ADAPTERS` (`dx`/`dweight`, `dX`/`dW`, `dQ/dK/dV`, …) |
| Every differentiable WS1 op enumerable | Registry + `test_required_ops_are_enumerable`; all 13 runnable adapters execute the full 13-cell matrix |
| Pack / KV rule | Pack registered (`layout_supported`), inactive tokens contribute 0; KV `absent_not_required` |
| Missing Triton required node is red | Status matrix marks C2 `missing_required` embedding / lm_head / logp as tracked red; CLI refuses them |
| No cross-profile borrow | Declared CUDA and Triton candidate paths must differ |
| No `atomicAdd` | Source-file audit on listed BI candidates (zero `atomicAdd` in `csrc/`) |
| No private thresholds | All compares go through the C1 resolver |
| Diagnostics | max abs/rel per named gradient, first failing op/tensor/config pair |
| C4 ≠ EXIT | This document does not claim C8/C10/C11 or full-model #150 |

## The matrix is falsifiable

Each C2 cell now hands the operator a genuinely different physical input. For
`rms_norm` with `hidden=8`:

| Config | Operator calls | Row shapes |
| --- | --- | --- |
| `BN/full` | 1 | `(59, 8)` |
| `BN/chunked` | 10 | `(7,8) (4,8) (7,8) (7,8) (2,8) (7,8) (6,8) (7,8) (7,8) (5,8)` |
| `B1-singleton_aggregate/full/s0` | 1 | `(11, 8)` |
| `BN/padded_right` / `BN/padded_left` | 1 | `(80, 8)` |
| `BN/permuted` | 1 | `(59, 8)` in permuted sample order |

`tests/validation/operators/test_gradient_invariance.py::TestPhysicalLayout` locks this in: a
layout-sensitive synthetic operator must be judged **red**, a
logical-identity-only operator must be judged **green**, `B=N` must be one
batched call, chunking must split it, and padding must reach the operator.
Without those guards a regression to layout-blind adapters makes every bitwise
verdict a tautology.

CPU-safe contract regression:

```bash
.venv/bin/python -m pytest -q \
  tests/contracts/test_tolerance_contract.py \
  tests/validation/operators/test_ws1_workload.py \
  tests/validation/operators/test_forward_invariance.py \
  tests/validation/operators/test_gradient_invariance.py \
  tests/validation/operators/test_op_checks.py
```

Result: `133 passed` (`test_gradient_invariance` 26 passed). `mypy` and
`flake8`/`black`/`isort` clean on the C4 files.

## Runtime verification

Every runnable adapter was swept against its C2-declared candidate on both
required profiles, on NVIDIA GeForce RTX 4070 Ti SUPER (`sm89`):

```bash
.venv/bin/python tools/validation/operators/sweep_gradient_invariance.py
```

The sweep classifies every cell (`green` / `red_verdict` / `red_no_backward` /
`blocked_hardware` / `blocked_c2` / `skipped`) and exits non-zero unless all
non-skipped cells are green. Single cells still run through
`tools/validation/operators/check_gradient_invariance.py --op … --candidate … --backend-profile …`.

Current tally: `green=8, red_verdict=6, red_no_backward=1, blocked_hardware=4,
blocked_c2=3, skipped=4`.

**Re-running on Hopper.** The four `blocked_hardware` cells are the only ones a
different GPU can resolve. On an H20 (`sm90`) the extension must first be built
with SM90 kernels, otherwise the candidates still fail to load:

```bash
KERNEL_ALIGN_FORCE_SM90=1 pip install -e .
python tools/validation/operators/sweep_gradient_invariance.py
```

Hopper does **not** change the two open findings below or the three Triton
`missing_required` nodes — those are implementation gaps, not hardware gaps.

| Op | `cuda_bf16` | `triton_cuda_bf16` |
| --- | --- | --- |
| `attention` | **green** | **green** |
| `silu` | **green** | **green** |
| `swiglu` | **green** | **green** |
| `rope` | needs Hopper (`cuda-sm90`) | **green** |
| `batch_invariant_logp` | needs Hopper (`cuda-sm90`) | **green** |
| `rms_norm` / `qk_norm` | red — `dweight` | red — `dweight` |
| `det_gemm` | red — `dW` | red — `dW` |
| `embedding` / `lm_head` | needs Hopper (`cuda-sm90`) | C2 `missing_required` |
| `logp` | red — **no backward** | C2 `missing_required` |
| `linear_logp` | skip (`optional_fused`, no C2 node) | skip |
| `pack` | profile-independent (CPU contract test) | profile-independent |

7 of 26 cells green. Detail for the reds:

- **`dweight` / `dW` chunk + singleton aggregate** — `rms_norm` and `qk_norm`
  `dweight` max abs `1.77621841e-04` (chunk) and `3.07083130e-04` (N× B=1
  aggregate), identical on both profiles; `det_gemm` `dW` the same class. `dx`,
  `dX`, permutation and padding are all `0.0` bitwise. See the open finding
  below.
- **`logp` has no backward** — `FusedLogpGenericOp` is not a
  `torch.autograd.Function`, so its output has no `grad_fn`. Reported as
  `MissingBackwardError` → a categorised red, not an autograd stack trace.
- **Hopper-only cells** — the `cuda_bf16` profile declares `cuda-sm90`
  candidates for `embedding`, `lm_head`, `rope` and `batch_invariant_logp`.
  Complete CUDA-profile evidence requires a Hopper GPU with
  `KERNEL_ALIGN_FORCE_SM90=1`; this box cannot produce it, and the CLI refuses
  rather than falling back.
- **`pack`** — `layout_supported`, the same PyTorch op under both profiles and
  not a C2 backend node. C1 provenance requires
  `requested == actual == profile backend family`, so a per-profile gate could
  only pass by recording a backend that never ran. The CLI refuses; its
  gradient contract is covered on CPU instead.

The earlier "`0.0` everywhere, `rms_norm` only" evidence is **void**: it was
produced by adapters that ignored `config.physical_layout` and ran `B=N` as
N× `B=1`, so every cell compared one computation against itself.

The GPU gate also needs shapes the real kernels accept — the deterministic CUDA
attention requires `head_dim == 128`, so the CLI exposes `--n-heads`,
`--n-kv-heads` and `--head-dim` and defaults to a runnable shape.

## Historical finding — CUDA `logprob` had no backward

> **Historical snapshot only.** This residual was open at the C4 landing
> (`596feb0`). Current C8 evidence
> (`docs/archive/design/ws1-c8-274-closeout-evidence.md`,
> `docs/archive/design/ws1-c8-execute.json`) reports `logp` green on both profiles at
> source commit `5c33dcd` with manifest `ws1-c2-v7`. Do not treat this section
> as a live blocker.

`FusedLogpGenericOp` previously called `_C.fused_logp` without a
`torch.autograd.Function`, so `dlogits` could not be produced. That gap is
closed by the row-local FP32 softmax VJP bridge; see `docs/archive/design/ws1-blockers.md`.

## Historical finding — RMSNorm `dweight` is not chunk/batch decomposable

> **Historical snapshot only.** At the C4 landing, kernel-level `dweight` /
> `dW` accumulation was shape-dependent. The current adapter protocol reduces
> logical-row FP32 contributions, so C8 evidence no longer treats this as a
> live red cell. Re-run the C8 sweep for authoritative status.

`dx` was bitwise invariant across the whole matrix on both profiles. Kernel
`dweight` was not, because of a row-count-dependent accumulation shape:

- CUDA: `csrc/cuda/norm/rmsnorm.cu:71-75` fixes `RMSNORM_DW_ROWS_PER_CHUNK = 256` and
  derives `chunks = ceil(T / 256)`; `rmsnorm_partial_dw_kernel` left-folds rows
  inside a chunk (`csrc/cuda/norm/rmsnorm.cu:181-196`).
- Triton: `_rmsnorm_bwd_dw_kernel` accumulates `acc += tl.sum(vals)` over
  `tl.range(0, T, BLOCK_T)` (`rl_engine/backends/shared/triton/norm/rmsnorm.py:48-58`).

Both are deterministic for a fixed `T`, but splitting the same tokens across
launches re-associates the sum: a left fold over 59 rows is not bitwise equal to
the sum of left folds over 11 + 16 + 13 + 19 rows. That is precisely the
`shape_dependent_bwd_accum = forbidden` property the adapter registry declares —
previously asserted only as a string, never as behaviour. `det_gemm`'s `dW`
failed the same way on both profiles at the C4 landing.

Historical tracked red at the C4 landing: Triton `embedding`, `lm_head`, and
plain `logp` were C2 `missing_required`. Those candidates are now declared and
green in C8 evidence (`docs/archive/design/ws1-c8-execute.json`).

## Parent boundary

This closes only the C4 harness, adapter registry, and canonical aggregation
contract that C8/C10 must reuse. It does not claim the full-model, KV-cache, or
CI EXIT requirements of #266.
