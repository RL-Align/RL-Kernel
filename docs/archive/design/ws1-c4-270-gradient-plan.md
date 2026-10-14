# WS1 C4 (#270) Landing Plan — Gradient-invariance harness & adapters

**Parent:** #266 · **Issue:** #270 · **Depends on:** C1 (#267), C2 (#268)
**Branch:** `feat/ws1-c4-gradient-invariance-270` (from `feat/ws1-c3-forward-invariance-269`)

C4 is a hard prerequisite of C10. C4 green alone is **not** WS1 EXIT.

---

## 1. Goal

Give every differentiable WS1 op — and later the full chain — **one**
training-style gradient comparison semantic under the C2 Batch/Chunk matrix,
so tests do not invent their own upstream grads, loss reduction, or
active-token denominator.

## 2. Locks from #270 and #266

| Item | Lock |
| --- | --- |
| API | `assert_gradient_batch_invariant(op, configs, contract) -> GradientInvarianceReport` |
| Judgments | `gradient_accuracy` (vs FP32 VJP) and `gradient_invariance` (cross-config) are separate; **no** silent forward inheritance; **no** private atol/rtol |
| Batch/Chunk invariance | bitwise after logical aggregation (`atol=0`, `rtol=0`) |
| Accuracy | only FP32-reference `gradient_accuracy` rows from C1 |
| Logical identity | C2 `(sample_id, token_position)`; compare only after restore |
| B1 vs BN | same sample/token multiset; N× B=1 `singleton_aggregate` in **fixed sample order** vs one B=N |
| Shared across configs | same upstream grad (keyed by logical identity), same `loss_reduction`, same **global** `active_token_count_across_all_samples` |
| Naming | `singleton_aggregate` is a C2 execution mode only — never a C1 `comparison_*_role` |
| Profiles | `cuda_bf16` and `triton_cuda_bf16` are independent; missing required Triton bwd is **red**, not N/A or fallback; neither profile may borrow the other |
| Adapters | real registered adapters (`GRADIENT_ADAPTERS` / `OP_SPECS`); name-only mention in a chain report does not count |
| Pack / KV | adapter required **only if** declared supported **and** differentiable |
| Defects | do **not** reopen #145–#151 / #153; open a Blocker if a sweep finds an untracked red |
| Out of C4 | full-model e2e (C9/C10), KV path (C6/C7), four-judgment evidence matrix (C8), CI gates (C11), new kernels |

C2 already pins Triton `embedding` / `lm_head` / plain `logp` as
`missing_required`. C4 must surface those as **tracked red**. It must not
implement the missing kernels and must not treat them as skip/N/A.

## 3. Deliverables

| Path | Role |
| --- | --- |
| `rl_engine/validation/operators/gradient_invariance.py` | Shared API, report schema, B1 aggregate, C1 thresholds |
| `rl_engine/validation/operators/gradient_adapters.py` | Enumerable adapters + stable grad names + status matrix + bwd audit list |
| `tools/validation/operators/check_gradient_invariance.py` | One GPU command per required profile |
| `tests/validation/operators/test_gradient_invariance.py` | CPU contract tests (no GPU required) |
| `docs/archive/design/ws1-c4-270-gradient-plan.md` | This plan |
| `docs/contributing/gtest-usage.md` | Point C4 at the shared API (no private thresholds) |

Reuse, do not fork: C1 resolver, C2 workload / `build_config_matrix`, C3
`ConfigSpec` / comparison helpers / provenance checks.

## 4. API and report

```text
assert_gradient_batch_invariant(
    op, configs=None, contract=None, *,
    grad_tensors, backend_profile, provenance, gold_fn, ...
) -> GradientInvarianceReport
```

`op(config, **op_kwargs)` returns either:

- `{grad_name: token_map | parameter_tensor}`
- `GradientObservation(grads=..., actual_backend, kernel_id, output_dtype, device)`

Token maps are `{ (sample_id, token_position): Tensor }` or a physical tensor
that the harness restores via C2. Parameter tensors are compared after the
singleton aggregate described below.

`GradientInvarianceReport` (C10 must reuse this schema):

- `accuracy_reports` — judgment `gradient_accuracy`
- `invariance_reports` — token VJPs and non-singleton parameter grads
- `singleton_aggregate_reports` — N× B=1 parameter grads vs BN
- provenance / candidate / device / CC / seed / fallback
- `loss_reduction`, `active_token_denominator`, `grad_tensor_names`
- `first_failing_op`, `first_failing_tensor`, `first_failing_config_pair`
- `passed` requires accuracy + invariance + aggregate + provenance + metadata

Logprob aggregates (`max_abs_dlogp` / `approx_kl0` / `clipfrac0`) judge
**outputs only**. They do not appear in this report.

## 5. Training-style VJP (fixed across configs)

From the C2 manifest:

- `loss_reduction = sum_over_active_tokens_then_optional_mean_by_active_count`
- denominator = `active_token_count_across_all_samples` of the **full** BN
  logical batch (not the local B=1 count)
- upstream `g[sample_id, token_position, ...]` is a pure function of logical
  identity (no layout-order RNG)
- inactive / pad tokens contribute 0

Then `sum_i ∇_θ L(B=1 sample i)` equals `∇_θ L(B=N)` for a batch-invariant op.
Each B=1 `dweight` is **not** compared to BN `dweight` by itself.

## 6. Required adapters (stable names)

| Op | Grad names | Kind |
| --- | --- | --- |
| `rms_norm` / `qk_norm` | `dx`, `dweight` | token, parameter |
| `det_gemm` | `dX`, `dW` | token, parameter |
| `attention` | `dQ`, `dK`, `dV` | token |
| `embedding` | `dweight` | parameter |
| `lm_head` | `dhidden`, `dweight` | token, parameter |
| `logp` / `batch_invariant_logp` | `dlogits` | token |
| `linear_logp` | `dhidden`, `dW` | token, parameter (optional fused path) |
| `rope` / `silu` | `dx` | token |
| `swiglu` | `dgate`, `dup` | token |
| `pack` | `dx` | token (packing is C2 `supported` and differentiable) |
| `kv_cache_attention` | — | **absent_not_required** (not declared supported+differentiable on the C2 training path; C6/C7 own KV) |

Attention / RoPE adapters must not mix samples into one flattened sequence.
They materialize per-sample (or padded) logical rows so Batch/Chunk compares
the same token multiset.

## 7. Status matrix and Blocker rule

For each `(backend_profile, adapter)`:

| C2 / capability | C4 status |
| --- | --- |
| `declared` + adapter + matching family candidate | runnable |
| `missing_required` (Triton embedding / lm_head / plain logp) | **tracked red** |
| required + no adapter, or declared + borrowed other profile | **untracked red** → C4 fails; open Blocker, do not reopen closed op issues |
| pack supported + differentiable | adapter required; not a C2 profile node |
| KV not declared supported | `absent_not_required` |

C4 unit tests fail on any **untracked** red. Tracked C2 `missing_required`
rows stay visible and keep the CLI red if someone tries to run them.

## 8. Bwd contract audit

Every BI candidate adapter lists its source files. Tests forbid `atomicAdd`
and record `shape_dependent_bwd_accum=forbidden`. This is an audit of
declared candidates, not a kernel rewrite.

## 9. Test plan (`tests/validation/operators/test_gradient_invariance.py`)

CPU-only, synthetic ops plus one real PyTorch `rms_norm` adapter:

- accuracy vs invariance use different C1 judgments
- invariance is bitwise; accuracy uses `gradient_accuracy`
- B1/BN share sample set, upstream identity, global denominator, fixed order
- parameter grads pass only after singleton aggregate
- missing active token / missing gold_fn hard-fail
- provenance + cross-profile fallback fail closed
- both profiles share the report schema
- every required differentiable op is enumerable with stable names
- status matrix: tracked red vs untracked red; pack present; KV absent
- CUDA and Triton declared candidates are distinct paths
- no `atomicAdd` in listed BI candidate sources
- no private thresholds; no `singleton_aggregate` comparison role

## 10. GPU evidence command (not EXIT)

```bash
python tools/validation/operators/check_gradient_invariance.py \
  --op rms_norm --candidate cuda \
  --backend-profile cuda_bf16 --json

python tools/validation/operators/check_gradient_invariance.py \
  --op rms_norm --candidate triton \
  --backend-profile triton_cuda_bf16 --json
```

CUDA unavailable, missing candidate, `missing_required`, SM90-on-non-SM90,
or provenance mismatch → exit red. This is C4 harness evidence, not C8/C10.

## 11. Explicit non-claims

C4 does **not** claim: full Qwen3-8B model, #150 matrix on the full model,
stateful KV / generate-rescore, C8 four-judgment greens, or WS1 EXIT.
