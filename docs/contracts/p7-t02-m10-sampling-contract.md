# P7 / T02: Sampling Params Gate and M10 Semantics — Implementation Spec

Status: Draft (WS1 primary implementation; WS2 continuation awaits a distributed environment). Gaps that consume T01 frozen artifacts are marked `[pending T01]`.

Metadata:

- Spec version: `p7-t02-spec.v1`
- Task: T02 (Sampling Params Gate and M10 numerical contract)
- Contract ref: `p7-task-contract.v3`
- Reviewer: T09
- Code baseline: `upstream/test-dsv4-cuda @ 1dd053f` (rebased from `origin/main @ 4406afc` on 2026-11-01; another rebase may be required once T01 publishes its frozen baseline, see §9)

> This document is T02's internal implementation spec, not an external contract. On conflict,
> `p7-task-contract.v3`, the accompanying schemas, the fixture manifest and the test commands
> take precedence. This spec turns only the contract's **T02 section**
> (`p7-task-selection.md#L197-L223`) and the **§2 shared M10 input/output contract**
> (`#L43-L80`) into implementable fields and decision rules.

---

## 1. Goal and scope

**One-line goal:** ensure both the training side and the rollout side use the same
**raw model logprob** semantics; any non-identity sampling setting, actual semantic drift,
LM Head LoRA, or undeclared backend must fail explicitly **before computation or comparison**.

**In scope:**

- Normalization of the canonical sampling record and the four-layer check
  (requested / normalized / materialized / runtime actual).
- The "call the existing `linear_logp` and validate its semantics" part of the M10 numerical
  contract (not a reimplementation of GEMM / log-softmax / `dlogits`).
- Parameter provenance fields and positive/negative tests.
- WS2 continuation: per-rank canonical sampling record completeness and consistency
  (this spec defines the fields; implementation is left to WS2).

**Out of scope (explicitly not P7/T02 per the contract):**

- No new GEMM kernel, TP LM Head, log-softmax or `dlogits` backward (`#L30`, `#L214`).
- No replay of sampling transforms (consistency mode requires identity; the main path does
  not replay, `#L33`).
- No expert LoRA implementation (only record and validate its identity, `#L32`).
- No materialization of the full `[token, vocab]` logits (`#L30`).

---

## 2. Dependencies and baseline

| Dependency | Status | Notes |
| --- | --- | --- |
| T01 frozen M10 input/output schema | [pending T01] | Contract `#L43-L80` already lists the fields; this spec instantiates them. Formal dtype/shape/layout follow the T01 manifest. |
| T01 boundary-contract manifest (M10 row) | [pending T01] | `comparison_mode`, `tolerance_contract_version`, fixture checksum |
| T01 frozen canonical hash version | [pending T01] | Used by `boundary_hash_index` |
| P1 shared fixed-K/batch-invariant GEMM arithmetic spec + bit-equivalence harness | [pending T01] | `#L173`; used to verify the LM Head path of the existing `linear_logp` |
| T01 provider stub + synthetic fixture + CPU oracle | [pending T01] | `#L168`; until they exist, the self-built stub in §7 stands in |
| Real provider actual readback from T05/T06 | WS2 only | Affects only WS2 actual validation; does not block WS1 (`#L22`) |

**Baseline conclusion:** work can start as soon as the T01 start kit exists (`#L601`).
Real readback from T05/T06 does not block WS1.

### 2.1 H20/SM90 pre-validation evidence (non-normative gate evidence)

This machine is `NVIDIA H20` (compute capability **9,0** = sm_90) with the CUDA toolkit 12.8
at `/usr/local/cuda-12.8`. After building the native extension with
`KERNEL_ALIGN_FORCE_SM90=1` (`_EXT_AVAILABLE=True`, with `fused_linear_logp_sm90` /
`_backward` / `_global_target` / `ordered_logp_merge` all present), every existing SM90 case
actually executed and passed:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export KERNEL_ALIGN_FORCE_SM90=1
export MAX_JOBS=16
python -m pip install -e . --no-build-isolation
python -m pytest tests/ops/logprob/test_linear_logp.py -k sm90 -v
# 24 passed, 29 deselected
```

Coverage includes `test_sm90_forward_matches_native_bf16`,
`test_sm90_strict_contract_is_bitwise_batch_invariant`,
`test_sm90_strict_contract_logits_return_and_padding_do_not_change_logp_bits`,
`test_sm90_strict_contract_backward_matches_logp_and_lse_reference`,
`test_sm90_rejects_out_of_range_target`, `test_sm90_falls_back_for_unsupported_inputs`
and other positive and negative paths.

**Attribution:** this is **pre-validation** performed before T01's bit harness exists, meant
to surface actual-kernel problems early. The formal bit-equivalence verdict required by the
WS1 gate still depends on T01's frozen P1 arithmetic spec + harness (`#L173`, `#L700`).
**Note**: the byte-exact M10 contract object is the output of the **same kernel on the same
input** from the training side and the rollout side, not kernel-versus-PyTorch-golden
(reduction kernels use a different reduction order than the golden, so forward cannot be
bitwise equal in the first place).

---

## 3. Canonical Sampling Record

### 3.1 Record structure

```text
canonical_sampling_record:
  record_version:   "p7-sampling-record-v1"
  case_id:          <stable case identity>
  engine:           megatron | miles_vllm | <producer>
  layers:
    requested:      <SamplingLayer>   # passed explicitly by the caller
    normalized:     <SamplingLayer>   # after framework normalization (e.g. top_k=0 -> DISABLED)
    materialized:   <SamplingLayer>   # actual arguments installed on the runtime object/tensor
    runtime_actual: <SamplingLayer>   # values read back at execution time (readback, not config)
  canonical_fingerprint: <hash(normalized four layers)>
  provenance:       <see §6>
  fallback_layers:  <typed tuple of layer names that resolved through a fallback>
```

Each `SamplingLayer` (contract `#L211`):

```text
SamplingLayer:
  temperature:  float
  top_k:        int | "DISABLED"
  top_p:        float | "IDENTITY"
  min_p:        float | "DISABLED"
  source:       config | api | framework_default | <producer>   # where this value came from
  raw:          <framework-raw expression>                      # value before normalization, for provenance
```

### 3.2 Canonical sentinel (fixed values)

The consistency-mode sentinel that every normalized layer must equal (`#L210-L211`):

| Parameter | Canonical value | Allowed framework-raw expressions (normalize before comparing) |
| --- | --- | --- |
| `temperature` | `1.0` | Exactly `1.0`; anything else → mismatch |
| `top_k` | `DISABLED` | `None` / `0` / `-1` → `DISABLED` (no vocab context during normalization, so no `>= vocab` decision) |
| `top_p` | `IDENTITY` | `None` / `1.0` → `IDENTITY` |
| `min_p` | `DISABLED` | `None` / `0.0` → `DISABLED` |

**Normalization rules (normalize before comparing; never compare raw values):**

- Numeric comparison uses canonical semantic equality, not string equality.
- Framework defaults (values not passed explicitly) must be expanded into explicit fields;
  "absent = don't care" is not allowed.
- If a producer expresses `top_k`/`top_p` as a per-row tensor, every row must be the sentinel.
- An absent `temperature` is a structural error (`SamplingParamsError`), never identity.
  `None` is an allowed raw form only for `top_k` / `top_p` / `min_p`.

**Cross-task naming and encoding conventions (must be unified before wiring T05/WS2):**

- **Layer name**: the upstream contract and the merged T05 gate use
  `requested / normalized / materialized / actual` (see `SAMPLING_LAYERS` in
  `rl_engine/alignment/cross_config/adapters/megatron_training_gate.py`). The field name in
  this repository's implementation is `runtime_actual`
  (`CanonicalSamplingRecord.runtime_actual`). Wiring must map `actual <-> runtime_actual`
  explicitly; do not pass the implementation object straight into the T05 gate, which would
  otherwise report `MISSING_PROVENANCE: sampling_record.actual`.
- **Sentinel encoding**: Python uses `None` internally (`TOP_K_DISABLED` / `TOP_P_IDENTITY` /
  `MIN_P_DISABLED`), but the **serialized cross-engine/cross-task form must be the strings
  `DISABLED` / `IDENTITY`** (matching T05's `IDENTITY_SAMPLING`). `to_dict()` does not
  perform that conversion yet (it emits `null`); the adapter work must add it together
  with a test.

### 3.3 Four-layer validation rules

In consistency mode (`#L209`):

1. Any layer with `temperature != 1.0` → `SAMPLING_PARAMS_MISMATCH`.
2. Any layer whose normalized `top_k != DISABLED`, `top_p != IDENTITY` or `min_p != DISABLED`
   → `SAMPLING_PARAMS_MISMATCH`.
3. Semantic drift between the four layers (requested == sentinel but runtime_actual != sentinel,
   or vice versa) → `SAMPLING_PARAMS_MISMATCH` (this is "actual semantic drift").
4. Any missing layer field / unreadable layer → `MISSING_PROVENANCE`.
5. Hitting a framework fallback / undeclared backend branch → `SILENT_FALLBACK`. Fallbacks are
   read from the typed `fallback_layers` field or from a layer whose `source` is `fallback`;
   the legacy `provenance["fallback_layers"]` map is still accepted for producers that do
   not set either.

**Key constraint:** a `configured value` must not masquerade as an `actual readback`
(DoD `#L743`). `runtime_actual` must come from execution-side readback and must not reuse
`materialized` directly.

**Implementation diagnostic priority** (fail-closed): missing layer → `MISSING_PROVENANCE`;
fallback hit → `SILENT_FALLBACK`; per-layer identity mismatch (including requested-versus-
runtime_actual drift) → `SAMPLING_PARAMS_MISMATCH`. Drift needs no separate path — any
non-identity layer is caught by the per-layer check, and `failing_layer` names the earliest
offending layer.

---

## 4. M10 input schema

Instantiated from contract `#L43-L61`. Where marked `[pending T01]`, dtype/shape follow the
frozen T01 manifest.

| Field | Type / shape | Required | Notes |
| --- | --- | --- | --- |
| `case/run/engine identity` | immutable/versioned | yes | Bound to the case_id |
| `final_decoder_rows` | BF16 `[N, H]` | yes | The only permitted logits source for M10 |
| `target_token_ids` | INT64 `[N]` | yes | Selected target |
| `active_token_mask` | BOOL `[N]` | yes | Only active action tokens are compared |
| `lm_head_weight_local` | BF16 `[V_local, H]` | yes | Local vocab shard |
| `token/absolute layer` | canonical global identity | yes | |
| `position/cache identity` | versioned state/page identity | yes | |
| `checkpoint/weight_id` | immutable identity + fingerprint | yes | |
| `failure_batch_context` | optional filter | no | run/checkpoint/sample/rank/module |
| `pre_update_weight_lock` | immutable fingerprint | yes | Drift invalidates the case |
| `vocab ownership` | `[vocab_start, vocab_end)` | yes | No gaps, overlaps or ambiguity |
| `real_vocab_size` | int, optional | no (defaults to the padded width) | Real tokens; every target must be `< real_vocab_size` |
| `rank/group/order` | versioned metadata | yes with TP | Required by ordered rank merge |
| `sampling requested/actual` | canonical record | yes | See §3 |
| `Foundation ABI versions` | five ABI version numbers | yes | SemanticTensor/RoutePlan/ExchangePlan/ExpertBatch/CombinePlan |
| `boundary trace` | versioned TraceEnvelope | yes | recorded or live |

### 4.1 M10 pre-checks (before computing logprob)

- **schema version**: `schema_version` must equal `p7-m10-input-v1`; a foreign version →
  `INVALID_PROFILE` before any field is interpreted.
- **vocab ownership check**: `[vocab_start, vocab_end)` covers `[0, global_vocab)` with no
  gaps or overlaps (`#L29`, `#L719`). Otherwise → `INVALID_PROFILE` or `MISSING_PROVENANCE`
  (per the T01 manifest).
- **single-device full vocabulary**: strict single-device scoring requires
  `vocab_start == 0` and `vocab_end == global_vocab_size`. A vocabulary shard needs a TP
  group, so shards are rejected here and belong to WS2.
- **selected target ownership**: every active target is owned by exactly one rank with an
  owner count of 1 (`_validate_global_targets` / `owner_count`, `#L215`).
- **padded vocab mask**: every target row (active or not) must be `< real_vocab_size`; a
  target in the padded region (`real_vocab_size <= target < global_vocab_size`) →
  fail-closed. The strict wrapper validates all rows, not only the active ones (`#L215`).
- **stop numerical attribution when the identity/profile gate fails** (`#L685`).

---

## 5. M10 output and numerical contract

### 5.1 Output fields (T02 subset of contract `#L65-L79`)

| Field | Notes |
| --- | --- |
| `selected_logprob` | FP32 `[N]`, `z[t] - logsumexp(z)` |
| `selected_logprob_bytes` | Raw bytes of the active tokens (the sole basis of the strict verdict) |
| `selected_target_summary` | Target value + global max/sum/LSE + ordered merge provenance |
| `actual_provenance` | backend/kernel/build/dtype/weight/shard/reduction/collective |
| `identity_discrete_gate` | Exact gate verdict + failing field |
| `first_mismatch` | On failure, keyed by `(absolute_layer, sublayer_site, pass, event_index)` + canonical token/rank |
| `result_cube` | T02 fills only the case/engine/level dimensions; boundary/config dimensions belong to T10/WS2 |

### 5.2 Numerical path: reuse the existing `linear_logp`

| Contract point | Real symbol / location |
| --- | --- |
| Unified non-TP / TP entry point | `tensor_parallel_linear_logp(hidden, lm_head_weight, target_ids, bias=None, *, tp_group, vocab_start_index=0, global_vocab_size=None)` |
| Non-TP single-device op | `NativeLinearLogpOp` (CPU/Triton-less fallback + oracle) |
| Local statistics (max/sum/target logit) | `_chunked_local_linear_logp_stats(...)` |
| Ordered rank merge | `_merge_tp_local_logp(local_lse, local_target_logit, *, tp_group)`; CUDA FP32 goes through `ordered_logp_merge` |
| Target ownership declaration | `_validate_global_targets` / `_assert_global_targets_async` |
| Vocab shard validation | `_validate_tp_vocab_partition(_cached)` |
| `BN=64` constant | `SM90_N_SPLIT_CONTRACT == 64` (i.e. the contract's `BN=64`) |
| Numerical contract version | `SM90_LINEAR_LOGP_CONTRACT_VERSION = "cuda-fused-linear-logp-sm90-contract-v1"` |

**Hard constraints:**

- The existing implementations above must be called; **falling back to any undeclared scorer
  is forbidden** (`#L216`).
- Consistency mode **does not replay sampling transforms** (`#L33`).
- The global max / sum / LSE and the rank merge order must be guaranteed by `linear_logp`
  itself; T02 only **validates** the shape and provenance of the ordered merge and does not
  write another reduction (`#L29`, `#L215`).
- The M10 numerical bit contract is determined by P1's fixed-K/batch-invariant GEMM arithmetic
  spec; T02 verifies the actual kernel with T01's bit harness, and a failure must be attributed
  to an **actual kernel / spec mismatch**, never to a replacement kernel newly written in P7
  (`#L214`).

### 5.3 LM Head LoRA rejection path

- When LM Head LoRA is detected (on the LM Head weight path) → strict mode returns
  `UNSUPPORTED_CAPABILITY` and **must not silently ignore it** (`#L31`, `#L217`).
- Keep the distinction: P5 expert LoRA is **not in the rejection scope**; it is part of the
  upstream model identity and must be recorded and validated (adapter/target/rank/scale,
  `#L32`). Implementations must not conflate the two.

---

## 6. Provenance fields (a deliverable)

`actual_provenance` covers at least (contract `#L69`, `#L35`):

```text
backend           actual execution backend (cuda/rocm/ascend/triton/cpu)
kernel            actual kernel / implementation symbol name
kernel_contract   e.g. SM90_LINEAR_LOGP_CONTRACT_VERSION
build             extension / library build identity
dtype             hidden / weight / accumulator / output
weight_id         weight fingerprint (including pre_update_weight_lock)
vocab_shard       [vocab_start, vocab_end) + world_size
reduction         BN constant + rank merge order
collective        TP group / ordered merge collective (when applicable)
fallback_used     whether a fallback / undeclared backend was hit
```

Rule: missing data or a silent fallback **must not be judged as a pass** (`#L35`, `#L683`).

---

## 7. Gate decision flow and stub design

### 7.1 Decision order (fail-fast, before computation/comparison)

```text
1. identity / discrete profile gate        -> INVALID_PROFILE / IDENTITY_DRIFT
2. sampling params gate (§3.3)             -> SAMPLING_PARAMS_MISMATCH / MISSING_PROVENANCE / SILENT_FALLBACK
3. LM Head LoRA detect                     -> UNSUPPORTED_CAPABILITY
4. M10 pre-checks (§4.1)                   -> MISSING_PROVENANCE / INVALID_PROFILE
5. Call the existing linear_logp to compute selected_logprob
6. Emit selected_logprob_bytes for the T09 byte verdict; diagnostics do not enter the verdict
```

Any failing step returns immediately and hands off to T09/T10/T11 without downstream
attribution (`#L683-L687`). Non-finite / zero-active fails closed immediately and never reaches
the tolerance comparator (`#L459`, `#L689`).

### 7.2 Stub / synthetic fixture (allowed by contract `#L601`, `#L641-L643`)

Until the T01 start kit exists, freeze this task's interface and tests with a self-built stub:

| Stub component | Content | Constraint |
| --- | --- | --- |
| synthetic fixture | Deterministic (fixed seed) generation of every §4 field (`final_decoder_rows` BF16 `[N,H]`, `target_token_ids` INT64 `[N]`, `active_token_mask`, `lm_head_weight_local` BF16 `[V_local,H]`, vocab ownership, canonical sampling record) | Must be synthetic/desensitized; real checkpoints or tokens are forbidden (`#L769`) |
| stub provider | Implements the interface only and returns the fixture; no real inference/training | Does not replace the numerical implementation |
| CPU oracle | Reuses `NativeLinearLogpOp` as the correctness oracle | No new oracle |
| stub sampling producer | Produces the four requested/normalized/materialized/runtime_actual layers, including negative samples | runtime_actual must have "readback" semantics and must not merely copy materialized |

**Critical: freeze the live-replacement ABI.** The stub and the real provider must share the
same schema/signature; after replacement, **the case ID, ABI, comparison policy, runner,
artifact and acceptance command stay unchanged** (`#L641-L643`). The case ID generation rule
must be pinned at stub time, otherwise WS2/Integration cannot reuse it.

**Stub boundary:** feed data and fill interfaces only; it is **not allowed** to write its own
`linear_logp` or sampling scorer just to make things pass.

---

## 8. Test plan

### 8.1 Positive

- A canonical sentinel record (`temperature=1.0 / top_k=DISABLED / top_p=IDENTITY /
  min_p=DISABLED`) with all four layers equal → PASS.
- Non-identity raw expressions that normalize to the sentinel (`top_k=0`, `top_p=None`,
  `min_p=0.0`) → PASS.
- M10: call the existing `linear_logp` on both the single-device and TP paths and compare
  against the CPU oracle; verify the actual kernel with T01's bit harness.
- `selected_logprob_bytes` is stable (repeated runs of the same case are byte-identical).

### 8.2 Negative (all fail-closed, `#L219`)

| Scenario | Expected status |
| --- | --- |
| `temperature != 1.0` | `SAMPLING_PARAMS_MISMATCH` |
| `top_k` enabled (not DISABLED) | `SAMPLING_PARAMS_MISMATCH` |
| `top_p != IDENTITY` | `SAMPLING_PARAMS_MISMATCH` |
| `min_p` enabled | `SAMPLING_PARAMS_MISMATCH` |
| requested=identity but runtime_actual non-identity (semantic drift) | `SAMPLING_PARAMS_MISMATCH` |
| Any missing layer field / unreadable layer | `MISSING_PROVENANCE` |
| Fallback hit / undeclared backend | `SILENT_FALLBACK` |
| LM Head LoRA present | `UNSUPPORTED_CAPABILITY` |
| Target lands in the padded vocab region | fail-closed (per the manifest) |
| vocab ownership gap/overlap/ambiguity | `INVALID_PROFILE` / `MISSING_PROVENANCE` |
| pre_update_weight_lock drift | `PRE_UPDATE_WEIGHT_DRIFT` |
| non-finite / zero-active | `NON_FINITE` / `ZERO_ACTIVE_TOKENS` |

### 8.3 WS2 continuation (implementation left to WS2; fields frozen by this spec)

- Every rank's canonical sampling record is complete and consistent (`#L218`).
- `actual_provenance` is present for every expected rank; a missing rank → `MISSING_RANK`.
- The same global token maps uniquely across parallel configurations; ambiguity →
  `AMBIGUOUS_GLOBAL_TOKEN_MAPPING`.

---

## 9. Open items / pending T01 confirmation (possible blockers)

1. **T01 frozen code baseline commit**: currently `upstream/test-dsv4-cuda @ 1dd053f`
   (includes the P1–P7 unified line, specifically `rl_engine/validation` and
   `rl_engine/backends`); if T01 pins a specific commit, another rebase is required.
2. **Formal dtype/shape/layout of the M10 input/output schema** (this spec infers them from
   contract `#L43-L80`).
3. **M10 row of the boundary-contract manifest**: `comparison_mode`,
   `tolerance_contract_version`, `fixture_checksum`.
4. **canonical hash version**: used by `boundary_hash_index`.
5. **P1 GEMM arithmetic spec + bit harness**: verifying the actual `linear_logp` kernel depends
   on it (`#L173`).
6. **Whether T01's provider stub / synthetic fixture / CPU oracle already have reusable
   implementations.**

If any of the above remains undecided and cannot be covered by the self-built stub in §7,
confirm with vensen.

---

## 10. Deliverables and acceptance (aligned with the contract)

**Deliverables** (`#L221`): gate, parameter normalization logic, provenance fields, positive
and negative tests.

**Acceptance** (`#L223`): any non-identity sampling setting, actual semantic drift, LM Head LoRA
or undeclared backend fails explicitly **before computation/comparison**; no silent
continuation.

**WS1 gate item** (`#L701`): the T02 sampling/M10 contract passes all positive and negative
tests.

**Shared DoD** (`#L739-L750`): configured must not masquerade as actual readback; schemas,
hashes and artifact payloads carry versions; reuse the designated infrastructure; the Owner
provides runnable commands plus result summaries; the Reviewer (T09) approves.

---

## 11. Quickstart

### 11.1 Environment setup

```bash
# Build the native extension once on an SM90 machine (includes strict linear_logp)
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export KERNEL_ALIGN_FORCE_SM90=1
export MAX_JOBS=16
python -m pip install -e . --no-build-isolation

# On this AutoDL host, reach github.com through the academic proxy first
source /etc/network_turbo
```

### 11.2 Minimal runnable example

```python
import torch

from rl_engine.validation.m10 import build_m10_stub, evaluate_m10_precheck, score_m10, M10Status
from rl_engine.validation.sampling_params import (
    CanonicalSamplingRecord,
    SamplingGateStatus,
    evaluate_sampling_gate,
    normalize_sampling_layer,
)

# 1) Sampling-parameter normalization + gate (pure CPU, no GPU dependency)
layer = normalize_sampling_layer(temperature=1.0, top_k=0, top_p=None, min_p=0.0)
record = CanonicalSamplingRecord(
    case_id="case-0", engine="stub",
    requested=layer, normalized=layer, materialized=layer, runtime_actual=layer,
)
assert evaluate_sampling_gate(record).status is SamplingGateStatus.PASS

# 2) M10 pre-checks (pure CPU)
inp = build_m10_stub(tokens=8, hidden=64, local_vocab=128)
assert evaluate_m10_precheck(inp).status is M10Status.PASS

# 3) M10 scoring (reuses strict linear_logp; requires the SM90 extension + CUDA)
score = score_m10(inp, device=torch.device("cuda"))
assert score.passed
raw = score.selected_logprob_bytes   # FP32 raw bytes of the active tokens, handed to T09 for the strict verdict
provenance = dict(score.provenance)  # actual backend / frozen contract / sampling fingerprint
```

**Swapping stub → real provider**: `StubM10Provider` freezes the
`read(case_id) -> M10Input` ABI; once T05/T06 land, only the implementation changes — the
case id, schema, runner and acceptance command stay the same (`#L641-L643`).

### 11.3 Tests

```bash
python -m pytest tests/validation/sampling_params tests/validation/m10 -q      # 64 passed
python -m pytest tests/ops/logprob/test_linear_logp.py -k sm90 -v               # 24 passed (H20/SM90)
```

### 11.4 fail-closed status reference

| Scenario | Status |
| --- | --- |
| Any missing layer / unreadable layer | `MISSING_PROVENANCE` |
| Fallback hit / undeclared backend | `SILENT_FALLBACK` |
| `temperature != 1.0`, non-identity `top_k`/`top_p`/`min_p`, four-layer drift | `SAMPLING_PARAMS_MISMATCH` |
| LM Head LoRA | `UNSUPPORTED_CAPABILITY` |
| Invalid shape/dtype/vocab ownership/padded target | `INVALID_PROFILE` |
| pre-update weight lock drift | `PRE_UPDATE_WEIGHT_DRIFT` |
| No active token | `ZERO_ACTIVE_TOKENS` |
| Non-finite value | `NON_FINITE` |

On failure `selected_logprob_bytes` is empty and no kernel is launched (fail-closed precedes
computation).
