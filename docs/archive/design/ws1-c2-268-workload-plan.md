# WS1 C2 (#268) Landing Plan — Canonical Workload & Logical Identity

**Parent:** #266 · **Issue:** #268 · **Depends on:** C1 (#267) contract roles only

**Branch:** `feat/ws1-c2-canonical-workload-268` (from `feat/ws1-c1-tolerance-contract-267@81ddd65`)

**Does not modify:** C1 branch tip

---

## 1. Goal (one sentence)

Freeze a **reproducible full-Qwen3-8B-Dense logical workload** (identity + fixtures + 2×2 Batch/Chunk matrix + backend profile map + representative `case_id`s) so later C3–C11 gates compare the **same** sample/token multiset after pad/pack/chunk transforms.

C2 does **not** implement #150 numerical asserts, full model forward, or multi-GPU.

---

## 2. Context from parent tree

| Item | Lock from #266 / #268 |
| --- | --- |
| Model | Full official **Qwen3-8B Dense** (no layer/hidden/head/vocab shrink) |
| Architecture pin | Config fingerprint + weight snapshot identity |
| Fixture scaling allowed | Seq length / padding / batch layout only |
| Primary matrix | `B1-singleton_aggregate/full`, `BN/full`, `B1-singleton_aggregate/chunked`, `BN/chunked` |
| Logical identity | `(sample_id, token_position)` recoverable after pad/pack/chunk |
| Gradient B1 vs BN | `singleton_aggregate` = N× B=1 of **same** N samples, fixed order + active-token denom |
| Naming | `singleton_aggregate` is **execution mode only** — never a C1 `comparison_*_role` |
| Backends | `cuda_bf16` + `triton_cuda_bf16`; every required chain node has expected candidate/path |
| Clip | `clip_interval` for `clipfrac0` co-located with aggregate pins (align C1 default `[0.8, 1.2]`) |
| Stochastic | Gate uses `dropout=0`; sampling out of logprob parity; undeclared RNG hard-fails |

### Official Qwen3-8B Dense fingerprint (source: HF `Qwen/Qwen3-8B` config)

| Field | Value |
| --- | --- |
| `model_id` | `Qwen/Qwen3-8B` |
| `num_hidden_layers` | 36 |
| `hidden_size` | 4096 |
| `intermediate_size` | 12288 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | 8 (GQA) |
| `head_dim` | 128 |
| `vocab_size` | 151936 |
| `rope_theta` | 1e6 |
| `rms_norm_eps` | 1e-6 |
| `hidden_act` | silu (SwiGLU MLP) |
| `tie_word_embeddings` | **false** |
| `attention_dropout` | 0.0 |
| QK-Norm | **enabled** in Qwen3 architecture (per-head q/k RMSNorm; not a config flag) |
| Config revision (pinned) | HF `x-repo-commit` at plan time: `b968826d9c46dd6066d109eabc6255188de91218` |

Weight snapshot: pin revision + SHA-256 of `model.safetensors.index.json` + all five
official LFS shard content SHA-256/size records.  The manifest also stores a reproducible
`sha256-of-sorted-shard-records-v1` aggregate, tensor payload bytes, and physical shard
bytes.  This is a full content-addressed weight identity without downloading 16 GB locally.

---

## 3. Deliverables (issue docking)

| Path | Role |
| --- | --- |
| `rl_engine/config/workloads/qwen3_8b.json` | SSOT: model identity, matrix, fixtures, backends, cases, clip/RNG policy |
| `rl_engine/config/workload.py` | Load/validate manifest; build logical samples; pad/pack/chunk; restore identity; fixture hash |
| `tools/validation/models/ws1_reference.py` | One command: emit workload ID + seed + dtype + fixture/reference identity payload |
| `tests/validation/operators/test_ws1_workload.py` | Schema + identity + matrix + naming + backend completeness |
| `docs/archive/design/ws1-c2-268-workload-plan.md` | This plan (closeout evidence pointer) |

Reuse, do not fork:

- C1 roles: `comparison_lhs_role` / `comparison_rhs_role` from `tolerance_contract.json` — **never** put `singleton_aggregate` or bare `baseline` there.
- Op defaults: `operator_inputs.py` dims must match manifest fingerprint.
- Candidate paths: `operator_specs.py` `candidate_paths` as the path vocabulary for profile maps.

---

## 4. Manifest schema (normative sections)

```text
version / workload_id / seed
model_identity
  model_id, revision, config_fingerprint{}, weight_snapshot{}, architecture_notes
chain_semantics
  execution_dtype, reference_dtype, temperature, loss_reduction,
  logprob_selection, clip_interval, aggregates[]
stochastic_policy
  dropout, sampling_in_logprob_parity, rng_source, undeclared_randomness
primary_matrix
  N, cells[{cell_id, batch_mode, prefill_mode, ...}]
  batch_permutation, chunk{size, require_ge_2_chunks, non_divisible_case}
fixtures
  samples[], short/long/varlen, left/right pad, packing status
logical_identity
  key=(sample_id, token_position), restore_after[]
capabilities
  packing, qk_norm, required_chain_ops[{op, status}]
backend_profiles
  cuda_bf16 / triton_cuda_bf16 → required_nodes[{node, expected_backend_id, expected_kernel_config_id, algorithm_property}]
representative_cases[]
  case_id, family(gemm|attention|logprob), shape pins, backend pins, algorithm property
```

### Primary matrix cells (fixed IDs)

| cell_id | batch_mode | prefill_mode |
| --- | --- | --- |
| `B1-singleton_aggregate/full` | B=1 × N runs → aggregate | full prefill |
| `BN/full` | B=N single run | full prefill |
| `B1-singleton_aggregate/chunked` | B=1 × N → aggregate | chunked prefill |
| `BN/chunked` | B=N | chunked prefill |

Fixed: `N=4` ( >1 ), target sample order fixed, at least one chunk size that yields ≥2 chunks and a non-divisible remainder case.

### Backend profiles

Enumerate every on-chain required node for full-model topology (#266 §5):

`embedding`, `rms_norm`, `det_gemm` (Q/K/V/O/gate/up/down), `qk_norm` (elementwise/RMS), `rope`, `attention`, `swiglu`/`silu`, `lm_head`, `logprob`/`batch_invariant_logp`/`linear_logp` as declared.

For each profile:

- Expected `backend_id` + `kernel_config_id` (or path id from `operator_specs`).
- Missing Triton candidate for a **required** node → status `red` / `missing_required` (not N/A, not silent fallback).

### Representative cases (stable `case_id`)

1–3 per family, full-model graph/weights identity, seq may be short:

| Family | Property exercised |
| --- | --- |
| GEMM | Multiple flattened-token `M`, incl. non-tile-aligned; no-Split-K path |
| Attention | Prefill/decode, GQA 32/8/128, multi KV len + non-tile-aligned; no-Split-KV |
| Logprob | Vocab/reduction crossing at least one declared block boundary |

Changing any pinned field → new `case_id` / revision.

---

## 5. Workload API (Python)

```text
load_manifest() / validate_manifest()
build_logical_batch(manifest=None, *, cell_id=None, sample_ids=None) -> LogicalBatch
  samples: list[LogicalSample]  # sample_id, token_ids, positions, loss_mask, ...
apply_padding(batch) / apply_chunking(batch) / apply_packing(batch) -> physical layout
restore_logical_order(physical, values) -> aligned values keyed by (sample_id, token_position)
singleton_aggregate_plan(N samples) -> execution schedule for B1×N vs BN
fixture_hash(batch|manifest) -> stable hex
matrix_cell_ids() / get_matrix_cell(manifest, cell_id)
profile_required_nodes(profile_id)
get_case(case_id)
```

Rules:

- After pad/pack/chunk, compare **only** after `restore_logical_order`.
- B1 `singleton_aggregate` and BN share the **same** logical sample/token multiset and fixed aggregation order + active-token denominator.
- Hard-fail on undeclared stochastic sources when building gate fixtures.

---

## 6. Reference command

```bash
python tools/validation/models/ws1_reference.py \
  --workload-id <id> \
  --seed <seed> \
  --dtype bf16 \
  [--cell-id BN/full] \
  [--emit-json path|-]
```

Emits: workload_id, seed, dtype, fixture_hash, model identity pins, cell descriptor,
clip_interval, profile ids, and deterministic logical/pad/chunk/pack/short/long tensor digests.
Does **not** run full 8B forward (owned by C9/C10); may emit tensor fixture digests for token/mask tensors only.

---

## 7. Test plan (`tests/validation/operators/test_ws1_workload.py`)

| Test group | Asserts |
| --- | --- |
| Schema | Every numerics-affecting field present; no forbidden comparison roles |
| Model identity | Full fingerprint; no shrink fields; weight pin present |
| Repro | Same workload_id → same fixture_hash / sample multiset |
| Logical identity | pad / chunk / (pack if supported) restore `(sample_id, token_position)` |
| Aggregate | B1 singleton plan multiset == BN multiset; fixed order |
| Naming | `singleton_aggregate` not in C1 role sets; no bare `baseline` in report fields |
| Matrix | 2×2 cells fixed; N>1; perm; multi-chunk non-divisible |
| Clip / RNG | clip_interval pinned; dropout=0; undeclared RNG rejected |
| Profiles | Both profiles list all required nodes; missing Triton required → red |
| Cases | Stable case_id; expected+actual path fields schema; algorithm property present |
| CLI | `ws1_reference.py` exits 0 and prints workload_id/seed/dtype/hash |

CPU-only; no GPU / no weight download required for C2 unit tests.

---

## 8. Implementation order

1. **Manifest JSON** with full pins (model, matrix, fixtures, profiles, cases).
2. **`ws1_workload.py`** loader + validators + logical batch + pad/chunk restore + hash.
3. **`tools/validation/models/ws1_reference.py`** thin CLI.
4. **Tests** green on CPU.
5. Wire exports in `rl_engine/validation/common/__init__.py` (minimal public surface).
6. Short evidence comment map on PR / issue #268 (acceptance checklist).

---

## 9. Explicit non-goals (stay out)

| Out | Owner |
| --- | --- |
| Four-judgment numerical asserts / #150 matrix green | C10 |
| Forward harness + backend provenance runtime | C3 |
| Gradient harness | C4 |
| Full model assembly / real 8B run | C9 |
| Stateful KV / generate-rescore | C6/C7 |
| CI gate jobs | C11 |
| Multi-GPU | WS2 |

---

## 10. Acceptance ↔ evidence map

| #268 AC | Evidence |
| --- | --- |
| Manifest pins numerics fields | `ws1_manifest.json` + schema tests |
| Full Qwen3-8B Dense identity | `model_identity` section |
| Same workload_id → same identity | `fixture_hash` tests |
| Transforms restore logical identity | pad/chunk restore tests |
| B1 singleton vs BN same multiset | aggregate plan tests |
| Naming boundary vs C1 roles | forbidden-role tests + contract cross-check |
| 2×2 + perm + multi-chunk | matrix section + tests |
| clip_interval pinned | manifest + tests |
| Dropout/RNG policy | stochastic_policy + hard-fail test |
| Short + rep fixtures hit candidates | representative_cases + profile map |
| Stable case_id | cases + tests |
| expected backend/kernel pins | cases + profiles |
| One reference command | `tools/validation/models/ws1_reference.py` |
| Packing / QK-Norm / ops status | capabilities |
| Both profiles enumerate nodes | backend_profiles tests |

---

## 11. Risk notes

1. **Weight identity without multi-GB download:** pin HF revision, index SHA-256, every
   shard's official LFS content SHA-256/size, and a reproducible aggregate digest.
2. **Triton gaps:** declare `missing_required` honestly for nodes without Triton candidates (e.g. some embedding/lm_head paths) — C2 records red status; does not invent fallbacks.
3. **Packing:** because `NativePackOp` exists, C2 marks packing supported, freezes the
   variable-length packed fixture, and round-trips its logical identity even though packing
   is outside the primary 2×2 matrix.
4. **C1 alignment:** re-export clip_interval default from C1; dual-write in manifest so C2 is self-contained for gates.

---

## 12. Done definition for this branch

- [x] Docking files land on `feat/ws1-c2-canonical-workload-268` without rewriting C1 tip history.
- [x] `pytest tests/validation/operators/test_ws1_workload.py -q` green — 33 passed (CPU).
- [x] `python tools/validation/models/ws1_reference.py` emits workload_id / seed / dtype / fixture digests.
- [x] Closeout evidence: `docs/archive/design/ws1-c2-268-closeout-evidence.md`.
- [x] #268 residual closeout: per-sample `completion_lens`, TF32 ref, report naming, fixture-derived representative shapes, and executable CUDA/Triton actual provenance.
- [x] Explicit non-claim: does not close #266 or turn Triton missing_required green.
