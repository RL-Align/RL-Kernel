# P7 T05: Megatron Training Provider and Readback

Status: **P1 in progress** (P0 interfaces and observation skeleton; P1 fail-closed gate). Draft
schemas carry a `-draft` version suffix until the P7 Foundation owner (T01)
publishes the public schemas.

T05 binds the Megatron training side of the M10 selected-token logprob to
evidence: which inputs the existing strict `linear_logp` route actually
received, which code path actually ran, and which pre-update weights and
failure-batch slice the result belongs to. It reuses the existing route
(`LinearCrossEntropyModule` output patch → VIME request → provider →
`LinearLogpWrapper`) and never adds a second LM-head hook, re-runs a kernel, or
changes the training path.

## Evidence sources

Every captured field is tagged with where its value came from. Only
`observed` and `upstream` evidence may certify a strict result; `declared` and
`configured` values are kept for cross-checks and must never stand in for
observed values.

| Source | Meaning | Example |
|---|---|---|
| `observed` | Value actually passed to or returned by this call | `request.context.hidden` raw-byte digest |
| `declared` | Self-reported by code | wrapper `arithmetic_contract`, operator `fallback` flag |
| `configured` | Caller-supplied configuration | VIME `request.metadata["tp_rank"]` |
| `upstream` | Producer (P1–P6) evidence forwarded verbatim | `TraceEnvelope` verdicts |

## Field sources (VIME `LinearLogpRequest`, tree `f1e20b7`)

| Field | Source | Notes |
|---|---|---|
| `logits` | observed | Local logits `[T, V_local]`; input of the reuse path |
| `target_ids` | observed | `[T]`; target shift is applied upstream by VIME |
| `log_prob_keep_mask` | observed | Non-`None` means top-p replay; rejected in consistency mode (P1) |
| `temperature` | observed | Value passed in; the value given to the kernel is cross-checked in P1 |
| `with_entropy`, `with_entropy_grad`, `chunk_size` | observed | |
| `token_layout.{world_size,rank,layout}` | observed | Row → global-token mapping is derived in P1 with the existing CP helpers |
| `context.hidden` | observed | Final decoder rows, BF16 `[N, H]` |
| `context.projection.{weight,bias}` | observed | LM-head local shard; checked against the weight lock |
| `context.vocab_partition.*` | observed | Ownership `[local_start, local_start + local_size)`, real/padded vocab |
| `tensor_parallel_group` coordinates | observed | Read from `torch.distributed` for the group passed in |
| `metadata` | configured | Caller-declared TP coordinates; cross-checked, never certifying |
| `result.logp`, `result.entropy` | observed | |
| `result.{backend_id,contract_id,provenance}` | declared | |
| `operator.provenance` | declared | Contains hard-coded `fallback`/`logits_materialized`; not trusted |
| `linear_logp.provenance` | declared | `strict_entrypoint`, `arithmetic_contract` constants |

Not present in the request and therefore not observable at this boundary
(P1/P2 must source them or report them missing): active-token mask,
position/cache identity, checkpoint/weight identity, global token ids,
four-layer sampling record, executed GEMM spec/harness fingerprint, global
max/sum, and the actual canonical virtual-TP split.

## Read-only observation

```python
from rl_engine.integrations.megatron_training_readback import TrainingLogpObserver

observer = TrainingLogpObserver(digest_tensors=True)
observer.install()        # wraps MegatronLogpOperator.__call__ at class level
...                       # training runs unchanged
snapshots = observer.snapshots   # one immutable TrainingCallSnapshot per call
observer.uninstall()      # restores the exact original function
```

- Evidence is frozen when each call returns; later mutation of the request,
  the operator or its provenance does not change a snapshot.
- Tensors are recorded as identities (shape, stride, dtype, device, raw-byte
  SHA-256), never by value; `+0` and `-0` hash differently.
- A capture failure is recorded on the snapshot and never propagates into
  the training call; operator exceptions are recorded and re-raised.
- `digest_tensors=True` copies tensors to the host for hashing, which adds
  synchronization and host traffic; enable it only for evidence runs.
- Install is idempotent; a second observer on the same class is refused.

## Fail-closed gate (P1)

`rl_engine/alignment/cross_config/adapters/megatron_training_gate.py` validates
one frozen snapshot plus a `TrainingCallContext` (row layout, active mask,
checkpoint/weight identity, weight lock, LM-head LoRA state, four-layer
sampling record), all of which the VIME request does not carry. It never
reads live objects and never recomputes numerics. Order, identity first:

1. Snapshot complete (no capture/operator error, selected logprob captured).
2. Input schema: BF16 `[N, H]` hidden, BF16 `[V_local, H]` shard, INT64 `[N]`
   targets, logits `[N, V_local]` → `SCHEMA_MISMATCH`.
3. Row identity: each local row is the logits row `(sample_id, position)`;
   its target is the token at `position + 1` (VIME owns the shift). `single`,
   `zigzag` (via `canonical_cp.packed_gather_indices`) and `allgather` (VIME's
   contiguous per-rank split of the concatenated samples) are supported;
   padding rows are `None`; other layouts → `UNSUPPORTED_CAPABILITY`.
4. TP coordinates from the actual group; configured metadata that disagrees →
   `IDENTITY_DRIFT`; vocab ownership must equal the rank's equal shard.
5. Weight lock on the observed shard digest → `PRE_UPDATE_WEIGHT_DRIFT`; an
   LM-head bias (not covered by the lock) → `UNSUPPORTED_CAPABILITY`.
6. Executed path from `strict_entrypoint` cross-checked with the provider's
   `execution` block; provenance shapes must match this call's observed
   inputs (stale evidence → `MISSING_PROVENANCE`); non-strict backend or any
   fallback flag → `SILENT_FALLBACK`.
7. Sampling: keep mask, sparse nucleus or non-identity temperature →
   `SAMPLING_PARAMS_MISMATCH`; the four-layer record must exist and its
   normalized/materialized/actual layers must be identity.
8. LM-head LoRA must have been inspected (`None` → `MISSING_PROVENANCE`) and be
   absent (`True` → `UNSUPPORTED_CAPABILITY`).
9. Active mask BOOL `[N]`, padding rows inactive, at least one active token,
   FP32 output, all active values finite.

The result, `TrainingM10Evidence`, carries the FP32 `[N]` selected logprob,
row keys, and the active tokens' raw FP32 bytes and their SHA-256 for the
byte comparator (T09). `certifying_sources` lists only observed/upstream
fields.

## Execution evidence and the missing-evidence ledger (P2a)

Each validated call also carries:

- `build`: wrapper and kernel module source SHA-256 and kernel constants
  (`declared`), the loaded `rl_engine._C` binary SHA-256 and the running torch
  build (`observed`). `SM90_N_SPLIT_CONTRACT` is a vocab split count, not the
  bit-relevant BN tile width.
- `runtime`: RNG digests before and after the call (CPU, current CUDA device
  if already initialized, Megatron RNG tracker states) and whether the call
  consumed RNG; the current CUDA stream. Recorded, not gated.
- `target_owner_ranks`: the owner TP rank of every active target, derived from
  the observed target values and vocab partition; an active target in the
  padded tail is `IDENTITY_DRIFT`.
- `canonical_configured`: the canonical virtual-TP layout parsed from
  `RL_KERNEL_STRICT_CANONICAL_TP` / `_VOCAB_SIZE` with the kernel's rules.
  Configuration only, never evidence of the executed split; illegal values are
  `IDENTITY_DRIFT`.
- `missing_evidence`: what the contract requires but this boundary cannot
  observe. Reporting it correctly does **not** complete the item.

| Field | Status | Required interface | Depends on |
|---|---|---|---|
| `gemm.executed_spec` | `UPSTREAM_EVIDENCE_MISSING` | Executed LM-head GEMM spec (accumulator, K order, tile, split-K, cast point) bound to the fixed-K harness | T01 + strict kernel owner |
| `gemm.harness_fingerprint` | `UPSTREAM_EVIDENCE_MISSING` | Fingerprint of the T01 fixed-K/batch-invariant harness run on this build | T01 |
| `reduction.lse` | `MISSING_PROVENANCE` | This call's LSE in the provider result | VIME provider (additive delta, T02 review) |
| `reduction.global_max_sum` | `MISSING_PROVENANCE` | Per-row global max and sum (merge combines per-rank LSE only) | Strict kernel owner + T02 |
| `reduction.padded_vocab_mask_applied` | `MISSING_PROVENANCE` | Kernel readback of the real-vocab bound it masked | Strict kernel owner + T02 |
| `reduction.canonical_split_actual` (TP) | `MISSING_PROVENANCE` | Executed canonical summary intervals / layout match | Strict kernel owner |
| `reduction.redistribution_collective` (TP) | `MISSING_PROVENANCE` | Whether and how logits were all-gathered for redistribution | Strict kernel owner |
| `reduction.merge_order_actual` (TP) | `MISSING_PROVENANCE` | Rank order actually used to merge summaries | Strict kernel owner |

## Upstream correlation (P2b)

`rl_engine/alignment/cross_config/adapters/megatron_training_upstream.py`
binds exactly one recorded P1–P6 `TraceEnvelope` payload to each declared
`UpstreamRequirement` (producer, upstream contract, boundary, pinned schema
version, absolute layer, pass, sublayer site) for an
`UpstreamCorrelationKey` (case, run, rank, the call's active global tokens).
Payloads are forwarded verbatim (deep copy plus digest) and ordered by the
producer's runtime order `(absolute_layer, sublayer_site, pass, event_index)`,
never by M number. T05 does not re-judge producer verdicts.

| Condition | Status |
|---|---|
| No envelope for a required boundary, or a required field absent | `UPSTREAM_EVIDENCE_MISSING` |
| No producer verdict | `UPSTREAM_VERDICT_MISSING` |
| Schema version differs from the pin, unknown comparison mode, or two envelopes claim one boundary | `UPSTREAM_CONTRACT_MISMATCH` |
| Wrong run or case | `STALE_RUN_METADATA` |
| Wrong rank, or the call's active tokens not covered | `IDENTITY_DRIFT` |

The required field list (`DRAFT_REQUIRED_ENVELOPE_FIELDS`) is a T05 draft and
is injectable until T01 freezes the public `TraceEnvelope` schema.

## Gradient write evidence (P2c)

`rl_engine/integrations/megatron_training_gradients.py` wraps one Megatron
DDP instance read-only. The step window runs from the end of this step's
`zero_grad_buffer` to the end of `finish_grad_sync`, i.e. after all
accumulated micro-batches and the gradient sync, before the optimizer step and
the next zeroing; `end_step()` outside a complete window is
`MISSING_PROVENANCE`.

A write is decided from markers, never from values (contributions can cancel
across micro-batches, so a zero `main_grad` or an unchanged digest proves
nothing):

| Accumulate-hook calls | `grad_added_to_main_grad` | Decision |
|---|---|---|
| > 0 | `True` | written via fused wgrad into `main_grad` |
| > 0 | `False` | written via autograd, moved into `main_grad` by DDP |
| 0 | `False` | not written |
| 0 | `True`, or still `True` right after zeroing (TE CUDA graphs) | unknown → `MISSING_PROVENANCE` |

Each record keeps the parameter's DDP buffer index, bucket and range.
Optional per-bucket digests are extra evidence and never decide a single
parameter. `check_base_grad_absent` fails a written base parameter with
`IDENTITY_DRIFT`; `check_emitted_against_producer` cross-checks the observed
emitted set with the producer's declared set (absent →
`UPSTREAM_EVIDENCE_MISSING`, different → `UPSTREAM_CONTRACT_MISMATCH`).
Single-GPU runs verify the real DDP/fused paths; cross-rank reduce-scatter and
distributed-optimizer shard ownership need multiple GPUs.

## M10 export for T06 pairing and the T09 byte verdict (draft)

`rl_engine/alignment/cross_config/adapters/megatron_training_export.py`
writes one directory per validated call (schema
`rlkernel.p7.t05.m10_export.v1-draft`, replaced by the T01 schema once frozen):

| File | Content |
|---|---|
| `m10.json` | Canonical JSON: schema, side, `row_keys` in local row order (`[sample_id, position]`, `null` for padding), BOOL `active_token_mask`, `byte_encoding`, `active_bytes_sha256`, evidence identity and fingerprint, optional upstream evidence |
| `m10_active.bin` | FP32 selected logprob of active rows only, local row order, little-endian bytes of each value's `int32` view |

Files are created exclusively and never overwritten. `load_m10` rejects a
missing file (`INCOMPLETE_ARTIFACT`), a digest or length mismatch or an
active padding row (`CORRUPT_ARTIFACT`), and an unknown schema or byte
encoding (`SCHEMA_MISMATCH`). `self_test_compare` compares two exports token
by token on raw bits (it sees `+0`/`-0` and 1-ULP differences); it is a
format self-test only and carries no verdict authority. The M10 pairing with
T06 and the byte verdict by T09 remain WS1 acceptance items.

## Failure batch and pre-update weight lock (shared with T06)

`rl_engine/alignment/cross_config/adapters/megatron_training.py` defines the
draft interface T05 proposes for the shared T05/T06 failure-batch contract:

- `FailureBatchManifest`: entries keyed by
  `(run_id, checkpoint_id, sample_id, rank, module)` with a stable case ID,
  canonically ordered and fingerprinted, bound to one `PreUpdateWeightLock`.
- `FailureBatchFilter` / `select_failure_batch`: exact-match selection per
  dimension; an empty selection is `IDENTITY_DRIFT`, not a pass. The selection
  fingerprint covers manifest, filter, weight lock and selected case IDs.
- `PreUpdateWeightLock`: checkpoint/weight IDs plus a raw-byte digest of every
  LM-head local shard; any drift is `PRE_UPDATE_WEIGHT_DRIFT`.
- `T05ContractError`: carries one of the contract's fail-closed status names.

## Tests

CPU-only, run in CI (`.github/workflows/ci.yml`):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -rs \
  tests/test_megatron_training_provider.py \
  tests/test_megatron_training_readback.py \
  tests/test_megatron_training_readonly.py \
  tests/test_megatron_training_gate.py \
  tests/test_megatron_training_recorded.py \
  tests/test_megatron_training_evidence.py \
  tests/test_megatron_training_upstream.py \
  tests/test_megatron_training_export.py \
  tests/test_megatron_training_gradients.py \
  tests/test_megatron_training_ranks.py \
  tests/test_megatron_training_gradient_ranks.py \
  tests/test_megatron_training_pairing.py
```

Synthetic fixtures live in `tests/p7_doubles/` (integer-formula values exactly
representable in BF16; digests pinned in `tests/p7_doubles/fixtures/manifest.json`).
`tests/p7_doubles/` also holds test doubles for T01 artifacts that are not yet
published; they are not a public ABI. The VIME request-shape test runs only
when VIME is importable and is skipped otherwise.

GPU (SM90), skipped without a Hopper GPU, the extension, Megatron-LM and VIME:

- `tests/test_megatron_training_gpu.py` (G1): the real strict route passes the
  gate, the CPU double's provenance keys match `LinearLogpWrapper`, repeats are
  bitwise identical and batch composition does not change the bytes.
- `tests/test_megatron_training_gpu_live.py` (G2/G3): a tiny Megatron GPT
  (wrapped in `Float16Module` as Megatron's `get_model` does) with VIME's
  LM-head capture hook. G2: the live snapshot and `RecordedTrainingProvider`
  agree bitwise, and observation adds or removes no CUDA kernel. G3: real
  Megatron DDP over two micro-batches; write markers for the LM head, no write
  on the frozen embedding, and observation leaves `main_grad` bitwise
  unchanged. The fused-wgrad case is skipped when `fused_weight_gradient_mlp_cuda`
  is not installed.

The strict tensor-parallel entry points need the deterministic SM90 GEMM
compiled in. Build the extension with both flags, otherwise the strict route
fails closed ("built without KERNEL_ALIGN_DET_GEMM_SM90=1; refusing naive
fallback"):

```bash
KERNEL_ALIGN_FORCE_SM90=1 KERNEL_ALIGN_DET_GEMM_SM90=1 \
  pip install --no-build-isolation --no-deps -e .
```
