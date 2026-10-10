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
3. Row identity via `canonical_cp.packed_gather_indices` (padding rows are
   `None`); unknown layouts → `UNSUPPORTED_CAPABILITY`.
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
  tests/test_megatron_training_gate.py
```

Synthetic fixtures live in `tests/p7_doubles/` (integer-formula values exactly
representable in BF16; digests pinned in `tests/p7_doubles/fixtures/manifest.json`).
`tests/p7_doubles/` also holds test doubles for T01 artifacts that are not yet
published; they are not a public ABI. The VIME request-shape test runs only
when VIME is importable and is skipped otherwise.
