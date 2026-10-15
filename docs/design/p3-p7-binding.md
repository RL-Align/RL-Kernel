# P3 Router P7 `cross_config` Binding: Design Record

| Field | Value |
|---|---|
| Document status | Delivered |
| Document date | 2026-10-15 |
| Work package | T08 (CP/DP/PP/EP ownership, placement, P7 `cross_config`) |
| Test baseline | 24 passed (binding); 117 passed / 2 skipped across placement + ownership + binding |
| Scope | A `RuntimeMaterializer` and consumer hooks; no planner, runner, tolerance or collective |
| Depends on | #498, plus `rl_engine/alignment/cross_config` already on the branch |

> **What this is.** The adapter that lets P7's existing control plane drive
> the router's ownership and placement contracts. The contract says T08
> supplies a materializer hook and RoutePlan evidence and explicitly does not
> write a second planner, runner or tolerance; this module is bounded by that.

---

## 1. What P7 Already Owns

`Planner` builds the case grid, `RuntimeTools` materializes a case,
`PairedRunner` executes, `_resume` handles restart, and `ResultCube` collects.
None of that is rebuilt here. The only extension point P7 exposes is the
`RuntimeMaterializer` protocol:

```python
class RuntimeMaterializer(Protocol):
    runtime_kind: str
    @property
    def implementation_fingerprint(self) -> str: ...
    def materialize(self, normalized, descriptors) -> AdapterMaterialization: ...
```

`RuntimeTools` enforces a strict contract on what comes back: exactly one
`KnobApplication` per flattened path, no duplicates, no unknown paths, every
path backed by a descriptor, lifecycle and `critical` matching the descriptor,
and — for anything claimed `APPLIED` — `actual == materialized` and
`materialized == requested` unless the descriptor is `derived`.

The tests drive the real `RuntimeTools` rather than a stand-in. Mocking it
would prove the adapter satisfies a mock, which is precisely the thing not
worth knowing.

## 2. Knobs: Additive, Following The WS2 Precedent

`V1_KNOBS` was written against a HuggingFace/FSDP pair. It has `batch.size`,
rollout-side TP and CP, dtypes and a few engine switches. It has **no**
training-side pipeline, data or expert parallel size, so the configurations
the router must be compared across cannot be expressed in it at all.

`adapters/knobs.py` already set the precedent for this situation: extend
`V1_KNOBS`, never edit it, because deleting or redefining an entry breaks
existing cross-config tests. `P3_ROUTER_KNOBS` follows that exactly —
`V1_KNOBS` entries are reused by identity, and nine paths are added:

| Path | Lifecycle | Why |
|---|---|---|
| `training.{tensor,context,pipeline,data,expert}_parallel_size` | `PROCESS` | the five axes ownership composes over |
| `training.sequence_parallel` | `PROCESS` | flips TP from `replica` to `partition_by_sequence` |
| `router.placement_version` | `ENGINE_CONSTRUCTION` | placement identity, compared literally |
| `router.placement_strategy` | `ENGINE_CONSTRUCTION` | `block` / `interleave` |
| `router.placement_semantics` | `ENGINE_CONSTRUCTION` | which reading of `physical_expert_id` |

The parallel sizes are `PROCESS`-scoped: a process cannot change its EP group
mid-run, and P7 uses the lifecycle to decide what it may reuse. A test pins
that a materialized router case reports `process` isolation, so a future
loosening of that scope cannot pass silently.

`router.placement_semantics` is carried as a knob for a specific reason: the
two readings of `physical_expert_id` (see `p3-placement-map.md`) must never be
spanned by one comparison. Making it a knob means P7's own identity machinery
separates them instead of the router having to notice.

## 3. Design Decisions

### D1: A knob the router does not model is `UNOBSERVABLE`, not `APPLIED`

The materializer owns the parallel sizes, the placement knobs and
`batch.size`. Everything else in a case — `rollout.dtype`,
`rollout.enable_prefix_caching`, `training.attention_backend` — belongs to the
Megatron or vLLM adapter.

Reporting those as `APPLIED` would be a lie with consequences: `APPLIED` means
the value was observed in the running system, and `RuntimeTools` would then
treat a router-only materialization as a fully executable whole-system one.
`UNOBSERVABLE` is both honest and load-bearing — it is in `require_executable`'s
rejected set, so a case mixing engine knobs with the router adapter is refused
rather than half-run. The reason string names the owning adapter.

The practical consequence is that the router materializer expects
*router-scoped* cases. A test pins the mixed case degrading correctly.

### D2: Problems are collected, then reported per knob

`router_view` returns `(config, placement, problems)` instead of raising. A
raise inside `materialize` would collapse the whole case into one opaque
error with no per-path attribution, and `RuntimeTools` requires one
application per path regardless. Each offending knob therefore carries
`ERROR` with the same reason text, and the aggregate status follows P7's own
precedence rules.

### D3: An unbuildable configuration is `ERROR`, never `FALLBACK`

`FALLBACK` means "we ran something else instead". There is no "something
else" for a parallel configuration: if `expert_parallel_size=0` or the
placement strategy has no generator, there is no router view at all. Marking
it `FALLBACK` would let a non-strict run proceed against a configuration that
was never built.

### D4: No placement is declared rather than invented

A case that names no `router.placement_version` yields `placement=None`, and
the binding records `None`. Defaulting to the identity placement would make
two configurations look comparable on a dimension neither declared.

### D5: Two kinds of empty ownership are kept apart

`materialize_rank_rows` distinguishes a rank that owns *other* layers but not
this one (a different pipeline stage — `MISSING_RANK`) from a rank that owns
nothing at all (more DP ranks than sequences — `ZERO_ACTIVE_TOKENS`, the
kit's own name for it). Both are legal states of a legal configuration;
returning an empty slice for either would make "this rank has no work" look
like "this rank finished its work".

This surfaced from a test: with one sequence and `dp=3`, ranks 1 and 2 own
nothing, which is degenerate but not a defect.

### D6: Row slicing delegates to `mocks.materialize_tokens`

The start kit left that hook for exactly this, and it already defines what a
sliced case is. Re-implementing the slice here would be a second definition
of row selection, which the DoD forbids. This module supplies the token set;
the kit supplies the slice.

## 4. Consumer Hooks

| Function | Role |
|---|---|
| `rank_token_manifest(config, layout)` | `{rank: {layer: (token, ...)}}` — P7's rank ownership manifest, one entry per layer because a rank under `pp=1` owns several |
| `materialize_rank_rows(case, config, layout, rank)` | the manifest applied to a recorded case, via `mocks.materialize_tokens` |
| `config_from_binding(binding)` | recover the `ParallelConfig` from what the adapter *reported it built*, not by re-reading the case |
| `layout_from_case(case, layers)` | a single-sequence `TokenLayout` over a case's active tokens; padding never enters the decision space |

`config_from_binding` reads the binding rather than the case on purpose: a
cross-config comparison should run against what was materialized, so a
divergence between requested and materialized shows up as a comparison input
rather than being silently normalized away. It fails closed with
`UPSTREAM_CONTRACT_MISMATCH` on a binding from another adapter.

## 5. Test Matrix

| Case | Expected |
|---|---|
| valid five-axis case | `APPLIED`, executable in strict mode |
| topology reported | `world_size == 32`, all five sizes, `sequence_parallel` |
| ownership table surfaced | the six contract modes, verbatim |
| `sequence_parallel=True` | TP mode flips to `partition_by_sequence` |
| config recovered from the binding | round-trips to the same `describe()` |
| foreign binding | `UPSTREAM_CONTRACT_MISMATCH` |
| no placement declared | `APPLIED`, placement `None` |
| ungeneratable placement strategy | `ERROR`, `require_executable` refuses |
| `expert_parallel_size=0` | `ERROR` |
| engine knob present | `UNOBSERVABLE`, reason names the adapter, not executable |
| path without a descriptor | `SCHEMA_MISMATCH` |
| identical cases / different EP | reusable / not reusable |
| isolation scope | `process` |
| absent dimensions | default to 1 |
| rank order override | honoured, changes the decomposition |
| manifest vs `expected_decisions` | identical for every rank |
| rank rows under `cp=N` | each rank gets exactly its token, union is the whole case |
| rank asked for a layer it does not own | `MISSING_RANK` |
| over-provisioned DP rank | `ZERO_ACTIVE_TOKENS` |
| `layout_from_case` | active tokens only, no padding sentinel |

## 6. Boundary

- No collective is implemented or wrapped.
- No second planner, runner, tolerance or ResultCube.
- The materializer builds the router's *view* of a configuration. Engine
  construction stays with the Megatron and vLLM adapters; a case needing both
  needs both adapters, which is P7's composition question, not the router's.
- `TokenLayout` is still supplied by the caller. Deriving it from a live job's
  batch and model configuration remains open.

## 7. Reproduction

```bash
cd RL-Kernel  # repo root
python -m pytest tests/p3/test_p7_binding.py -q
# -> 24 passed

python -m pytest tests/p3/test_placement.py tests/p3/test_ownership.py \
                 tests/p3/test_p7_binding.py -q
# -> 117 passed, 2 skipped

python -m ruff check  rl_engine/p3/p7_binding.py tests/p3/test_p7_binding.py
python -m ruff format --check rl_engine/p3/ tests/p3/
```
