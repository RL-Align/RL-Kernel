# P3 Token Ownership Under Combined Parallelism: Design Record

| Field | Value |
|---|---|
| Document status | Delivered |
| Document date | 2026-10-15 |
| Work package | T08 (CP/DP/PP/EP ownership, placement, P7 `cross_config`) |
| Target | DSV4-Flash MoE router |
| Test baseline | 40 passed (ownership); 93 passed / 2 skipped with the placement map |
| Scope | Token ownership and cross-config comparison; no collectives, no second P7 runner |
| Depends on | #498 (`P3Verdict`, `require`, `mocks.OWNERSHIP`) |

> **What this is.** `mocks.py` records *what* each parallel dimension does to
> token ownership. This is the module that turns that table into behaviour.
> Section 2 is the modelling decision everything else follows from.

---

## 1. Why The Table Is Not Enough

The contract assigns each dimension a mode:

```
tp: replica       sp: partition_by_sequence    cp: partition
dp: partition     pp: replica_by_layer         ep: not_a_token_dimension
```

Each is clear in isolation and none of them is checkable in isolation. A rank
under `tp=2, cp=2, dp=2, pp=2, ep=2` is simultaneously a sequence-replica, a
sequence-partition, a batch-partition, a layer-replica and token-irrelevant.
There are 32 ranks and the question "should rank 19 hold token 5 at layer 2?"
has one right answer that no single row of the table gives.

Composition is the work.

## 2. The Unit Is A Decision, And The Model Is Carriers

### 2.1 `(absolute_layer, global_token_id)`, not a token

A token-keyed model cannot express `pp: replica_by_layer` at all. The same
token routed at layer 1 and layer 2 is two routing decisions, and under
pipeline parallelism they live on different stages. Keying by token forces
either "this stage is missing most of its tokens" or "layers are not checked",
and both are wrong.

This also matches the semantic hash, which is per routing decision.

### 2.2 One function: `carriers`

`carriers(config, layout)` maps every decision to the set of global ranks that
must hold it. Each dimension either shrinks that set (partitioning) or leaves
it alone (replication), so composition is an intersection and the six modes
need no special cases downstream:

| Finding | Verdict |
|---|---|
| a rank holds a decision it does not carry | `AMBIGUOUS_GLOBAL_TOKEN_MAPPING(63)` |
| a carrier is missing a decision it owes | `INCOMPLETE_ARTIFACT(13)` |
| two carriers of one decision disagree | `ROUTE_SEMANTIC_FINGERPRINT_MISMATCH(59)` |

Three rules cover every configuration. Without the carrier set each
combination would need its own predicate, and the predicates would disagree
with each other at the edges.

`carriers` is built by **inverting** `expected_decisions` over the world
rather than by composing per-dimension rules a second time. One derivation
means the two directions cannot drift apart — a class of bug that is invisible
until a configuration exercises exactly the case where they differ.

## 3. Declared, Not Inferred

Two things could be guessed from the data. Both are declared instead, because
a validator that guesses them agrees with the truth on symmetric
configurations and silently disagrees on the rest — stable in testing, wrong
in production.

**`rank_order`** — how a global rank decomposes into coordinates. Default
`("tp", "cp", "ep", "dp", "pp")`, fastest-varying first. Under `tp=2, pp=2`,
rank 1 is `tp=1` in this order and `pp=1` under a different one; nothing in
the artifact distinguishes them.

**Sequence split order** — CP cuts the sequence first, then SP subdivides each
CP chunk inside the TP group. Reversing it produces a different, equally
self-consistent assignment. `test_cp_then_sp_split_the_sequence_in_that_order`
pins the chosen one.

`sp` is a flag rather than a size, because sequence parallelism subdivides
along the existing TP group and has no axis of its own. Setting it changes TP
from a replica dimension into a partitioning one, which is exactly what the
contract means by "`tp` is `replica` or `partition_by_sequence`". The mode is
read through `mocks.OWNERSHIP` rather than restated, so that table stays the
single source.

## 4. Design Decisions

### D1: `ep` is listed apart from `tp` although the arithmetic is identical

Both replicate, so carrier computation treats them the same. They are kept
separate because the *defect* they expose is different: a fingerprint
difference that tracks the EP coordinate means expert placement leaked into
the routing decision, which is a different bug with a different owner from a
TP replica drifting. `replica_dimensions` names the responsible dimensions in
the failure message, so the report points at EP when EP is implicated.

### D2: Gate order is rank set -> per-rank ownership -> replica agreement

A dependency order, not a preference. Ownership computed over a contaminated
rank set is meaningless. Comparing carriers before knowing each rank holds the
right decisions would report a partition error as a numeric disagreement, and
send it to the wrong owner.

Within the rank-set gate, duplicate/unexpected ranks and mixed groups outrank
a simultaneous missing rank: contaminated evidence makes the missing-rank
finding an artefact of the contamination rather than a fact about the run.

Within per-rank ownership, unauthorized decisions outrank gaps: holding
another rank's decision means the partition itself is wrong, and the gaps
elsewhere are its consequence.

### D3: Ownership is judged per rank, over all its layers at once

A rank reports one evidence record per layer it owns. Judging a single record
against the rank's full expected set reads every other layer of that rank as a
gap — this was a real defect during development, caught by the `pp=1` control
where one rank legitimately owns four layers. `check_rank_ownership` takes the
rank and all of its records.

### D4: Replica agreement compares every carrier, never a deduplicated copy

Folding carriers to one representative before comparing would hide exactly the
failure replication exists to rule out: a single rank drifting while its peers
agree. Every carrier's fingerprint is collected and any split is reported with
the rank groups on each side.

### D5: Cross-config comparison is token-centric

Never rank-against-rank. The whole point of changing the parallel layout is
that rank N holds different tokens afterwards; `rank 0 vs rank 0` compares
unrelated work. Each side is folded to `(layer, token) -> fingerprint` first,
and only then compared. Both sides are validated for ownership before any
comparison, so a broken side is reported as broken rather than as drift.

## 5. Test Matrix

Every negative case injects exactly one defect.

| Injected defect | Verdict |
|---|---|
| a rank absent from the evidence | 20 `MISSING_RANK` |
| duplicate `(rank, layer)` record | 22 `STALE_RUN_METADATA` |
| rank outside the world | 22 |
| mixed group labels | 22 |
| rank holds a decision it does not carry | 63 |
| rank missing a decision it owes | 13 |
| one TP replica drifting | 59 (message names `tp`) |
| one EP replica drifting | 59 (message names `ep`) |
| cross-config: side missing decisions | 13 |
| cross-config: ghost decision | 63 |
| cross-config: decision fingerprint differs | 59 |
| layout repeating a `global_token_id` | 63 |
| empty sequence or layer space | 11 `SCHEMA_MISMATCH` |
| non-positive dimension size, bad `rank_order` | 11 |
| rank outside `[0, world_size)` | 20 |
| stale evidence **and** a missing rank | **22** (integrity precedes) |
| unauthorized decision **and** a gap | **63** (partition precedes) |
| broken rank set **and** a ghost decision | **20** (rank set precedes) |

Controls (must **not** be flagged):

| Case | Why it is legal |
|---|---|
| pipeline stages holding disjoint layers | `pp: replica_by_layer` |
| two TP ranks holding the same token | `tp: replica` |
| partitioned configuration with nothing to agree on | no carrier has a peer |
| two configurations where rank 0 holds different tokens | D5 |

Positive coverage pins each dimension's effect on the carrier set
(`per_rank` size and whether anything is replicated), the combined
`tp2 cp2 dp2 pp2 ep2` case (world 32, four carriers per decision = `tp x ep`),
and `coordinates`/`global_rank` round-tripping across the whole world.

## 6. Boundary and Pending Work

- No collectives are implemented or wrapped here, and no second P7 runner is
  introduced. This module answers "who should hold what", nothing else.
- `TokenLayout` describes the decision space explicitly (sequences plus
  layers). Deriving it from a live job's batch and model configuration is not
  done; the validator takes it as input.
- The P7 `cross_config` materializer is still unwired. `check_cross_config`
  provides the comparison; connecting it to P7's existing planner/runner and
  `mocks.materialize_tokens` is the remaining T08 item.
- Capacity overflow does not interact with ownership here: an overflowed slot
  is still a decision the owning rank must report.

## 7. Reproduction

```bash
cd RL-Kernel  # repo root
python -m pytest tests/p3/test_ownership.py -q
# -> 40 passed

python -m pytest tests/p3/test_ownership.py tests/p3/test_placement.py -q
# -> 93 passed, 2 skipped  (skips are the native-bitmath interop cases)

python -m ruff check  rl_engine/p3/ownership.py tests/p3/test_ownership.py
python -m ruff format --check rl_engine/p3/ownership.py tests/p3/test_ownership.py
```
