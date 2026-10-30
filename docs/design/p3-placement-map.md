# P3 Placement Map: Design and Implementation Record

| Field | Value |
|---|---|
| Document status | Delivered; one contract question open (Section 2) |
| Document date | 2026-10-15 |
| Work package | T08 (CP/DP/PP/EP ownership, placement, P7 `cross_config`) |
| Target | DSV4-Flash MoE router (E=256, K=6) |
| Test baseline | 53 passed, 2 skipped (native bitmath unavailable on the dev host) |
| Scope | The `logical <-> physical` expert placement map and its gates |
| Depends on | #498 (`P3Verdict`, `require`, `assembler`, `checker`) |

> **What this is.** The design record for the expert placement map: the
> versioned bijection relating the router's canonical expert id to the
> physical id recorded in the Envelope, under *both* readings of
> "physical id" currently in circulation. Section 2 is the open question and
> is the part reviewers should read first.

---

## 1. Why A Placement Map Exists

`logical_expert_id` is the canonical route key: the router's semantic output,
which must not change when expert parallelism moves an expert to another
device. `physical_expert_id` is Envelope metadata describing where that expert
currently lives; it is audited by the artifact hash and excluded from the
semantic hash.

Keeping them apart is the point. Without an explicit object relating the two,
nothing verifies that a recorded `physical_expert_id` actually corresponds to
the `logical_expert_id` on the same row: a dispatch-table off-by-one would
send a token to the wrong expert while every semantic comparison still passed,
because both sides would be wrong in the same way.

The start kit already carries that relation as a flat list plus a version, and
`checker.validate_plan` already gates it. This module adds the EP-rank
dimension, generation from an EP configuration, a provenance fingerprint,
granular diagnostics, and — the substantive part — an explicit declaration of
which numbering a map was built under.

## 2. Open Question: What Is `physical_expert_id`?

Two readings are in circulation, and they disagree.

**Global permutation.** `physical_expert_id` is unique across all `E` experts,
so the map is a permutation of `range(E)`. This is what `assembler.assemble`
builds by default and what `checker.validate_plan` enforces today:

```python
require(len(mapping) == E and sorted(mapping) == list(range(E)),
        P3Verdict.INVALID_PLACEMENT_MAP, "archived placement not bijective")
```

**Rank-local.** The task contract (§1.2) defines `physical_expert_id` as *"the
rank-local id under the current EP placement"*. Under that reading ids index
the hosting rank's own expert table, so they repeat across EP ranks and the
bijection is `logical <-> (ep_rank, physical)`.

The two are not reconcilable. With `EP=8` and `E=256`, rank-local ids are
`0..31` on each of the eight ranks, so `sorted(mapping)` is
`[0]*8 + [1]*8 + ...`, and the current check rejects a correct placement.

This module does not pick a winner. A map declares its `semantics` and
validation applies the matching rule, so:

- a map recorded under one reading can never be silently judged under the
  other (`test_rank_local_table_is_rejected_when_declared_global`);
- the fingerprint separates the two, so the readings cannot collide in
  provenance;
- whichever way the question is resolved, the implementation already exists.

P4 needs the answer too: it decides what a dispatch target id means.

## 3. The Object

```
PlacementMap
  version          placement identity, e.g. "ep8-block-v1" (compared literally)
  semantics        global_permutation | rank_local
  ep_size          number of EP ranks
  ep_rank_of       indexed by logical_expert_id -> ep_rank
  physical_of      indexed by logical_expert_id -> physical_expert_id
  strategy         explicit | identity | block | interleave   (provenance)
  num_experts      E
  schema_version   p3-placement.v1
```

Construction paths, all returning the same type:

| Constructor | Use |
|---|---|
| `identity()` | the single-rank default the assembler already builds |
| `from_logical_to_physical()` | adopt a plan already recorded under the flat list |
| `from_pairs()` | explicit `logical -> (ep_rank, physical)`, irregular placements |
| `from_ep_config()` | generate `block` / `interleave` from an EP configuration |
| `from_plan_fragment()` | rebuild from a RoutePlan's `placement_map` |

A global physical id is defined as the expert's position in the rank-major
concatenation of all per-rank tables. That is the only numbering that agrees
with the rank-local one about where an expert sits *inside* its own rank
(`test_both_semantics_agree_on_position_within_a_rank`). For `block` it
collapses to the identity permutation; for `interleave` it is a non-trivial
one.

## 4. Design Decisions

### D1: Both semantics in one type, declared not inferred

Inferring the reading from the data is possible for most tables and wrong for
the rest: a single-expert-per-rank layout satisfies both rules, and an
EP=1 placement is identical under either. An inferred reading would therefore
be stable in testing and wrong in production. The map carries its semantics;
validation never guesses.

### D2: Construction never validates

A map is a claim; the gates test it. `from_pairs` turns a missing logical id
into a `(-1, -1)` hole rather than raising, because a map that cannot be
constructed also cannot be *rejected with a verdict* — and the project's
fail-closed discipline requires defects to surface as verdict codes rather
than as exceptions in a caller's stack. It also lets each negative fixture
build exactly one broken map, which is how the gates are proved.

`from_ep_config` is the exception: a non-positive shape or an ungeneratable
strategy raises `INVALID_PLACEMENT_MAP` immediately, because there is no map
to report on.

### D3: Two verdicts, because the repair paths differ

`PLACEMENT_MAP_VERSION_MISMATCH(65)` means the evidence was produced under a
*different* placement, so validating the table in hand proves nothing about
it and the fix is to fetch the right map. `INVALID_PLACEMENT_MAP(64)` means
this table is broken and the fix is to repair it. Collapsing them would route
both to the same wrong repair.

`validate_plan_placement` therefore gates `version -> structure -> rows`, a
dependency order rather than a preference.

### D4: Envelope `rank` is deliberately not compared with `ep_rank`

This is the decision most likely to look like a missing check, so it is
recorded explicitly.

The contract classifies EP as `not_a_token_dimension`. An Envelope `rank` is a
*token-dimension* coordinate — which DP/CP/TP/PP rank recorded this token —
while `ep_rank` is an *expert-dimension* coordinate. Under any combined
configuration a token recorded on one rank routinely routes to experts hosted
on another; that is normal operation. Comparing them would fail every
realistic configuration. Pinned as a control test, not left implicit.

Token ownership is a separate concern and already has hooks in `mocks.py`.

### D5: `plan_fragment` extends the plan additively

`version` and `logical_to_physical` keep their meaning and position, so a
consumer reading only those two is unaffected. The extra keys
(`semantics`, `ep_size`, `logical_to_ep_rank`, `strategy`,
`placement_fingerprint`) are additive, so this needs no contract delta and no
schema major bump.

A fragment *without* the extended keys is read as the global reading at
`ep_size=1` — that is what such a plan was recorded under, and assuming
anything else would reinterpret old evidence.

### D6: Defects are collected, not short-circuited

`placement_problems` returns every structural defect. The gate only needs to
know whether the list is empty, but a caller repairing a table wants the whole
list. Checks run in the order defects corrupt the table — shape, then range,
then the semantics-specific rule — and shape failures suppress per-entry
findings, which would otherwise be noise generated by the shape error itself.

## 5. Test Matrix

Every negative case injects exactly one defect.

| Injected defect | Verdict |
|---|---|
| unmapped logical expert (gap) | 64 |
| two logical experts on one `(ep_rank, physical)` slot | 64 |
| repeated physical id under `global_permutation` | 64 |
| `ep_rank` outside `[0, ep_size)` | 64 |
| rank-local slots not `0..n-1` | 64 |
| table length disagrees with `num_experts` | 64 |
| rank table and physical table of different length | 64 |
| empty placement version | 64 |
| unknown semantics / unknown strategy | 64 |
| lookup of a logical id outside the table | 64 |
| declared version != map version | 65 |
| Envelope physical disagrees with the map | 64 |
| padding row carrying a physical id other than `-1` | 64 |
| Envelope version drift inside one plan | 65 |
| core and envelope of different length | 64 |
| plan recorded under `interleave`, judged against `block` | 64 |
| rank-local table declared as `global_permutation` | 64 |
| version mismatch **and** broken map | **65** (version precedes structure) |
| broken map **and** bad row | **64 structural** (structure precedes rows) |

Controls (must **not** be flagged): Envelope `rank` differing from every
`ep_rank`; padding rows with `-1` on both sides; an irregular but bijective
explicit table; a plan fragment round-tripping without an external map.

Interop: a generated global placement (`block` and `interleave`) is driven
through the real `assemble` and accepted by both `validate_plan_placement` and
the start kit's own `checker.validate_plan` — the compatibility claim is that
under the global reading nothing existing has to change.

## 6. Boundary and Pending Work

- Token ownership is not decided here. `mocks.py` holds the `OWNERSHIP` table
  and the `materialize_tokens` / `validate_ownership` hooks; turning that
  table into per-mode dispatch (including `pp: replica_by_layer` and
  `ep: not_a_token_dimension`) is the next T08 change.
- `expected_token_set` generation from a parallel configuration is still T08.
- No P7 `cross_config` materializer is wired yet.
- If the rank-local reading wins, `assembler.assemble` and
  `checker.validate_plan` need their bijection check routed through
  `placement_problems` instead of `sorted(mapping) == list(range(E))`. That is
  a one-line change in each, deliberately not made here while the question is
  open.

### Platform note

Two integration tests skip where the native bitmath library cannot be loaded.
On this Windows development host `rl_engine/p3/provider.py` (`import fcntl`)
and `rl_engine/p3/bitmath.py` (ctypes load of a Unix build product) are not
importable at all, so the whole of `tests/p3` except this module fails
collection. The unit layer here builds plans synthetically and runs anywhere;
the interop test runs wherever the kit itself runs.

## 7. Reproduction

```bash
cd RL-Kernel  # repo root
python -m pytest tests/p3/test_placement.py -q
# -> 53 passed, 2 skipped (skips are the native-bitmath interop cases)

python -m ruff check  rl_engine/p3/placement.py tests/p3/test_placement.py
python -m ruff format --check rl_engine/p3/placement.py tests/p3/test_placement.py
```
