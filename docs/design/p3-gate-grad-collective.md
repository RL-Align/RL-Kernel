# Gate Parameter-Gradient Collective Provenance: Design Record

| Field | Value |
|---|---|
| Document status | Delivered |
| Document date | 2026-10-21 |
| Work package | T08, contract §4 ("记录 DP 下 gate parameter-grad collective 的 bucket 顺序/算法") |
| Test baseline | 45 passed (collective); 166 passed / 2 skipped across the T08 modules |
| Scope | A provenance record and its gates. **No collective is implemented or wrapped.** |
| Depends on | #498 |

---

## 1. Why P3 Records Something It Does Not Own

P3 produces `dz` and stops. P1 owns the gate GEMM backward, and the all-reduce
that sums `dW_gate` across data-parallel ranks belongs to the training
framework. The contract is explicit in three places that P3 implements neither
(§2.1, §2.7, §4).

It is recorded because it is a **reduction, and a reduction has an order**.
Bucket composition, bucket order, the algorithm and the accumulation dtype
together fix the floating-point summation order of the gate weight gradient.
Change any of them and `dW_gate` moves in the last bits, which moves the next
step's gate logits, which moves router scores — and the router is a discrete
decision layer, so a last-bit move near the Top-K boundary flips a selection.

By the time anyone sees it, that looks like a router defect. The record exists
so an unexplained routing change can be checked against the collective
*before* anyone goes looking inside the router. It is an attribution tool, not
a computation.

## 2. Design Decisions

### D1: Configured and actual are separate fields

The contract repeats that a configured value may not stand in for an actual
readback (§2.3, and `fallback_reason` non-empty is fail-closed). The record
carries `configured_algorithm` and `algorithm` as two fields, and a
difference with no `fallback_reason` is `SILENT_FALLBACK(66)` — a silent
substitution, not a rounding of the truth.

`source` names how the readback was obtained. A record without it is not an
observation, so it is treated as missing evidence rather than as weaker
evidence; there is no third category between "observed" and "not observed".

### D2: Absent evidence is not blank evidence

A missing field fails closed with `MISSING_PROVENANCE(67)` instead of
defaulting. A default would make two unrelated runs compare equal, which is
the one outcome a provenance record must never produce.

### D3: DP=1 needs no record

No all-reduce runs, so there is nothing to observe.
`CollectiveProvenance.none()` says exactly that and validates. Demanding
evidence of a collective that does not run would be fail-closed theatre, and
theatre trains people to route around the gate.

### D4: A missing record is a gap; a malformed record is a defect

In the P7 binding these take different paths:

| Situation | Status | Why |
|---|---|---|
| DP>1, no record supplied | `UNOBSERVABLE` on `training.data_parallel_size` | nobody failed to build anything; the router was handed no observation of a reduction P1 owns |
| DP>1, record supplied but inconsistent | `ERROR` | someone did hand over evidence and it does not hold together |

`UNOBSERVABLE` is in `require_executable`'s rejected set, so a DP case with no
collective evidence is not strictly executable. The DP size is the knob that
reports the hole, because it is the knob whose numeric effect the missing
record hides.

### D5: An unrecognised algorithm is rejected

`KNOWN_ALGORITHMS` is a closed set. A reduction nobody has named cannot be
reasoned about, and "unattributable" is precisely the state this record
exists to prevent. A new implementation is addable, but through a contract
delta rather than by appearing in a field.

### D6: The fingerprint covers order, not observation

`source` and `notes` are excluded from `fingerprint()`. How the same behaviour
was observed is not a numeric difference; including it would make one
collective read back two ways look like drift. Everything that can move the
summation order is included, and a test parametrizes over all nine such
fields.

### D7: A difference between configurations is attribution, not failure

`collective_delta` returns descriptions, never raises. Two configurations with
different DP sizes *must* have different collectives. The delta names what
moved — including the first position at which the reduction order diverged —
so a downstream drift can be checked against it.

`require_same_reduction` is the separate, stricter gate: in a one-at-a-time
experiment where the collective is not the declared axis, any difference is an
uncontrolled variable and the comparison would misattribute its effect to
whatever *was* declared. It validates both sides first, so a broken side is
reported as broken rather than as a difference.

## 3. Test Matrix

Every negative case injects exactly one defect.

| Injected defect | Verdict |
|---|---|
| unsupported schema version | 11 `SCHEMA_MISMATCH` |
| `dp_size < 1`, `gradient_accumulation_steps < 1` | 11 |
| rank list shorter than `dp_size`, or a duplicate rank | 11 |
| unrecognised algorithm / reduce op | 11 |
| bucket with no parameters, or no dtype | 11 |
| one parameter reduced in two buckets | 11 |
| bucket indices not distinct and ascending | 11 |
| any of six required fields empty | 67 `MISSING_PROVENANCE` |
| no buckets | 67 |
| `fallback_reason` set | 66 `SILENT_FALLBACK` |
| configured != actual with no reason | 66 |
| malformed **and** incomplete | **11** (structure precedes completeness) |
| incomplete **and** a declared fallback | **67** (completeness precedes fallback) |
| comparison where the collective is not held constant | 69 `UPSTREAM_CONTRACT_MISMATCH` |

Controls (must **not** be flagged or alter the fingerprint):

| Case | Expected |
|---|---|
| DP=1 with an empty record | valid |
| the same collective read back by a different `source` | identical fingerprint |
| differing `notes` | identical fingerprint |
| two valid records with different `dp_size` | a delta, both still valid |

Binding integration: DP>1 with no record is `UNOBSERVABLE` and not strictly
executable; DP=1 needs none; a supplied record's fingerprint reaches
`side_configs`; a malformed record is `ERROR`.

## 4. Boundary

- Nothing here calls, configures or wraps a collective.
- The record is supplied by whoever can observe the framework; this module
  defines its shape, validates it, fingerprints it and diffs it.
- Bucket contents are taken as given. Verifying that the declared buckets
  match the parameters a framework actually registered needs a live training
  job and belongs with P1.

## 5. Reproduction

```bash
cd RL-Kernel  # repo root
python -m pytest tests/p3/test_collective_provenance.py -q
# -> 45 passed

python -m pytest tests/p3/test_placement.py tests/p3/test_ownership.py \
                 tests/p3/test_p7_binding.py tests/p3/test_collective_provenance.py -q
# -> 166 passed, 2 skipped
```
