# MoE Router Validation Infrastructure: Design and Implementation Record

| Field | Value |
|---|---|
| Document status | Delivered (infrastructure live; external anchors pending) |
| Document date | 2026-09-09 (development 2026-09-07 ~ 09-09) |
| Last reviewed | 2026-10-07 |
| Target | DSV4-Flash MoE router (H=4096, E=256, K=6, eps=1e-20, scale=1.5) |
| Test baseline | 129 passed / 1 skipped; anchor-free CLI smoke 12/14 exit=67 (formal L3b fails closed) |
| Scope | Validation infrastructure only — no router operator code |

> **What this is.** The design record for the MoE router validation
> infrastructure: the validation ladder, canonical fingerprints, first-mismatch
> localization, and the negative-fixture matrix. Reviewers can jump to
> Section 4 for the design decisions; maintainers start at Section 6 for the
> pending external anchors; Section 7 reproduces the tests.

---

## 1. Why Router Validation Exists (First Principles)

In RL post-training (GRPO/PPO), gradients are driven by the policy ratio

$$
r_\theta(y\mid x)=\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{old}}(y\mid x)}
$$

and KL terms. The numerator and denominator are
computed by two physically distinct engines — the rollout engine and the
training engine. Any logprob difference between them enters the ratio and is
learned by the policy as if it were a real signal: the classic reward-hacking
entry point.

For Dense models the logprob path is a continuous function: a 1-ULP weight
difference produces a ~1-ULP logprob difference, bounded and tolerance-able.
The MoE router is a **discrete decision layer**: which K experts a token
visits is the output of a top-K. A 1-ULP router-score difference near the
top-K boundary **flips a selection** — the token enters a different expert
sub-network and the hidden-state divergence is O(1), not O(ulp).

Consequences that shape everything below:

1. Discrete router outputs (selection, order, tie-break) must be **bit-exact**
   across engines; tolerance has no meaning at the selection level.
2. When a mismatch occurs, the report must localize the **first divergence**
   and name the **owning component** — averaged errors are not actionable.
3. The infrastructure must be able to develop and CI-run **before** live
   engine recordings exist (deterministic synthetic producers).

## 2. The Object Under Validation

```
Hash layers (0-2):    ids_i = tid2eid[input_token_id, i]; no learned Top-K.
                      Selected scores still use the common FP32 gather,
                      fixed-tree normalization, and route-weight path below.

Learned layers (>=3): z -> z' = round_point(z) -> s = sqrt(softplus(z'))
                      q = s + correction_bias
                      ids = stable_topk(q)     # q desc, logical_expert_id asc
                      a_i = s[ids_i]           # weight source is pre-bias score
                      Z = fixed FP32 6-way tree sum + eps
                          ((a0+a1)+(a2+a3))+(a4+a5)  — reduction order frozen
                      w_i = (a_i / Z) * scale
```

Concrete learned-router example (all unlisted experts have selection scores
below `0.8`):

| slot | expert | `s` | bias | selection score `q` | weight score `a` | route weight |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3  | 0.9 | 0.4  | 1.3 | 0.9 | 0.22131148 |
| 1 | 7  | 1.2 | 0.0  | 1.2 | 1.2 | 0.29508197 |
| 2 | 2  | 1.0 | 0.1  | 1.1 | 1.0 | 0.24590164 |
| 3 | 5  | 0.8 | 0.2  | 1.0 | 0.8 | 0.19672131 |
| 4 | 11 | 1.5 | -0.6 | 0.9 | 1.5 | 0.36885246 |
| 5 | 1  | 0.7 | 0.1  | 0.8 | 0.7 | 0.17213115 |

The frozen reduction gives
`S = ((0.9 + 1.2) + (1.0 + 0.8)) + (1.5 + 0.7) = 6.1`, then each route
weight is `1.5 * a_i / (6.1 + 1e-20)`. Selection uses the biased `q`; weight
normalization deliberately returns to the pre-bias `s`. The final route
weights therefore sum to `1.5`, not `1.0`.

### 2.1 Dense/GEMM versus MoE forward and backward

A simplified Dense feed-forward path applies the same matrices to every token:

$$
h=\phi(XW_1),\qquad Y=hW_2.
$$

MoE retains GEMMs but adds routing around the expert GEMMs. If $E_i(X)$ is
the output of expert $i$, its simplified forward path is

$$
z=XW_{\mathrm{gate}},\qquad (I,w)=\operatorname{Router}(z),\qquad
Y=\sum_{i=0}^{5}w_iE_{I_i}(X).
$$

Backward therefore splits into an expert path and a router path. For upstream
output gradient $G=\partial L/\partial Y$, the route-weight gradient is

$$
g_i=\frac{\partial L}{\partial w_i}
=\left\langle G,E_{I_i}(X)\right\rangle,
$$

where the inner product is over the hidden dimension. Expert/combine code
produces this upstream `dweights`; router backward begins at that boundary.

### 2.2 Router backward and the meaning of `dz`

Let

$$
Z=S+10^{-20},\qquad p_i=\frac{a_i}{Z},\qquad w_i=1.5p_i,
\qquad g_i=\frac{\partial L}{\partial w_i}.
$$

The common term uses the frozen six-way reduction tree:

$$
c=((g_0p_0+g_1p_1)+(g_2p_2+g_3p_3))+(g_4p_4+g_5p_5),
$$

and each selected slot receives

$$
\frac{\partial L}{\partial a_i}=\frac{1.5}{Z}(g_i-c).
$$

Because $a_i=s_{I_i}$, slot gradients scatter-add back to their expert score
in slot order:

$$
\frac{\partial L}{\partial s_e}
=\sum_{i:I_i=e}\frac{\partial L}{\partial a_i}.
$$

This also defines duplicate-expert behavior for Hash routing. Correction bias,
Hash lookup, Top-K, tie-breaking, and expert selection carry no gradient; the
selected set is held fixed during backward.

For $s=\sqrt{\operatorname{softplus}(z')}$, the final score-transform
backward is

$$
d_{\mathrm{sp}}=
\begin{cases}1,&z'>20,\\ \sigma(z'),&z'\le20,\end{cases}
\qquad
\frac{\partial L}{\partial z'}=
\begin{cases}
+0.0,&\partial L/\partial s=0,\\
(\partial L/\partial s)\,d_{\mathrm{sp}}/(2s),&\text{otherwise},
\end{cases}
\qquad
\frac{\partial L}{\partial z}=\frac{\partial L}{\partial z'}.
$$

`dz` denotes $\partial L/\partial z$: the gradient of the loss with respect
to the gate-GEMM logits, not the gradient of the entire MoE block. The final
equality is the specified straight-through rule for the forward rounding
point. P3 returns `dz`; P1 owns the subsequent gate-GEMM backward. Conceptually,
for $z=XW_{\mathrm{gate}}$,

$$
\frac{\partial L}{\partial X}\bigg|_{\mathrm{gate}}
=dz\,W_{\mathrm{gate}}^{\mathsf T},\qquad
\frac{\partial L}{\partial W_{\mathrm{gate}}}=X^{\mathsf T}dz.
$$

The full MoE input gradient also includes the expert path (and any residual
path). T09 validates byte-exact `dz` equality and rejects any reported
selection-path gradient; it does not implement these GEMM backward operations.

All danger surfaces sit at **discontinuities**: near-tie ordering of q, the
reduction order of Z, where the bias is applied, the logit rounding point.

## 3. Deliverables

```
rl_engine/moe/
├── naive_topk.py                 total-order Top-K checker (independent rewrite)
└── t01_verdicts.py             frozen verdict codes + priority arbitration

rl_engine/moe/validation/
├── t01_fingerprint.py                 canonical serialization: semantic/artifact hashes
├── first_mismatch.py              six-tuple localization + owner attribution
├── comparison.py                  four-stage ordered comparator
├── runner.py                      shared ValidationReport + L1/L2/L3a/L3b (WS1)
│                                 + rank completeness / cross-config (WS2)
├── paired_check.py                Torch paired check (anchor-pending gated)
└── t01_synthetic_producer.py      T01 seeded deterministic producer

scripts/check_router.py            router validation CLI and orchestration

tests/
├── test_router_naive_topk.py     ordering semantics, tie-break, K-generality
├── test_router_comparison.py      stage order, halt semantics, +/-0.0, padding
├── test_router_validation_runner.py  per-stage boundaries, WS2, producer helpers
├── test_router_validation_failures.py single-defect injection -> exact verdict
├── test_check_router.py           CLI selection, reports, and exit codes
└── test_router_paired_check.py    diagnostics never flip verdicts
```

## 4. Design Decisions

Each entry: problem -> decision -> why the alternative was rejected.

### D1: Independent naive Top-K, never the implementation under test

The cross-check's value is independence. If the checker reused the production
top-K, same-origin defects would be systematically masked. Semantics: total
order `(q desc, logical_expert_id asc)`, slots keep sort order, comparisons on
exact FP32 values (losslessly widened to double). `k` is a runtime argument —
K=6 is only the DSV4-Flash convenience default; routers with other K need no
code change here. Fixture families: random / near-tie (ULP staircase) /
exact-tie (an all-equal row must output ids 0..k-1 — the easiest case to get
wrong).

### D2: Verdict codes frozen first

All later modules (comparator, ladder, CLI exit codes) reference the code
table; freezing it first prevents per-module magic numbers. Bands encode who
holds the evidence: device 1-2 (a kernel can only atomicMin these),
provider 10-22 (identity/schema/upstream), runner 50-72 (case-level).
`primary_verdict` arbitrates multi-error cases by fixed priority
(infrastructure/identity/schema -> upstream evidence -> discrete plan ->
numeric bytes -> fingerprint -> diagnostics) — the priority encodes
causality: an identity error means the numeric diffs downstream are its
effects and must not mask it. New codes append-only, never reordered.

### D3: Four-stage comparator with identity gate first

Stage order `identity -> discrete -> score/weight -> gradient` is a
dependency order, not a preference: different weights mean different
functions (halt — numeric comparison is meaningless); discrete selection
determines which weights exist; weights feed gradients through the chain
rule. Calling any stage method after a halt raises `RuntimeError` instead of
silently skipping — "did not finish comparing" must never masquerade as
"finished comparing". The formal L3b runner requires score tensors as explicit
inputs, so callers cannot silently reduce stage three to a weight-only check.

### D4: Canonical serialization and the dual hash

Two hashes answer two different invariance questions:

- **semantic hash** (per routing decision — the unit is
  `(absolute_layer, global_token_id)`, so the same token routed at two
  layers is two decisions and is never merged; decisions ascending,
  padding excluded, run/engine/attempt/rank excluded): did the policy
  experience the same routing *decisions*? Must be invariant under
  batch/pack/padding/launch changes — and under any physical row
  permutation, because rows enter the hash sorted by `topk_index` within
  each decision unit.
- **artifact hash** (all rows including padding + Envelope fields): is the
  recording itself reproducible? Padding does not affect the loss, but
  unstable padding fill means the launch is not reproducible, so it is
  audited here and only here.

Explicit schema version (`router-canonical.v1`) is embedded in the hash
header: when the official schema lands, the in-place replacement changes the
version string, the two hashes are naturally unequal, and the change can
never be mistaken for semantic drift.

### D5: Six-tuple localization with an exhaustive attribution table

`MismatchKey = (absolute_layer, site, pass_direction, event_index,
global_token_id, rank)` — layer, computation point, forward/backward, event
ordinal, token, rank: an executable debug address. The site -> owner table is
static and exhaustive (`score -> router-score`, `hash_lookup -> tid2eid-lookup`,
`topk -> stable-topk`, `selection -> learned-selection`, `weight ->
route-plan`, `handoff -> combine-plan`, ...); an unknown site raises
`UnknownSiteError` — refusing to guess, because a wrong attribution sends the
bug to the wrong owner and wastes everyone's time. Backward events are only
legal on gradient-carrying sites; a backward event at `topk` means the trace
itself is corrupt.

### D6: The ladder is ordered by debugging cost

| Stage | Compares | Hash / gate | Catches |
|---|---|---|---|
| L1 | same-config repeat | artifact hash (all bytes) | nondeterminism: races, uninitialized memory, atomic ordering |
| L2 | batch/pack/padding/launch perturbation | per-decision semantic hash | layout leakage: semantics varying with physical layout |
| L3a | candidate vs bit-defined oracle | row-wise byte-exact | operator implementation defects |
| L3b | recorded dual-engine | four-stage comparator | engine-vs-engine divergence |

L2 missing decisions -> `INCOMPLETE_ARTIFACT` (the `(layer, token)`
unit is named); extra decisions -> `AMBIGUOUS_GLOBAL_TOKEN_MAPPING`. `runner.py` owns
`ValidationReport`/`make_pass`/`make_fail` as the single construction point so
`paired_check` imports them from one place instead of reaching into another
runner's private helpers (an earlier layout violated this and was refactored
out).

### D7: Two token-ownership modes under parallelism

WS2 first requires the actual rank set to match the expected set exactly.
Duplicate ranks, unexpected ranks, or artifacts from multiple `group` labels
are stale/mixed evidence (`STALE_RUN_METADATA`); this integrity failure takes
priority over a simultaneous missing-rank symptom. A clean but incomplete set
is `MISSING_RANK`.

Cross-config comparison is token-centric, never `rank 0` versus `rank 0`.
Each rank is normalized to
`(absolute_layer, global_token_id) -> semantic_hash`; the base configuration
defines the expected decision set. Checks then run in order: missing unit,
extra/ghost unit, duplicate ownership in partition mode, and carrier-by-carrier
semantic hash equality.

CP/DP partition tokens (each decision unit on exactly one rank; duplicates
mean ambiguous mapping). TP uses replica ownership (a token may appear on
several ranks and every existing carrier's semantic hash must match the base —
comparing a deduplicated single copy would miss a single-rank drift). The first
offending `(layer, token, rank)` is named. `base_config` and `other_config` are
diagnostic labels; `ownership` selects the rule. Replica mode proves equality
of existing carriers, not a configured per-token replica count; rank-set and
placement/ownership evidence remain separate gates. `FORBIDDEN_LOCAL_SHARD_TOPK`
covers the shard-only bug class: top-K over a local shard is not top-K over all
E experts.

### D8: Non-finite gate before byte gates

Without it, `NaN != NaN` reports `ROUTE_WEIGHT_BYTES_MISMATCH` — attributing
the defect to "bytes differ" when the real defect is "something produced a
NaN", two completely different repair paths. The gate runs before any byte
comparison on active values; padding rows are exempt (only active rows are
judged); the router's own `NON_FINITE(1)` is never conflated with
`UPSTREAM_NON_FINITE(18)`.

### D9: Hash/Learned mode exclusivity is tested explicitly

A layer is one mode at a time. Mode disagreement between the two sides must
halt at the identity gate (never reach numeric stages); flipping only the
mode field must change the per-token semantic hash (otherwise the
exclusivity constraint is vacuous) and be caught by L2.

### D10: Paired check is diagnostic-only and anchor-pending

The raw Torch reference must run on every formal golden and record
diagnostics (max/mean abs diff) — to catch "the golden itself drifted from
Torch" early. The diff can never flip a strict verdict; missing execution
evidence is `MISSING_PROVENANCE` (fail-closed, not green); a Torch crash is
missing evidence, not a pass. Shape/dtype mismatches and non-finite
diagnostics are also invalid evidence because no meaningful elementwise
comparison was completed. A finite non-zero diff still passes this evidence
gate, but carries a diagnostic warning and never overrides L3a/L3b. The
recorded absolute-difference statistics must also be non-negative and
internally coherent (mean no greater than maximum); empty tensors and
non-tensor reference outputs are not comparison evidence. The
fixture manifest is looked up at a fixed repository path and never guessed —
a hardcoded guess that happens to hit a stale cache would silently fabricate
an "evidence complete" conclusion. Until the manifest is published, the
lookup returns empty and integration tests skip.

### D11: CLI exit code semantics and stream discipline

Exit code = the first failing verdict's numeric code (0 = all pass), so CI
can classify failure categories directly from `$?`. In `--json` mode the
structured report goes to stdout and the human summary to stderr, so
`| jq` pipelines never break on a summary line.

## 5. Test Matrix Design

Every negative case injects exactly **one** defect — multi-defect injection
makes "correct verdict" coincidental (one defect's verdict can mask the
other's); single-defect injection proves the attribution logic itself.
Control cases pin the inverse: row reorder / padding add-remove / Envelope
changes **must** be judged invariant (false positives from the validation
infrastructure are as fatal as false negatives).

| Injected defect | Verdict | Intercepted at |
|---|---|---|
| tie-break violation (desc id on exact ties) | 55/56 | cross-check / L3a |
| mode exclusivity violation | 10 halt | identity gate |
| router_mode field flip only | 59 | L2 semantic hash |
| weight bitflip (active row) | 51 | L3a |
| score bitflip | 52 | L3a |
| identical active NaN payload on oracle and candidate | 1 halt | L3a non-finite gate |
| L3b score-only drift with equal weights | 52 | L3b score/weight stage |
| semantic field change | 59 | L2 |
| artifact field change (incl. padding) | 60 | L1 |
| stale run/attempt metadata | 60 | L1 |
| missing provenance (identity field absent) | 67 halt | identity gate |
| identity drift (checkpoint/weight change) | 10 halt | before later stages |
| round-point policy inconsistency (both auditable, mutually different) | 14 halt | identity gate |
| tie-break policy inconsistency | 56 halt | identity gate |
| L2 lost token | 13 | L2 |
| L2 ghost token | 63 | L2 |
| WS2 missing rank | 20 | WS2-rank |
| WS2 duplicate/unexpected rank or mixed group | 22 | WS2-rank |
| WS2 partition duplicate ownership | 63 | WS2-cross |
| paired tensor shape/dtype mismatch or NaN/Inf statistic | 67 halt | paired evidence gate |
| WS2-cross ghost token (owned by no base rank) | 63 | WS2-cross |
| WS2 replica single-rank semantic drift | 59 | WS2-cross |
| forbidden silent-fallback flag | 66 halt | provenance |
| selection-gradient leak | 61 | L3b (owner: router-backward, selection-gradient gate) |
| active NaN/Inf value | 1 halt | non-finite pre-gate |
| padding add/remove masking a Core change | 59 | L2 (control: legal padding not flagged) |

## 6. Boundary and Pending External Anchors

This infrastructure validates routers; it does not implement them.

- `naive_topk.py` is a cross-check baseline, not a production kernel — it
  enters no provider path.
- `t01_fingerprint.py`'s `router-canonical.v1` is a minimal stand-in
  serialization; when the official assembler/schema is published it is
  replaced in place (maintaining a second field table is forbidden) and the
  version string is bumped.
- `t01_synthetic_producer.py` uses only seeded deterministic data.
- Known localization limitation: the byte-gate and non-finite-gate
  `FirstMismatch` records carry placeholder coordinates (layer/token/rank
  `-1`); at those gates the comparator holds flattened tensors and the
  tensor->token mapping is not available. The flat index and side are
  reported in `detail`; structural six-tuple localization at those gates
  arrives with the start-kit trace schema.

Pending external anchors (all fail-open for development, fail-closed for
evidence):

1. Golden fixture manifest publication -> the `paired_check` gated
   integration tests activate automatically.
2. Official canonical schema -> in-place replacement in `t01_fingerprint.py`
   with a version bump.
3. Live dual-engine recordings -> L3b switches from synthetic streams to
   recorded traces without code change.

## 7. Reproduction

```bash
cd RL-Kernel  # repo root
# all router validation tests
python -m pytest tests/test_router_*.py tests/test_check_router.py -q
# -> 129 passed, 1 skipped

# CLI smoke (2 cases x 4 WS1 ladders + rank/cross-config WS2 checks)
python scripts/check_router.py --cases smoke
# -> ... passed=12/14 exit=67 (expected while Miles anchor/recordings are pending)

# structured output (stdout is pure JSON, pipeable)
python scripts/check_router.py --cases smoke --json | jq '.[0].stage'

# exit code = first failing verdict code (CI-classifiable)
python scripts/check_router.py --stages L1; echo $?
```

## 8. Module Dependency Graph

```
scripts/check_router.py (router validation entry point and orchestration)
        │
        ▼
runner.py (runners + ValidationReport) ── paired_check.py <- runner layer
             │
             ▼
comparison.py (TraceComparator, 4 stages)     <- comparison primitive
  first_mismatch.py (six-tuple + attribution)  <- localization primitive
  t01_fingerprint.py (semantic/artifact hash)      <- serialization primitive
             │
             ▼
        t01_verdicts.py                     <- frozen codes (bottom, no deps)

t01_synthetic_producer.py -> runner (T01 fixtures)
naive_topk.py        -> independent; referenced only by tests & cross-check
```

No cycles; each layer depends only downward. `t01_verdicts.py` sits at the
bottom and is referenced by every layer — the structural expression of D2
("frozen first").
