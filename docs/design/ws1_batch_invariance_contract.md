# WS1 Batch-Invariance Contract

Status: RFC

Tracking issues:

- [#101: Batch-invariant RL kernel suite for train-inference consistency](https://github.com/RL-Align/RL-Kernel/issues/101)
- [#83: RL-Kernel roadmap Q3-Q4](https://github.com/RL-Align/RL-Kernel/issues/83)
- [#108: WS1 numerical contract](https://github.com/RL-Align/RL-Kernel/issues/108)
- [#96: Batch-invariant deterministic logprob CUDA kernel](https://github.com/RL-Align/RL-Kernel/issues/96)
- [#111: WS2 cross-config alignment](https://github.com/RL-Align/RL-Kernel/issues/111)

## Motivation

RL post-training requires that the same token sequence score the same way regardless of what
else happens to be in the batch. Rollout engines use dynamic batching, chunked prefill, paged
KV, and prefix caching. Reference and reward scoring use different packing and padding. Training
uses micro-batches, gradient accumulation, and different row order. None of these are supposed to
change the logprob of a given token, and all of them can.

The target identity is:

```text
same model + same token sequence + same policy state
    => identical active-token outputs, regardless of surrounding batch shape
```

This is a different failure axis from cross-engine drift. WS2 ([#111]) measures drift between two
*execution paths* — rollout vs training — and its primary metric is
`dlogp = train_recomputed_logp - rollout_old_logp`. WS1 measures drift within a *single* path when
only the batch shape around a sequence changes.

The relationship is a dependency, not an overlap. The WS2 ablation ladder begins at `A0: fully
aligned reference` and states that if A0 fails, the failure is not yet a cross-config problem. A0
is only meaningful if the underlying kernels are batch-invariant. If they are not, A0 is
nondeterministic under batching and every WS2 measurement stacked above it is noise. **This RFC
defines the contract that makes A0 trustworthy.**

Batch invariance is also the axis where the strictest possible guarantee is actually achievable.
When only the batch shape changes — same kernel, same dtype, same reduction tree, same device —
there is no legitimate source of floating-point difference. Any difference is a bug in the
reduction schedule, not an inherent cost of the hardware. WS1 therefore demands bitwise identity
where WS2 correctly refuses to.

## Relationship To Adjacent Contracts

| Contract | Compares | Axis | Requirement |
| --- | --- | --- | --- |
| WS1 accuracy ([#108]) | candidate kernel vs fp32 gold | implementation quality | per-dtype tolerance |
| **WS1 invariance (this RFC)** | **kernel vs itself under a shape transform** | **batch shape** | **bitwise** |
| WS2 cross-config ([#111]) | rollout path vs training path | engine / parallelism | #108 `logprob` tolerance |

This RFC does not define a new tolerance table. Tolerance-based comparisons inherit the per-dtype
table owned by [#108] (`rl_engine/kernels/gtest/tolerance_contract.json`). The only threshold this
RFC governs is the `batch_invariance` entry already present in that file:

```json
"batch_invariance": {"atol": 0.0, "rtol": 0.0}
```

That entry exists today and is read by nothing. `op_checks.py::_resolve_tolerance` only consults
`contract["accuracy"]`. Ratifying and wiring it is the central deliverable here.

## Scope

This RFC defines what batch invariance means, which transforms must preserve it, which comparisons
are bitwise and which are tolerance-based, how a violation is reported and localized, and the
minimal P0 suite with its deferred extensions.

Out of scope for this document:

- Defining a second numerical tolerance table. The accuracy table stays owned by [#108].
- Cross-engine and cross-parallelism alignment. That is WS2 / [#111].
- Layer-wise probing of a full model forward pass. That is WS4.
- Implementing new kernels. Concrete implementations are owned by their own issues; [#96] is the
  CUDA deterministic selected-logprob path referenced by this contract.
- Fixing any specific invariance violation found by the suite.
- Multi-GPU test infrastructure. Every P0 check in this RFC is single-device by construction.

## Definitions

**Logical sequence.** A token sequence plus everything that determines its outputs independently
of batch shape: token ids, position ids, the model checkpoint, and the sampling/scoring parameters.
Two logical sequences are identical when all of these match exactly.

**Active tokens.** The positions whose outputs the contract governs. For scoring paths this is the
response/action positions selected by the action mask. Prompt positions, padding positions, and
masked-out positions are excluded from every comparison. This matches the WS2 measurement contract.

**Transform.** A change to the batch, layout, or execution schedule *around* a logical sequence
that leaves the logical sequence itself unchanged. Transforms are the independent variable of this
contract.

**Invariance-declared operator.** An operator that claims to satisfy this contract. Not every
operator does — `NativeGemmOp` is a deliberately non-deterministic baseline and is exempt.
Declaration is explicit and per-operator; see *Harness Integration*.

## Transform Taxonomy

The following transform classes must not change active-token outputs. Each is a `source_class`
value compatible with the WS2 `KnobDefinition` model, so a WS1 axis can be lifted into a WS2
ablation without redefinition.

### T1: Batch population

Change how many sequences share the launch. `B=1` versus `B=N`; adding or removing unrelated
sequences; growing the batch past kernel tiling and bucket boundaries.

This is the canonical case. It catches split-K, atomic accumulation, and any reduction whose tree
shape depends on grid size.

### T2: Batch position

Permute row order while keeping the population fixed. A sequence at row 0 must score identically
at row 7.

This catches accumulation orders that depend on block index and any reliance on launch order.

### T3: Neighbor content

Keep population and position fixed; change the *token values* of other rows. A sequence must not
notice what its neighbors contain.

This catches shared normalization state, cross-row reductions, and workspace reuse without proper
isolation. `test_deterministic_logp.py::test_deterministic_logp_ignores_batch_noise_bitwise_cuda`
is the existing model for this class.

### T4: Padding layout

Change pad side (left vs right), pad amount, and pad token value. Also includes the fully-masked
row case, where an entire sequence is inactive.

This catches masking implemented as arithmetic rather than exclusion, and any kernel where padded
lanes contribute to a reduction.

### T5: Execution schedule

Change how the sequence is split across launches without changing the sequence: chunked-prefill
boundary placement, prefill/decode split points, and KV-cache handoff. Scoring `[0:S]` in one
launch must equal scoring `[0:C]` then `[C:S]` against the resulting cache.

This class governs attention specifically, including the exported LSE, which must be invariant on
the same terms as the attention output.

### T6: Selection and packing layout

Change how active positions are addressed without changing which they are: dense selection versus
indexed selection, and varlen packing versus padded layout.

`test_deterministic_logp.py::test_deterministic_logp_indexed_matches_dense_bits_cuda` is the
existing model. Indexed writes must additionally leave inactive output rows untouched.

### T7: Cache reuse (P1)

Prefix-cache hit versus miss for a shared prompt prefix. Deferred from P0 because it requires a
live engine rather than a kernel-level fixture; see *Deferred Extensions*.

## Comparison Relations

Every check in the suite is exactly one of three relations. This is the core classification #101
asks for, and it determines the threshold without further negotiation.

### I-relation: invariance — bitwise

```text
op(fixture, transform=T) == op(fixture, transform=identity)
    on active tokens, exact equality
```

Held constant: operator, backend, device, dtype, and every logical input. The **only** difference
is the transform. Threshold is `batch_invariance: {atol: 0.0, rtol: 0.0}` — bitwise, not
`allclose`.

There is no legitimate source of difference under an I-relation. The kernel, the arithmetic, and
the data are identical; only the surrounding shape moved. A nonzero difference is a defect in the
reduction schedule. Tolerance here would hide exactly the bug the suite exists to find.

Applies to: T1-T6, for every invariance-declared operator, forward and backward.

### A-relation: accuracy — tolerance

```text
op(fixture) ~= gold_fp32(fixture)
```

Threshold from the [#108] `accuracy` table by `op_class` and dtype. This is the existing
`run_operator_suite` behavior and is unchanged by this RFC.

### P-relation: backend parity — tolerance

```text
op_backendX(fixture) ~= op_backendY(fixture)
```

Different backends legitimately use different instruction sequences and tiling. Bitwise identity
across CUDA, Triton, and PyTorch is not a realistic requirement and is not demanded. Threshold is
the [#108] `accuracy` table for the operator class and dtype.

**Both backends must independently satisfy the I-relation.** Parity is measured between two
implementations that are each internally batch-invariant. A backend that fails I is not eligible
for a P comparison; report the I failure and stop.

### Summary

| Relation | Varies | Threshold | Rationale |
| --- | --- | --- | --- |
| I | batch shape only | **bitwise (0, 0)** | no legitimate source of difference |
| A | implementation vs fp32 gold | #108 accuracy | finite precision is expected |
| P | backend vs backend | #108 accuracy | different instruction schedules are legitimate |

The distinction that makes this tractable: **I-relations do not need a reference implementation.**
The kernel is its own oracle. This is why the P0 suite runs on one device, in seconds, with no
engine dependency — and why it can gate PRs where WS2 cannot.

## Fixture Contract

An invariance fixture is a logical sequence plus a declared set of transforms. The generator's job
is to guarantee that applying a transform changes only what the transform names.

```text
InvarianceFixture:
    fixture_id:        content-derived stable id
    op_name:           key into OP_SPECS
    op_class:          elementwise | reduction | attention | logprob
    dtype:             float32 | bfloat16 | float16
    device:            cpu | cuda | rocm
    seed:              int

    logical:
        token_ids:         [S] or [B, S]
        position_ids:      optional, explicit when the op consumes them
        target_ids:        for logprob ops
        attention_mask:    [B, S]
        action_mask:       [B, S], defines the active token set
        tensor_inputs:     op-specific, generated from seed

    reference_row:     which row of the baseline holds the sequence under test

    transforms:        list of declared transform instances (T1..T6)
```

Requirements:

1. **Transform isolation.** Applying a transform must change only its declared dimension. The
   generator emits the baseline and the transformed batch from the same seed and the same logical
   tensors; it must not regenerate random inputs per case. An undeclared change invalidates the
   case, mirroring the WS2 `IsolationValidator` rule.
2. **Extraction is exact.** Pulling the sequence under test out of a transformed batch is indexing,
   never recomputation. No gather that reorders, no cast, no contiguity change that could alter the
   compared values.
3. **Active-token masking is the generator's responsibility.** Comparisons receive already-masked
   tensors so that no downstream check accidentally compares padding.
4. **Deterministic and serializable.** `fixture_id` is derived from content, so a failing case can
   be re-run standalone from its id and reported in an issue without attaching tensors.

This extends `rl_engine/kernels/gtest/operator_inputs.py`, which already generates seeded
per-operator inputs. It adds the baseline/transform pairing and the active-token mask; it does not
replace the existing generator.

## Drift Report

Every I-relation failure emits one record. The schema is fixed so that CI, local runs, and issue
reports are the same artifact.

```json
{
  "relation": "invariance",
  "status": "fail",
  "fixture_id": "logp-bf16-cuda-a91c3f",
  "operator": "batch_invariant_logp",
  "op_class": "logprob",
  "chain_index": 8,
  "transform": {"class": "T1", "name": "batch_population", "baseline": 1, "variant": 16},
  "backend": "cuda",
  "arch_key": "sm90",
  "device": "cuda:0",
  "dtype": "torch.bfloat16",
  "direction": "forward",
  "threshold": {"atol": 0.0, "rtol": 0.0},
  "max_abs_error": 3.0517578125e-05,
  "mean_abs_error": 4.1e-07,
  "n_mismatched": 12,
  "n_active": 4096,
  "first_mismatch": {"sequence_row": 3, "token_position": 117, "vocab_index": 40213},
  "launch": {"batch_size": 16, "seq_len": 256, "chunk_size": null, "padding_side": "right"},
  "seed": 123
}
```

`first_mismatch` reports the lowest `(sequence_row, token_position)` in index order, not the
largest error. The first divergence is the debugging entry point; the largest error is usually
downstream of it.

The existing `OutputCheck` dataclass in `op_checks.py` already carries `max_abs_error`,
`mean_abs_error`, `max_rel_error`, and `passed`. This schema is that structure plus transform,
backend, launch, and position metadata.

## First-Divergence Localization

#101 asks how a failure reports the first divergent operator. WS1 answers this at operator
granularity, using a declared forward-chain order. Full layer-wise probing of a live model remains
WS4.

The chain is a static ordering over invariance-declared operators:

```text
0  embedding
1  rope
2  rms_norm
3  attention          (output + LSE)
4  det_gemm / linear
5  silu / swiglu
6  lm_head
7  linear_logp
8  logp / batch_invariant_logp
9  loss reductions    (grpo_loss, ratio_kl)
```

The suite runs every operator against the same transform set under the same fixture family. When
multiple operators fail, the report names the **lowest `chain_index`** as the localization result:

```text
first divergent operator: rms_norm (chain_index 2, T4 padding_layout, bf16, cuda)
downstream also failing:  attention, lm_head, logp
verdict: investigate rms_norm; downstream failures are not independently actionable
```

This is a triage ordering, not a proof of causation. An operator early in the chain that violates
invariance will propagate to everything after it, so downstream failures carry no independent
information until the earliest one is fixed. Operators are compared independently on identical
fixtures, so this ordering is a reporting convention rather than a data dependency — an operator
can fail in isolation without any upstream failure.

## Suite Definition

### P0: the minimal suite

Required for the contract to be considered implemented.

| Operator | T1 pop | T2 pos | T3 neigh | T4 pad | T5 sched | T6 select | bwd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| embedding | R | R | R | R | - | - | R |
| rope | R | R | R | R | - | - | R |
| rms_norm | R | R | R | R | - | - | R |
| attention (+LSE) | R | R | R | R | R | - | R |
| det_gemm | R | R | R | R | R | - | R |
| silu / swiglu | R | R | R | R | - | - | R |
| lm_head | R | R | R | R | - | - | R |
| linear_logp | R | R | R | R | - | R | R |
| logp / batch_invariant_logp | R | R | R | R | - | R | R |
| grpo_loss | R | R | R | R | - | R | R |
| ratio_kl | R | R | R | R | - | R | R |

`R` = required, `-` = not applicable to that operator's inputs.

Backends: PyTorch, Triton, CUDA — each independently, wherever the operator has that path
registered. P-relation parity is checked between all registered pairs.

Dtypes: `float32`, `bfloat16`, `float16` for A and P relations. I-relations run on all three; note
that bitwise invariance is dtype-independent by construction, since it compares a kernel against
itself.

### P1: deferred, with reasons

- **T7 prefix-cache invariance.** Needs a live vLLM engine, so it cannot run in the PR lane. Today
  `enable_prefix_caching` is tested only as a config flag (`test_vllm_rollout_sampler.py`); its
  numerical effect is untested. Nightly GPU lane.
- **Prefix-shared attention.** No dedicated operator exists yet. Blocked on implementation.
- **ROCm parity.** `rl_engine/kernels/ops/rocm/` currently contains three files covering attention
  only. Add operators to the P-relation matrix as they land; the Triton path already provides ROCm
  coverage for most operators.
- **Sampling invariance.** Requires deciding what invariance means under a stochastic operator —
  most likely invariance of the sampling *distribution* and of greedy/`temperature=0` selection,
  rather than of sampled tokens. Needs its own design note.

### P2: deferred

- Cross-engine invariance (rollout vs training) — owned by WS2 / [#111].
- Multi-GPU and TP>1 invariance — owned by WS2.
- Model-level layer-wise probes — owned by WS4.
- FP8 and quantized representations.

## Current Coverage

Measured against the repository at the time of writing. This is the starting point, not a claim
about the target state.

| Transform | Status | Evidence |
| --- | --- | --- |
| T1 batch population | covered for 8 ops | `test_batch_invariant_logp.py:180`, `test_deterministic_logp.py:276`, `test_det_gemm.py:42` |
| T2 batch position | covered for 6 ops | `test_deterministic_logp.py:306`, `test_deterministic_attention_cuda.py:328` |
| T3 neighbor content | partial | `test_batch_invariant_logp.py:212`, `test_deterministic_logp.py:373` |
| T4 padding layout | covered for 9 ops | `test_det_gemm.py:67`, `test_triton_batch_invariant_attention.py:138` |
| T5 execution schedule | attention + det_gemm only | `test_deterministic_attention_cuda.py:384`, `test_det_gemm.py:56` |
| T6 selection layout | CUDA logp only | `test_deterministic_logp.py:402` |

Gaps this RFC closes:

1. **No shared axis vocabulary.** The transform classes above exist today as hand-written tests
   repeated across roughly thirty functions in a dozen files, with the phrase "Axis A — batch
   invariance, bitwise" copy-pasted as a comment in at least five of them. There is no way to answer
   "which (operator x transform x backend) cells are green?" without grepping.
2. **The declared threshold is unwired.** `batch_invariance: {atol: 0.0, rtol: 0.0}` is in
   `tolerance_contract.json` and read by no code path.
3. **`grpo_loss`, `ratio_kl`, and sampling have no invariance tests at all**, despite being named
   in #101 and sitting at the end of the chain where drift is amplified into the policy ratio.
4. **No drift report and no localization.** Failures surface as bare pytest assertions.

Note that the substance is largely present and the coverage is real. What is missing is a declared
matrix, a wired threshold, and a report format — this is consolidation work more than new
verification.

## Harness Integration

The existing `run_operator_suite` implements the A-relation: it calls a candidate and a gold
function on the same inputs and compares. The I-relation has a different shape — one operator, two
input layouts, exact comparison — so it needs a sibling runner rather than a new flag.

```text
run_invariance_suite(
    suite_name,
    *,
    operators:  Sequence[OperatorSpec],     # from OP_SPECS
    transforms: Sequence[TransformSpec],    # T1..T6
    backends:   Sequence[str],
    dtypes:     Sequence[torch.dtype],
    contract:   Mapping | None = None,      # tolerance_contract.json
) -> InvarianceReport
```

Two additions to existing structures:

```text
OperatorSpec.invariance: InvarianceDeclaration | None
    chain_index:         int
    transforms:          frozenset[str]     # which classes apply
    backward:            bool
    exempt_reason:       str | None         # required when invariance is None
```

An operator with `invariance=None` must state why. `NativeGemmOp` declares
`exempt_reason="non-deterministic reference baseline"`. This makes exemption a deliberate, reviewed
act rather than an omission, and it makes the matrix self-describing: every registered operator is
either covered or explicitly excused.

`_resolve_tolerance` gains an I-relation branch that reads `contract["batch_invariance"]` and
compares with exact equality rather than `torch.allclose`.

Existing hand-written tests are not deleted. They remain as targeted regressions for specific bugs;
the matrix provides the systematic floor beneath them.

## Commands

### Contributor, before opening a PR

```bash
# Full local matrix on whatever device is present. Seconds on one GPU, no engine required.
python -m rl_engine.kernels.gtest.invariance --slice pr

# Narrow to one operator while iterating.
python -m rl_engine.kernels.gtest.invariance --op batch_invariant_logp --backend cuda

# Re-run exactly one reported failure.
python -m rl_engine.kernels.gtest.invariance --fixture-id logp-bf16-cuda-a91c3f
```

The `pr` slice is the P0 matrix at small shapes, single device, all registered backends.

### CI

```bash
# PR lane, blocking. CPU + single GPU, small shapes.
python -m rl_engine.kernels.gtest.invariance --slice pr --report-json invariance-pr.json

# Nightly GPU lane. Large shapes, bucket boundaries, all dtypes, backend parity.
python -m rl_engine.kernels.gtest.invariance --slice nightly --report-json invariance-nightly.json
```

The PR lane is blocking because it is cheap and its failures are unambiguous: a bitwise I-relation
failure is a defect, never a tolerance judgment call. This is the practical payoff of the
bitwise/tolerance split — it produces a gate that needs no interpretation.

## Decision Rule

When a check fails, classify in this order:

1. **I-relation failure.** Defect in the operator's reduction schedule. Not a tolerance question,
   not a hardware question. Fix the kernel.
2. **I passes, A fails.** Implementation accuracy problem against the fp32 gold. Route to the
   [#108] accuracy contract.
3. **I passes on both backends, P fails.** Legitimate schedule difference exceeding the accuracy
   tolerance. Decide whether the tolerance or the backend is wrong; do not relax the I threshold.
4. **Multiple I failures across operators.** Act on the lowest `chain_index` only. Downstream
   failures are not independently actionable until it is fixed.
5. **All WS1 checks pass, WS2 still reports drift.** WS1 has done its job: the base scoring path is
   sound, and the drift is genuinely cross-config. Proceed to the WS2 ablation ladder with a
   trustworthy A0.
6. **A WS2 A0 reference fails while WS1 passes.** The mismatch is in engine integration, masks,
   tokenizer, or checkpoint identity — not in kernel invariance. WS1 passing narrows the WS2 search
   space, which is the main reason this contract is worth having.

## Follow-Up Issues

Sized to be independently reviewable. Each is openable from this document without reopening the
contract.

| # | Scope | Depends on |
| --- | --- | --- |
| 1 | `TransformSpec` + `InvarianceFixture` generator over `operator_inputs.py`; T1-T4 only | - |
| 2 | `run_invariance_suite` + wire `batch_invariance` in `_resolve_tolerance`; exact-equality comparator | 1 |
| 3 | `InvarianceDeclaration` on `OperatorSpec`; declare or exempt every operator in `OP_SPECS` | 2 |
| 4 | Drift report schema + JSON emission + `--fixture-id` replay | 2 |
| 5 | T5 execution-schedule transforms for attention and det_gemm, incl. LSE | 1 |
| 6 | T6 selection/packing transforms; dense-vs-indexed and varlen-vs-padded | 1 |
| 7 | Invariance coverage for `grpo_loss` and `ratio_kl` (currently zero) | 3 |
| 8 | Chain-index localization and the multi-failure verdict | 4 |
| 9 | CLI + `pr`/`nightly` slices + CI wiring | 3, 4 |
| 10 | Backend-parity P-relation matrix across registered backends | 3 |
| 11 | (P1) T7 prefix-cache invariance, nightly engine lane | 9 |
| 12 | (P1) Sampling invariance design note | - |

Issues 1-4 are the critical path; 5-10 parallelize behind them.

## Completion Criteria For #101

- [ ] The I/A/P relations are implemented, and the I-relation compares bitwise using the
      `batch_invariance` entry rather than an accuracy tolerance.
- [ ] Every operator in `OP_SPECS` either declares an `InvarianceDeclaration` or an
      `exempt_reason`.
- [ ] The P0 matrix runs green on CPU and on a single CUDA device.
- [ ] A violation produces a drift report record that can be replayed from `fixture_id` alone.
- [ ] Multi-operator failures report a single lowest-`chain_index` verdict.
- [ ] One contributor command and one CI command exist and are documented.
- [ ] `grpo_loss` and `ratio_kl` have invariance coverage.
- [ ] WS2 can cite "WS1 P0 matrix green" as a precondition for a trustworthy A0 reference.

[#83]: https://github.com/RL-Align/RL-Kernel/issues/83
[#96]: https://github.com/RL-Align/RL-Kernel/issues/96
[#101]: https://github.com/RL-Align/RL-Kernel/issues/101
[#108]: https://github.com/RL-Align/RL-Kernel/issues/108
[#111]: https://github.com/RL-Align/RL-Kernel/issues/111
