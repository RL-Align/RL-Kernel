# Module Debug Matrix

The cross-configuration debug surface uses a fixed replay and changes exactly
one factor at a time. It compares two edges of the same frozen rollout:

1. training score versus rollout prefill;
2. rollout prefill versus rollout decode.

The source of truth is
`rl_engine.validation.ablation.debug_matrix.module_debug_matrix`. The
manifest is reporting metadata, not a second collection of runtime flags.

## Comparison gates

Before reporting numerical drift, the run must prove that the selected tokens,
active mask, model state, and logical ownership map are unchanged. A failed
gate records `comparable=false`; it does not produce a drift number. This keeps
an input or sharding error from being misdiagnosed as kernel rounding.

## First-line axes

| Module | Baseline | Gate rows | Diagnostic rows | Invariant control |
| --- | --- | --- | --- | --- |
| Attention | `A0` | `A4` head/sequence ownership | `A1` position/RoPE, `A2` Q/K preprocessing, `A3` mask boundary, `A5` KV state, `A6` precision/rounding, `A7` block plan/merge | `C0`-`C2` |
| FFN / GEMM | `F0` | `F1` TP weight ownership | `F2` SwiGLU rounding, `F3` K-reduction/Split-K, `F4` token collective | `FC0` batch-invariant row replay |
| Selected-token logp | `L0` | `L1` vocabulary ownership/padding domain, `L2` selected token/active mask | `L3` vocabulary LSE tile/merge | `LC0` batch-invariant row replay |

The logp wrapper receives local logits and therefore owns the vocabulary
log-sum-exp, selected-token gather, mask, padding, and TP merge. It does not
own the language-model-head GEMM. The FFN/GEMM wrapper owns the K reduction and
SwiGLU rounding; both must be diagnosed before attributing a logp difference to
the vocabulary merge.

## Numerical program

For each diagnostic row, record the same inputs and constants, intermediate
precision and final write, reduction grouping and logical order, and every
state handoff. Scheduler choices that do not change the numerical program or
logical ownership are intentionally not exposed as mismatch axes.

Use the compact row first. The operator-specific secondary probes remain
available for a focused follow-up only after the representative row reproduces
the mismatch.

## Report artifacts

The matrix reporter is owned by RL-Kernel because it consumes the sealed
cross-configuration attempt contract. It reads `actual.json`,
`comparison.json`, and `token_diffs.pt` only after `COMPLETE` validation, so
the report displays actual selected operators and the materialized topology,
not merely requested flags. See
[Cross-Configuration Drift Report](../../usage/cross-config-drift-report.md) for
the `.rlk-drift` bundle, desktop viewer, static image, and trace entry points.
