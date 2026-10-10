# Attention Debug Matrix

The post-training Attention tool replays one frozen rollout, keeps the sample
identity fixed, changes one factor at a time, and reports both mismatch metrics
and a small set of invariance controls.
This is intentionally separate from the runtime knob catalog in
`configs/experiments/cross_config/cross_config_qwen3_8b_megatron_tp2_cp2_vllm.json`.

## Matrix shape

`rl_engine.reference.attention.debug_matrix` is the source of truth for
the compact matrix manifest:

| Rows | Role | Meaning |
| --- | --- | --- |
| `A0` | baseline | Strict deterministic replay with the same train/rollout identity. |
| `A1`-`A3`, `A5`-`A7` | root-cause probes | One representative probe for each numerical or stateful Attention category. |
| `A4` | comparability gate | Head/sequence ownership must match before a numerical comparison is meaningful. |
| `C0`-`C2` | invariant controls | Cases that must remain exactly zero; a nonzero result invalidates the run. |

The seven `A` rows are:

| Row | Category | Representative probe |
| --- | --- | --- |
| `A1` | Position / RoPE | `position_ids` |
| `A2` | Q/K preprocessing | `qk_norm_disabled` |
| `A3` | Mask / sequence boundary | `causal_mask` |
| `A4` | Topology / head ownership | `tp_head_ownership`; reject as `comparable=false`, do not report a drift scalar. |
| `A5` | KV-cache identity / layout | `kv_page_order` |
| `A6` | Numerical policy | `accum_dtype` |
| `A7` | Distributed schedule | `merge_order` |

The representative row is a fast first-line diagnosis. The existing taxonomy
keeps the secondary probes for a second pass, so users do not need to run all
21 probes for every post-training incident.

## Replay contract

Every row reuses the same:

- checkpoint and model version;
- token IDs, selected-token IDs, masks, and positions;
- KV-cache and packing metadata;
- pre-update model state;
- train/rollout sample ordering.

The matrix is one-at-a-time. It does not create a Cartesian product of Attention
knobs. Each diagnostic row records its own phase-local baseline, which allows
rollout-only and train-only debugging to be compared independently. A changed topology is a
gate failure (`comparable=false`), not a meaningful numerical drift sample. A changed
Split-KV plan or merge order is useful as a diagnostic injection (`A7`), but is not an
accepted production comparison until the actual plan again matches on both sides.

## Metrics

The replay report should include selected-token mismatch metrics:
`train_rollout_logprob_abs_diff`, forward `mismatch_kl`, and `mismatch_k3_kl`.
The Attention artifact additionally records `out`, `lse`, `dQ`, `dK`, and `dV`
maximum absolute drift. These are diagnostics; pass/fail still uses the fixed
repository numerical contract and never a user-supplied tolerance.

`C0` (`tp_partition_control`), `C1` (`batch_composition_control`), and `C2`
(`prefill_decode_tail_control`) must be bitwise zero. They catch accidental
batch dependence, invalid preserved head ownership, and cache-position mistakes
before a root-cause row is trusted.

## Runtime knobs versus debug probes

The cross-configuration planner continues to own executable lifecycle knobs such
as TP/CP topology, CP communication, fusion boundary, and reduction policy.
Those knobs are materialized and read back by the runtime adapters. The compact
`A0`-`A7` matrix is a triage layer over that execution evidence; it does not
pretend that unsupported values such as `reduction_order=arrival` or BF16 CP
accumulation are production implementations.

The portable manifest is available from:

```python
from rl_engine.reference.attention.debug_matrix import attention_debug_matrix

manifest = attention_debug_matrix()
```
