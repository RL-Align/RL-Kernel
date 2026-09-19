# H100 cross-topology gradient repair (2026-09-19)

This patch fixes the canonical TP input-gradient reduction in the Megatron
LM head and QKV projections, including the Transformer Engine QKV path.
It does **not** establish cross-topology bitwise training equality.

The reported 0.021213% was the first-step gradient-norm difference between
training TP2/CP4 and TP4/CP2, both with rollout TP4/CP1. It was not a logprob
mismatch percentage. The original runs also had different audited weights
after the first update.

## Change

Compute column-projection input gradients at the configured canonical TP
leaf width, then add BF16 leaves in the same balanced binary tree used by
the physical TP collective. An unchunked physical-shard GEMM rounded at a
different point when TP changed. Vocabulary padding participates in the
same leaf partition. Forward arithmetic and user CLI options are unchanged.
Ordinary QKV and Transformer Engine QKV both use the repaired backward.

## Validation

56 focused CPU regression tests passed, including bitwise BF16 TP1/2/4/8
subtree comparisons, vocabulary padding, invalid trees, LM-head autograd,
and both installed QKV module paths. Tests use small operands; this is not
an end-to-end certificate for every topology.

H100 one-step runs used the same initial checkpoint and identical tokens,
rewards, masks and sample order: temperature=0.7, top-p=0.95, top-k=-1,
global batch=8, max response length=512, KL coefficient=0.01. All responses
hit the response cap; the nonzero update in this fixture comes from KL.
Each run passed its train/rollout logprob validation with zero mismatches.

| Run | First-step gradient norm |
| --- | ---: |
| graddiag-tp2cp4-r1 | 0.022590212059465119 |
| graddiag-tp4cp2-r1 | 0.022595004168846734 |
| gradfix-lm-tp2cp4-r1 | 0.02258917105830989 |
| gradfix-lmqkv-tp2cp4-r1 | 0.022594830279424053 |
| gradfix-lmqkv-tp4cp2-r1 | 0.022595004168846734 |

The same-version patched TP4/CP2 run reproduced its original gradient norm.
The relative gap fell from 0.021213211142%
to 0.000769598269%
(96.372% reduction). It is still nonzero.

At the LM-head boundary, after removing the power-of-two CP loss scale and
matching 3,940 identical hidden-input rows, differing dX elements fell from
42,400 to zero out of 16,138,240 compared elements. The 128-column sampled
dlogits also matched. This comparison is limited to matched diagnostic rows.

## Remaining gap

Final RMSNorm and LM-head parameter gradients still differ across CP2/CP4.
For the LM head, a CPU replay of saved operands confirms that the same
208 nonzero row contributions, partitioned into different CP groups and
rounded to BF16 before the parameter reduction, produce different results.
That replay diagnoses the reduction schedule; it is not a production fix.
Full CP parameter-gradient ordering and rounding still need to be unified.
Q/K norm reductions and FFN gathered-token ordering also require auditing.

These diagnostic runs include CPU copies and disk I/O, so their timings
are not performance acceptance evidence. ROCm and rollout CP>1 were not
validated by this repair. Post-update weight bitwise equality is not
certified. Full cross-topology training remains **not bitwise**.
