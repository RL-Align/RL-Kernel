# MiniMax-H3 Token Refiner Block

## Summary

RFC #420 row `token_refiner_block`: the two text-refiner blocks (`token_refiner.refiner_blocks.{0,1}`)
and the final RMSNorm that closes the refiner. Weights are the released checkpoint's
(`MiniMaxAI/MiniMax-H3@42ed227`, shard 14 of 14). A refiner block is a plain pre-norm
transformer block with **no AdaLN, no RoPE and no timestep input**, 14 nodes:

```text
norm1 -> q_proj, k_proj, v_proj -> q_norm, k_norm -> attention -> o_proj -> residual_attn
residual_attn -> norm2 -> gate_up -> swiglu -> down -> residual_mlp
```

The row owns the two residual adds (`residual_attn`, `residual_mlp`, FP32 add of the
BF16 operands rounded once to BF16) and the graph itself; every other node is bound to
the operator that already exists for it: `h3_rmsnorm`, `h3_qk_rmsnorm_d128` (the
[per-head norms](h3-qk-rmsnorm-d128.md)), `h3_qkv_gemm` (Q/K/V, from the
[projection rows](h3-projection-rows.md)), attention (`H3FullAttentionOp`, #553), the
O/gate_up/down GEMM rows (#554, #535; SM90 only, the interim GEMM elsewhere) and the still-`interim`
SwiGLU, which is replaced when its row lands, without changing the graph.

## Entry Points

```python
from rl_engine.validation.models import h3_refiner as hr, h3_block as hb

ops = hb.CandidateOps()
text = hr.run_refiner("candidate", weights, hidden, ops)      # 2 blocks + final_norm
report = hr.run_refiner_case(weights, block=0, tokens=257)    # per-node evidence
```

`operator_trace.block_observer(trace, block_index=b, stage="token_refiner",
bindings=hb.bindings(ops, hr.NODES))` records the refiner nodes under their fingerprint paths (`token_refiner.refiner_blocks.b.<node>`), so
`trace.coverage(paths=...)` can be checked against `fingerprint.build_graph()`.

## Contract

- Node names and owner rows equal `fingerprint.REFINER_NODES` (a test fails otherwise).
- Bitwise repeatable; each batch row independent of its neighbours.
- Nodes that promise it (norms) equal diffusers' result on diffusers' inputs.
- Every node, the block output and the 2-block stack have FP32-golden error
  <= 1.25 x diffusers' error + 1e-3.
- Backward: the gradient reaching the input and each of the 10 parameter leaves is
  repeatable and within 1.5 x diffusers' error + 5e-3 of the FP32 golden's.
- A fused `[Q;K;V]` rollout layout gives bitwise the same Q/K/V as the strict one
  (inherited from `h3_qkv_gemm`).

## Evidence

`tests/models/minimax_h3/test_h3_token_refiner.py` (needs `RL_KERNEL_H3_WEIGHTS` pointing at a
directory with the shard index and shard 14; CPU graph tests run anywhere, CUDA tests skip
without weights). On H100 (Hopper) 15 passed. Tokens 1, 64 and 257 per block, both blocks.

## Limits

The block needs Triton: the interim GEMM and the GEMM rows' backends are Triton kernels, so
`CandidateOps` cannot be built without it.

A refiner trace is not yet strict-clean, for two separate reasons:

- **Interim bindings.** SwiGLU is `interim` until its row lands. So are the
  O/gate_up/down GEMMs off SM90, and attention where the `H3FullAttentionOp` extension cannot be
  loaded (it then runs the interim `DeterministicAttentionOp`).
- **Unobserved fields, on every node.** `block_observer` only learns each node's name, so it
  records the accumulator, the split policy and the fallback state as not observed, row-bound
  nodes included. `assert_strict` (by default `require_observed=True`) reports those too.
  Replacing the interim bindings alone does not clear them; the operators have to report their
  own. `violations(require_observed=False)` checks the bindings alone.

Fail-closed strict mode is therefore enforced per node by the trace, not by the block.

## Hopper evidence (H100)

These outputs and figures are not part of this diff; they are on the fork's `pr_assets` branch (links in the PR description).
`reports/experiments/h3-token-refiner-h100/report.json` (from `tools/validation/models/h3_refiner_report.py`, real block-0 weights) records per-node agreement,
batch/repeat invariance, relative error vs FP64 and block time at 12, 257 and 925 tokens.
Time: 3.7 ms vs 0.54 ms (12 tokens), 8.5 vs 0.62 ms (257), 32.3 vs 1.42 ms (925) for RL-Kernel vs provider. These timings are
historical: they were measured when full attention and the O/gate_up/down GEMMs still ran their interim bindings, and almost all of
the gap was those nodes; the rows this PR adds (QKV, norms, residual) were within about 2x of the provider. They have not been
re-measured with `H3FullAttentionOp` and the #535/#554 GEMM rows bound.
