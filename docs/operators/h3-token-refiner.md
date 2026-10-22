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
the operator that already exists for it: `h3_rmsnorm`, `h3_qkv_gemm` (Q/K/V, from the
[projection rows](h3-projection-rows.md)) and the still-`interim` stand-ins for attention,
SwiGLU and the O/gate_up/down GEMMs (claimed by other contributors; they are replaced when
those rows land, without changing the graph).

## Entry Points

```python
from rl_engine.validation.models import h3_refiner as hr, h3_block as hb

ops = hb.CandidateOps()
text = hr.run_refiner("candidate", weights, hidden, ops)      # 2 blocks + final_norm
report = hr.run_refiner_case(weights, block=0, tokens=257)    # per-node evidence
```

`operator_trace.block_observer(trace, block_index=b, stage="token_refiner")` records the
refiner nodes under their fingerprint paths (`token_refiner.refiner_blocks.b.<node>`), so
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

Attention, the O/gate_up/down GEMMs and SwiGLU are `interim` until their rows land, so a
refiner trace is not yet strict-clean (`assert_strict` reports exactly those nodes).
Fail-closed strict mode is therefore enforced per node by the trace, not by the block.
