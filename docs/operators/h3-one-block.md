# MiniMax-H3 One Block, Forward and Backward

## Summary

`ws1_one_h3_block` (RFC #420, WS1 closeout) runs **block 0 of the pinned MiniMax-H3 checkpoint**
node by node, forward and backward, and reports where it first departs from diffusers. The block
is a fixed graph of 17 nodes, from the AdaLN projection to the second gated residual. It runs on
packed FL2VA layouts that are **bitwise equal to the pinned diffusers pipeline's** `position_ids`
and modality tags.

On one B200, at S = 264, 925 and 3160:

- **Determinism.** Every node, every gradient reaching a node, and all 14 leaf gradients are
  bitwise repeatable.
- **Batch invariance.** Batch row 0 of a two-row batch has the single-row bytes at every node,
  forward and backward.
- **Agreement with diffusers.** Every node that promises it is bitwise equal to diffusers on
  diffusers' inputs. The first drift from diffusers is `adaln_projection`, the first reduction
  whose tree differs from cuBLAS.
- **Accuracy.** The block output is exactly as close to the FP32 golden as diffusers' (relative
  error 5.2e-3 / 7.9e-3 / 9.0e-3).

Only three of the 17 nodes are bound to this stack's row operators. The others use **interim**
deterministic operators until their own rows land, or the provider replay where no kernel exists
yet (see Bindings). The block-level promises above hold with whatever a node is bound to. Speed
and the exact rounding of interim nodes belong to those operators, not to this row.

## Entry Point

```python
from rl_engine.validation.models.h3_block import CandidateOps, run_block_backward, run_block_case
from rl_engine.validation.models.h3_weights import load_h3_block_weights

weights = load_h3_block_weights("cuda")  # conditioning + block-0 tensors, sha256-checked
ops = CandidateOps()
forward = run_block_case(weights, layout="small", ops=ops)    # per node, first_*_drift
backward = run_block_backward(weights, layout="small", ops=ops)
```

The CLI writes the JSON evidence and the plot script renders it:

```bash
python tools/weights/prepare_h3_weights.py --out <dir> --block   # shard 1 of 14, ~4.8 GB
export RL_KERNEL_H3_WEIGHTS=<dir>
python tools/validation/models/h3_block_replay.py --layouts tiny,small,medium --out report.json
python tools/validation/models/plot_h3_block.py report.json
```

## Graph and Bindings

`rl_engine/validation/models/h3_block.py` `NODES` (graph order):

| Node | Owning RFC row | Binding today |
|---|---|---|
| `adaln_projection` | `adaln_projection_3mod` | **row**: `H3AdaLNProjectionCudaOp` (registry) |
| `norm1`, `norm2` | `h3_rmsnorm` | **row**: `H3RMSNormCudaOp.forward_modulated` (registry) |
| `q_proj`, `k_proj`, `v_proj` | `h3_qkv_gemm` | interim: Triton FP32-accumulate GEMM, one BF16 rounding |
| `q_norm`, `k_norm` | `h3_qk_rmsnorm_d128` | interim: `H3RMSNormCudaOp.forward` on the 128-wide heads |
| `rope_q`, `rope_k` | `h3_mm_rope_3axis_partial` | reference: the provider replay (no kernel exists yet) |
| `attention` | `h3_full_attention` | interim: `DeterministicAttentionOp(causal=False)` |
| `o_proj` | `h3_attention_o_gemm` | interim: Triton FP32-accumulate GEMM |
| `residual_attn`, `residual_mlp` | `adaln_gate_residual` | **row**: `H3GateResidualCudaOp` (registry) |
| `ffn_gate_up` | `h3_ffn_gate_up_gemm` | interim: Triton FP32-accumulate GEMM |
| `swiglu` | `h3_swiglu` | interim: `SwiGLUCudaOp` (FP32 math, one rounding) |
| `ffn_down` | `h3_ffn_down_gemm` | interim: Triton FP32-accumulate GEMM |

When a row lands, replacing its node's `candidate` callable is the whole integration. The node's
report entries then become that operator's block-level evidence.

The interim GEMMs use `_triton_gemm_fp32`, one fixed-order FP32 K loop with no split-K, and round
to BF16 once. They do not use `TritonDetGemmOp.__call__`, whose K tree has BF16 nodes. Measured on
B200:

| Shape | Tree path err | FP32 K loop → BF16 | cuBLAS | Tree | FP32 K loop |
|---|---|---|---|---|---|
| 264 × 5376 → 7168 | 7.0e-3 | 2.4e-3 | 2.4e-3 | 10.4 ms | 1.2 ms |
| 925 × 14336 → 5376 | 7.5e-3 | 2.6e-3 | 2.6e-3 | 47 ms | 2.7 ms |

Both paths are bitwise repeatable and row-invariant. Error is max|err| / max|ref| against FP64.

## What Each Node Reports

Every node runs three ways: the provider replay (`h3_provider`, diffusers@f53d552 op for op), the
candidate (chained, and *isolated* on the provider's inputs), and the FP32 golden
(`rl_engine/reference/minimax_h3/block.py`: FP32 math, TF32 off, no BF16 storage between nodes).

- **Forward:** `repeat_bitwise_equal` and `batch_row_bitwise_equal` (row 0 of a B = 2 batch whose
  row 1 has a different input). Also chained and isolated agreement with the provider (bitwise,
  max abs, first differing packed row), the candidate's and provider's `max|err| / max|golden|`,
  and the sha256 of the node's output.
- **Backward:** for the block input, `temb` and the 12 block parameters: repeat bitwise, and the
  candidate's and provider's error against the golden. For the gradient reaching each node's
  output, in reverse graph order: repeat bitwise and batch-row bitwise. Parameter gradients sum
  over the batch, so only activation gradients take part in the batch check.
- **First drift:** `first_drift`, `first_isolated_drift`, `first_repeat_drift`,
  `first_batch_drift`, `first_grad_repeat_drift` and `first_grad_batch_drift` each name the first
  node where that comparison stops being bitwise.
- **Permutation probe (control H-C3, reported, not promised):** permute the packed rows together
  with positions and AdaLN indices, and compare the output to the permuted original.

Inputs: `temb` comes from the provider's conditioning path (the conditioning rows are checked by
`h3_chain`). The block input is a seeded BF16 stand-in, because the packed input projections are
other rows. Text rows use AdaLN table row 0 and media rows use row 1.

## Evidence

[`report.json`](https://github.com/fusheng-ji/RL-Kernel/blob/6d6193d94a83266d5f5f62bb9d90fb3215716ee3/pr539/reports/experiments/h3-one-block-b200/report.json) was generated from a clean tree
at `41e4293` on an otherwise idle B200 (torch 2.13.0+cu130):

![one-block evidence](https://raw.githubusercontent.com/fusheng-ji/RL-Kernel/6d6193d94a83266d5f5f62bb9d90fb3215716ee3/pr539/reports/experiments/h3-one-block-b200/figure.png)

| Layout | S | First drift / first isolated drift | Repeat / batch drift (fwd, grad) | Output err: RL-Kernel / diffusers | Fwd ms: RL-Kernel / diffusers | Fwd+bwd ms |
|---|---|---|---|---|---|---|
| tiny | 264 | `adaln_projection` / `adaln_projection` | none | 5.21e-3 / 5.21e-3 | 7.4 / 0.61 | 22.7 / 2.8 |
| small | 925 | `adaln_projection` / `adaln_projection` | none | 7.87e-3 / 7.87e-3 | 30.1 / 1.12 | 81.7 / 5.0 |
| medium | 3160 | `adaln_projection` / `adaln_projection` | none | 8.97e-3 / 8.97e-3 | 225 / 3.8 | 577 / 11.5 |

- **Forward accuracy.** Per node, the candidate's error is within 0.93–1.20× diffusers'. The
  largest is `attention` at S = 3160, which is the interim attention operator's.
- **Gradient accuracy.** The candidate is better than diffusers on `adaln_proj.weight`,
  `adaln_proj.bias` and `temb`, by up to 7×: diffusers' `index_select` backward accumulates the
  table gradient in BF16 atomics, while the row operators use FP32 segment sums. The candidate is
  within 1.16× on the other leaves, except `attn.norm_q.weight` at S = 3160 (9.5e-3 vs 5.5e-3).
  At S = 925 the same leaf is 2× *more* accurate than diffusers (1.3e-2 vs 2.4e-2), so this
  128-element gradient's error moves with BF16 rounding rather than favouring one side.
- **Permutation probe (H-C3).** The output changes in 79 / 450 / 2199 of the rows, by at most 8.
  Attention sums over keys in packed order, so a consistent permutation is not byte-preserving with
  the interim attention operator. This is for `h3_full_attention` to settle.
- **Speed.** The candidate is 12–60× slower than diffusers forward. Almost all of it is the
  interim GEMMs and attention (FP32 scores materialised). The three row operators are each
  faster than or equal to diffusers in their own evidence.

## Existing implementations (RFC #420 reuse rule)

- **Is there an existing block-level harness?** diffusers' `MiniMaxH3TransformerBlock` is the
  executable reference. It is replayed op for op as the provider, so diffusers is not imported and
  the replay pins the reference's dtypes and ordering. The Qwen3-8B chain gate from #315
  (`rl_engine/validation/models/chain_gate.py`) has per-node token digests and first-drift
  localisation, but its graph, shapes and acceptance are Qwen3-specific (causal attention, no
  AdaLN). This row reuses its method (per-node digests, graph-ordered first drift) on the H3
  `h3_chain` stage pattern.
- **Interim operators.** These are reused from RL-Kernel and are not new code. The only choice
  made here is the FP32-accumulate entry of the Triton GEMM instead of its BF16 tree, for the
  accuracy and speed measured above.

## Tests

`tests/models/minimax_h3/test_h3_one_block.py`:

- **CPU:**
  - the three layouts hash to the pinned diffusers `build_packed_sequence` output;
  - the graph is well formed;
  - the partial MM-RoPE leaves channels 96–127 unchanged.
- **GPU, `tiny` and `small` forward:**
  - every node repeats and is batch-row bitwise;
  - promised nodes are isolated-bitwise to diffusers;
  - node and output errors are at most 1.25 × diffusers' + 1e-3.
- **GPU, `tiny` backward:**
  - every leaf repeats;
  - dX is batch-row bitwise;
  - no gradient drift;
  - leaf errors are at most 1.5 × diffusers' + 5e-3, or, for a BF16 leaf, at most 6 ulps
    (of its largest golden element) above diffusers'. A stored gradient's worst element moves
    in whole ulps, so the relative rule alone is seed- and torch-version-sensitive;
    `tools/validation/models/h3_block_ulp_sweep.py` measured at most +4.15 ulps over
    tiny/small × 10 seeds on torch 2.13 and 2.9.1, and +5.15 ulps over 34 single-node
    candidate/provider swaps (reports: `reports/experiments/h3-one-block-b200/ulp-sweep-torch*.json`).

```bash
RL_KERNEL_H3_WEIGHTS=<dir> CUDA_VISIBLE_DEVICES=0,1 python -m pytest tests/models/minimax_h3 -q
```

## Known Limitations

- **Single platform.** CUDA only. `DeterministicAttentionOp` and the Triton GEMM also build for
  ROCm, but the block has not been run there.
- **Interim nodes.** Eleven of the 17 nodes are interim or reference bindings. Their speed and
  rounding are not the owning rows' final ones, and the MM-RoPE nodes are the provider replay
  itself, so they carry no RL-Kernel evidence until `h3_mm_rope_3axis_partial` lands.
- **Stand-in inputs.** The block input is a seeded stand-in and `temb` is the provider's.
  Real packed projections arrive with the input-projection rows.
- **Not covered here.** Block 0 only (`ws1_full_h3_transformer` covers 50 blocks and the refiner),
  and one sequence per batch row (H3's batch is a replication axis).
