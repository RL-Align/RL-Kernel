# DSv4 Operator Performance

Every DSv4 operator on the SM90 batch-invariant kernel set, against its non-invariant baseline
(goal D7: each within 1.3x). H200; ratio = ours / baseline. The ranges span the token counts
on each operator's page (typically T = 1 to 16384). Lower is better.

Each row's numbers come from the operator page and its benchmark, and every operator in the
table is bitwise batch-invariant (the backward rows are deterministic and batch-invariant).

| Area | Operator | Baseline (not BI) | Ratio | Page / commit |
|---|---|---|---|---|
| GEMM | `Fp8Linear` (DeepGEMM FP8, dense projections) | cuBLAS BF16 | 0.43-1.09x | `988538b3` |
| GEMM | `bf16_gemm_nt` (compressor / indexer score projections) | cuBLAS | 0.90-1.28x | `8b0ef05c` |
| GEMM | `fp8_group_quant` | DeepGEMM helper + layout conversion | ~0.08x | [fp8-group-quant](fp8-group-quant.md) |
| Attention | Indexer stable top-k | `torch.topk` | 0.31-0.96x | [stable-topk](stable-topk.md) |
| Attention | Indexer Q RoPE + FP8 | vLLM CuTe kernel | 1.00-1.14x | [indexer-q](indexer-q.md) |
| Attention | Indexer compressor | vLLM Triton | 1.05-1.14x | [compressor](compressor.md) |
| Attention | Main compressor C4 | vLLM Triton | 0.98-1.07x | [compressor](compressor.md) |
| Attention | Main compressor C128 | vLLM Triton | 0.38-0.41x | [compressor](compressor.md) |
| Attention | FlashMLA sparse forward (pinned) | eager gather + softmax | 0.04-0.11x | `1aa0652a` |
| Attention | Attention layer, C4 decode 64 x 4K (graph) | earlier glue: 5.2 ms | 0.47 ms | `b5b6d671` |
| MoE | Router (gate GEMM + top-6 / hash) | cuBLAS + vLLM Triton top-k | 1.03-1.22x | [dsv4-router](dsv4-router.md) |
| MoE | Router dX | scatter + `torch.mm` | 0.48-1.03x | [dsv4-router](dsv4-router.md) |
| MoE | Route plan (counting sort) | PyTorch sort + searchsorted | 0.12-0.48x | [dsv4-moe](dsv4-moe.md) |
| MoE | Combine (slot sum) | `index_add_` + merge | 0.06-0.41x | [dsv4-moe](dsv4-moe.md) |
| MoE | dX fan-in | `index_add_` | 0.16-0.91x | [dsv4-moe](dsv4-moe.md) |
| MoE | P5 routed forward (two-launch) | per-expert cuBLAS BF16 loop | 0.39-1.02x | [dsv4-moe](dsv4-moe.md) |
| MoE | P5 routed backward (dX) | per-expert cuBLAS BF16 loop | 0.44-0.74x | [dsv4-moe](dsv4-moe.md) |
| MoE | Shared expert (DeepGEMM FP8) | eager BF16 | 0.71-0.78x | `ec257d43` |
| mHC | `mhc_pre` + fused norm | vLLM `mhc_pre_tilelang` + norm | 1.09-1.18x | [dsv4-mhc](dsv4-mhc.md) |
| mHC | `mhc_post` | vLLM `mhc_post_tilelang` | 0.92-1.02x | [dsv4-mhc](dsv4-mhc.md) |
| mHC | Train step (pre + post, fwd + bwd) | naive PyTorch autograd | 0.07-0.09x | [dsv4-mhc](dsv4-mhc.md) |
| Head | Selected-token logprob (fwd) | cuBLAS BF16 logits + FP32 `log_softmax` | 0.79-1.05x | [dsv4-logprob](dsv4-logprob.md) |
| Head | Selected-token logprob (fwd + bwd) | naive PyTorch autograd | 0.84-1.04x | [dsv4-logprob](dsv4-logprob.md) |

Nothing measured exceeds 1.3x.

- `mhc_pre` (1.09-1.18x) and the router (1.03-1.22x) are the closest to the limit.
- The `mhc_pre` cost is DeepGEMM's TF32 GEMM with a fixed split count.
- The router cost is the DeepGEMM gate GEMM against cuBLAS plus vLLM's top-k.

## Not measured yet

- The attention q / o-side kernels (`q_norm_rope_kv_insert`, `OutputProjection`) and the whole
  `Dsv4Attention` layer, against vLLM's attention layer on the same shapes. That needs vLLM's
  attention module with its forward context. The C4 decode figure above compares only against
  our own earlier glue.
- The decoder layer and the full model forward against vLLM. This comes with the layerwise
  gate.
- The P5 `fused` single-launch path: 10.1 ms against 3.2 ms for two-launch at 32768 rows. It is
  not used (two-launch is the default) and is listed in the design doc's Phase 5.
