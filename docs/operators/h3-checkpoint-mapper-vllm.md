# MiniMax-H3 Checkpoint Mapper and vLLM-Omni Rollout Adapter

## Summary

RFC #420 integration rows, validated on H100 against the installed vLLM-Omni name tables:

| Row | Where | What it fixes |
| --- | --- | --- |
| `h3_checkpoint_mapper` | `models/minimax_h3/checkpoint_mapper.py` | Diffusers <-> vLLM-Omni native names and layouts |
| `vllm_runtime_adapter_h3` | `integrations/engines/rollout/vllm/h3_omni.py` | strict fixed-order kernels behind vLLM-Omni's `qkv_proj` / `out_proj` / `fc1` / `fc2` |

## Checkpoint mapper

`transformer_blocks.N` -> `blocks.N`, `token_refiner.refiner_blocks.N` -> `token_refiner.blocks.N`;
`to_q/k/v` -> one `attn.qkv_proj` (`[Q;K;V]` rows); `to_out.0` -> `out_proj`; `ff.net.0.proj`
(value-first) -> `mlp.fc1` (gate-first: halves swapped); `ff.net.2` -> `mlp.fc2`; model-level names
from vLLM-Omni's `fasth3` table. All 638 checkpoint tensors map; the round trip is bitwise;
unmapped names and incomplete q/k/v sets fail closed. A test compares the tables with
`vllm_omni.diffusion.models.minimax_h3.fasth3` when importable (passes with vLLM-Omni installed).

## vLLM-Omni adapter

`install_h3_rollout(model, collective=None, tp=1)` replaces the four fused linears' forwards
(returning vLLM's `(output, None)`): `qkv_proj` / `fc1` use `h3-projection-gemm-v1` (a fused column
equals the same column alone, so fused == separate bitwise, tested); `out_proj` / `fc2` use the
canonical row-parallel reduction of [`h3-parallel-v1`](h3-parallel.md). Wrong dtype, non-CUDA
weights or a model without those modules raise in strict mode. Attention, rotary, QK norm,
SwiGLU and AdaLN stay vLLM-Omni's and are reported `external`. `handle.remove()` restores.

## Evidence

`tests/models/minimax_h3/test_h3_checkpoint_mapper_vllm.py`. The name-table test compares with `vllm_omni.diffusion.models.minimax_h3.fasth3` when it is importable and is skipped otherwise.
