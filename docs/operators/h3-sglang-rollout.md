# MiniMax-H3 SGLang Rollout Adapter

## Summary

RFC #420 row `sglang_runtime_adapter_h3`: the strict H3 projections inside SGLang's own H3 DiT
(`sglang.multimodal_gen`, `MiniMaxH3DiTModel`, checked at SGLang 0.5.21).

| Where | What |
| --- | --- |
| `integrations/engines/rollout/sglang/h3.py` | `install_h3_sglang_rollout`, `check_sglang_server_args`, `report` |
| `validation/models/h3_sglang_rollout.py` | block 0 on SGLang's modules: native, bound and training arithmetic |
| `tools/validation/models/h3_sglang_rollout_report.py` | the JSON report at TP1 and TP2 |

## What is bound

SGLang's block and token-refiner projections have the paths, fused layouts and `(output, bias)`
return of vLLM-Omni's, and SGLang's loader maps the Diffusers checkpoint exactly as
`models/minimax_h3/checkpoint_mapper` does (all 638 tensors, bitwise; tested when SGLang is
importable). The adapter therefore binds the kernels of the
[vLLM-Omni adapter](h3-checkpoint-mapper-vllm.md):

| SGLang module | Layout | Bound to |
| --- | --- | --- |
| `attn.qkv_proj` | `[Q; K; V]`, each sharded by heads | `h3-projection-gemm-v1` |
| `mlp.fc1` | `[gate; up]` | `h3-projection-gemm-v1` |
| `attn.out_proj` | row parallel | canonical 8-unit reduction of [`h3-parallel-v1`](h3-parallel.md) |
| `mlp.fc2` | row parallel | canonical 8-unit reduction of `h3-parallel-v1` |

This is the arithmetic of the strict training block (`h3_tp.run_tp_block_rows`). With `tp > 1`,
pass a deterministic collective over SGLang's TP group.

Attention, rotary, QK norm, SwiGLU and AdaLN stay SGLang's and are reported `external`.
`report()` also says whether SGLang's AdaLN cache replaces the AdaLN projections.

## What is refused

Before any forward changes, `install_h3_sglang_rollout` refuses what would bypass a bound forward
or change what it computes; one error names every refused module:

- a compiled module (`torch.compile` or `nn.Module.compile`), also without `strict`;
- a LoRA wrapper at a projection path (it adds the LoRA output in its own forward);
- a quantized linear, including the mxfp8 input path;
- a TP size other than `tp`, a column-parallel module that gathers its output, a row-parallel
  module that does not take sharded input or reduce its result;
- in strict mode, the VDN-H3 hybrid attention and the compression gate (not the released dense H3);
- CPU or non-BF16 weights and biased projections, as in the vLLM-Omni adapter.

Without `strict`, a refused projection keeps SGLang's forward and is listed in `handle.skipped`.

`check_sglang_server_args(server_args)` refuses `enable_torch_compile`, `regional_compile`,
`enable_breakable_cuda_graph`, `cache_dit_config`, quantization and `lora_path`.

## Evidence

NVIDIA B200, torch 2.13.0+cu130, SGLang 0.5.21 (`sglang-kernel` 0.4.7), pinned
`MiniMax-H3@42ed227` block 0 loaded through SGLang's loader, 261 rows, prefix of 37 rows.

| Projection | Bound = training, TP1 / TP2 | Bound TP2 = TP1 | SGLang native = training, TP1 / TP2 (max abs diff) | Native prefix = rows, TP1 |
| --- | --- | --- | --- | --- |
| `qkv_proj` | yes / yes | yes | yes / yes | yes |
| `out_proj` | yes / yes | yes | no (0.0625) / no (0.125) | no |
| `fc1` | yes / yes | yes | yes / yes | yes |
| `fc2` | yes / yes | yes | no (0.25) / no (0.25) | no |

The bound projections give the same rows on a prefix of the rows, and `remove()` restores
SGLang's forward bitwise. SGLang's own `out_proj` / `fc2` also differ between TP1 and TP2.

Tests: `tests/models/minimax_h3/test_h3_sglang_rollout.py`. The refusal and server-setting tests
run on CPU; the SGLang tests skip when `sglang.multimodal_gen` is not importable (it is not a CI
dependency).
