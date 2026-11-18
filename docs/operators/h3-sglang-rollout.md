# MiniMax-H3 SGLang Rollout Adapter

## Summary

RFC #420 row `sglang_runtime_adapter_h3`: the strict H3 operators inside SGLang's own H3 DiT
(`sglang.multimodal_gen`, `MiniMaxH3DiTModel`, checked at SGLang 0.5.21).

| Where | What |
| --- | --- |
| `integrations/engines/rollout/sglang/h3.py` | `install_h3_sglang_rollout`, `check_sglang_server_args`, `report` |
| `validation/models/h3_sglang_rollout.py` | block 0, the final layer and the refiner norms on SGLang's modules: native, bound and training arithmetic |
| `tools/validation/models/h3_sglang_rollout_report.py` | the JSON report at any TP sizes (TP1 is the reference) |

## What is bound

SGLang's block and token-refiner projections have the paths, fused layouts and `(output, bias)`
return of vLLM-Omni's, and SGLang's loader maps the Diffusers checkpoint exactly as
`models/minimax_h3/checkpoint_mapper` does (all 638 tensors, bitwise; tested when SGLang is
importable). The adapter binds the kernels of the
[vLLM-Omni adapter](h3-checkpoint-mapper-vllm.md) to the projections, and the training rows of
the AdaLN path and the norms:

| SGLang module | Bound to |
| --- | --- |
| `attn.qkv_proj` (`[Q; K; V]`), `mlp.fc1` (`[gate; up]`) | `h3-projection-gemm-v1` |
| `attn.out_proj`, `mlp.fc2` (row parallel) | canonical 8-unit reduction of [`h3-parallel-v1`](h3-parallel.md) |
| `blocks.N.adaln_proj.linear` | the deterministic GEMV of `adaln_projection_3mod`, on its TP column shard |
| `blocks.N` norm1 / norm2 with their modulation | `h3_rmsnorm` (`forward_modulated`) |
| `blocks.N` both gated residuals | `adaln_gate_residual` |
| `final_layer.adaln_proj.linear` + `final_layer.norm` | `final_adaln_out` (the same GEMV and modulated norm) |
| `token_refiner.blocks.N.norm1` / `norm2`, `token_refiner.final_norm` | `h3_rmsnorm` |

- **The AdaLN input.** SGLang's `adaln_input` is `silu(t_emb).to(bfloat16)`, which is the
  activation `adaln_projection_3mod` / `final_adaln_out` compute from `temb` before their GEMV.
- **TP.** The AdaLN GEMV runs on SGLang's TP column shard, which is the TP layout of
  `adaln_projection_3mod`: each column depends only on its own weight row, and SGLang's
  all-gather of the shards is a copy. With `tp > 1`, pass a deterministic collective over
  SGLang's TP group for the row-parallel projections.
- **Why the block's `forward` is replaced.** SGLang applies the norm and its modulation as two
  steps. The bound block applies them as the training block does, and still calls `attn` / `mlp`
  as SGLang's modules. The final layer's FP32 output heads (`video_out` / `audio_out`) stay
  SGLang's.

Attention, rotary, QK norm and SwiGLU stay SGLang's, as do the FP32 boundary projections (patch
embeddings, time embedder, output heads). The report lists them as `external`.

## What is refused

Before any forward changes, `install_h3_sglang_rollout` refuses what would bypass a bound forward
or change what it computes. In strict mode one error names every refused module:

- an SGLang version other than those in `TESTED_SGLANG_VERSIONS` (0.5.21), or no SGLang, also
  without `strict`: the bound block and final-layer forwards replace SGLang's own and call its
  `adaln_proj`, `attn`, `mlp` and `_project` with the signatures they were validated on, so a
  changed SGLang module would otherwise be bypassed without an error. A version is added after
  the report tool passes on it;
- a compiled module (`torch.compile` or `nn.Module.compile`), also without `strict`;
- a LoRA wrapper at a bound linear (it adds the LoRA output in its own forward);
- a quantized linear; a block with quantized `qkv_proj` / `fc1` (SGLang's mxfp8 modulation path);
- a TP size other than `tp`, a column-parallel module that gathers its output, a row-parallel
  module that does not take sharded input or reduce its result;
- the AdaLN cache (`adaln_proj is None`: precomputed tables replace the projection), and a bound
  block or final layer called with precomputed `adaln_params`;
- FP32 AdaLN weights (curve-grid checkpoints), the distilled PDD output heads, and in strict mode
  the VDN-H3 hybrid attention and the compression gate (none of these is the released dense H3);
- CPU or non-BF16 weights, biased projections, and norms without an explicit `eps`.

Without `strict`, a refused module keeps SGLang's forward and is listed in `handle.skipped`.

`check_sglang_server_args(server_args)` refuses `enable_torch_compile`, `regional_compile`,
`enable_breakable_cuda_graph`, `cache_dit_config`, quantization and `lora_path`.

## Evidence

NVIDIA B200 (one GPU per TP rank), torch 2.13.0+cu130, SGLang 0.5.21 (`sglang-kernel` 0.4.7),
pinned `MiniMax-H3@42ed227` block 0, final layer and refiner final norm loaded through SGLang's
loader, 261 rows, a prefix of 37 rows, two timesteps. The block runs with stand-ins for its attention and
MLP outputs; the refiner blocks' norm weights are seeded (not in the pinned extract).

| Output | Bound = training, TP1 / TP2 / TP4 | Bound TP2, TP4 = TP1 | SGLang native = training, TP1 / TP2 / TP4 (max abs diff) | Native TP2, TP4 = TP1 |
| --- | --- | --- | --- | --- |
| `qkv_proj` | yes / yes / yes | yes | yes / yes / yes | yes |
| `out_proj` | yes / yes / yes | yes | no (0.0625) / no (0.125) / no (0.125) | no |
| `fc1` | yes / yes / yes | yes | yes / yes / yes | yes |
| `fc2` | yes / yes / yes | yes | no (0.25) / no (0.25) / no (0.25) | no |
| block AdaLN table | yes / yes / yes | yes | no (0.03125) at every TP size | yes |
| block norm1 (modulated) | yes / yes / yes | yes | no (0.125) at every TP size | yes |
| block norm2 (modulated) | yes / yes / yes | yes | no (0.25) at every TP size | yes |
| block output | yes / yes / yes | yes | no (0.0625) at every TP size | yes |
| final AdaLN table | yes / yes / yes | yes | no (0.03125) at every TP size | no |
| final layer output | yes / yes / yes | yes | no (0.0078) / no (0.0234) / no (0.0234) | no |
| refiner norms (3) | yes / yes / yes | yes | yes / yes / yes | yes |

- **Row invariance.** Every bound output gives the same rows on a prefix of the rows. SGLang's
  own `out_proj` / `fc2` do not at TP1.
- **`remove()`** restores SGLang's forward bitwise.
- **Where SGLang's drift comes from.** Given the training AdaLN table, SGLang's own
  norm-then-modulation and gated residual equal `h3_rmsnorm` / `adaln_gate_residual` bitwise at
  these inputs, and its plain `nn.RMSNorm` equals `h3_rmsnorm`. So all of the native drift in the
  norms, the block output and the final layer comes from SGLang's AdaLN projection (a cuBLAS GEMM).
  Like the native Q/K/V and gate/up, the native norm and gate equalities are measured, not
  guaranteed; the adapter binds them so they do not depend on SGLang's kernels.

### What the equality covers

A bound module runs the training operator's own kernel, whose reduction order is fixed and does not
depend on the row count, the row position or the TP shard. So it gives the training bytes for the
same inputs, under these conditions:

- **The same inputs.** Some inputs of the bound operators come from external ones: `out_proj`
  takes the attention output, `fc2` the SwiGLU output, the gated residuals the attention and MLP
  outputs, and the AdaLN projections `adaln_input`, which SGLang computes from its time embedder.
  Where those differ from training, so do the bound outputs; this is not yet a bitwise SGLang
  trajectory.
- **The same GPU architecture.** The accumulation order inside one BF16 tensor-core instruction is
  the architecture's, so training and rollout are compared on the same architecture. The adapter
  was run on B200 only.
- **`strict`**, a validated SGLang version, and server settings that pass
  `check_sglang_server_args` (the adapter does not run a server).

TP8 is not measured. The bound kernels' TP layouts (whole weight rows per column shard, the fixed
8-unit row-parallel reduction) do not depend on the TP size, which TP2 and TP4 confirm.

Tests: `tests/models/minimax_h3/test_h3_sglang_rollout.py`. The refusal and server-setting tests
run on CPU; a CUDA test checks the bound AdaLN paths against the training operators on stand-in
modules; the SGLang tests skip when `sglang.multimodal_gen` is not importable (it is not a CI
dependency).
