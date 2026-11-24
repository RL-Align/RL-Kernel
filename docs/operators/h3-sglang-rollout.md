# MiniMax-H3 SGLang Rollout Adapter

## Summary

RFC #420 row `sglang_runtime_adapter_h3`: the strict H3 operators inside SGLang's own H3 DiT
(`sglang.multimodal_gen`, `MiniMaxH3DiTModel`, checked at SGLang 0.5.21).

| Where | What |
| --- | --- |
| `integrations/engines/rollout/sglang/h3.py` | `install_h3_sglang_rollout`, `check_sglang_server_args`, `check_sglang_sampling_params`, `report` |
| `integrations/engines/rollout/sglang/plugin.py` | the SGLang diffusion plugin: the adapter inside SGLang's own workers |
| `validation/models/h3_sglang_rollout.py` | block 0, the final layer and the refiner norms on SGLang's modules: native, bound and training arithmetic; SGLang's whole DiT with the readback |
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
| `attn.qkv_proj` (`[Q; K; V]`) | `h3-projection-gemm-v1` |
| `mlp.fc1` (`[gate; up]`) | the gate/up row's fixed FP32-tile GEMM (`fixed_fp32_gemm`, [`h3-parallel-v2`](h3-parallel.md)) |
| `attn.out_proj`, `mlp.fc2` (row parallel) | canonical 8-unit reduction of [`h3-parallel-v2`](h3-parallel.md) |
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

- **SGLang's TP batching of the AdaLN projections.** Under TP, `MiniMaxH3DiTModel` projects the
  AdaLN tables of all blocks in one batch and hands each block its precomputed table, which a
  bound block refuses: without a change, every TP > 1 forward of the whole model fails. The
  adapter turns that batching off on the model (`_can_batch_block_adaln`, put back by `remove()`),
  so each block runs its bound projection. The bytes are the same: the batched path calls the same
  bound `adaln_proj.linear` and all-gathers the same column shards, once for all blocks instead of
  once per block.

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
`enable_breakable_cuda_graph`, `cache_dit_config`, quantization and `lora_path`, and the settings
that move or shard the DiT's weights: `dit_cpu_offload`, `dit_layerwise_offload`,
`use_fsdp_inference`, and `cpu_offload_components` / `layerwise_offload_components` naming the
transformer.
`check_sglang_sampling_params(sampling_params)` refuses a request's `enable_spectrum` and
`enable_teacache`: both skip transformer steps.

## Readback

The handle reads back what ran, not only what was installed:

- `handle.calls`: how often each bound forward ran since install (or `handle.calls.clear()`);
- `handle.implementations`: the training kernel each bound module runs;
- `handle.overrides`: the model-level change (the batching switch above);
- `handle.readback(forwards)`: every bound module that did not run exactly once per DiT forward. A
  path of SGLang's model that goes around a bound forward, or a step it skips, shows up here;
- `handle.assert_ran(forwards)`: raises on any of those.

`report(model, handle)` includes all of these.

## In an SGLang server

RL-Kernel declares an SGLang diffusion plugin (`pyproject.toml`, entry-point group
`sglang.multimodal_gen.plugins`, `rl_kernel_h3`). SGLang runs its plugins in every worker before
the worker loads a model. The plugin does nothing unless `RL_KERNEL_SGLANG_H3_STRICT=1`:

```bash
RL_KERNEL_SGLANG_H3_STRICT=1 RL_KERNEL_SGLANG_H3_READBACK_DIR=/path/to/readback \
  sglang serve --model-path <MiniMax-H3> --dit-cpu-offload false ...
```

Then, in every worker, each `MiniMaxH3DiTModel` that `TransformerLoader.load_customized` returns
gets:

- `check_sglang_server_args` on the worker's settings (SGLang may turn `dit_cpu_offload` on by
  itself on small GPUs; the strict path refuses it);
- `install_h3_sglang_rollout` in strict mode at SGLang's TP size, with a `DeterministicCollective`
  over SGLang's TP group (`RL_KERNEL_SGLANG_H3_COLLECTIVE_BYTES` sizes its buffer, default 512 MiB:
  a row-parallel reduction exchanges FP32 partials of rows x 5376);
- a forward guard. Before each DiT forward it refuses the request's Spectrum / TeaCache. After it,
  it asserts the readback: every bound forward ran once per DiT forward so far. A bound forward
  that SGLang went around, or a step it skipped, stops the worker instead of serving a non-strict
  result. With `RL_KERNEL_SGLANG_H3_READBACK_DIR`, each worker writes its report to
  `rank<global rank>.json` after every forward.

The handle is `model.rl_kernel_h3_handle`.

## Evidence

These measurements were taken under `h3-parallel-v1`, before the gate/up row's fixed FP32-tile
GEMM (`fc1`) and the O / down rows' kernel (`out_proj`, `fc2`) of `h3-parallel-v2`. They show the
bound forwards equal to the training graph of that contract; they have not been rerun under v2.

NVIDIA B200 (one GPU per TP rank), torch 2.13.0+cu130, SGLang 0.5.21 (`sglang-kernel` 0.4.7),
pinned `MiniMax-H3@42ed227` block 0, final layer and refiner final norm loaded through SGLang's
loader, 261 rows, a prefix of 37 rows, two timesteps. The block runs with stand-ins for its attention and
MLP outputs; the refiner blocks' norm weights are seeded (not in the pinned extract).

| Output | Bound = training, TP1 / TP2 / TP4 / TP8 | Bound TP2, TP4, TP8 = TP1 | SGLang native = training, TP1 / TP2 / TP4 / TP8 (max abs diff) | Native TP2, TP4, TP8 = TP1 |
| --- | --- | --- | --- | --- |
| `qkv_proj` | yes / yes / yes / yes | yes | yes / yes / yes / yes | yes |
| `out_proj` | yes / yes / yes / yes | yes | no (0.0625) / no (0.125) / no (0.125) / no (0.125) | no |
| `fc1` | yes / yes / yes / yes | yes | yes / yes / yes / yes | yes |
| `fc2` | yes / yes / yes / yes | yes | no (0.25) / no (0.25) / no (0.25) / no (0.25) | no |
| block AdaLN table | yes / yes / yes / yes | yes | no (0.03125) / no (0.03125) / no (0.03125) / no (0.03125) | no at TP8 |
| block norm1 (modulated) | yes / yes / yes / yes | yes | no (0.125) / no (0.125) / no (0.125) / no (0.125) | no at TP8 |
| block norm2 (modulated) | yes / yes / yes / yes | yes | no (0.25) / no (0.25) / no (0.25) / no (0.125) | no at TP8 |
| block output | yes / yes / yes / yes | yes | no (0.0625) / no (0.0625) / no (0.0625) / no (0.0625) | no at TP8 |
| final AdaLN table | yes / yes / yes / yes | yes | no (0.03125) / no (0.03125) / no (0.03125) / no (0.03125) | no |
| final layer output | yes / yes / yes / yes | yes | no (0.0078) / no (0.0234) / no (0.0234) / no (0.0234) | no |
| refiner norms (3) | yes / yes / yes / yes | yes | yes / yes / yes / yes | yes |

- **The whole DiT.** SGLang's `MiniMaxH3DiTModel` with 2 blocks at H3's widths and seeded weights
  (`full_model_target`), on one denoising step packed by SGLang's own request builder
  (`minimax_h3_packed_sequence`, `MiniMaxH3DenoiseBranch`), at TP1, TP2, TP4 and TP8. The adapter binds 27
  forwards (2 blocks x 6, 2 refiner blocks x 6, the final layer's 2, the refiner's final norm).
  Over two forwards each bound forward ran exactly twice on every rank, the output is finite and repeats
  bitwise, and `remove()` restores SGLang's output and its batching switch. The weights are seeded,
  so this checks that every bound path runs, not the bytes; the bytes are the table above.
- **The plugin in SGLang's worker lifecycle** (`plugin_target`), at TP1, TP2, TP4 and TP8, on every
  rank: RL-Kernel's entry point, declared in a `dist-info` as an install declares it, is found by SGLang's own
  `load_plugins()` / `apply_plugin_hooks()` and hooks `TransformerLoader.load_customized`. The
  loader's body is replaced by one that returns the seeded whole DiT, since there is no H3
  checkpoint here. The adapter is installed with the collective over SGLang's TP group; two guarded
  forwards read back clean and write the readback file; a request with Spectrum is refused before
  the forward; with one bound forward put back to SGLang's own, the next forward is stopped
  (`final_layer: ran 2 times in 3 forwards`); `remove()` restores the model's forward. Not run: a
  server launched on the released checkpoint.
- **Row invariance.** Every bound output gives the same rows on a prefix of the rows, at every TP
  size. SGLang's own `out_proj` / `fc2` do not at TP1, and its `qkv_proj` (TP4, TP8) and `fc1`
  (TP8) stop being row-invariant at larger TP sizes.
- **`remove()`** restores SGLang's forward bitwise.
- **TP size.** SGLang's own `out_proj`, `fc2` and final-layer table change with the TP size. At
  TP8 its block AdaLN table does too (its GEMM sees a narrower column shard), and with it its norms
  and block output. Every bound output is the same at TP1, TP2, TP4 and TP8.
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

Tests: `tests/models/minimax_h3/test_h3_sglang_rollout.py`. The refusal and server-setting tests
run on CPU; a CUDA test checks the bound AdaLN paths against the training operators on stand-in
modules; the SGLang tests skip when `sglang.multimodal_gen` is not importable (it is not a CI
dependency).
