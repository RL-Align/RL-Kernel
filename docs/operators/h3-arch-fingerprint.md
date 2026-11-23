# MiniMax-H3 Architecture Fingerprint

## Summary

`h3_arch_fingerprint` (RFC #420, foundation) is the single, machine-checked description of
the model every other H3 row refers to: the pinned checkpoint, its config, the exact
tensor table, the mixed-precision map, the scheduler contract and the strict operator
graph with the RFC row that owns each node. It is pure Python (stdlib only) and has no
kernel; its job is to make "which model, which graph, which row" a testable fact.

Pinned model: `MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe`
(`FL2VA` = `transformer/`, `Ref2VA` = `transformer_ref/`; identical config and tensor table).

## What Is Pinned

| Fact | Value |
| --- | --- |
| Architecture | `MiniMaxH3Transformer3DModel`, 33,122,992,896 parameters, 638 tensors |
| Blocks | 50 transformer blocks (17 graph nodes each) + 2 token-refiner blocks (14 nodes each) |
| Widths | hidden 5376; 56 heads x 128 = 7168 attention inner dim (differs from hidden); FFN 14336, gate/up fused to 28672 |
| AdaLN | 3 modalities (video 0, text 1, audio 2), row = `timestep_index * 3 + tag`, projection 2688 -> 96768 |
| RoPE | MM-RoPE rotates 96 of 128 channels; 32 pass through |
| FP32 modules | `proj_in`, `audio_proj_in`, `time_embedder`, `proj_out`, `audio_proj_out`; FP32 compute without parameters: `time_proj`, `rope` |
| BF16 | every other module and the packed residual stream |
| Scheduler | rectified-flow Euler, `eta = 0`, video shift 12.0, audio shift 3.0, `t = 1 - sigma` |

The public sampler is an ODE (`eta = 0`), so the fingerprint carries no log-probability.
That is the subject of the `stochastic_transition_contract` row.

## How It Is Kept Honest

1. `tools/weights/snapshot_h3_checkpoint.py` reads the Hub tree, configs and safetensors
   headers at the pinned revision (no weights downloaded) and writes
   `rl_engine/models/minimax_h3/pinned_checkpoint.json`.
2. `expected_tensor_table(config)` derives every tensor name, dtype and shape from
   `config.json` alone. `build_fingerprint()` raises unless its digest and parameter
   count equal the checkpoint headers for both variants.
3. `validate_diffusers_model(model)` compares a live (meta-device is enough) Diffusers
   model against the table; `validate_checkpoint_headers(dir)` compares local shards.
4. `arch_fingerprint.json` is the committed golden. `tools/validation/models/h3_fingerprint.py check`
   fails when it differs from the recomputed fingerprint.

## Entry Points

```python
from rl_engine.models.minimax_h3 import fingerprint as fp

fingerprint = fp.build_fingerprint()           # raises on any inconsistency
graph = fp.build_graph()                       # 17-node blocks, refiner, stem, tail
fp.validate_diffusers_model(model)             # class, config, state-dict names/shapes
fp.validate_checkpoint_headers(path)           # local safetensors headers, dtype and shape
```

```bash
python tools/validation/models/h3_fingerprint.py print|write|check
python tools/validation/models/h3_fingerprint.py checkpoint $RL_KERNEL_H3_WEIGHTS/hf/transformer
```

## Graph And Row Ownership

`BLOCK_NODE_NAMES` equals `rl_engine.validation.models.h3_block.NODE_NAMES`, and the owning
row of every block node equals the `Binding.rfc_row` there (both are tested). The graph is
the vocabulary used by `h3_operator_trace` (what must be traced) and by
`h3_trajectory_artifact` (what a sealed run covers). Refiner residual adds are owned by
`token_refiner_block`; the stem rows `audio_input_projection_fp32`, `text_context_projection`
and the tail row `audio_output_projection_fp32` are visible in the graph even before their
kernels land.

## Tests

`tests/models/minimax_h3/test_h3_arch_fingerprint.py` (CPU, seconds): analytic table equals the
checkpoint digest for both variants; mixed precision per tensor; graph counts, uniqueness,
order and ownership against `h3_block`; scheduler shifts against the released configs;
corrupted shard dtype/shape/unknown names, an edited fixture and a changed config all fail
closed; local shard headers and a meta-device Diffusers model are checked when available.

## Out Of Scope

No numerical operator, no weights, no runtime dispatch. Changing the pinned revision means
re-running the snapshot tool, refreshing the golden file and updating dependent rows.
