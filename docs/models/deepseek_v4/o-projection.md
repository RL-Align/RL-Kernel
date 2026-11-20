# DSV4 item 23: grouped output projection

This is an opt-in MI300X (`gfx942`) operator boundary, not a replacement
implementation of GEMM or RoPE. Its immutable order is:

1. Native aiter inverse RoPE on the **interleaved suffix** of each head.
2. Consecutive groups of **eight globally numbered heads**.
3. `wo_a` for each group, with a BF16 boundary.
4. Concatenation in increasing global group order.
5. The local column shard of `wo_b`, producing a **FP32 TP partial**.

The vLLM reference is commit
`e1107b01cc676f18a329576a2f5e7b2a7ba16987`,
`vllm/models/deepseek_v4/nvidia/ops/o_proj.py`. Its non-TMA FP8 BI
path uses per-group block-scaled matmul. This adapter keeps that structure;
it does not attempt to run NVIDIA DeepGEMM/TMA on AMD.

## Reuse, not new numerical kernels

| Stage | Reused implementation |
|---|---|
| Inverse RoPE / its adjoint | aiter `rope_cached_bwd` / `rope_cached_fwd` |
| BF16 grouped `wo_a` / dX | aiter `batched_gemm_bf16`, fixed launch config |
| BF16 `wo_b` and parameter-gradient GEMMs | item-13 `AiterIndexerProjectionBackend` |
| FP8 quantization / checkpoint normalization | item-13 vLLM group-128 quantizer and OCP-to-FNUZ normalization |
| FP8 matmul | item-13 aiter block-scaled GEMM with fixed K recipe |
| Weight-gradient ordering | existing RLK `CanonicalBackwardSession` |
| TP reference merge | increasing-rank FP32 `torch.add`; transport remains P4 |

There are no new Triton/HIP arithmetic kernels, automatic PyTorch-matmul
fallbacks, atomic dW accumulations, or token-count-dependent tuning choices.
Torch views, masking, concatenation and metadata snapshots are layout/control
operations. The explicitly injected FP64 backend exists **only in tests**.

## Geometry and checkpoint contract

Construct `OProjectionContract` with explicit `num_heads`, `head_dim`,
`rope_dim`, `o_lora_rank`, and `hidden_size`; no model variant is guessed.
TP shards must contain complete groups of eight heads. Rank `r` owns a
contiguous, increasing range of global heads.

* Attention input: contiguous BF16 `[M, local_heads, head_dim]`.
* `wo_a` master: contiguous BF16 `[local_groups, rank, 8*head_dim]`.
* `wo_b` master: contiguous BF16 `[hidden, local_groups*rank]`.
* Each precision independently selects `bf16` or `block_fp8_ue8m0`.
* FP8 checkpoint data must be raw, **unshuffled** OCP E4M3 or FNUZ.
  Scales are unpacked FP32, positive power-of-two, one per 128x128 block.
  Packed exponent bytes and framework-private shuffled weights are rejected.
* FP8 inference may omit the corresponding master. Any VJP through that
  stage requires an explicit BF16 master; it is not reconstructed from FP8.
* `prepare_weights` snapshots FP8 data/scales and retains stable master views.
  Source, scale, packed storage and master mutations invalidate the binding.
  After an optimizer update, prepare a new version and bind matching rows.
* Version-counter-free inference tensors are not accepted as weights. Prepare
  ordinary versioned tensors outside inference mode, then use inference mode
  for execution.

The loader must supply real checkpoint data. This API never manufactures an
FP8 checkpoint by casting the BF16 master.

## Logical rows, RoPE and lifetime

`OProjectionRows.bind` takes int64 `[request_id, global_position]` keys,
FP32 `[positions, rope_dim/2]` cos/sin caches, the explicit global head IDs,
and the model version. It owns snapshots of keys and selected angles.
It rejects duplicate active identities, out-of-range positions, malformed
padding, non-unit/non-finite selected rotations and a changed head order.

Padding is exactly `(-1,-1)` and contributes zero output/gradient. The
unrotated prefix is preserved. Empty input does not launch native kernels.
Admission performs host validation; forward does not read token values back
to the CPU. Caller-side source cache changes cannot mutate an admitted
snapshot. Mutating the bound snapshot itself is rejected.

## Backward and batch invariance

```python
from rl_engine.contracts.operators.o_projection import OProjectionContract
from rl_engine.models.deepseek_v4.o_projection import OProjection, OProjectionRows
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session

# This is an explicit example geometry, not a model-config auto-detection.
c = OProjectionContract(16, 128, 64, 128, 128)
op = OProjection(c, layer_id="decoder.0")
weights = op.prepare_weights(wo_a, wo_b, version="optimizer-step-10")
rows = OProjectionRows.bind(
    c, logical_keys, cos_cache, sin_cache,
    head_ids=c.head_ids, version=weights.version,
)
with canonical_backward_session() as session:
    result = op(attention_output, weights, rows=rows)
    result.partial.backward(local_output_gradient)
    session.validate_complete()
```

Register **all forward microbatches before any backward** in a session.
Use disjoint logical keys for the same parameter. Parameter gradients are
formed once from canonically sorted rows, not by summing microbatch dW.
Inverse RoPE uses the native adjoint. Grouped dX uses native batched GEMM.
The `wo_b` bridge is the existing item-13 autograd implementation.

As in item 13 / mLite, dY is cast to BF16 and input/master gradients are
BF16. The FP8 profile deliberately uses **BF16-master surrogate VJPs**;
it does not claim a mathematical derivative of discrete quantization or
gradients for checkpoint bytes/scales. Exact BI is required only at a fixed
contract, checkpoint version, hardware/provider fingerprint and TP partition.
FP64 reference comparisons use numerical tolerances, separately from bytewise
train/no-grad/inference, permutation, chunk and backward-order comparisons.

## TP and engine boundary

`merge_tp_partials` validates complete/unique ranks, identical row order,
angles, model version, layer and implementation fingerprints, then adds in
increasing rank order. This local reference is **not RCCL transport** and
does not establish invariance across TP sizes or arbitrary all-reduce trees.
Do not apply an engine's additional reduction to an already merged result.

Opt-in hooks in both
`rl_engine.integrations.engines.rollout.vllm.o_projection` and
`rl_engine.integrations.engines.train.megatron.o_projection`
re-export the same `bind_output_projection` boundary. They accept explicit
loader-owned tensors, preserve the typed partial result, and support an
expected train/rollout implementation fingerprint. They do not monkeypatch
an installed engine or claim full-model execution.

## Supported envelope and exclusions

Validated: MI300X, eager BF16/FP8 profiles, native RoPE and grouped GEMM,
forward/VJPs, empty/padded/tail rows, fixed logical ordering, local TP1/2/4
partial assembly, and the reported large synthetic geometry.

Not signed off: Graph capture (rejected), actual multi-rank RCCL, CP/DCP
ownership, quantized attention-output input, fused inverse-RoPE quantization,
shuffled checkpoint ABIs, LoRA, full vLLM/Megatron/verl/vime model integration,
and equality across different TP sizes. These are not hidden behind fallbacks.
The provider requires the two installed aiter cached-RoPE binaries and
fingerprints them as well as Python/Triton sources and the shared VJP code.

See `reports/o-projection-mi300x-20261029.md` for measured acceptance.
