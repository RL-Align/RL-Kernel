# DSv4 Decoder Layer and Model (forward)

The DeepSeek-V4 model forward on the batch-invariant SM90 kernel set: embedding, the
decoder layers (mHC around attention and MoE), then `hc_head` with the final norm. This is
the forward for rollout and actor recompute.

The LM head and selected-token log-probs are P7. The attention backward does not exist
yet: the pinned FlashMLA has no SM90 sparse backward.

| Piece | Where |
|---|---|
| Config (`config.json` fields, fail-closed check) | `rl_engine/models/deepseek_v4/config.py` |
| RoPE cache | `rl_engine/models/deepseek_v4/rope.py` |
| Layer + model | `rl_engine/models/deepseek_v4/model.py` (`Dsv4DecoderLayer`, `Dsv4Model`) |
| Checkpoint loader | `rl_engine/models/deepseek_v4/loader.py` (`load_model`, `checkpoint_spec`) |
| Tests | `tests/models/deepseek_v4/test_dsv4_model.py` |
| Check against DeepSeek's inference code | `scripts/check_dsv4_model_ref.py` |
| Toy checkpoint | `scripts/make_dsv4_toy_checkpoint.py` |

## Forward

It follows vLLM's `DeepseekV4Model` in its batch-invariant mode (unfused pre / block / post,
`ISEEKYAN/vllm@1a6542c`):

```text
streams = embed[input_ids] copied to 4 streams                         BF16 [T, 4, H]
per layer:
    x, post, comb = mhc_pre(streams, hc_attn_*, attn_norm)
    streams       = mhc_post(attention(x), streams, post, comb)
    x, post, comb = mhc_pre(streams, hc_ffn_*, ffn_norm)
    streams       = mhc_post(moe(x, input_ids), streams, post, comb)
out = hc_head(streams, hc_head_*, norm)                                 BF16 [T, H]
```

Each operator is documented on its own page:
- attention (T07): design doc, Phase 2;
- MoE: [DSv4 MoE](dsv4-moe.md) and [DSv4 MoE Router](dsv4-router.md);
- mHC with the norms fused: [DSv4 mHC](dsv4-mhc.md).

Layer `i` gets its type from the config:
- **Attention:** `compress_ratios[i]` is 0 (sliding window only), 4 (C4 with the indexer) or
  128 (C128).
- **Routing:** hash routing (`tid2eid[input_ids]`) for `i < num_hash_layers`, scored top-6
  for the rest.

**Batch invariance.** A token's output bytes do not depend on which other tokens share its
step. They also do not depend on whether its request runs as one prefill, as chunked
prefill, or as decode steps.

**First layer.** The embedding enters as four identical streams. vLLM's broadcast kernel
instead sums `hc_attn_fn` over the four streams first. The two are equal in exact
arithmetic but not in rounding; this model keeps the `dsv4-mhc-v1` contract.

**RoPE.** The model follows DeepSeek's inference code and Megatron-Core:
- **Layers with a compressor (C4 / C128)** use YaRN with `compress_rope_theta`. Their cache
  reproduces vLLM's `DeepseekV4ScalingRotaryEmbedding` op for op on the GPU and is bit-equal
  to vLLM's (tested).
- **Sliding-window-only layers** use plain RoPE with `rope_theta`. Their cache follows
  `precompute_freqs_cis` of DeepSeek's `model.py` op for op and is bit-equal to it (tested).
  A 128-token window never reaches the distances YaRN rescales.

**Difference from vLLM.** vLLM (`ISEEKYAN/vllm@1a6542c`) builds a YaRN rotary embedding
(factor 16) for the window-only layers as well; this was checked by building
`build_deepseek_v4_rope` from the checkpoint config. The rollout patch must therefore pass
this model's RoPE table to every layer. Within a window the two differ by at most about
0.04 rad on the low-frequency dimensions, but they are not the same model.

## Loading

`load_model(path_or_mapping, config, max_model_len=..., num_layers=..., ep_group=...)`:

- Reads the `deepseek-ai/DeepSeek-V4-Flash` layout. `checkpoint_spec(config)` lists every
  tensor's name, dtype and shape, and a test checks it against the real Flash headers.
- Fuses as vLLM's `load_weights` does:
  - `fused_wqa_wkv = [wq_a; wkv]`;
  - each compressor `[wkv; wgate]`;
  - experts `w1 = [w1 (gate); w3 (up)]`;
  - the shared expert likewise.
- Uses FP8 and MXFP4 weights as stored (E8M0 scales to FP32 exactly); nothing is
  requantized. `attn_sink` is padded with `-inf`.
- `num_layers` keeps the first layers, for the layerwise gate. `ep_group` keeps each rank's
  slice of the routed experts.
- `Dsv4Config.check_supported()` rejects anything the kernels do not implement:
  - mHC constants, top-6, `sqrtsoftplus`, SwiGLU limit 10;
  - FP8 128x128 UE8M0 weights, MXFP4 experts;
  - compress ratios other than 0 / 4 / 128;
  - expert counts other than 256 or 384 (the router kernel's).

  A toy checkpoint therefore keeps 256 experts.

`Dsv4Model.forward_batch(input_ids, batch, caches, return_streams=False)` takes a
`Dsv4Batch` (host-built step layout) and the per-layer paged caches from
`model.new_caches(...)`. `return_streams=True` also returns each layer's output streams.

## Results (H200)

`tests/models/deepseek_v4/test_dsv4_model.py` (7 tests) uses four layers of the real Flash
types (window, window, C4, C128; hash routing on the first three) with random weights. It
checks that:
- the final hidden state is byte-identical across batch / alone / chunked prefill / decode;
- loading from a sharded safetensors directory equals loading from a mapping;
- the RoPE caches are bit-equal to vLLM's (compressed layers) and to DeepSeek's
  `precompute_freqs_cis` (window layers);
- the spec equals the Flash headers;
- the config check fails closed.

**Against DeepSeek's inference code.** `scripts/check_dsv4_model_ref.py` runs both on a toy
checkpoint (4 layers, 256 experts, Flash shapes, random weights). This is a wiring check,
not a byte check.

With the same input at layer 0:

| Step | Rel L2 vs reference | Note |
|---|---|---|
| mHC pre + norm | 0 | byte-equal |
| post / comb | ~1e-6 | |
| attention | 4.2% | output magnitude equal to 0.05% |
| MoE | 2.5% | |

These gaps are the vLLM-style precision choices:
- the KV cache is FP8 (fp8_ds_mla);
- `o` is quantized to FP8 before `wo_a`, which the reference keeps in BF16;
- MXFP8 activations, against the reference's per-128 FP8.

End to end at T = 1024 and 2048, the stream error grows smoothly with no jump at any layer:

| Layer | Rel L2 | Cosine |
|---|---|---|
| 0 | 8.6% | 0.996 |
| 1 | 12% | 0.993 |
| 2 | 14% | 0.990 |
| 3 | 20% | 0.981 |

Using vLLM's YaRN on the window layers instead changes these numbers by about 0.2%. T
stays ≤ 2048 so that the indexer selects every compressed entry on both sides.

## Limitations

- Forward only; attention has no backward yet.
- No LM head (P7).
- The MoE block syncs with the host (the MXFP8 finite check; with EP, the split sizes), so
  a full decode step is not CUDA-graph capturable yet. The attention layer alone is.
