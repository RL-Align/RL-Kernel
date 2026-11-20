# Native original compressed-KV codec for C4 and C128

`bind_native_compressor(compressor, preprocess)` retains the real native main
compressor Parameters and its C4/C128 sparse configuration. The loaded instance
uses the pinned vLLM production state writer and fused pooling/norm/GPT-J RoPE/
OCP FP8 cache encoder, then its real decoder. C128 uses the actual two-stage
prefill implementation. Buffers and typed production metadata belong to this
actor execution; no captured rollout values enter the runtime.

The native VJP recomputes only local FP32 pooling, norm and rotation, using an
explicit identity STE for the quantized cache boundary. Gradients reach the
original projected KV/score, APE and norm Parameters. No model reference forward,
Megatron-Lite dependency, dense weight substitute or upstream source patch is
used. The FP32 projection adapter and original FP32 GPU rotary table are separate
companions. The indexer compressor remains a separate boundary.

MI300X/gfx942 original four-layer checkpoint, eager TP=PP=EP=CP=1:

* Earlier native compressed KV differs in 14,324/16,384 C4 BF16 values even when
  main Q/recent KV match. Original projected operands are FP32 in rollout.
* Original-capture C4 regression matches all 16,384 BF16 values; C128 matches
  all 512. All four input/APE/norm gradients are finite and nonzero (2 passed).
* Real TrainingWorker rerun executes C4 and C128 production bindings. With the
  learned-router companion, the first generated response token's FP32 raw score
  matches exactly. Of the two response tokens, the decode score still differs
  (1 mismatch); no full four-layer or post-update acceptance is claimed.

Scope: full ordinary unpadded teacher forcing, original head dimension 512,
C4 overlap and C128 non-overlap. Prefixes shorter than the ratio preserve the
native no-compressed-token branch. Eager only; no Graph/persistent-state API is
added. Real-worker batch-position and post-update coverage remain separate gates.
