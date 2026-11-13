# Original DSV4 quantized checkpoint import into native MCore

`load_native_checkpoint` explicitly imports a single-file four-layer DSV4
checkpoint at TP=PP=EP=CP=1, before DDP and optimizer registration. It requires
every source tensor to be consumed. The MI300X run loaded all 6,281 tensors.

Original FP4 expert bytes and block32 E8M0 scales remain packed. The existing
staged expert training composition receives the explicit packed projection and
input-VJP provider. Original FP8 projection bytes/block128 scales also remain
resident. Their forward reuses AITER's production group128 FNUZ quantizer and
a fixed CK no-split-K GEMM; input VJP is explicit Triton. FNUZ weight preparation
maps negative zero and doubles E8M0 scales, preserving the original values.
No BF16 base-weight cache or TE FP4 capability override is used.

The native grouped `wo_a` call site is bound on the real model instance by
RLK's native adapter. No upstream source patch is required. The model retains
its native attention, compressor, indexer, router and expert structure. The P1
adapter supplies mHC, norm, final head and raw logprob with ROCm split mode.

The opt-in native verl engine enables fused latent QNorm-to-FP8. It preserves
the norm Parameter, uses the production AITER fused operator without a BF16
requantization boundary, and supplies the same codes/scales to the main and C4
indexer query projections. The optional main QKV codec calls the production
vLLM GPT-J QNorm/RoPE/cache encoder independently in the actor, gathers its
original FP8-DS-MLA KV representation and supplies explicit input VJPs. Rotary
tables are generated on GPU in FP32 and protected from model BF16 conversion.
Original checkpoint bytes/scales are unchanged; no actor output is copied to
rollout.

Quantized bases are frozen; original BF16/FP32 parameters stay trainable. This
is input-gradient training through quantized bases, not FP4 base-weight training.
The opt-in engine, version propagation and raw-score boundary live in RLK's
separate native verl adapter.

MI300X/gfx942 validation with the original four-layer checkpoint:

* Fused latent QNorm regression from real worker operands: FP8 codes/scales
  match the rollout bytes, while the former BF16 requantization fails; finite
  nonzero input/weight gradients and same-row batch-position checks pass.
* Main QKV replay against real rollout: 4,227,072 original BF16 query elements
  match exactly, with finite nonzero query/KV gradients.
* Real pristine-upstream TrainingWorker and independent rollout rerun confirms
  all 4,227,072 layer-0 query elements match. The next difference is attention
  output: 435,594 / 4,227,072 BF16 elements differ after inverse RoPE.
* Full strict raw FP32 score comparison remains 2 compared / 2 mismatched;
  the gate correctly prevents optimizer/synchronization acceptance afterward.

These fixes move the first difference; they do not establish four-layer parity.
The previous patched-image baseline completed an Adam update and export, but
that is separate from pristine-upstream post-update acceptance. See
`tests/backends/rocm/test_qnorm_fp8.py` and `test_qkv_preprocess.py`; set
`RLK_DSV4_CHECKPOINT` and `RLK_DSV4_TRACE_ROOT` to the original checkpoint and
retained real-worker trace. Missing GPU/fixtures are skips, never parity passes.
