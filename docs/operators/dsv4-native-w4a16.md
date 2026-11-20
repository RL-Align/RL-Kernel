# Preserve original W4A16 expert boundaries in native MCore

The pinned gfx942 rollout's actual original-MXFP4 path consumes BF16 activations.
The earlier native packed importer composed an MXFP8 activation quantizer instead:
identical 3,170,304 BF16 inputs became different values in 2,975,670 positions
before GEMM1. This is a contract difference, not an FP4 capability workaround.

`bind_native_w4a16(moe)` retains the real native router/dispatcher and references
its already-loaded original E2M1/E8M0 weight buffers without copies. It binds
fixed BF16-activation / packed-FP4 GEMMs with FP32 accumulation and the actual
BF16 projection boundaries. No dense weight cache is built. The existing
MXFP8xMXFP4 profile remains available separately.

The original routed activation uses vLLM's pinned `swiglu_limit_func`, including
its ROCm intermediate BF16 conversion and real autograd. It differs from the
shared expert's single fused C++ activation. Route probabilities multiply the
FP32 down-projection result immediately before its BF16 store. Explicit packed
weight input VJPs and probability gradients propagate into native training;
quantized bases remain frozen.

MI300X/gfx942, original four-layer checkpoint, eager TP=PP=EP=CP=1:

* Original active expert 4: gate/up and probability-weighted down each match
  524,288 captured rollout BF16 elements exactly, with finite nonzero input VJP.
* Actual routed activation regression matches 1,585,152 BF16 elements and has
  a nonzero finite backward. Substituting the shared fused activation fails.
* Real independent verl actor/rollout rerun matches all layer-0 routed gate/up
  (3,170,304), activation (1,585,152) and weighted down (3,170,304) BF16 elements.
* Final FFN output still differs in 142,413/528,384 elements: native combine
  remains a separate boundary. Full raw FP32 scores are still 2/2 mismatched.

Tests: `test_packed_w4a16_capture.py` (2 passed),
`test_native_w4a16_activation.py` (1 passed). Set original checkpoint and actual
worker trace fixture paths. This does not establish complete four-layer,
real-worker batch-position or post-update parity, or train packed base weights.
