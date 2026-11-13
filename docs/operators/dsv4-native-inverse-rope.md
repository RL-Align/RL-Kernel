# Preserve the fused inverse-RoPE conversion boundary

`bind_native_inverse_rope(attention)` binds only the loaded native instance's
inverse-RoPE call. It retains the native scheduler, parameters and grouped
projection. The optional original QKV adapter supplies the FP32 GPU-built
rotary table. The existing vLLM fused GPT-J kernel evaluates FP32 products and
writes BF16 once; the adapter supplies an explicit transpose VJP.

With the original checkpoint and fixed sparse-forward companion on MI300X:
the actual layer-0 pre-rotation attention outputs match, while the old native
inverse rotation differs in 164,724/4,227,072 BF16 elements. First differing tail
value: rollout 0.1220703125 (`3dfa`), actor 0.12109375 (`3df8`). The captured-input
regression fails for the retained old output and passes byte-for-byte with the
fused operator, with finite nonzero input gradients. The real dual-worker rerun
then matches all 4,227,072 layer-0 post-inverse-RoPE elements.

Scope: ordinary contiguous unpadded positions, eager CP=1, original BF16
residuals and FP32 rotary tables. Packed/CP/fused-native-RoPE paths fail closed.
Full raw scores still mismatch 2/2; the next projection boundary needs separate
validation. This does not establish four-layer bitwise parity or post-update
acceptance.
