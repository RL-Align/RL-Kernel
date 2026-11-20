# Native DSV4 sparse decode on gfx942

The pinned AMD vLLM model selected fused paged decode plus inverse RoPE, while
native MCore used the fixed sparse prefill math and a BF16 boundary before
inverse RoPE. With identical layer-0 decode norm, projections and Q, the original
four-layer capture differed in 1,467/32,768 group-projection input elements.
The first difference was 1.8046875 (`0x3fe7`) versus 1.8125 (`0x3fe8`).

`bind_rollout_decode` binds only the real model instance's decode sparse call.
It reads the actual CSR/dense selection, hybrid paged FNUZ sliding cache and OCP
compressed cache, preserves the selected set and canonical chronological order,
and independently executes `NativeSparseAttention`. The original standalone
inverse RoPE executes once after the BF16 attention result. No actor tensors or
scores enter rollout computation. The original FP4/FP8 checkpoint is retained.
The scope is ordinary eager one-token decode, gfx942, TP=PP=EP=CP=1.

Warmup can expose zero-length CSR segments and sink-only rows. Such rows retain
all `-1` selection entries. A zero workspace row is never selected. Hybrid cache
allocation uses each layer's logical packed-page size with the allocator's
larger physical stride; it must not treat the physical stride as payload size.

The real MI300X two-response-token probe passes both generation and fixed
sequence replay with original FP32 byte comparisons (2 compared, 0 mismatches).
The first complete real TrainingWorker backward/optimizer step changes final
norm SHA256 from `93f667f4d9fad24cc6ea302b6b960e22e9f0f5763890cb5f7103819f240ccfe7`
to `78151302cfb422bcead059cc542763dbbf194369e62c745d112bf605c2f68e49`.
71 actual tensors are exported, loaded into the same live rollout PID, and its
norm digest matches. The updated fixed sequence also compares 2/2 tokens with
zero differing FP32 bytes. Expanded prompts/batches and an independent container
repeat remain separate acceptance gates.

The optional observer saves actual decode operands and a separate legacy output
for diagnosis. `test_native_decode_capture.py` replays the four layers, repeats
execution and duplicates rows to check the production sparse boundary. Set
`RLK_DSV4_TRACE_ROOT` to a real trace directory. GPU skips are not passes.
