# Original FP8 grouped output projection on gfx942

The pinned vLLM AMD model prepares fused FP8 `wo_a` only on gfx950. Its gfx942
fallback decodes the checkpoint weight to a BF16 cache and evaluates einsum.
`bind_original_group_projection(attention, checkpoint, layer)` instead imports
the original E4M3/E8M0 block128 weight into `CheckpointFP8Group`, preserving
packed bytes/scales and the fixed AITER group128 FNUZ / CK no-split-K profile.
The same binding also imports the original `wo_b` (all rows at TP1, H/TP rows
with full K at TP>1) and, through `bind_original_input_projection`, replaces the
replicated fused `wq_a|wkv` GEMM with the actor's two original projections.
vLLM runs both GEMMs through AITER bpreshuffle, which picks a kernel from tuned
`(gfx, cu_num, M, N, K)` rows. MI300X (304 CUs) has no rows for these shapes and
happened to match the actor; MI308X (80 CUs) has rows that switch between
triton, CK, asm and opus kernels by M. Bind after loading and before warmup on
the actual attention instance. No framework source patch is needed.

Inverse RoPE remains exactly once in `forward_mqa`: the gfx950-only native
fused path stays unset and this grouped projection consumes already unrotated
activations. The module does not create a BF16 base-weight cache or change the
checkpoint quantization. Requires the original packed projection companion #688.

MI300X/gfx942 original-checkpoint evidence, ordinary eager TP=PP=EP=CP=1:

* Identical layer-0 grouped inputs previously gave 1,005,132/1,056,768 different
  BF16 outputs. First: rollout -0.216796875 (`be5e`) versus actor -0.2158203125
  (`be5d`).
* Captured-input GPU regression with original weight/scale bytes matches the
  actual actor result exactly and rejects the retained BF16 fallback output;
  input VJP is finite and nonzero (1 passed).
* Real independent four-layer actor/rollout rerun matches all 1,056,768 layer-0
  `wo_a` outputs, 528,384 `wo_b` outputs and 528,384 FFN input-norm elements.
* Shared FFN gate/up, activation, down projection and output also match.
  The next difference is routed FFN; full raw scores remain 2/2 mismatched.

These results validate these boundaries only. They do not establish full
four-layer parity, real-worker batch-position coverage or post-update acceptance.
