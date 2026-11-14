# Native dispatcher binding for fixed Top6 output combine

`bind_native_combine(moe)` retains native dispatch, all-to-all and shared-expert
scheduling. On this loaded dispatcher only, its ordinary unpermute boundary
uses the existing RLK P6 operator: ascending expert-ID slots 0–5, FP32 adds and
one final BF16 conversion. Probabilities already applied in the expert down
projection are not reapplied. P1 still owns residual/mHC post.

The adapter validates the actual expert-major routing map against the native
sorted row identities and prepares the existing versioned P6 mapping from
those tensors. It preserves autograd through the existing P6 gather VJP.
Fused, padded, alternate inverse-map and EP-invariant paths fail closed.

MI300X/gfx942 with original four-layer W4A16 weights, eager TP=PP=EP=CP=1:
all layer-0 per-expert projection/activation outputs match, while the former
native BF16 scatter combine gives 142,413/528,384 different final FFN values.
First: rollout 10.125 (`4122`), actor 10.1875 (`4123`). Actual-output regression
matches the real rollout, rejects the old native unpermute result, repeats five
times with identical bytes and has nonzero finite input VJP (1 passed).

Real independent actor/rollout rerun then matches all checked attention and FFN
boundaries of layers 0 and 1 for the retained 129-position prefill. The next
confirmed difference is layer-2 compressed KV. Full raw FP32 scores still
mismatch 2/2; no four-layer, real-worker batch-position or post-update claim.
