# Preserve native DSV4 hash-routing probability arithmetic

The real native hash router and ROCm vLLM select the original same six expert
IDs and produce identical FP32 gate logits, but their score/normalization paths
round differently. On the retained layer-0 prefill, 598/774 selected probabilities
differ: first rollout 0.046862900257110596 (`3d3ff350`) versus native
0.046862777322530746 (`3d3ff32f`).

`bind_native_hash_route(router)` preserves the native gate Parameter, hash table,
router interface and dispatcher. It binds only the original Top6 sqrt-softplus
hash probability boundary to the same production vLLM hash kernel, executed
independently on actor operands. Original integer table entries are represented
losslessly in the kernel's int32 index ABI; model weight formats are unchanged.
An explicit normalized-score VJP propagates into the original FP32 gate logits.

MI300X/gfx942 validation with the original four-layer checkpoint:

* The captured actual logits/table/token regression matches the real rollout
  probabilities and IDs exactly. Nonzero finite logits VJP agrees with an
  independent FP64 autograd oracle (1 passed).
* Real verl actor rerun executes all three original hash layers. The layer-0
  774 selected probabilities now match exactly, as do the 33,024 gate logits.
* Full response raw FP32 scores still mismatch 2/2; the next routed-expert
  projection/activation/combine boundaries require separate validation.

Scope: Top6 hash layers, original finite FP32 logits, eager TP=PP=EP=CP=1,
no forced benchmark routing or replay override. Learned routing is separate.
No four-layer, real-worker batch-position or post-update parity claim is made.
