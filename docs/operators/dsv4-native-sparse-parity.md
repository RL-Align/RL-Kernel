# Native DSV4 sparse attention arithmetic on MI300X

`bind_native_sparse(attention.core_attention)` retains the real native CSA
compressor, C4 indexer and selected sparse set. Only its unfused sparse math
boundary is bound on that model instance. `bind_rollout_sparse(attention)`
retains vLLM's real prefill scheduler, paged cache decoding and selected set.
The initial rollout adapter handles the pinned gather-prefill path; decode
still uses the native vLLM implementation.

Both independent engines execute the existing, hash-verified AITER DSV4
training forward: H32/K32 tiles, one split, BF16 probabilities and FP32 LSE.
Valid window positions precede compressed positions in ascending logical order,
with 640 fixed slots. Invalid slots remain masked. Packed weights, cache codec,
compression ratio and selected token sets are unchanged. The explicit backward
reuses RLK's existing AITER CSR/query gradients and ordered sink reduction.
The pinned current vLLM public gather ABI and AITER 0.1.29 delta file are accepted
alongside the previous audited signatures/hashes; no dependency is upgraded.

Original four-layer checkpoint on gfx942, TP=PP=EP=CP=1, eager:

* Before binding, layer-0 Q (4,227,072 BF16 elements), gathered KV (66,048),
  FP32 sink (64) and selected positions match, but 301,848 pre-inverse-RoPE
  attention output elements differ. First: rollout -0.2578125 (`be84`),
  actor -0.259765625 (`be85`).
* After binding, all 4,227,072 layer-0 pre-inverse-RoPE output elements match
  in the real verl actor and independent real rollout.
* Original-operand GPU forward repeat and nonzero finite Q/KV/sink VJP pass.
  This is operator coverage, not the required real-worker batch-position suite.
* Full raw score comparison still mismatches 2/2 tokens. Inverse RoPE and
  subsequent projections are separate boundaries. No four-layer parity claim.

Use the native/QKV companions and the actual-worker harness in
`examples/operators/dsv4_4l`. Checkpoint and traces remain external read-only
fixtures. A missing GPU or fixture is a skip, never a passing numeric result.
