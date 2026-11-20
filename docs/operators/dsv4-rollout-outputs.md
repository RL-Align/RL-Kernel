# DSV4 rollout vocabulary head and raw logprob binding

Call `bind_output_head(wrapper, worker)` on the real loaded vLLM DSV4 causal-LM
wrapper before warmup. The initial eager TP1 binding uses the original BF16 head
weight and the same production `TritonLMHeadOp.forward_fp32` used by the native
training adapter. Scaled/soft-capped logits and sharded/padded vocabulary heads
are rejected explicitly.

The worker's selected-score call uses
`selected_raw_logprobs(wrapper, logits, token_ids, weight_version=version)`.
Each returned column is a full-vocabulary raw score from `TritonLogpOp`, not a
sampling-truncated distribution. No temperature/top-k/top-p transform is applied.
The worker checks the requested version and records actual first execution paths
and original output dtype. Head, norm and raw scores now have owned RLK paths on
both engines; this does not align the remaining attention/MoE arithmetic.

The existing worker binder for the inner transformer still handles mHC head and
final norm with `backend="rocm", mode="split"`. No #662 code is used, and AITER
remains available to other operators.

MI300X validation includes original-vocabulary score/gradient accuracy and the
real verl-owned vLLM worker. Four-layer train/rollout scores remain mismatched;
no bitwise-parity claim is made by this integration.
