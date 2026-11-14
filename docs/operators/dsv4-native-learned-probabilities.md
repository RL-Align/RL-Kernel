# Preserve learned Top6 probability arithmetic

The native and rollout learned routers produce identical FP32 logits and select
the same expert set, but the score/normalization arithmetic rounds differently.
On the original layer-3 prefill, 721/774 probabilities differ: first rollout
0.30137887597084045 (`3e9a4e55`) versus native 0.30137908458709717 (`3e9a4e5c`).

`bind_native_learned_route(router)` adapts only this instance's existing Top6
score call. Native gate Parameters, selection bias, auxiliary-loss, masking and
dispatch code remain in use. The pinned production sqrt-softplus kernel executes
independently on actor operands, with a selected-score normalized-probability
VJP. Selection bias remains non-gradient metadata.

MI300X/gfx942 original-checkpoint GPU regression matches actual rollout weights
and IDs exactly, with nonzero finite logits gradients checked against independent
FP64 autograd (1 passed). Real TrainingWorker rerun matches all 774 layer-3
probabilities. The first response raw FP32 score then matches exactly; the second
response (decode) still differs. This is not four-layer acceptance.

Initial scope: eager TP=PP=EP=CP=1, ungrouped original sqrt-softplus Top6, no router
replay/fusion override. Hash routing uses its separate instance adapter. Complete
batch-position and post-update tests remain separate gates.
