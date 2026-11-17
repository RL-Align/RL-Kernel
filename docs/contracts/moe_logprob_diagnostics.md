# MoE logprob diagnostics: task 63 foundation

This change adds an offline, CPU-safe diagnostic for issue 563 task 63. It
compares one captured candidate selected-logprob vector against a fixed capture
and an independent Python/FP64 recomputation. It records the first difference
with layer, logical token, slot, expert and rank coordinates.

The entry point is `rl_engine.validation.models.moe_logprob.compare_moe_logprobs`.
`MoELogprobCase` preserves bounded provenance for logits, target IDs, routing
weights, mapping IDs, checkpoint/model version and the declared probability
rule. Raw tensors remain artifacts referenced by SHA-256; they are not expanded
into JSON reports.

Identity and value failures are separate. A changed token or expert mapping is
reported as `invalid_identity` with `token_indices_mismatch` or
`expert_indices_mismatch`; a changed value is reported as `bit_mismatch` or
`accuracy_mismatch`. Stale mapping and parameter versions fail closed before
comparison. The candidate backend labels and rank are recorded as assertions;
this module does not independently attest them.

Cross-config case and attempt IDs use the artifact store's normalized single
path-component rule. Whitespace-padded, dotted, or slash-containing IDs are
rejected before they can disagree with the artifact directory identity.

The fixed capture is compared byte-for-byte. The independent oracle uses Python
`math.fsum` in FP64 and the shared WS1 logprob absolute threshold. `passed`
requires both `fixed_tree_passed` and `accuracy_passed`, which are exposed
separately so a report cannot hide which criterion failed. The probability rule
is deliberately versioned as `log_softmax_fp32_selected`; no claim is made that
CPU `torch.log_softmax` or accelerator fast-math is bit-identical across devices.
The independent oracle remains in shifted coordinates, `(logit - max) -
log(sum(exp(logit - max)))`, so large finite logit offsets do not erase the
normalization correction. Reference and candidate selected logprobs must be
finite; matching NaN/Inf bytes are never accepted.

Reports also expose `inputs_preserved`, `routing_weights_preserved`,
`mapping_run_id_verified`, `device_execution_verified`,
`producer_identity_verified` and `parameter_content_verified`. The first two
mean that the captured identities were compared and the latter three remain
false for this offline tool; backend labels and version strings are evidence
fields, not independent attestation.

This is an offline foundation. It does not capture verl, execute Train/Prefill/
Decode/Verify, run EP collectives, or qualify an Ascend backend. A later adapter
must bind real model hooks and persist complete rank artifacts before task 63's
full acceptance criteria can be claimed.
