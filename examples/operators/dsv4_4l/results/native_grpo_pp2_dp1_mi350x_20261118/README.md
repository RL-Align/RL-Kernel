# Native four-layer GRPO on MI350X

One single-node run on **four MI350X (`gfx950`) GPUs** completed three continuous
native verl GRPO/PPO optimizer updates and passed the unchanged independent
`verify_grpo.py` verifier on 2026-11-18. This report supplements the
[MI300X baseline](../native_grpo_pp2_dp1_20261117/README.md); it does not resolve
that report's historical failure or standalone GPU setup errors.

## Scope and results

Actor uses physical GPUs 4–5 and rollout uses 6–7. Both use PP2/DP1,
TP1/EP1/CP1, original layer ownership 0–2 / 3, eager execution and
`rocm_triton/split`. The checkpoint, delivered data, training configuration,
optimizer settings and exact FP32 parity criteria are unchanged. Packed FP4/FP8
weights and scales remain frozen; selected nonquantized parameters are updated.
This is **not LoRA**.

The original [acceptance JSON](grpo-acceptance.json) records:

| Policy version | Completed updates | Expected / compared audit observations | Bit mismatches | Mean / max absolute difference | Invariance and sync |
| --- | ---: | ---: | ---: | --- | --- |
| v0 | 0 | 1,188 / 1,188 | 0 | 0 / 0 | PASS |
| v1 | 1 | 1,188 / 1,188 | 0 | 0 / 0 | PASS |
| v2 | 2 | 1,188 / 1,188 | 0 | 0 / 0 | PASS |
| v3 | 3 | 1,188 / 1,188 | 0 | 0 / 0 | PASS |

These are the independent fixed/fresh audit counts, not a sum including the
separate native training-batch comparisons. The verifier rechecks the retained
raw evidence. Both actor PP ranks recorded finite effective gradients,
optimizer-state continuity and changed parameters for each of the three steps.
The same rollout PIDs persisted across all four policy versions.

The final v3 stress phase completed **1,000 real requests**, with 14,869 expected
and compared token observations, zero bit mismatches and zero mean/max absolute
difference. Actor and rollout state readbacks remained unchanged. This phase
performed no additional optimizer updates.

The launched GRPO job exited zero after **594.495 seconds**, including startup,
audits and stress. The subsequent independent verifier also exited zero. This
timing is not a performance comparison. One completed run does not establish
independent-run training reproducibility or training benefit.

## Adaptation and provenance

See the [MI35x adaptation guide](../../../../../docs/usage/dsv4-mi35x-validation.md)
for the changes, official architecture references, build commands, targeted
tests and known limitations. The adaptations select OCP FP8 on gfx950, enable
the individually tested operator paths, and retain native BF16/unfused W4A16
MoE execution in the scoped rollout loading context. The latter requires the
accompanying [verl patch](../../patches/verl-mi35x-w4a16.patch).

The supplied vLLM and Transformer Engine libraries were built for gfx942.
Rebuilding their required libraries for gfx950 uses the supplied source and
dependencies. No vLLM or TE algorithm source was edited, no dependency was
upgraded, and no architecture override was used to disguise the GPU.

- [Provenance](provenance.json) records supplied/derived image IDs, runtime
  source identities, vLLM binary hashes, checkpoint and acceptance hashes.
- [Tested source hashes](tested-source-sha256.json) identify the changed runtime
  files as stored in the tested image, including their original line endings.
  RL-Kernel and verl are honestly recorded as patched/dirty trees.
- [TE binaries](te-binaries.json) records the rebuilt common/backend libraries,
  supplied source identity and focused test counts. Its 24,730 recorded source
  files remained unchanged during the rebuild.

The derived image tag is local to the validation host; it has not been published
to a registry. Full driver logs, raw comparisons, policy snapshots and failed
attempts remain in the retained validation workspace. The compact evidence
bundle's digest is recorded in `provenance.json`; the bundle is not a substitute
for the full raw run directory.

## Validation limits and preserved failures

567 targeted RL-Kernel cases passed, including CPU dispatch/loading-scope
checks. TE's upstream multi-tensor L2-norm matrix passed 288 cases. A separate
seven-step AdamW probe matched PyTorch's reference and verified finite optimizer
moments; multi-tensor gradient scaling matched exactly. These probes are
distinct from the three updates in the real GRPO run.

Earlier adaptation runs remain failed attempts: initialization guards in runs
001–002, initial routed-MoE parity in 003, and a gfx942-only TE gradient-norm
kernel in 004. Run 004 passed its initial v0 audit but completed no optimizer
update. The broader CPU-workflow test list in the ROCm image had four failures,
and the first TE optimizer-test collection lacked `expecttest`; the adaptation
guide records these without counting them as passes.

MI300X was not rerun for this patch. MI355X hardware, multi-node execution,
long-context/full-model training and arbitrary parallel layouts are unvalidated.
