# MiniMax-H3 Trajectory Artifact

## Summary

`h3_trajectory_artifact` (RFC #420, foundation) is the sealed record that end-to-end claims
are made about: "native and strict first differ at step 3", "strict is bitwise identical
across runs". It stores digests, not tensors, so it is small and diffable; tensors can be kept
beside it when you need to know *how far* apart two runs are.

It builds on `h3_arch_fingerprint` (the artifact is bound to the fingerprint digest) and can
reference an operator trace digest (`h3_operator_trace`), so what ran and what it produced
travel together. The core is stdlib; torch is imported only to hash or load tensors.

## Contents

Per artifact: variant (`FL2VA` / `Ref2VA`), label (`native`, `strict`, ...), seed, steps,
modalities (`video`, `audio`), per-modality shift, exact FP32 sigma grids (stored as
`float.hex` of the FP32 value), per-modality digest of the initial noise (dtype, shape,
sha256), optional operator-trace digest and free metadata.

Per step: the sigma and next sigma used, the digest of each modality's velocity and of the
latent after the step, and `chain_sha256 = H(previous chain, step body)`. The record is sealed
by `artifact_sha256`.

Tensor digests are sha256 over dtype, shape and raw bytes (any device, any stride), so one
flipped bit, a reshape or a dtype change is a different digest.

## Entry Points

```python
from rl_engine.models.minimax_h3 import trajectory as tj

rec = tj.TrajectoryRecorder(variant="FL2VA", label="strict", seed=7,
                            sigma_grids={"video": grid_v, "audio": grid_a},
                            initial_noise={"video": z_v, "audio": z_a},
                            shifts={"video": 12.0, "audio": 3.0},
                            operator_trace_sha256=trace.digest, keep_tensors=True)
for i in range(steps):
    rec.record_step(i, velocity={"video": v_v, "audio": v_a}, latent={"video": x_v, "audio": x_a})
artifact = rec.seal();  rec.save_tensors("run.safetensors")
open("run.json", "w").write(tj.dumps(artifact))

tj.verify(artifact)                              # fail closed
tj.first_drift(a, b, tensors_lhs=pa, tensors_rhs=pb)   # {"step", "kind", "modality", "mismatch_count", "max_abs_diff", ...}
tj.drift_report(a, b)                            # every differing (step, kind, modality)
```

```bash
python tools/validation/models/h3_trajectory.py verify run.json
python tools/validation/models/h3_trajectory.py compare native.json strict.json --require-equal
```

## Guarantees

* `verify` rejects: a changed seal, an edited/dropped/reordered step (hash chain), a sigma
  that disagrees with the grid, a grid that is not non-increasing or leaves `(0, 1]`, a final
  latent that differs from the last step, a schema mismatch, and an artifact recorded
  against a different architecture fingerprint.
* A comparison first checks that the two runs were *given* the same variant, seed, step
  count, shifts, sigma grids and initial noise; otherwise it raises, because a drift
  between different inputs says nothing about the operators (`require_same_inputs=False`
  overrides, deliberately).
* Within a step the velocity is compared before the latent it feeds, and video before audio,
  so the first drift names the earliest observable cause.

## Tests

`tests/models/minimax_h3/test_h3_trajectory_artifact.py` (CPU): deterministic artifacts; the
seal/chain on every kind of edit including a re-sealed one; recorder contract (order,
completeness, modalities, grids); exact FP32 grids; digest sensitivity to bytes, dtype and
shape; first drift on a synthetic Euler trajectory (step, kind, modality, propagation to
later latents) and with kept tensors (mismatch count, max abs diff); incomparable inputs.

## Out Of Scope

It does not define the sigma schedule (`h3_sigma_schedule`), the Euler step (`h3_ode_step`)
or the stochastic transition (`stochastic_transition_contract`); it records what those
produced.
