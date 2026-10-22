# MiniMax-H3 Full Transformer and Dual Trajectory

## Summary

RFC #420 rows `ws1_full_h3_transformer` and `ws1_dual_trajectory`, validated on H100 (Hopper)
with the pinned checkpoint (`MiniMaxAI/MiniMax-H3@42ed227`, all 14 shards, sha256-verified,
638 tensors, 33,122,992,896 parameters, one GPU, BF16).

- `tools/validation/models/h3_full_transformer.py`: the full 33B forward and backward on a
  packed input through the RL-Kernel block graph (50 real blocks) **and** the boundary rows
  (`audio_input_projection_fp32`, `audio_output_projection_fp32`, `text_context_projection`,
  `token_refiner_block`); `proj_in`, `proj_out`, `time_embedder`, `norm_out` stay on the
  provider and the report says so (`boundary_bindings`).
- `tools/validation/models/h3_dual_trajectory.py`: a 4-step flow-matching Euler sampler
  (video + audio, FP32 latents) run in two layouts - the request alone, and as row 1 of a
  batch of 2 - sealed with the trajectory artifact and compared with `drift_report`.

## Result (H100, `reports/experiments/h3-full-transformer-h100/`)

| Check | Provider | RL-Kernel |
| --- | --- | --- |
| Full model bitwise repeatable (outputs, input grads, all 638 parameter grads) | yes | **yes** |
| Video / audio output vs provider | - | rel 2.1e-3 / 1.8e-2 |
| Same, provider with another SDPA kernel vs default | rel 1.9e-3 / 1.6e-2 | - |
| Dual trajectory single vs batched (4 steps, 16 tensors) | **16 / 16 differ**, first at step 0 (max abs 0.36) | 8 / 16 differ (max abs 2.9e-6) |
| Same, with `proj_in`/`proj_out` on the fixed-order GEMM (diagnostic) | - | **0 / 16 differ, final latents equal** |

Reading: the candidate stays inside the spread of independent provider kernel choices
(`provider_variants_vs_provider`). The residual strict drift is entirely the provider's
video `proj_in`/`proj_out` (cuBLAS FP32, row-count dependent); binding them to the same
fixed-order GEMM makes the full 33B sampler bitwise layout-invariant. Those two rows belong
to another contributor, so the diagnostic binding is labelled in the artifact metadata
(`diagnostic_video_projections`) and is not claimed as a row.

## Limits

- No FP32 full-model golden (132 GB of weights); agreement is against the provider and its
  kernel-choice spread. Per-block FP32 golden evidence is in the one-block and refiner rows.
- Attention, O/gate_up/down GEMMs, SwiGLU and QK norm are still `interim` bindings (see the
  operator trace); `time_embedder` and `norm_out` use the provider in this harness.
- Parameter-gradient fingerprints are GPU-side position-weighted checksums, not SHA-256.
- Sampler is the deterministic Euler ODE; the stochastic `cps` transition is a separate row.
