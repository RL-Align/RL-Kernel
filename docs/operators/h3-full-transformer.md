# MiniMax-H3 Full Transformer and Dual Trajectory

## Summary

RFC #420 rows `ws1_full_h3_transformer` and `ws1_dual_trajectory`, validated on H100 (Hopper)
with the pinned checkpoint (`MiniMaxAI/MiniMax-H3@42ed227`, all 14 shards, sha256-verified,
638 tensors, 33,122,992,896 parameters, one GPU, BF16).

- `tools/validation/models/h3_full_transformer.py`: the full 33B forward and backward on a
  packed input through the RL-Kernel block graph (50 real blocks) **and** the boundary rows
  (`video_input_projection_fp32`, `video_output_projection_fp32`, `audio_input_projection_fp32`,
  `audio_output_projection_fp32`, `text_context_projection`, `token_refiner_block`,
  `timestep_mlp_fp32` for `time_embedder`, `final_adaln_out` for `norm_out`); the report lists
  every binding (`boundary_bindings`). `--provider-boundaries` keeps all of them on the
  provider. `time_proj` has no parameters and stays provider arithmetic. `time_embedder` and
  `norm_out` must resolve to their CUDA ops: if the registry falls back to the PyTorch reference
  (for example an `_C` built without the H3 kernels), strict mode raises before binding anything.
- `tools/validation/models/h3_dual_trajectory.py`: a 4-step flow-matching Euler sampler
  (video + audio, FP32 latents) run in two layouts - the request alone, and as row 1 of a
  batch of 2 - sealed with the trajectory artifact and compared with `drift_report`.

## Result (H100, `reports/experiments/h3-full-transformer-h100/`)

The numbers below were measured before `proj_in`/`proj_out` were bound to the merged video
projection rows (#562, #646); the "diagnostic" row used the `h3_projection` GEMM instead. A
rerun of both tools with the bound rows replaces them and has not been done yet.

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
fixed-order GEMM makes the full 33B sampler bitwise layout-invariant. The harness now binds
those two modules to `video_input_projection_fp32` / `video_output_projection_fp32`.

## Limits

- No FP32 full-model golden (132 GB of weights); agreement is against the provider and its
  kernel-choice spread. Per-block FP32 golden evidence is in the one-block and refiner rows.
- Attention, O/gate_up/down GEMMs, SwiGLU and QK norm are still `interim` bindings (see the
  operator trace); `time_proj` uses the provider in this harness.
- Parameter-gradient fingerprints are GPU-side position-weighted checksums, not SHA-256.
- Sampler is the deterministic Euler ODE; the stochastic `cps` transition is a separate row.

## Hopper evidence chart

These outputs and figures are not part of this diff; they are on the fork's `pr_assets` branch (links in the PR description).
Three seeds of the single-vs-batched Euler control (`reports/experiments/h3-full-transformer-h100/dual_trajectory_seed{11,22,33}[_diag].json`, plus the original
`dual_trajectory.json`) and the strict-vs-provider comparison of the 33B forward/backward
(`full_transformer.json`).

* provider: the two paths differ at every step (max abs 0.25-0.54 on step 0).
* RL-Kernel strict: the first difference is 4.3e-6 to 4.8e-6 and comes from the video `proj_in`/`proj_out`, which were provider arithmetic in that run; the difference then grows through the sampler on some seeds.
* strict with the video projections on the `h3_projection` GEMM (the removed `--diag-video-projections` diagnostic): bitwise equal at every step on all three seeds.
* the wall-time panel includes warm-up; steady-state step times are a follow-up.
