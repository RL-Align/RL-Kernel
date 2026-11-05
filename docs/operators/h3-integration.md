# MiniMax-H3 Integration Rows (runtime modes, overlay, Diffusers adapter, training provider)

## Summary

RFC #420 integration rows, validated on H100 against the real verl-omni scheduler:

| Row | Where | What it fixes |
| --- | --- | --- |
| `h3_runtime_modes` | `models/minimax_h3/runtime.py` | `RL_KERNEL_MODE` strict / audit / auto / off -> bind the RL-Kernel graph or not; strict raises instead of falling back, audit reports, case id `R/R` or `P/P` |
| `h3_ablation_overlay` | same | `AblationOverlay.parse("node:attention,boundary:token_refiner")` keeps parts on provider arithmetic; the run is labelled `ablation(...)`, never `R/R`; unknown names are rejected |
| `diffusers_runtime_adapter_h3` | same (`H3DiffusersAdapter`) | binds `MiniMaxH3Transformer3DModel` inside `with adapter.active():` (forward **and** backward, so checkpoint recomputation sees the same operators) and reports bindings |
| `h3_training_provider` | same (`H3TrainingProvider`) | actor-side `step()` scope + per-sample CPS transition |
| `h3_lora_gradient_gate` | same | LoRA over a strict projection: A/B/dX bitwise repeatable, forward rows independent of batching, base frozen, FP64-close |
| `stochastic_transition_contract` | `models/minimax_h3/transition.py` | the `cps`, `noise_level=0.8` step, layout invariant |

## Stochastic transition contract

```text
std = sigma_prev * sin(0.8 * pi / 2)
mean = (x - sigma v) * (1 - sigma_prev) + (x + v (1 - sigma)) * sqrt(sigma_prev^2 - std^2)
x_prev = mean + std * eps        log_prob = mean over non-batch dims of -(x_prev - mean)^2
```

- elementwise FP32 in this order; **bitwise equal** to
  `FlowMatchSDEDiscreteScheduler.sample_previous_step(sde_type="cps")` (`prev_sample_mean`, tested
  against the verl-omni checkout);
- `eps = step_noise(seed, step, sample_id, modality, shape)`: one generator per sample (SHA-256 of
  the key), so noise never depends on batch order or composition;
- the log-prob mean is a fixed-order GEMM reduction per sample (the scheduler's `torch.mean` order is
  shape dependent; ours agrees to FP32 rounding and is batch invariant);
- `noise_level = 0` (non-selected steps) yields `x_prev == mean`, `log_prob == 0`; scoring a stored
  rollout (`prev_sample=`) reproduces the sampling log-prob bitwise.

Open item for the maintainer: the formulas above are the scheduler's current CPS law; if the
recipe changes the noise law, only `transition.py` and its test change.

## Evidence

`tests/models/minimax_h3/test_h3_runtime_integration.py` (runtime modes, overlay, adapter, transition, LoRA gate). The transition tests compare with the verl-omni scheduler when it is importable and are skipped otherwise.
