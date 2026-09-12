# VIME Qwen3-8B TP4/CP2 200-round consistency experiment

This example measures train/rollout numerical consistency at two independent
layers: VIME's framework-level reuse of rollout log-probabilities and
RL-Kernel's operator-level alignment of Attention, dense FFN, and linear logp.
It is designed for one 8×H100 node. Megatron uses TP4/CP2 across all eight
GPUs; two TP4 vLLM engines share those GPUs through VIME colocated offload.

The optimization algorithm is explicitly fixed to GRPO with
`--advantage-estimator grpo`. DAPO-Math-17k is only the prompt/answer dataset;
it does not select the DAPO training algorithm. The rule reward is computed by
VIME's `deepscaler` reward implementation.

The experiment is fail-closed. A run is accepted only when its Ray job
succeeds, every expected operator route has runtime execution evidence, no
fallback or Triton route is observed for an R/R arm, the requested number of
steps is present, and vLLM CUDA Graph evidence matches the manifest.

All launch commands are maintained in [`REPRODUCTION.md`](REPRODUCTION.md).
This README describes the experiment and its acceptance boundary; it does not
duplicate host-specific launch commands.

## Ablation matrix

| Group | VIME `--use-rollout-logprobs` | Attention / FFN / logp | Purpose |
|---|---:|---|---|
| G00 | off | P/P | Production baseline |
| G10 | on | P/P | Framework-level consistency only |
| G01 | off | R/R | RL-Kernel operator-level consistency only |
| G11 | on | R/R | Framework-level plus operator-level consistency |

`P/P` selects the production implementation on training and rollout. For
Megatron linear logp this means that no external provider is configured and
VIME calls its native `calculate_log_probs_and_entropy` implementation
directly. `R/R` selects RL-Kernel on both sides and installs the strict
RL-Kernel linear-logp provider. All four groups use the same prompts, initial
checkpoint, sampling settings, seeds, TP4/CP2 topology, and batch sizes.

The CUDA module ablation is the `M000`-`M111` matrix defined by the same
`run_arm.py` and `experiment_matrix.json`; it is intentionally kept under this
TP4/CP2 example rather than duplicated in a CUDA-only directory. The canonical
command is in [`REPRODUCTION.md`](REPRODUCTION.md#cuda-module-ablation).

The ROCm P/R and Attention-only runners are separate because they select HIP,
AITER/CK, and RCCL-specific routes that cannot pass the CUDA validation gates.
The Attention-only runner is a historical attribution diagnostic, not a second
definition of the module matrix.

Do not interpret G10/G11 as evidence that train and rollout recomputation is
bitwise equal: framework reuse changes which stored logp enters the RL loss.
The direct numerical claim comes from G01/G11 and the runtime comparison
metrics.

## Required gates

- NVIDIA H100 × 8; colocated actor/rollout GPUs 8; actor TP=4, CP=2, PP=1;
  two rollout engines with TP=4 each.
- Keep the TP4 Megatron actor resident and offload rollout during training.
  This avoids remapping live NCCL parameter buffers while still fitting Qwen3-8B
  on each 80GB H100.
- Pin Megatron's production attention backend to Transformer Engine `fused` for
  CP2/P2P. Backend auto-selection is host-dependent and would make G00/G10
  incomparable across environments. On hosts exposing multiple CUDA runtime
  majors, use Transformer Engine 2.18 or newer and select the CUDA 12 runtime
  explicitly with `CUDNN_FRONTEND_CUDART_LIB_NAME`.
- GRPO, BF16, `top_p=1.0`, temperature 1, no dropout, fixed training and rollout seeds.
- A 7168-token response budget, one prompt with eight GRPO samples per step,
  and full uniform activation recomputation
  (`recompute-num-layers=1`). Do not enable expandable CUDA allocator segments:
  deterministic TP collectives require CUDA IPC-capable staging allocations.
- vLLM CUDA Graph mode `FULL_DECODE_ONLY`, not eager, with exact capture sizes
  `1..(rollout_batch_size × n_samples_per_prompt)`.
- Megatron and vLLM integration readbacks with positive call counts for every
  configured route. A production Megatron logp route instead requires VIME's
  native-backend runtime marker and rejects any provider hook or provider
  readback.
- Production routes reject provenance whose actual backend is RL-Kernel, even
  if an outer integration layer labeled the call as production.
- R/R runs must report zero bitwise mismatches, zero max absolute logp
  difference, CUDA execution, and no fallback or Triton provenance.
- Append-only run directories. A passing validator creates `COMPLETE`; failed
  attempts remain available for audit and are not overwritten.

The current VIME debug dump does not include training `log_probs` in
`rollout_data`. `validate_run.py` therefore uses VIME's runtime `torch.ne`,
maximum, and mean absolute-difference metrics. Counts are reconstructed from
the sample means and global batch size. The report marks offline tensor
comparison as unavailable instead of claiming it was performed.

## Recommended phases

The phase definitions are frozen in `experiment_matrix.json`.

| Phase | Steps | Seeds | Decision |
|---|---:|---|---|
| short | 8 | 1234 | Catch state transition, weight-update, and cache issues |
| precision | 30 | 1234, 2345, 3456 | Estimate drift distribution before the long run |
| convergence | 200 | 1234 | Primary PR evidence and learning/performance curves |

Use the paired 200-step runs for the main claim. A bitwise invariant does not
need seed averaging; the three paired 30-step seeds test repeatability and
provide uncertainty estimates for reward, throughput, and overhead. Run groups
in the same seed order and compare paired seeds; report the mean and a 95%
confidence interval. Never merge runs from different code revisions,
checkpoints, prompt hashes, or CUDA Graph settings in one estimate.

## Prepare DAPO-Math-17k

`prepare_dapo_data.py` downloads or converts the official Parquet file and
emits VIME `prompt`/`label` JSONL. It deduplicates by `extra_info.index` and
writes source/output hashes and row counts to a sibling manifest.

```bash
python examples/vime_qwen3_8b_tp4_cp2_200/prepare_dapo_data.py \
  --download \
  --source /data/dapo-math-17k.parquet \
  --output /data/dapo-math-17k.vime.jsonl
```

The converter requires `pyarrow`. The small
`qwen3_8b_multiround_math.jsonl` file is a developer fixture and must not be
used for experiment or reward claims.

## Reproduction

Use [`REPRODUCTION.md`](REPRODUCTION.md) for the complete host setup, data and
checkpoint preparation, CUDA 200-step launch, CUDA module matrix, ROCm
entrypoints, Ray log capture, validation, and performance analysis commands.
The runbook is the single source of truth for commands and paths.

The plotting step requires Matplotlib. It produces:

- `consistency.png`: mean/max absolute logp difference, mismatch rate, and
  mismatch count per step;
- `learning.png`: raw reward with moving average, PPO KL, entropy, and response
  truncation ratio;
- `optimization.png`: GRPO policy-gradient loss, clipped ratio fraction, PPO
  KL, and gradient norm;
- `performance.png`: end-to-end step time, rollout time, and actor throughput.

The summary table also reports total active-token exposure, cumulative
bitwise mismatch count, token-weighted mean absolute difference, maximum
absolute difference, reward, truncation, step time, and throughput. For R/R,
the strongest claim is `mismatch_count = 0` over the stated token exposure;
reward and speed are secondary quality and cost measurements.

## Published convergence results

The sealed 200-step G10/G11 results, per-step data, reproducible plotting
script, and consistency figures are published in
[`results/convergence_s1234_g10_g11`](results/convergence_s1234_g10_g11/README.md).
G00 and G01 were paused, so the publication is explicitly a two-arm interim
result rather than a completed four-arm ablation. Performance is omitted
because the immutable G11 and final G10 runs used different repository and
Transformer Engine revisions.

The diagnostic (non-causal) stage-timing comparison can be regenerated from
the two sealed `run.log` files with
[`analyze_performance.py`](analyze_performance.py), following the commands in
[`REPRODUCTION.md`](REPRODUCTION.md#performance-analysis-commands).
