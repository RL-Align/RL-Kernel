# Vime Qwen3-8B ROCm Attention ablation

The canonical launch command is maintained in
[`../vime_qwen3_8b_tp4_cp2_200/REPRODUCTION.md`](../vime_qwen3_8b_tp4_cp2_200/REPRODUCTION.md#rocm-entry-points).
This file documents the ROCm-specific acceptance contract; keep host paths and
commands in the shared runbook.

This is the ROCm end-to-end counterpart of PR230's production/RL-Kernel
operator matrix. It launches the real Vime orchestration once per Attention
cell and requires runtime evidence from both sides:

| Case | Megatron training | vLLM rollout |
|---|---|---|
| `P/P` | framework-native | framework-native |
| `P/R` | framework-native | RL-Kernel AITER/CK |
| `R/P` | RL-Kernel AITER/CK | framework-native |
| `R/R` | RL-Kernel AITER/CK | RL-Kernel AITER/CK |

FFN and Logp remain fixed at `P/P`, so only the Attention implementation
changes. Each cell starts in a fresh process and inherits the same model,
checkpoint, prompt data, seeds, token limits, and one-rollout pre-update state.

This is not an operator microbenchmark. The subprocess must run both vLLM
rollout and Megatron training. A return code of zero is insufficient: the
runner fails the cell if either framework emitted no executed Attention
readback, selected the wrong P/R route, reported fallback, or failed to prove
the strict ROCm runtime on an R side.

## Required host state

- A ROCm PyTorch build with visible AMD GPUs.
- AITER with `aiter.ops.mha.mha_fwd` and `mha_bwd` available.
- Vime, Megatron-LM, vLLM and RL-Kernel importable by every Ray worker.
- A frozen Qwen3 model/checkpoint and prompt file.
- A Vime launcher that honors `RL_KERNEL_ABLATION_OUTPUT_DIR` for case-local
  output, so one cell cannot update the input checkpoint used by the next.
- Megatron startup wired through
  `rl_engine.integrations.megatron_runtime.initialize_from_environment`, so
  the training worker installs the selected P/R plan and emits its readback.

The executable run requires these immutable input variables:

```bash
export MODEL_ROOT=/models/Qwen3-8B
export TORCH_DIST_ROOT=/models/Qwen3-8B_torch_dist
export VIME_CKPT=/checkpoints/qwen3-8b-pre-update
export PROMPT_DATA=/data/dapo-math-17k.jsonl
export NUM_ROLLOUT=1
export TRAIN_SEED=1234
export ROLLOUT_SEED=42
```

The parent launcher must propagate the P/R and readback variables into Ray
workers. `rocm_python_entrypoint.sh` is provided for launchers that replace
their Python executable. Point `RL_KERNEL_REAL_PYTHON` at the real interpreter
and configure Vime to invoke this wrapper.

## Run contract

Use the shared runbook for the dry-run and full-matrix commands. `--case R/R`
is repeatable for debugging a subset; the final acceptance run should execute
all four cells.

## Evidence and pass boundary

Each case directory contains the combined process log and the unmodified JSON
readbacks emitted by `FrameworkOperatorIntegration`. The aggregate is a human-
readable `summary.md`; no generated result JSON is checked into the repository.

For each R side, accepted evidence includes:

- semantic backend `rlkernel.attention.deterministic.v1`;
- `runtime_platform=rocm`;
- actual runtime `rlkernel.rocm.attention.aiter_ck_ag_rs.v1`;
- AITER/CK fixed no-Split-KV schedule;
- no native, PyTorch-reference, Triton, or fallback execution.

The rollout route consumes vLLM's paged cache, reconstructs logical KV order,
and invokes the same strict AITER/CK core as the training route. The training
route binds the Megatron CP process group to the RCCL AG/RS transport and
preserves the explicit global position order used by the PR230 contract.
