# Reproduce and profile the Qwen3-8B TP4/CP2 experiment

This runbook records the commands used for the 200-step G10/G11 experiment and
the commands used to analyse its performance. Run every training replica under
a **new** run ID. The published G11 run
`g11-convergence-s1234-tp4-20260901e` is immutable and must never be submitted
again or overwritten.

The historical pair is suitable for consistency validation and descriptive
performance diagnosis. It is not a causal performance ablation: G10 and G11
used different RL-Kernel, VIME, and Transformer Engine revisions. A strict
operator-overhead claim requires a separately predeclared pair with identical
software, prompts, generated token lengths, and profiler settings.

## Frozen configuration

| Item | G10 | G11 |
|---|---|---|
| Run ID | `g10-convergence-s1234-tp4-20260901j` | `g11-convergence-s1234-tp4-20260901e` |
| RL-Kernel | `d2173e8d948e8cf062ac36be32bdf53bac75daa0` | `5403df6e3c5244343438916248ccfcc597dd96f6` |
| VIME | `1a113710e80aa7cfc271caa9bd90bcf348a7af08` | `a013293fb6dfdc5cd27152b54f64209ea2691d26` |
| Megatron-LM | `1dcf0dafa884ad52ffb243625717a3471643e087` | same |
| Transformer Engine | 2.18 local wheel | 2.11 local wheel |
| Attention backend | `fused` | `auto` |
| Attention / FFN / logp | P/P, P/P, native VIME | R/R, R/R, strict RL-Kernel |
| Framework logp | rollout logp reused | rollout logp reused |
| CUDA Graph | `FULL_DECODE_ONLY`, capture sizes 1 through 8 | same |

Both runs used Python 3.11.15, PyTorch 2.9.1, vLLM 0.16.0, Ray 2.57.0,
Megatron TP4/CP2, two TP4 vLLM engines, BF16, one prompt and eight samples per
step, a 7168-token response limit, GRPO, and the `deepscaler` rule reward. A
reference checkpoint was supplied through `--ref-load`; this does not imply a
non-zero KL penalty unless that penalty is enabled by the training config.

## Host paths and preflight

The following values reproduce the original host layout. Change them together
when using another machine.

```bash
set -euo pipefail

export EXPERIMENT_ROOT=/home/ellm/ljj/vime_qwen3_8b_tp4_cp2_200_experiment
export DATA_ROOT=/data/ellm/vime_qwen3_8b_tp4_cp2_200_experiment
export RLK_ROOT=$EXPERIMENT_ROOT/RL-Kernel
export VIME_ROOT=$EXPERIMENT_ROOT/vime
export MEGATRON_ROOT=/home/ellm/ljj/Megatron-LM
export EXAMPLE_ROOT=$RLK_ROOT/examples/vime_qwen3_8b_tp4_cp2_200
export RUNTIME_ROOT=/home/ellm/workspace/ljj/.conda/envs/rlk-attention-engines
export PYTHON=$RUNTIME_ROOT/bin/python3.11
export RAY=$RUNTIME_ROOT/bin/ray
export CUDA_RUNTIME_ROOT=$RUNTIME_ROOT/lib/python3.11/site-packages/nvidia/cuda_runtime
export TE218_ROOT=/home/ellm/ljj/.te-2.18.0-site-packages
export TE211_ROOT=/home/ellm/ljj/.te-2.11.0-site-packages
export RUNTIME_SITE=/home/ellm/ljj/.runtime-site-packages
export CUDA_PYTHON_SITE=/home/ellm/ljj/.cuda-python-12.8-site-packages
export HF_MODEL_ROOT=/home/ellm/ljj/checkpoints/Qwen3-8B_vime_rlkernel_tp2_cp2
export TORCH_DIST_ROOT=/home/ellm/ljj/checkpoints/Qwen3-8B_torch_dist
export PROMPT_DATA=$DATA_ROOT/datasets/dapo-math-17k.vime.jsonl
export RAY_API_SERVER_ADDRESS=http://127.0.0.1:8265

test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 8
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
df -h /dev/shm "$DATA_ROOT"
git -C "$RLK_ROOT" status --short
git -C "$VIME_ROOT" status --short
git -C "$MEGATRON_ROOT" status --short
"$PYTHON" - <<'PY'
import importlib.metadata as metadata
import platform

print("python", platform.python_version())
for package in ("torch", "vllm", "ray", "sympy", "pylatexenc"):
    print(package, metadata.version(package))
PY
```

Formal runs must start from clean repositories. Do not use `--allow-dirty` for
published evidence.

## Checkout or clone the sources

Clone once, then select the revisions listed above before launching each arm.
The commands below create the original directory layout without changing an
existing checkout.

```bash
test -d "$RLK_ROOT/.git" || git clone https://github.com/RL-Align/RL-Kernel.git "$RLK_ROOT"
test -d "$VIME_ROOT/.git" || git clone https://github.com/RL-Align/vime.git "$VIME_ROOT"
test -d "$MEGATRON_ROOT/.git" || git clone https://github.com/NVIDIA/Megatron-LM.git "$MEGATRON_ROOT"

git -C "$RLK_ROOT" fetch origin
git -C "$VIME_ROOT" fetch origin
git -C "$MEGATRON_ROOT" fetch origin
git -C "$MEGATRON_ROOT" checkout --detach 1dcf0dafa884ad52ffb243625717a3471643e087
```

## Prepare the prompt data and checkpoint

```bash
mkdir -p "$DATA_ROOT/datasets"
"$PYTHON" "$EXAMPLE_ROOT/prepare_dapo_data.py" \
  --download \
  --source "$DATA_ROOT/datasets/dapo-math-17k.parquet" \
  --output "$PROMPT_DATA"

test "$(sha256sum "$PROMPT_DATA" | cut -d ' ' -f 1)" = \
  73e2166517fd635e1157aff17202f86a5cced44ca1669e6f49d2d63a59bf509d
```

The actor and reference begin with the same Qwen3-8B weights. If the
`torch_dist` checkpoint is not already present, convert it with the model
arguments shipped by the selected VIME revision:

```bash
if [ ! -d "$TORCH_DIST_ROOT" ]; then
  cd "$VIME_ROOT"
  source scripts/models/qwen3-8B.sh
  PYTHONPATH="$MEGATRON_ROOT" "$PYTHON" tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$HF_MODEL_ROOT" \
    --save "$TORCH_DIST_ROOT"
fi
test -d "$HF_MODEL_ROOT"
test -d "$TORCH_DIST_ROOT"
```

## Start Ray

The Ray start invocation was not sealed in either run manifest. The following
single-node command matches the observed dashboard, 8-GPU resource count,
200 GB object store, and temporary directory. Record the command used on a new
host because changing the object store or spilling path can affect timing.

```bash
if ! "$RAY" status >/dev/null 2>&1; then
  "$RAY" start --head \
    --include-dashboard=true \
    --dashboard-host=127.0.0.1 \
    --dashboard-port=8265 \
    --num-gpus=8 \
    --object-store-memory=200000000000 \
    --temp-dir=/tmp/rlk-mxs-perf
fi
"$RAY" status
```

## Submit one exact arm

This function is the complete `run_arm.py` invocation used to construct the
Ray runtime environment and full `train.py` command. `run_arm.py` writes the
expanded `ray_command`, `train_command`, environment, revisions, hardware,
dataset hash, seeds, and topology to `manifest.json` before submission.

```bash
submit_arm() {
  local group=$1
  local run_id=$2
  local te_root=$3
  shift 3

  env -u PYTHONPATH \
    CUDNN_FRONTEND_CUDART_LIB_NAME="$CUDA_RUNTIME_ROOT/lib/libcudart.so.12" \
    "$PYTHON" "$EXAMPLE_ROOT/run_arm.py" \
      --group "$group" \
      --num-rollout 200 \
      --seed 1234 \
      --rollout-seed 1234 \
      --run-id "$run_id" \
      --output-root "$DATA_ROOT/runs/convergence" \
      --rl-kernel-root "$RLK_ROOT" \
      --vime-root "$VIME_ROOT" \
      --megatron-root "$MEGATRON_ROOT" \
      --model-root "$HF_MODEL_ROOT" \
      --ref-load "$TORCH_DIST_ROOT" \
      --prompt-data "$PROMPT_DATA" \
      --python "$PYTHON" \
      --ray-bin "$RAY" \
      --extra-pythonpath "$RUNTIME_SITE" \
      --extra-pythonpath "$CUDA_PYTHON_SITE" \
      --extra-pythonpath "$te_root" \
      --ld-library-path "$CUDA_RUNTIME_ROOT/lib:$te_root/transformer_engine/wheel_lib" \
      "$@"
}
```

Use a fresh replica ID for G10. These checkouts reproduce the final G10 stack:

```bash
git -C "$RLK_ROOT" checkout --detach d2173e8d948e8cf062ac36be32bdf53bac75daa0
git -C "$VIME_ROOT" checkout --detach 1a113710e80aa7cfc271caa9bd90bcf348a7af08
export G10_REPLICA_ID=g10-convergence-s1234-replica-$(date -u +%Y%m%d%H%M%S)
submit_arm G10 "$G10_REPLICA_ID" "$TE218_ROOT"
```

The following G11 command is intentionally `--dry-run`: it reconstructs and
records the historical command without launching a duplicate G11 job. Never
reuse the sealed ID. Removing `--dry-run` is only appropriate for a separately
approved replica with a new ID and output directory.

```bash
git -C "$RLK_ROOT" checkout --detach 5403df6e3c5244343438916248ccfcc597dd96f6
git -C "$VIME_ROOT" checkout --detach a013293fb6dfdc5cd27152b54f64209ea2691d26
export G11_AUDIT_ID=g11-convergence-s1234-audit-$(date -u +%Y%m%d%H%M%S)
submit_arm G11 "$G11_AUDIT_ID" "$TE211_ROOT" --dry-run
```

To audit the fully expanded command without executing it:

```bash
export AUDIT_MANIFEST=$DATA_ROOT/runs/convergence/$G11_AUDIT_ID/manifest.json
"$PYTHON" - "$AUDIT_MANIFEST" <<'PY'
import json
import shlex
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
print(shlex.join(manifest["ray_command"]))
print("\nTRAIN COMMAND\n", shlex.join(manifest["train_command"]))
PY
```

## Capture logs, status, and validation evidence

For a submitted run, wait for a terminal Ray state, then save logs and validate.
The validator seals only a successful run that passes backend, native/provider,
CUDA Graph, mismatch, fallback, traceback, and step-count gates.

```bash
export RUN_ID=$G10_REPLICA_ID
export SUBMISSION_ID=vime200-$RUN_ID
export RUN_DIR=$DATA_ROOT/runs/convergence/$RUN_ID

while ! "$RAY" job status --address="$RAY_API_SERVER_ADDRESS" "$SUBMISSION_ID" \
  | grep -Eq 'SUCCEEDED|FAILED|STOPPED'; do
  sleep 20
done
"$RAY" job status --address="$RAY_API_SERVER_ADDRESS" "$SUBMISSION_ID" \
  > "$RUN_DIR/ray-status.txt"
"$RAY" job logs --address="$RAY_API_SERVER_ADDRESS" "$SUBMISSION_ID" \
  > "$RUN_DIR/run.log"

grep -Ei 'actual_backend|linear-logp|native.*backend|cuda.graph|capture.size|mismatch|abs.diff|fallback|oom|spill|traceback' \
  "$RUN_DIR/run.log" > "$RUN_DIR/runtime-audit.txt" || true
"$PYTHON" "$EXAMPLE_ROOT/validate_run.py" --run-dir "$RUN_DIR" --seal
test -f "$RUN_DIR/COMPLETE"
```

## Collect learning and consistency results

```bash
"$PYTHON" "$EXAMPLE_ROOT/collect_results.py" \
  --runs-root "$DATA_ROOT/runs" \
  --output-dir "$DATA_ROOT/results"

"$PYTHON" "$EXAMPLE_ROOT/plot_results.py" \
  --rounds-csv "$DATA_ROOT/results/rounds.csv" \
  --phase convergence \
  --output-dir "$DATA_ROOT/results/figures"
```

## CUDA module ablation

The CUDA module matrix is part of this TP4/CP2 example. It is not a separate
CUDA directory: `run_supplement_suite.py` drives the eight `M000`-`M111` arms
through the same validated `run_arm.py` and `validate_run.py` path.

Run it only with a fresh suite ID and an empty output directory:

```bash
export MODULE_SUITE_ID=cuda-module-$(date -u +%Y%m%dT%H%M%SZ)
export MODULE_ROOT="$DATA_ROOT/runs/module/$MODULE_SUITE_ID"
mkdir -p "$MODULE_ROOT"

env -u PYTHONPATH \
  CUDNN_FRONTEND_CUDART_LIB_NAME="$CUDA_RUNTIME_ROOT/lib/libcudart.so.12" \
  "$PYTHON" "$EXAMPLE_ROOT/run_supplement_suite.py" \
    --phase module \
    --suite-id "$MODULE_SUITE_ID" \
    --output-root "$MODULE_ROOT" \
    --rl-kernel-root "$RLK_ROOT" \
    --vime-root "$VIME_ROOT" \
    --megatron-root "$MEGATRON_ROOT" \
    --model-root "$HF_MODEL_ROOT" \
    --ref-load "$TORCH_DIST_ROOT" \
    --prompt-data "$PROMPT_DATA" \
    --python "$PYTHON" \
    --ray-bin "$RAY" \
    --extra-pythonpath "$RUNTIME_SITE" \
    --extra-pythonpath "$CUDA_PYTHON_SITE" \
    --extra-pythonpath "$TE218_ROOT" \
    --ld-library-path "$CUDA_RUNTIME_ROOT/lib:$TE218_ROOT/transformer_engine/wheel_lib"
```

The resulting summary is `$MODULE_ROOT/$MODULE_SUITE_ID.summary.json`. The
matrix uses eight rollouts, one prompt with eight samples, seed 1234, and the
same TP4/CP2 validation gates as the long run. Do not copy the old TP2/CP2
commands; that example has been retired.

## ROCm entry points

ROCm is a separate backend and must not use the CUDA `run_arm.py` command: the
CUDA runbook requires `nvidia-smi`, CUDA Graph evidence, and Transformer Engine
libraries. The maintained ROCm launchers remain available, but their commands
are documented here so the repository has one command index:

```bash
# Full ROCm VIME operator matrix (P/P, P/R, R/P, R/R)
python examples/vime_qwen3_8b_rocm_ablation/run.py \
  --run \
  --output-dir /tmp/rocm-vime-ablation \
  -- bash /path/to/vime/scripts/run-qwen3-8B-rocm.sh

# Historical Attention-only cross-configuration diagnostic
python examples/vime_rocm_attention_ablation/run.py \
  --vime-root /work/vime \
  --rl-kernel-root /work/RL-Kernel \
  --megatron-root /work/Megatron-LM-vime \
  --model-root /app/model/Qwen3-8B \
  --reference-checkpoint /app/model/Qwen3-8B_torch_dist \
  --prompt-data /app/model/dapo-math-17k/dapo-math-17k.jsonl \
  --run-dir /work/RL-Kernel/runs/vime-rocm-attention-$(date -u +%Y%m%dT%H%M%SZ) \
  --run
```

The Attention-only runner was introduced earlier for ROCm route attribution
(PR #385) and fixes FFN/Logp while varying only the Attention P/R pairing. It
is not the CUDA/ROCm `M000`-`M111` module matrix and its results must not be
presented as the primary TP4/CP2 ablation. Keep it only when reproducing that
historical diagnostic; use the full-path matrix for current evidence.

## Performance analysis commands

The performance analyser reads the emitted per-step timers instead of rounded
progress-bar durations. It separates rollout generation, weight update,
wake/offload residual, actor training, and train residual; controls for response
or total token length with OLS; reports token-normalized rollout and actor
throughput; and bootstraps mean timing gaps with 20,000 draws.

Create a lightweight analysis environment if the training environment does not
contain NumPy and Matplotlib:

```bash
export PERF_VENV=$EXPERIMENT_ROOT/.perf-analysis-venv
test -x "$PERF_VENV/bin/python" || python3 -m venv "$PERF_VENV"
"$PERF_VENV/bin/python" -m pip install --upgrade pip
"$PERF_VENV/bin/python" -m pip install numpy==2.4.1 matplotlib==3.10.8
```

Run the sealed historical comparison without copying or editing either log:

```bash
export G10_RUN=$DATA_ROOT/runs/convergence/g10-convergence-s1234-tp4-20260901j
export G11_RUN=$DATA_ROOT/runs/convergence/g11-convergence-s1234-tp4-20260901e
export PERF_OUT=$DATA_ROOT/results/performance_g10_g11

"$PERF_VENV/bin/python" "$EXAMPLE_ROOT/analyze_performance.py" \
  --g10-log "$G10_RUN/run.log" \
  --g11-log "$G11_RUN/run.log" \
  --output-dir "$PERF_OUT"

column -s, -t < "$PERF_OUT/summary.csv" | less -S
"$PERF_VENV/bin/python" -m json.tool "$PERF_OUT/summary.json" \
  > "$PERF_OUT/summary.pretty.json"
```

Outputs are:

- `step-metrics.csv`: all parsed and derived metrics for every step and arm;
- `summary.csv` and `summary.json`: descriptive statistics, stage gap shares,
  length-control regressions, fully truncated and steady-step subsets, speedups,
  and bootstrap intervals;
- `performance-decomposition.{png,pdf}`: mean step-time stack;
- `length-controlled-scaling.{png,pdf}`: time versus token length and OLS fits;
- `stage-time-series.{png,pdf}`: raw and 10-step moving-average stage times;
- `token-normalized-throughput.{png,pdf}`: rollout and actor throughput.

For live diagnostics on a newly approved run, collect utilization and I/O in a
separate shell. These samplers do not alter the training command, but their
overhead and sampling interval must be identical across compared arms.

```bash
mkdir -p "$RUN_DIR/perf-monitor"
nvidia-smi dmon -s pucvmet -d 1 -o DT > "$RUN_DIR/perf-monitor/nvidia-smi-dmon.log" &
export GPU_MONITOR_PID=$!
iostat -xz 1 > "$RUN_DIR/perf-monitor/iostat.log" &
export IO_MONITOR_PID=$!

# After the Ray job reaches a terminal state:
kill "$GPU_MONITOR_PID" "$IO_MONITOR_PID"
wait "$GPU_MONITOR_PID" "$IO_MONITOR_PID" 2>/dev/null || true
"$RAY" memory --stats-only \
  > "$RUN_DIR/perf-monitor/ray-memory.txt" || true
```

Before interpreting a speedup, check response-length balance, truncation ratio,
CUDA Graph evidence for both engines, disk spilling, OOM/fallback/traceback
events, and the exact repository/TE revisions. Do not attribute the historical
G10/G11 timing gap to one operator because those controls are not matched.
