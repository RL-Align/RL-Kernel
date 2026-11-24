# RL-Kernel Train–Rollout Consistency: A Practical Guide

In RL post-training, rollout generates responses and records a log probability
(LogP) for each token. Training recomputes those scores before updating the
weights. RL-Kernel checks whether the two independently computed scores have
identical bits when the weights and inputs match.

This guide runs a short Qwen3-8B check with vime, vLLM and Megatron-LM on
NVIDIA H100 or AMD MI300X. It compares training and rollout on each platform
separately. For the numerical background and reported results, see the
[technical article](2026-09-18-rl-kernel-amd-vime-qwen3-8b-train-rollout-bitwise-consistency.md).

## Requirements

Start inside a prepared GPU environment. Installing RL-Kernel alone does not
install the complete training stack. Reserve all eight GPUs for this experiment.

| | NVIDIA | AMD |
|---|---|---|
| Reference hardware | 8 × H100 80 GB, one node | 8 × MI300X, one node |
| Model | Qwen/Qwen3-8B | Qwen/Qwen3-8B |
| Training and rollout | vime, Megatron-LM and vLLM | Patched vime, Megatron-LM and vLLM |
| Environment setup | [CUDA requirements](../usage/qwen3-vime-consistency.md#supported-reference-configuration) | [ROCm requirements](../usage/qwen3-vime-consistency.md#rocm-mi300x-and-gfx942) and [companion builds](https://github.com/RL-Align/RL-Kernel/blob/main/rl_engine/integrations/orchestrators/vime/patches/rocm_attention/README.md) |

Use Bash and keep the following commands in the same shell. Put model files,
datasets, checkpoints and outputs on a data disk. Container users must use paths
visible inside the container.

## 1. Install RL-Kernel

```bash
git clone https://github.com/RL-Align/RL-Kernel.git
cd RL-Kernel

RL_KERNEL_REQUIRE_EXT=1 python -m pip install --no-deps --no-build-isolation -e .
python -c "import rl_engine._C as C; print(C.__file__)"
git rev-parse HEAD
```

The import should print the compiled extension path. Record the Git revision
with your results. Build prerequisites are in the [installation guide](../getting_started/installation.md).

Set two local paths. The model directory must contain the Hugging Face weights,
configuration and tokenizer.

```bash
export RLK_WORKSPACE=/data/rlk-guide
export RLK_MODEL=/data/models/Qwen3-8B
mkdir -p "$RLK_WORKSPACE"
```

If the model is not already available, download it once using huggingface\_hub:

```bash
python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-8B", local_dir=os.environ["RLK_MODEL"])
PY
```

## 2. Configure your environment

Choose one backend. For ROCm, the source paths must point to the patched and
built companion checkouts described above. The vLLM changes include compiled
code; changing a Python path alone is insufficient.

=== "NVIDIA H100 (CUDA)"

    ```bash
    export RLK_BACKEND=cuda
    export RLK_PROFILE_TEMPLATE=configs/integrations/orchestrators/vime/dense/qwen3-8b-tp4-cp2.json
    ```

=== "AMD MI300X (ROCm)"

    ```bash
    export RLK_BACKEND=rocm
    export RLK_PROFILE_TEMPLATE=configs/integrations/orchestrators/vime/rocm_attention/mi300x-qwen3-8b.json
    export RLK_VIME=/data/rocm-repro/vime
    export RLK_MEGATRON=/data/rocm-repro/Megatron-LM
    export RLK_VLLM=/data/rocm-repro/vllm
    ```

Create .rlk-profile.json once. It records your interpreter and local paths so the
launcher can reuse them. If you already have this file, check that
`requirements.backend` matches `RLK_BACKEND` and that its paths point to the
selected environment. If either
differs, move the old file aside and rerun the Python block with the selected
template. Skip that block only after these checks. It refuses to overwrite an
existing profile.
Unset RLK\_REPRO\_PROFILE to select this checkout's local file.

```bash
unset RLK_REPRO_PROFILE
python - <<'PY'
import json
import os
import sys
from pathlib import Path

profile = json.loads(Path(os.environ["RLK_PROFILE_TEMPLATE"]).read_text())
workspace = Path(os.environ["RLK_WORKSPACE"]).resolve()
profile["paths"].update({
    "workspace": str(workspace),
    "rl_kernel_root": str(Path.cwd()),
    "python": sys.executable,
    "model_root": str(Path(os.environ["RLK_MODEL"]).resolve()),
    "output_root": str(workspace / "runs"),
})
if os.environ["RLK_BACKEND"] == "rocm":
    profile["paths"].update({
        "vime_root": str(Path(os.environ["RLK_VIME"]).resolve()),
        "megatron_root": str(Path(os.environ["RLK_MEGATRON"]).resolve()),
        "vllm_root": str(Path(os.environ["RLK_VLLM"]).resolve()),
        "ref_load": str(workspace / "checkpoints/Qwen3-8B_torch_dist"),
        "prompt_data": str(workspace / "data/dapo-math-17k.vime.jsonl"),
    })
with Path(".rlk-profile.json").open("x") as handle:
    json.dump(profile, handle, indent=2)
PY
```

Finish preparation for the same backend:

=== "NVIDIA H100 (CUDA)"

    ```bash
    ./rlk prepare --download-data --convert-checkpoint

    ray start --head --num-gpus=8 \
      --include-dashboard=true --dashboard-host=127.0.0.1 \
      --dashboard-port=8265 --object-store-memory=200000000000
    export RAY_API_SERVER_ADDRESS=http://127.0.0.1:8265

    ./rlk doctor --mode consistency
    ```

    prepare selects the pinned vime and Megatron-LM sources, prepares the data and
    converts the model. Use a new workspace or clean companion checkouts.
    It does not install the full Python stack or download model weights.

    The Ray command reserves 200 GB for its object store. Adjust this for host
    RAM and /dev/shm. If you reuse your own Ray head, set
    `RAY_API_SERVER_ADDRESS` to its dashboard address. Continue when doctor
    reports Doctor: PASS. Extra runtime paths are covered in the
    [CUDA setup guide](../usage/qwen3-vime-consistency.md#prepare-once).

=== "AMD MI300X (ROCm)"

    ```bash
    mkdir -p "$RLK_WORKSPACE/data" "$RLK_WORKSPACE/checkpoints"
    python rl_engine/integrations/orchestrators/vime/experiments/dense/prepare_dapo_data.py \
      --download \
      --source "$RLK_WORKSPACE/data/dapo-math-17k.parquet" \
      --output "$RLK_WORKSPACE/data/dapo-math-17k.vime.jsonl"

    (
      set -euo pipefail
      source "$RLK_VIME/scripts/models/qwen3-8B.sh"
      PYTHONPATH="$RLK_MEGATRON${PYTHONPATH:+:$PYTHONPATH}" \
        python "$RLK_VIME/tools/convert_hf_to_torch_dist.py" \
          "${MODEL_ARGS[@]}" --hf-checkpoint "$RLK_MODEL" \
          --save "$RLK_WORKSPACE/checkpoints/Qwen3-8B_torch_dist"
    )
    ```

    Run conversion once, with a new destination. If the data and checkpoint
    already exist, set prompt\_data and ref\_load in your profile to those
    locations and skip this block.

    ROCm validates inputs during run and manages its own Ray instance. Keep
    the profile's Ray ports free. The CUDA prepare and doctor commands do not
    support this ROCm profile; standalone validate, report and verify are also
    CUDA-only. For the first check, keep rollout CP at 1, temperature positive
    and top-k disabled. See the
    [CUDA and ROCm support audit](../usage/cuda-rocm-consistency-audit.md)
    for supported settings.

## 3. Preview and run

Use the README Quick Start settings. This guide changes only `--steps` to 3
for the short check and uses a run ID to locate its report. Consistency mode is
the default. The Bash array keeps the same settings for the native comparison.

```bash
RUN_ARGS=(
  --tp 4 --rollout-tp 4
  --temperature 0.7 --top-p 0.95
  --lr 5e-7 --kl-coef 0.01
  --max-response-len 6912 --max-tokens-per-gpu 4096
)

./rlk plan "${RUN_ARGS[@]}" --steps 3
```

Check the resolved paths and runner\_command in the printed JSON. plan does
not start a GPU job. TP and CP describe how the eight GPUs share model and
sequence work. This example uses training TP4 with CP2 and rollout TP4 with
CP1. Keep these settings for the first check.

Run three steps with a unique output name:

```bash
export RLK_RUN_ID="consistency-$(date -u +%Y%m%dT%H%M%SZ)"
./rlk plan "${RUN_ARGS[@]}" --steps 3 --run-id "$RLK_RUN_ID" \
  > "$RLK_WORKSPACE/$RLK_RUN_ID.plan.json"
./rlk run "${RUN_ARGS[@]}" --steps 3 --run-id "$RLK_RUN_ID"
```

run waits for completion and validates the results. Model loading, compilation
and warmup can take time. On ROCm, watch arms/r-r/launcher.log in the run directory
for progress if the main terminal is quiet.

## 4. Check the result

Read the report from the output location resolved in the saved plan. This also
handles existing profiles that use a default output directory. Keep the same
profile and environment between plan and run.

```bash
python - <<'PY'
import json
import os
from pathlib import Path

run_id = os.environ["RLK_RUN_ID"]
plan = json.loads((Path(os.environ["RLK_WORKSPACE"]) / f"{run_id}.plan.json").read_text())
run_dir = Path(plan["paths"]["output_root"]) / run_id
reports = [run_dir / name for name in ("run-validation.json", "single-arm-summary.json")
           if (run_dir / name).is_file()]
if len(reports) != 1:
    raise SystemExit(f"Expected one validation report in {run_dir}; check the run log.")
report = json.loads(reports[0].read_text())
print(f"Run directory: {run_dir}")
print(json.dumps(report, indent=2))
if report.get("passed") is not True:
    raise SystemExit("Validation failed: inspect the report and run log.")
PY
```

The top-level passed field must be true. A completed training run alone does
not prove consistency. Review the complete validation report.

| Backend | Consistency evidence | Report and log |
|---|---|---|
| CUDA | runtime\_scalar\_logprob.bitwise\_zero is true | run-validation.json; run.log |
| ROCm | metrics.bitwise\_equal is true; metrics.bitwise\_mismatch\_count is 0; frozen\_sources\_match is true | single-arm-summary.json; arms/r-r/validation.json; arms/r-r/launcher.log |

The reports also check execution evidence and source records. CUDA and ROCm use
different evidence formats, so retain the original reports and the machine profile.
If the report is missing or passed is false, check the run log and report errors.
Keep the profile and source revisions when reporting a failure.

## 5. Compare with native execution

After the consistency run finishes, run the native baseline sequentially on the
same node. Keep the settings fixed and change the mode and output name:

```bash
./rlk run --mode native "${RUN_ARGS[@]}" --steps 3 \
  --run-id "native-$(date -u +%Y%m%dT%H%M%SZ)"
```

Native mode uses vime's native operators. Its validator can pass even when
training and rollout scores differ; zero mismatch is a consistency-mode
requirement. The ROCm native log is arms/p-p/launcher.log.

A short pair is useful for checking the workflow. It is not a performance
benchmark: warmup, response lengths and generated trajectories affect timing.
See the [ROCm performance report](../usage/rocm-sparse-performance.md) for measured
results, historical 200-step evidence and the native execution-record limitation.

For a longer consistency run, reuse the same settings:

```bash
./rlk run "${RUN_ARGS[@]}" --steps 200
```

## Current model work and next integrations

Qwen3-8B on CUDA and ROCm is our starting point. We are extending the same
training and rollout consistency checks across models, RL frameworks and hardware:

- Models in progress: [DeepSeek-V4 Flash MoE](2026-08-09-dsv4-flash-moe-consistency-roadmap.md),
  [MiniMax-H3](https://github.com/RL-Align/RL-Kernel/issues/420),
  Qwen3-Next and Muse-Glimmer.
- Next framework integrations: AReaL, verl, verl-omni, miles and
  [Alibaba ROLL](https://github.com/alibaba/ROLL). Each integration needs a
  reproducible training and rollout run and a clear numerical comparison.
- Hardware vendor adaptation: MetaX, T-Head Zhenwu and NVIDIA for broader
  model and backend coverage. MUSA work in the miles roadmap and
  [Ascend with verl](https://github.com/RL-Align/RL-Kernel/issues/563)
  remain separate ongoing tracks.

These are ongoing or planned projects, not claims of completed support. We will
publish supported configurations and results after validation.

## Contribute

Start by reproducing this guide and sharing the hardware, source revisions and
validation report. You can also improve a confusing step, report a small
reproducible failure or help with one of the integration tracks above.

- [Contribution guide](../contributing/contributor-guide.md)
- [Detailed setup and supported configurations](../usage/qwen3-vime-consistency.md)
- [H100 validation results](../usage/h100-matrix-validation.md)
