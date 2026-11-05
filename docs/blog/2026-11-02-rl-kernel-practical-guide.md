# RL-Kernel Train–Rollout Consistency: A Practical Guide

In RL post-training, one engine generates responses and another trains on them.
Both assign probabilities to the generated tokens. **Before the weights change,
the same tokens should receive the same scores under the same policy.** Different
execution paths can break that agreement, even when the model weights match.

RL-Kernel aligns these computations across training and rollout. This guide
explains why that matters, then runs a short Qwen3-8B consistency check with
**vime, vLLM and Megatron-LM** on NVIDIA H100 or AMD MI300X.

## Why exact consistency matters in post-training

Rollout records a log probability, or **LogP**, for each generated token.
Training independently recomputes LogP for those tokens. Differences between the
two can enter policy ratios and KL-related measurements. They can make the policy
appear to have changed before an optimizer update and, if large enough, alter
which tokens reach a clipping threshold.

**Bitwise train–rollout consistency** means the independently computed LogP
values have identical bits at the declared comparison point. The comparison must
use the same weights, tokens, positions, masks and probability definition,
including any temperature or sampling transformations. Training does not simply
reuse rollout's stored scores.

This gives post-training developers:

- **A cleaner learning signal.** Remove execution-induced score differences from
  the comparison between rollout and training at the same policy version.
- **A clear debugging target.** Zero mismatch provides a concrete check when
  changing kernels, engine versions or supported parallel configurations.
- **A controlled basis for optimization.** Compare faster implementations against
  an explicit numerical requirement while measuring end-to-end cost separately.

Batch invariance (BI) asks whether changing a batch changes a computation's
result. It is one part of this problem. Post-training also needs agreement across
**two engines and execution phases**, with aligned policy versions and token
state. BI inside one engine alone does not establish that agreement.

Exact scores remove one source of numerical mismatch. They do not by themselves
guarantee higher rewards, faster convergence or identical training trajectories.
Those outcomes need separate experiments. This guide checks training against
rollout **within each tested platform**; it does not compare NVIDIA outputs with
AMD outputs. See the [technical explanation and published results](2026-09-18-rl-kernel-amd-vime-qwen3-8b-train-rollout-bitwise-consistency.md)
for the comparison boundaries and evidence.

## Requirements

Start inside a **prepared GPU environment**. Installing RL-Kernel alone does not
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

Choose one backend. For ROCm, the source paths must point to the **patched and
built** companion checkouts described above. The vLLM changes include compiled
code; changing a Python path alone is insufficient.

=== "NVIDIA H100 / CUDA"

    ```bash
    export RLK_BACKEND=cuda
    export RLK_PROFILE_TEMPLATE=configs/integrations/orchestrators/vime/dense/qwen3-8b-tp4-cp2.json
    ```

=== "AMD MI300X / ROCm"

    ```bash
    export RLK_BACKEND=rocm
    export RLK_PROFILE_TEMPLATE=configs/integrations/orchestrators/vime/rocm_attention/mi300x-qwen3-8b.json
    export RLK_VIME=/data/rocm-repro/vime
    export RLK_MEGATRON=/data/rocm-repro/Megatron-LM
    export RLK_VLLM=/data/rocm-repro/vllm
    ```

Create .rlk-profile.json once. It records your interpreter and local paths so the
launcher can reuse them. **If you already have this file, check its paths and
skip the Python block.** The block refuses to overwrite an existing profile.
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

=== "NVIDIA H100 / CUDA"

    ```bash
    ./rlk prepare --download-data --convert-checkpoint

    ray start --head --num-gpus=8 \
      --include-dashboard=true --dashboard-host=127.0.0.1 \
      --dashboard-port=8265 --object-store-memory=200000000000

    ./rlk doctor --mode consistency
    ```

    prepare selects the pinned vime/Megatron sources, prepares the data and
    converts the model. Use a new workspace or clean companion checkouts.
    It does not install the full Python stack or download model weights.

    The Ray command reserves 200 GB for its object store. Adjust this for host
    RAM and /dev/shm, or reuse a suitable Ray head you own. Continue when doctor
    reports **Doctor: PASS**. Extra runtime paths are covered in the
    [CUDA setup guide](../usage/qwen3-vime-consistency.md#prepare-once).

=== "AMD MI300X / ROCm"

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
    support this ROCm profile.

## 3. Preview and run

Define the experiment settings once. The Bash array keeps the same settings
for the consistency run and native comparison.

```bash
RUN_ARGS=(
  --tp 4 --rollout-tp 4
  --temperature 0.7 --top-p 0.95
  --lr 5e-7 --kl-coef 0.01
  --max-response-len 6912 --max-tokens-per-gpu 4096
)

./rlk plan --mode consistency "${RUN_ARGS[@]}" --steps 3
```

Check the resolved paths and runner\_command in the printed JSON. plan does
not start a GPU job. These settings use training TP4/CP2 and rollout TP4/CP1
on the same eight-GPU node. Keep the topology fixed for the first check.

Run three steps with a unique output name:

```bash
export RLK_RUN_ID="consistency-$(date -u +%Y%m%dT%H%M%SZ)"
./rlk plan --mode consistency "${RUN_ARGS[@]}" --steps 3 --run-id "$RLK_RUN_ID" \
  > "$RLK_WORKSPACE/$RLK_RUN_ID.plan.json"
./rlk run --mode consistency "${RUN_ARGS[@]}" --steps 3 --run-id "$RLK_RUN_ID"
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

**The top-level passed field must be true.** A completed training process or
one zero-looking metric is insufficient. Review the complete validation report.

| Backend | Consistency evidence | Report and log |
|---|---|---|
| CUDA | runtime\_scalar\_logprob.bitwise\_zero is true | run-validation.json; run.log |
| ROCm | metrics.bitwise\_equal is true; metrics.bitwise\_mismatch\_count is 0; frozen\_sources\_match is true | single-arm-summary.json; arms/r-r/validation.json; arms/r-r/launcher.log |

The reports also check execution evidence and source records. CUDA and ROCm use
different evidence formats, so retain the original reports and the machine profile.

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
./rlk run --mode consistency "${RUN_ARGS[@]}" --steps 200
```

## Troubleshooting

| Symptom | Next step |
|---|---|
| Extension import or build fails | Check that Python, PyTorch and the CUDA/ROCm toolchain match the installation requirements. |
| Profile already exists or plan shows the wrong paths | Inspect .rlk-profile.json and RLK\_REPRO\_PROFILE. Keep a working profile rather than overwriting it. |
| CUDA doctor fails | Fix the named path, version or Ray connection before launching. |
| ROCm companion checks fail | Check the patched vime, Megatron and vLLM sources and their installed builds. |
| ROCm finds an unrelated Ray cluster | Coordinate use of the node. Do not stop another user's job. |
| Output directory already exists | Choose a new run ID. |
| Report is missing or passed is false | Inspect the backend's run log and all report errors. Keep the profile and source revisions when reporting the issue. |

ROCm supports plan and run with validation inside run. prepare, doctor, validate,
report and the weight-update verify workflow are CUDA-only. For the initial ROCm
check, keep rollout CP at 1, temperature positive and top-k disabled. See the
[CUDA/ROCm support audit](../usage/cuda-rocm-consistency-audit.md) before changing
these settings.

## Next steps and contributions

A useful first contribution is to reproduce this run and report the exact
hardware, source revisions and validation result. Documentation improvements and
small reproducible failures also help the project.

- [Contribution guide](../contributing/contributor-guide.md)
- [Detailed setup and supported configurations](../usage/qwen3-vime-consistency.md)
- [H100 validation results](../usage/h100-matrix-validation.md)
- [DSV4 Flash MoE roadmap](2026-08-09-dsv4-flash-moe-consistency-roadmap.md)
