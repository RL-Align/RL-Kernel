# Your First RL-Kernel Run: From Setup to a Consistency Check

RL-Kernel helps training and rollout agree on the numbers they compute. This
guide walks through a small Qwen3-8B experiment with **vime for training and
vLLM for rollout**. You will prepare the inputs, preview a command, run three
steps, and read the validation result.

You do not need to understand GPU kernels to follow the walkthrough. The goal
is simple: **run a real experiment and know what its result means**.

## What are we checking?

During rollout, the model generates an answer and records a log probability
for each generated token. During training, the model computes those scores
again for the same tokens, using the corresponding model weights.

We want the two sets of scores to match exactly. RL-Kernel's consistency mode
uses its numerical paths on both sides. Training recomputes the scores; it
does not simply copy the rollout scores.

In this guide, a passing check means the training and rollout log probabilities
agree for the tested run on that machine. It does not mean that NVIDIA and AMD
produce identical outputs to each other, or that every model and parallel
configuration has been validated.

## Before you start

Choose **one** hardware path:

| | NVIDIA path | AMD path |
|---|---|---|
| Reference hardware | One node with 8 × H100 80 GB | One node with 8 × MI300X, gfx942 |
| Model | Qwen/Qwen3-8B | Qwen/Qwen3-8B |
| Starting environment | A compatible vime CUDA environment | A compatible vime ROCm environment with the companion patches |
| First run | Three training/rollout steps | Three training/rollout steps |
| Ray startup | Start Ray before running | The launcher manages its own Ray instance |

**This walkthrough starts inside a prepared GPU environment.** Installing
RL-Kernel alone does not install the full training stack. A laptop, a CPU-only
installation, or an arbitrary vLLM image is not enough for this experiment.

For CUDA, the bundled profile checks Python 3.11.15, PyTorch 2.9.1, vLLM 0.16.0,
Ray 2.57.0, and Transformer Engine 2.18.0. See the
[CUDA environment requirements](../usage/qwen3-vime-consistency.md#supported-reference-configuration).
For ROCm, use the
[MI300X setup and companion source instructions](../usage/qwen3-vime-consistency.md#rocm-mi300x-and-gfx942).
The recorded ROCm stack uses Python 3.12 and PyTorch 2.12 / ROCm 7.14;
the patched vime, Megatron and vLLM sources are part of that setup.

Reserve all eight GPUs for the run. Keep model files, converted checkpoints,
datasets and output on a data disk with enough free space. Prepare downloads
and builds before a live presentation.

## 1. Get RL-Kernel and choose your paths

Run the commands in Bash, inside your GPU environment. If it is a container,
the paths below must be visible **inside that container**.

```bash
git clone https://github.com/RL-Align/RL-Kernel.git
cd RL-Kernel

# Change these two paths for your machine.
export RLK_DEMO_WORKSPACE=/data/rlk-demo
export RLK_DEMO_MODEL=/data/models/Qwen3-8B
mkdir -p "$RLK_DEMO_WORKSPACE"

# Use the Python environment that already contains your training stack.
RL_KERNEL_REQUIRE_EXT=1 python -m pip install --no-deps --no-build-isolation -e .
python -c "import rl_engine._C as C; print(C.__file__)"
git rev-parse HEAD
```

The import should print the compiled extension's location. Save the Git
revision with your results so that you know which source version you used.
For build prerequisites, see [Installation](../getting_started/installation.md).

`RLK_DEMO_MODEL` must contain the Hugging Face model files, including the
weights, configuration and tokenizer. If you do not already have them, download
them once using `huggingface_hub` from your environment:

```bash
python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download("Qwen/Qwen3-8B", local_dir=os.environ["RLK_DEMO_MODEL"])
PY
```

## 2. Save your machine configuration once

A **profile** is a small JSON file holding your backend and local paths.
Saving it as `.rlk-profile.json` lets `./rlk` find the right environment without
editing a training script. This file is ignored by Git.

Choose the tab for your hardware. The snippets create a new local profile and
refuse to overwrite an existing one. If you already have a working profile,
keep it and skip profile creation. `RLK_REPRO_PROFILE`, when set, selects a
different file; unset it below to use this checkout's local profile.

=== "NVIDIA H100 / CUDA"

    From the RL-Kernel directory, in the same shell as step 1:

    ```bash
    unset RLK_REPRO_PROFILE
    python - <<'PY'
    import json
    import os
    import sys
    from pathlib import Path

    source = Path("configs/integrations/orchestrators/vime/dense/qwen3-8b-tp4-cp2.json")
    profile = json.loads(source.read_text())
    workspace = Path(os.environ["RLK_DEMO_WORKSPACE"]).resolve()
    profile["paths"].update({
        "workspace": str(workspace),
        "rl_kernel_root": str(Path.cwd()),
        "python": sys.executable,
        "model_root": str(Path(os.environ["RLK_DEMO_MODEL"]).resolve()),
        "output_root": str(workspace / "runs"),
    })
    with Path(".rlk-profile.json").open("x") as handle:
        json.dump(profile, handle, indent=2)
    PY

    ./rlk prepare --download-data --convert-checkpoint
    ```

    `prepare` checks out the pinned vime and Megatron revisions, prepares
    DAPO-Math-17k prompts, and converts the local model to the Megatron checkpoint
    format. It does not create a Python environment or download the model.
    Use a new workspace; existing companion checkouts must be clean because
    preparation selects the profile's source revisions.

    Next, start Ray on your reserved node. The reference command allocates
    **200 GB** for the object store; reduce that value if necessary for your
    host RAM and `/dev/shm`. Reuse a suitable Ray head you already own instead
    of starting a second one.

    ```bash
    ray start --head \
      --include-dashboard=true \
      --dashboard-host=127.0.0.1 \
      --dashboard-port=8265 \
      --num-gpus=8 \
      --object-store-memory=200000000000

    ./rlk doctor --mode consistency
    ```

    Continue only when the final line says `Doctor: PASS`. If Transformer
    Engine or CUDA Python is installed outside this environment, configure
    the extra paths described in the
    [setup guide](../usage/qwen3-vime-consistency.md#prepare-once).

=== "AMD MI300X / ROCm"

    First finish the [ROCm companion setup](https://github.com/RL-Align/RL-Kernel/blob/main/rl_engine/integrations/orchestrators/vime/patches/rocm_attention/README.md).
    This is required even if an unmodified vLLM installation already runs
    inference. The patches cover vime, Megatron and vLLM; the vLLM base patch
    changes compiled code, so a Python path change alone is not sufficient.

    Use your patched, built companion checkouts below. The vLLM instructions
    include a public source reconstruction path; do not try to fetch the
    historical local build commit from upstream.

    ```bash
    # Change these to your prepared companion source directories.
    export RLK_DEMO_VIME=/data/rocm-repro/vime
    export RLK_DEMO_MEGATRON=/data/rocm-repro/Megatron-LM
    export RLK_DEMO_VLLM=/data/rocm-repro/vllm

    unset RLK_REPRO_PROFILE
    python - <<'PY'
    import json
    import os
    import sys
    from pathlib import Path

    source = Path("configs/integrations/orchestrators/vime/rocm_attention/mi300x-qwen3-8b.json")
    profile = json.loads(source.read_text())
    workspace = Path(os.environ["RLK_DEMO_WORKSPACE"]).resolve()
    profile["paths"].update({
        "workspace": str(workspace),
        "rl_kernel_root": str(Path.cwd()),
        "python": sys.executable,
        "vime_root": str(Path(os.environ["RLK_DEMO_VIME"]).resolve()),
        "megatron_root": str(Path(os.environ["RLK_DEMO_MEGATRON"]).resolve()),
        "vllm_root": str(Path(os.environ["RLK_DEMO_VLLM"]).resolve()),
        "model_root": str(Path(os.environ["RLK_DEMO_MODEL"]).resolve()),
        "ref_load": str(workspace / "checkpoints/Qwen3-8B_torch_dist"),
        "prompt_data": str(workspace / "data/dapo-math-17k.vime.jsonl"),
        "output_root": str(workspace / "runs"),
    })
    with Path(".rlk-profile.json").open("x") as handle:
        json.dump(profile, handle, indent=2)
    PY
    ```

    Prepare the data and reference checkpoint once. If you already have these
    files, set `prompt_data` and `ref_load` in your local profile to their
    existing locations and skip this block.

    ```bash
    mkdir -p "$RLK_DEMO_WORKSPACE/data" "$RLK_DEMO_WORKSPACE/checkpoints"
    python rl_engine/integrations/orchestrators/vime/experiments/dense/prepare_dapo_data.py \
      --download \
      --source "$RLK_DEMO_WORKSPACE/data/dapo-math-17k.parquet" \
      --output "$RLK_DEMO_WORKSPACE/data/dapo-math-17k.vime.jsonl"

    # Run this conversion once, with a new destination directory.
    (
      set -euo pipefail
      source "$RLK_DEMO_VIME/scripts/models/qwen3-8B.sh"
      PYTHONPATH="$RLK_DEMO_MEGATRON${PYTHONPATH:+:$PYTHONPATH}" \
        python "$RLK_DEMO_VIME/tools/convert_hf_to_torch_dist.py" \
          "${MODEL_ARGS[@]}" \
          --hf-checkpoint "$RLK_DEMO_MODEL" \
          --save "$RLK_DEMO_WORKSPACE/checkpoints/Qwen3-8B_torch_dist"
    )
    ```

    Do not run the CUDA `prepare` or `doctor` commands with this ROCm profile.
    The ROCm runner checks its inputs when starting and manages its own Ray
    instance. It refuses to stop an unrelated Ray cluster. Keep the profile's
    Ray ports available, or choose free ones in its `defaults` section.

## 3. Preview the command

Both hardware paths now use the same command. Run it from the RL-Kernel
directory:

```bash
./rlk plan --mode consistency \
  --tp 4 --rollout-tp 4 --temperature 0.7 --top-p 0.95 \
  --lr 5e-7 --kl-coef 0.01 \
  --max-response-len 6912 --max-tokens-per-gpu 4096 --steps 3
```

`plan` prints JSON containing `paths` and `runner_command`. Check that the
Python executable, model, companion repositories and output folder are yours.
**It does not launch a GPU job or prove that the environment is ready.**

You only need a few settings for this first run:

| Setting | Plain-language meaning |
|---|---|
| `--mode consistency` | Use RL-Kernel's consistency path. |
| `--tp 4` | Split training model computation across four GPUs per TP group. Training CP is inferred as 2 on this eight-GPU setup. |
| `--rollout-tp 4` | Use four GPUs per rollout engine; rollout CP stays at 1. |
| `--temperature 0.7 --top-p 0.95` | Choose the sampling settings used to generate answers. |
| `--steps 3` | Run three training/rollout steps instead of the default 200. |
| `--max-response-len 6912` | Limit the number of generated tokens per answer. |
| `--max-tokens-per-gpu 4096` | Set the training dynamic-batching token budget per GPU. |
| `--lr 5e-7 --kl-coef 0.01` | Set the learning rate and reference-policy KL coefficient. Keep these fixed for the walkthrough. |

The training and rollout workers share the node over the workflow; these
settings do not request sixteen GPUs. Leave the topology unchanged for your
first check.

## 4. Run three steps

Replace `plan` with `run`. Give this attempt a unique name so its output is
easy to find:

```bash
export RLK_DEMO_RUN_ID="first-consistency-$(date -u +%Y%m%dT%H%M%SZ)"

./rlk run --mode consistency \
  --tp 4 --rollout-tp 4 --temperature 0.7 --top-p 0.95 \
  --lr 5e-7 --kl-coef 0.01 \
  --max-response-len 6912 --max-tokens-per-gpu 4096 --steps 3 \
  --run-id "$RLK_DEMO_RUN_ID"
```

The command waits for completion and performs validation. Model loading,
compilation and warmup can take time; three steps are a small experiment,
not a promise of a three-minute demo. On ROCm, detailed progress is written to
the run's `arms/r-r/launcher.log`, so the main terminal may be quiet.

For a presentation, finish one rehearsal beforehand and keep its real result
available. If the live job is still running, show that result and say it is
from the rehearsal.

## 5. Read the result

The profiles above put outputs under `workspace/runs/<run-id>`. Use this
snippet to find the current run and print its report. It reads your local
profile, so it also works if you chose a different output folder there:

```bash
python - <<'PY'
import json
import os
from pathlib import Path

profile = json.loads(Path(".rlk-profile.json").read_text())
run_dir = Path(profile["paths"]["output_root"]) / os.environ["RLK_DEMO_RUN_ID"]
backend = profile["requirements"]["backend"]
filename = "single-arm-summary.json" if backend == "rocm" else "run-validation.json"
report_path = run_dir / filename
print(f"Run directory: {run_dir}")
report = json.loads(report_path.read_text())
print(json.dumps(report, indent=2))
if report.get("passed") is not True:
    raise SystemExit("Validation failed: inspect the report and run log.")
PY
```

The report must have **`"passed": true`**. A finished training process alone
is not a passing consistency check.

| Backend | What to inspect | Where to find details |
|---|---|---|
| CUDA | Top-level `passed`; `runtime_scalar_logprob.bitwise_zero` should be `true`. Review the other validation sections too. | `run-validation.json` and `run.log` |
| ROCm | Top-level `passed`; `metrics.bitwise_equal` should be `true`, `metrics.bitwise_mismatch_count` should be `0`, and `frozen_sources_match` should be `true`. | `single-arm-summary.json`, `arms/r-r/validation.json`, and `arms/r-r/launcher.log` |

The validation also checks evidence such as executed operators, compared
samples and source records. Read the full report if anything fails; do not
select only a zero-looking metric. The CUDA runtime scalar check and the ROCm
raw-byte check have different evidence formats, so keep the original reports.

## 6. Try the native comparison, then a longer run

After the consistency run succeeds, run the native baseline **sequentially**
on the same node. Change only the mode and run name:

```bash
./rlk run --mode native \
  --tp 4 --rollout-tp 4 --temperature 0.7 --top-p 0.95 \
  --lr 5e-7 --kl-coef 0.01 \
  --max-response-len 6912 --max-tokens-per-gpu 4096 --steps 3 \
  --run-id "first-native-$(date -u +%Y%m%dT%H%M%SZ)"
```

Native mode uses vime's native operators. Its validator can pass while
training and rollout scores differ: zero mismatch is the requirement for
**consistency mode**, not a requirement imposed on the native baseline.
On ROCm, the native log is under `arms/p-p/launcher.log`.

Do not treat this short pair as a performance benchmark. Warmup and generated
answer lengths affect timing, and the two modes can generate different
trajectories. The [ROCm performance report](../usage/rocm-sparse-performance.md)
separates current three-step evidence from historical 200-step results and
records the native execution-record limitation.

When you are ready for a longer consistency run:

```bash
./rlk run --mode consistency \
  --tp 4 --rollout-tp 4 --temperature 0.7 --top-p 0.95 \
  --lr 5e-7 --kl-coef 0.01 \
  --max-response-len 6912 --max-tokens-per-gpu 4096 --steps 200
```

This creates a new run directory automatically. It is a longer validation
attempt, not a guarantee that your machine reproduces a published timing.

## If something goes wrong

| What you see | What to do next |
|---|---|
| Extension import or build fails | Check that the active Python, PyTorch and CUDA/ROCm toolchain match. Follow the installation guide and rebuild. |
| Profile creation says the file exists | Keep your existing profile, inspect its paths, and skip the creation block. |
| `plan` shows another machine's paths | Check `.rlk-profile.json` and whether `RLK_REPRO_PROFILE` selects another file. |
| CUDA `Doctor: FAIL` | Fix the named path, version or Ray connection check before launching. |
| ROCm reports missing companion integration | Check all three patched companion sources and their installed builds, including the compiled vLLM base change. |
| ROCm finds an unrelated Ray cluster | Coordinate use of the node; do not stop another user's job. |
| Output directory already exists | Generate a new run name. Outputs are not overwritten. |
| No final report yet | Check the backend-specific run log. The job may still be running or may have failed before validation. |
| Validation says `passed: false` | Read the errors and nested checks. Keep the log, profile and source revisions when reporting the problem. |

The ROCm profile currently supports `plan` and `run`, with validation included
in `run`. `prepare`, `doctor`, `validate`, `report` and the weight-update
`verify` workflow are CUDA-only. For the first ROCm run, keep rollout CP at 1,
temperature positive and top-k disabled (`-1`, the default). See the
[support audit](../usage/cuda-rocm-consistency-audit.md) before changing these
settings.

## A short script for a live walkthrough

Use this after setup is complete. The five passages can be read aloud while
showing the corresponding parts of this page. Pause while commands run.

**Introduce the goal**

“Today I will show you how to run a small RL-Kernel example. We use Qwen3-8B,
with vime for training and vLLM for rollout. Rollout means generating answers
for training. We want both sides to agree on the scores for those answers.
We have prepared the environment and downloaded the model before this session.”

**Show the profile and plan**

“First, we save the paths for this machine in one profile. We only need to do
this once. Then we run the plan command. This shows what will run, but it does
not start training. Here we can check the model path and the output folder.”

**Start the short run**

“Now we change plan to run. We use three steps for this example. Training
computes the scores again, independently. It does not copy the scores from
rollout. The command also checks the result when the job finishes.”

**Show a completed report**

“This is the validation report for a completed run. First, we check that the
whole report passed. Then we look at the training and rollout comparison.
For a passing consistency run, the checked scores match exactly. This result
applies to this model, configuration and run on this machine.”

**Close with the next step**

“Once this works, we can try the native baseline or run more steps. Today we
focused on getting started and reading the result. Our wider goal is to bring
this consistency to more models, hardware platforms and post-training
frameworks. The links below explain the current results and ongoing work.”

## Learn more and contribute

- [Project results: Qwen3-8B training and rollout consistency](2026-09-18-rl-kernel-amd-vime-qwen3-8b-train-rollout-bitwise-consistency.md)
- [Detailed setup and supported configurations](../usage/qwen3-vime-consistency.md)
- [Current H100 validation results](../usage/h100-matrix-validation.md)
- [Current ROCm performance and validation scope](../usage/rocm-sparse-performance.md)
- [DSV4 Flash MoE roadmap — work in progress](2026-08-09-dsv4-flash-moe-consistency-roadmap.md)
- [Contributing to RL-Kernel](../contributing/contributor-guide.md)

The command-first structure of this walkthrough takes inspiration from the
[vLLM MoRIIOConnector usage guide](https://docs.vllm.ai/en/latest/features/moriio_connector_usage/).
MoRIIOConnector is a separate feature; it is not required for this walkthrough.
