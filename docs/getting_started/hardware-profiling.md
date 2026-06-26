# Hardware Profiling Guide

This guide explains how to profile RL-Kernel on NVIDIA Ampere GPUs such as A100,
A30, and A10. It focuses on a repeatable process for validating the runtime
environment, isolating the target GPU, running smoke-scale profiler checks, and
promoting the same workflow to larger benchmark shapes.

Use this page as a profiling checklist, not as a static benchmark report. Avoid
committing generated CSV, JSON, or notebook execution logs; regenerate those
artifacts on the target machine when you need fresh measurements.

## Scope

The Ampere profiling workflow validates that:

- PyTorch can see and use the selected CUDA device.
- RL-Kernel dispatch tests pass on the target environment.
- The profiler can run selected workloads and emit machine-readable reports.
- Generated reports record the expected backend, architecture, device index, and
  status fields.

It does not claim that every CUDA, driver, PyTorch, or GPU configuration is
supported. It also does not validate ROCm, Hopper-only kernels, or production
performance numbers unless those paths are explicitly run on the target hardware.

## 1. Inspect the Node

Start by recording the visible GPUs and runtime versions before selecting a
profiling device.

```bash
nvidia-smi
```

```bash
python - <<'PY'
import sys
print('python', sys.version)
try:
    import torch
    print('torch', torch.__version__)
    print('cuda_available', torch.cuda.is_available())
    print('cuda_version', torch.version.cuda)
    print('hip_version', getattr(torch.version, 'hip', None))
    if torch.cuda.is_available():
        print('device_count', torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print('device', i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
except Exception as exc:
    print('torch_error', repr(exc))
PY
```

For Ampere, the CUDA compute capability should be in the SM80 family. For
example, A100 reports capability `(8, 0)`.

## 2. Select One GPU

Choose an idle physical GPU and expose only that device to the profiler with
`CUDA_VISIBLE_DEVICES`. PyTorch will remap the selected physical GPU to logical
CUDA device 0 inside the process.

```bash
CUDA_VISIBLE_DEVICES=<physical_gpu_index> python - <<'PY'
import torch
print('cuda_available', torch.cuda.is_available())
print('device_count', torch.cuda.device_count())
if torch.cuda.is_available():
    print('logical_device_0', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
```

Keep the physical GPU index in your local notes so report metadata can be
interpreted correctly later.

## 3. Run Smoke Tests

Before collecting profiler output, run focused tests that confirm the dispatch
and profiler paths work in the selected CUDA environment.

```bash
CUDA_VISIBLE_DEVICES=<physical_gpu_index> python -m pytest rl_engine/tests/test_dispatch.py -v
```

```bash
CUDA_VISIBLE_DEVICES=<physical_gpu_index> python -m pytest tests/test_profiler.py -v
```

If these tests fail, fix the environment or implementation issue before treating
profiler output as meaningful.

## 4. Run a Smoke-Scale Profile

Use a small shape first. The goal is to validate profiler wiring, backend
selection, CSV/JSON writing, and status semantics. Do not use single-run smoke
latency as a publishable benchmark number.

```bash
CUDA_VISIBLE_DEVICES=<physical_gpu_index> python scripts/run_profile_suite.py \
  --device cuda \
  --dtype float32 \
  --batch-sizes 2 \
  --seq-lens 16 \
  --vocab-sizes 1024 \
  --workloads logp-native,logp-fused,sampling-native \
  --warmup 0 \
  --repeat 1 \
  --output-dir reports/ampere-smoke \
  --csv \
  --json
```

Inspect the generated CSV or JSON locally and confirm that each row records the
expected values for:

- `gpu_name`
- `gpu_architecture`
- `gpu_backend`
- `gpu_compute_capability`
- `gpu_device_index`
- `status`
- `notes`

A workload with `status=pass` means the selected implementation completed for
that shape. If the runtime log indicates a fallback path, describe the validated
path precisely and avoid claiming that a stricter fused kernel path was measured.

## 5. Profile Representative Shapes

After the smoke run succeeds, increase the workload sizes and repetitions to
match the target RL workload. Prefer multiple warmup and measured iterations for
performance comparisons.

```bash
CUDA_VISIBLE_DEVICES=<physical_gpu_index> python scripts/run_profile_suite.py \
  --device cuda \
  --dtype float16 \
  --batch-sizes 8,16,32 \
  --seq-lens 128,512 \
  --vocab-sizes 4096,128256 \
  --workloads logp-native,logp-fused \
  --warmup 5 \
  --repeat 20 \
  --output-dir reports/ampere-logp \
  --csv \
  --json
```

For sampling workloads, keep the top-k and top-p settings close to the target
serving or training configuration.

```bash
CUDA_VISIBLE_DEVICES=<physical_gpu_index> python scripts/run_profile_suite.py \
  --device cuda \
  --dtype float16 \
  --batch-sizes 64,128,256 \
  --vocab-sizes 128256 \
  --workloads sampling-native \
  --top-k 50 \
  --top-p 0.9 \
  --warmup 5 \
  --repeat 20 \
  --output-dir reports/ampere-sampling \
  --csv \
  --json
```

## 6. Validate Strict Fused Dispatch When Needed

If the goal is to claim that the strict fused LogP path works, build the
extension and require fused dispatch explicitly.

```bash
MAX_JOBS=2 python setup.py build_ext --inplace
CUDA_VISIBLE_DEVICES=<physical_gpu_index> python examples/grpo_single_gpu.py \
  --device cuda \
  --require-fused-logp \
  --steps 2 \
  --num-prompts 1 \
  --samples-per-prompt 2 \
  --prompt-len 2 \
  --completion-len 3 \
  --vocab-size 16 \
  --hidden-dim 8
```

Only report strict fused dispatch as validated after this command succeeds on
the target hardware.

## Reporting Guidance

When sharing results, include:

- GPU model and compute capability.
- Driver, CUDA runtime, PyTorch, and Python versions.
- The exact `CUDA_VISIBLE_DEVICES` mapping used for the run.
- The profiler command, workload shapes, dtype, warmup, and repeat count.
- Whether each workload used native, fused, fallback, or blocked dispatch.
- The generated report files as local artifacts, not committed documentation.

Keep committed docs focused on the process and commands. Generated logs and
point-in-time notebook outputs should stay outside the repository.
