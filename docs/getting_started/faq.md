# FAQ

This FAQ is the first stop for installation, CUDA, ROCm, vLLM, benchmark, and CI
questions. It favors commands that expose the current environment over static
compatibility tables, because GPU drivers, PyTorch wheels, and optional runtimes
change more often than RL-Kernel's public API.

## Start Here

### Which setup path should I use?

| Goal | Recommended install | GPU required | Notes |
| --- | --- | --- | --- |
| Read docs or edit docs | `pip install -r requirements-docs.txt` | No | Use `mkdocs build --strict -f mkdocs.yaml` before opening a PR. |
| Run CPU/mock tests | `pip install -e ".[dev]"` | No | Matches the default CI style: fallback and mocked integration coverage. |
| Run CUDA operators | `pip install -e ".[cuda]"` | Yes, NVIDIA | Requires a CUDA-enabled PyTorch wheel and a working CUDA toolchain for source builds. |
| Run ROCm operators | `pip install -e ".[rocm]"` | Yes, AMD | Requires a ROCm-enabled PyTorch wheel and ROCm compiler/runtime environment. |
| Run real vLLM rollout | `pip install -e ".[vllm]"` | Runtime-dependent | Core tests do not need vLLM; install this only where real vLLM is used. |

Do not install every optional extra by default. Install the smallest environment
that matches the work you are doing, then add GPU or engine extras only when
needed.

### What information should I collect before debugging?

Run this from the repository root:

```bash
python3 - <<'PY'
import importlib.util
import platform
import sys

import torch

print("python:", sys.version.replace("\n", " "))
print("platform:", platform.platform())
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("torch hip:", torch.version.hip)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device count:", torch.cuda.device_count())
    for index in range(torch.cuda.device_count()):
        print("device", index, torch.cuda.get_device_name(index))
        print("capability", index, torch.cuda.get_device_capability(index))
print("vllm installed:", importlib.util.find_spec("vllm") is not None)
print("triton installed:", importlib.util.find_spec("triton") is not None)
PY
```

For native extension build issues, also include:

```bash
nvcc --version
```

For ROCm issues, also include:

```bash
rocminfo
```

If a command is not found, include that fact in the issue. Missing tooling often
explains the failure more directly than a Python stack trace.

### What are the minimum versions?

RL-Kernel requires Python 3.10 or newer. The package metadata uses
`torch>=2.4.1`.

The important rule is that PyTorch must match your runtime:

- CPU-only work should use a CPU PyTorch wheel.
- NVIDIA work should use a CUDA-enabled PyTorch wheel compatible with the
  installed NVIDIA driver.
- AMD work should use a ROCm-enabled PyTorch wheel where `torch.version.hip` is
  not `None`.

## Installation

### What is the basic source install?

```bash
git clone https://github.com/RL-Align/RL-Kernel.git
cd RL-Kernel
pip install -e .
```

The examples on this page use `python3` for system-level commands. Inside an
activated virtual environment, `python` is also fine.

### Which optional extras exist?

```bash
pip install -e ".[cuda]"
pip install -e ".[rocm]"
pip install -e ".[vllm]"
pip install -e ".[dev]"
```

The optional extras install Python dependencies. They do not replace the platform
runtime itself. For example, `.[cuda]` does not install an NVIDIA driver or CUDA
toolkit, and `.[rocm]` does not turn a CUDA PyTorch wheel into a ROCm PyTorch
wheel.

### Why can install fail on a GPU machine?

RL-Kernel's source build probes PyTorch. If `torch.cuda.is_available()` is true,
the build path attempts to compile CUDA extensions. A GPU-visible PyTorch runtime
is not enough for source builds: the compiler toolchain also needs to be present.

Check:

```bash
nvcc --version
```

If `nvcc` is missing and you only need docs or CPU fallback tests, use a CPU-only
development environment. If you need CUDA kernels, install a matching CUDA toolkit
and retry the source build.

### My environment is externally managed and pip refuses to install. What now?

Some Linux distributions block global `pip install` into the system Python. Use a
virtual environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

For docs-only work:

```bash
python3 -m venv .venv-docs
. .venv-docs/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-docs.txt
mkdocs build --strict -f mkdocs.yaml
```

## CUDA

### How do I confirm PyTorch can see my NVIDIA GPU?

```bash
python3 - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY
```

If `cuda available` is `False`, debug the PyTorch/driver install before debugging
RL-Kernel.

### Why is the SM90 TMA fused LogP path not active?

The SM90 TMA path is only for TMA-capable architectures and only when the
extension was compiled with that path enabled. Source builds enable that path with
`KERNEL_ALIGN_FORCE_SM90=1`.

On non-SM90 NVIDIA GPUs, RL-Kernel should use the generic CUDA path when it is
available, then optional backends, then PyTorch fallback. For example, an Ada GPU
such as SM89 should not be expected to use the SM90 TMA kernel.

### What should I report for CUDA build failures on new GPU architectures?

Include the GPU compute capability, CUDA toolkit version, PyTorch CUDA version,
and the full compiler command or error. Newer architectures such as Blackwell
should not be debugged as Hopper-only SM90 issues unless the failing path was
explicitly enabled.

Run:

```bash
python3 - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY
nvcc --version
```

If the build only fails when `KERNEL_ALIGN_FORCE_SM90=1` is set, say that in the
issue. That flag opts into the SM90-specific path and is not required for the
generic CUDA fused-logp path.

### When should I add the `needs-gpu-ci` label?

Use `needs-gpu-ci` for pull requests that cannot be validated by CPU/mock tests:

- native CUDA or ROCm code under `csrc/`
- GPU dispatch behavior under `rl_engine/kernels/`
- GPU-only benchmark behavior
- changes where correctness depends on compiled extensions

Documentation-only changes and mocked integration tests should not need GPU CI.

## ROCm

### What should I verify before debugging ROCm?

Before opening a ROCm issue, verify all of these environment facts:

- AMD GPU supported by the installed ROCm stack
- ROCm-enabled PyTorch build
- ROCm compiler/runtime environment
- RL-Kernel ROCm optional dependencies when the relevant backend needs them

Installing `.[rocm]` in a CUDA or CPU-only PyTorch environment is not sufficient.

### How do I confirm I am using a ROCm PyTorch environment?

```bash
python3 - <<'PY'
import torch

print("torch:", torch.__version__)
print("hip runtime:", torch.version.hip)
print("cuda available:", torch.cuda.is_available())
PY
```

If `torch.version.hip` is `None`, the current Python environment is not a ROCm
PyTorch environment.

### Why does PyTorch still say `cuda` on ROCm?

PyTorch exposes ROCm devices through much of the `torch.cuda` API. RL-Kernel
distinguishes ROCm by checking `torch.version.hip`, not by expecting a separate
`torch.rocm` device namespace.

## vLLM And Rollout

### Do I need vLLM for core development?

No. vLLM is lazily imported by the rollout sampler. Core CI, dispatch tests, docs,
and mocked vLLM sampler tests do not require the real vLLM package.

Run the mocked vLLM coverage with:

```bash
python3 -m pytest tests/test_vllm_rollout_sampler.py -q
```

Install `.[vllm]` only for real rollout or vLLM-specific benchmark work.

### Do I need vLLM for the single-GPU GRPO example?

No. `examples/grpo_single_gpu.py` is a minimal single-device GRPO training script
and does not require vLLM, Ray, or DeepSpeed.

Run its smoke tests with:

```bash
python3 -m pytest tests/test_grpo_single_gpu_example.py -q
```

`--require-fused-logp` is stricter than the default example path. Use it only when
the CUDA extension has been built and the purpose of the run is to prove fused
CUDA dispatch. Without a built extension, strict mode should fail instead of
silently accepting a PyTorch fallback.

### How do I configure vLLM rollout when I do need it?

Use the rollout executor configuration fields consumed by
`VLLMSamplerConfig.from_model_config(...)`. At minimum, provide a model path or
model name. Prefix caching is enabled by default.

Example shape:

```python
from rl_engine.executors.rollout import RolloutExecutor

executor = RolloutExecutor(
    {
        "model": "/path/to/model",
        "sampler": {
            "num_generations": 4,
            "sampling_params": {"temperature": 0.7, "top_p": 0.9},
        },
        "vllm": {
            "engine_kwargs": {"dtype": "float16"},
        },
    }
)
```

GRPO usually generates multiple candidates for the same prompt. RL-Kernel expands
those requests so each candidate in a prompt group keeps the same prompt prefix,
which gives vLLM the request shape needed for prefix-cache reuse. The sampler also
accepts token prompt mappings, so smoke tests can avoid tokenizer downloads when
the vLLM runtime supports token prompts.

## Runtime Dispatch And Fallbacks

### How does RL-Kernel choose an operator backend?

Operators are selected through `rl_engine.kernels.registry.kernel_registry`.
The registry checks the detected platform and tries backends in priority order.
If an optional backend cannot be imported or instantiated, the registry records
that failure and tries the next candidate.

For supported operator types, the final fallback is a PyTorch-native
implementation.

### How do I validate dispatch locally?

```bash
python3 -m pytest rl_engine/tests/test_dispatch.py -v
```

This is the fastest check for fallback and registry behavior.

### How do I validate operator accuracy?

```bash
python3 -m pytest tests/test_op_accuracy.py -q
```

For focused areas, run the relevant test file:

```bash
python3 -m pytest tests/test_reference_ops.py tests/test_op_accuracy.py -q
```

GPU-specific test files may require a matching backend and may exercise known
open issues. Run those only when the target backend and issue state match the
change you are validating.

### Is a fallback warning always a bug?

No. Messages such as "falling back to native code" mean the optional extension or
backend was unavailable and RL-Kernel continued with another implementation.

Treat it as a bug when:

- the selected operation has no fallback and raises `No functional backend`
- a fallback produces incorrect output
- a GPU-specific PR expected a fused backend but dispatch selected PyTorch native
- a fallback is missing a method that the fused backend exposes

The native log-prob fallback is expected to keep public API parity with the fused
log-prob wrapper for dense, indexed, online, and `out` variants. Validate that
contract with:

```bash
python3 -m pytest tests/test_op_accuracy.py -q
```

## Benchmarks

### How should I run a small benchmark first?

Start with smoke-sized workloads:

```bash
python3 scripts/run_profile_suite.py --smoke --workloads logp-native --no-summary \
  --output-dir /tmp/rl-kernel-smoke-reports
python3 benchmarks/profiler.py --smoke --workloads logp-native --no-summary
```

For CUDA profiling, choose shapes that fit your GPU memory. Large vocabulary,
long sequence length, and high GRPO group size can allocate large input tensors
even when the fused operator has low extra VRAM overhead.

The commands above assume you are running from the repository root after installing
RL-Kernel or its development dependencies. `--no-summary` keeps the smoke path
focused on execution and report generation; install the project dependencies if
you want the terminal summary table.

### How should I interpret VRAM numbers in GRPO benchmarks?

GRPO log-prob benchmarks should report extra VRAM above the already allocated
input tensors. Counting the input logits tensor itself makes native and fused
paths hard to compare because both paths start with the same input allocation.

For GRPO operator reports, look for wording like "extra VRAM" or "peak overhead
above input tensors". If you add a new benchmark, record the baseline after input
allocation and measure only the additional memory used by the compute call.

When timing CUDA workloads, prefer CUDA events or the profiler helpers over raw
wall-clock timing around asynchronous GPU work.

### What does benchmark status `blocked` mean?

`blocked` means the requested workload could not run in the current environment,
for example because CUDA is unavailable, an optional package is missing, or the
requested backend is not supported on the detected hardware.

It is not the same as a correctness failure. Check the benchmark note, install the
missing optional dependency if needed, or select a CPU/PyTorch fallback workload.

### What benchmark metadata should I include in a report?

Include:

- command line
- commit SHA
- Python and PyTorch versions
- GPU model, memory, backend, and driver/runtime
- dtype
- batch size, sequence length, vocabulary size, and group size where relevant
- status (`pass`, `blocked`, `oom`, or failure)
- latency, memory, and any drift/correctness numbers emitted by the benchmark

This is enough for maintainers to tell whether a result is a kernel issue, an
environment issue, or a workload-sizing issue.

### How should I test sampling temperature behavior?

Do not validate sampling only at `temperature=1.0`. Temperature-specific bugs can
hide at the default value. For any sampling backend, the probability path should
match `softmax(logits / temperature)`; temperature should be applied once.

Run the FlashInfer temperature regression test with:

```bash
python3 -m pytest tests/test_sampler_temperature.py -q
```

## Documentation

### How do I build the documentation locally?

```bash
pip install -r requirements-docs.txt
mkdocs build --strict -f mkdocs.yaml
```

Use the strict build as the PR gate. Preview locally only after the strict build
passes.

### Why does the docs build warn about git revision timestamps?

New documentation files do not have committed git history yet, so
`mkdocs-git-revision-date-localized-plugin` may warn that it is using the current
timestamp. That warning is expected before the file is committed.

## CI And Pull Requests

### What does default CI run?

Default CI runs:

- pre-commit hooks
- mypy over `rl_engine/`
- CPU/mock dispatch tests
- strict MkDocs build

GPU CI is separate and label-gated.

### What should I run before opening a docs PR?

```bash
git diff --check
mkdocs build --strict -f mkdocs.yaml
```

If you changed code examples that are covered by tests, run the relevant pytest
file as well.

### What should I run before opening a kernel or executor PR?

Start with:

```bash
python3 -m pytest rl_engine/tests/test_dispatch.py -v
python3 -m pytest tests/test_reference_ops.py -q
```

Then run the focused tests for the component you changed. Add `needs-gpu-ci` if
local CPU/mock coverage cannot validate the behavior.

## Reporting Issues

### What should a good bug report include?

Include:

- the command that failed
- full error output
- environment snapshot from the "Start Here" section
- whether the same command works on CPU fallback
- whether the failure happens during install, import, dispatch, benchmark, or
  numerical comparison
- for GPU issues, whether the PR or branch has run GPU CI

### How do I separate environment failures from RL-Kernel failures?

Use this order:

1. Confirm Python and PyTorch import.
2. Confirm PyTorch sees the intended CPU/CUDA/ROCm backend.
3. Confirm optional tools such as `nvcc`, `rocminfo`, or `vllm` are installed only
   when the task requires them.
4. Run `python3 -m pytest rl_engine/tests/test_dispatch.py -v`.
5. Run the smallest relevant smoke benchmark or test.

If the failure appears before step 4, it is usually an environment or dependency
issue. If step 4 passes but a fused backend fails, it is likely a backend-specific
RL-Kernel issue.
