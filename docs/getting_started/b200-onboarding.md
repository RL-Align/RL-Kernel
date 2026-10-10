# NVIDIA B200 Onboarding

This guide follows the [A100 hardware profiling workflow](hardware-profiling.md)
on a single NVIDIA B200. It covers an isolated Python environment, driver and
compiler checks, a native CUDA extension build, profiler smoke checks, and the
synthetic single-GPU GRPO example. Run all commands from the repository root in
the same shell.

This is a deployment smoke check, not a production training or performance
report. For AMD installation, use the [ROCm instructions](installation.md#rocm-backend).

## 1. Check the Host and Select a GPU

```bash
nvidia-smi
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version
```

`nvidia-smi` reports the driver's CUDA compatibility; `nvcc --version` reports
the installed compiler. Both are needed: a PyTorch CUDA wheel alone does not
provide the compiler needed to build RL-Kernel.

B200 has compute capability `10.0` (SM100). Use a Blackwell-capable CUDA toolkit
and PyTorch build; do not reuse an A100 CUDA 12.4 environment. NVIDIA documents
native Blackwell compilation starting with CUDA 12.8 in its
[Blackwell compatibility guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html).
The commands below pair the CUDA 13.0 toolkit with the PyTorch 2.9.1 CUDA 13.0
wheel listed in [PyTorch's installation instructions](https://pytorch.org/get-started/previous-versions/#v291).

Choose an available physical GPU after inspecting `nvidia-smi`. The example
selects physical GPU 1; change it for your machine. It becomes logical device 0
inside every command below.

```bash
export CUDA_VISIBLE_DEVICES=1
```

## 2. Create an Isolated Environment

Use Python 3.11 with venv support:

```bash
python3.11 -m venv .venv-b200
source .venv-b200/bin/activate
python -m pip install --upgrade pip setuptools==81.0.0 wheel ninja
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130
```

Check both device discovery and actual CUDA execution before compiling:

```bash
python - <<'PY'
import sys
import torch
from torch.utils.cpp_extension import CUDA_HOME

print('python', sys.version)
print('torch', torch.__version__)
print('torch_cuda', torch.version.cuda)
print('cuda_home', CUDA_HOME)
assert torch.cuda.is_available(), 'CUDA is unavailable'
assert torch.cuda.device_count() == 1, 'Expose exactly one GPU'
print('gpu', torch.cuda.get_device_name(0))
print('capability', torch.cuda.get_device_capability(0))
assert torch.cuda.get_device_capability(0) == (10, 0), 'Expected SM100'
x = torch.ones(16, device='cuda')
assert (x + x).sum().item() == 32
torch.cuda.synchronize()
PY
```

## 3. Compile and Install RL-Kernel

Build the generic CUDA path for SM100. Leave the optional SM90/TMA build switches
unset for this workflow; those kernels require separate validation.

```bash
unset KERNEL_ALIGN_FORCE_SM90 KERNEL_ALIGN_DET_GEMM_SM90
export TORCH_CUDA_ARCH_LIST=10.0
export RL_KERNEL_REQUIRE_EXT=1
export MAX_JOBS=2
python -m pip install --no-build-isolation -e .
python -m pip install pytest
python -c "import rl_engine._C as c; assert hasattr(c, 'fused_logp'); print(c.__file__)"
```

`--no-build-isolation` lets the build use the installed CUDA-enabled PyTorch.
`RL_KERNEL_REQUIRE_EXT=1` prevents a failed native build from silently producing
a Python-only installation. Use a clean checkout when switching toolchains or
GPU architectures so an earlier extension cannot be mistaken for this build.

## 4. Run the Smoke Checks

```bash
python -m pytest \
  tests/test_extension_smoke.py \
  rl_engine/tests/test_dispatch.py \
  tests/test_profiler.py \
  tests/test_grpo_single_gpu_example.py -v
```

The extension smoke test must pass, not skip: it imports `_C` directly, launches
`fused_logp`, and synchronizes the GPU to catch incompatible binaries. Some
other tests exercise CPU or mocked dispatch; their success alone is not proof
that a B200 kernel launched.

Run the same small profiler shapes as the A100 workflow:

```bash
python scripts/run_profile_suite.py \
  --device cuda \
  --dtype float32 \
  --batch-sizes 2 \
  --seq-lens 16 \
  --vocab-sizes 1024 \
  --workloads logp-native,logp-fused,sampling-native \
  --warmup 0 \
  --repeat 1 \
  --output-dir runs/b200-onboarding \
  --csv --json
```

Inspect every report row, including `status` and `notes`; report generation alone
does not imply success. Expect GPU name `NVIDIA B200`, architecture `Blackwell`,
backend `cuda`, compute capability `10.0`, logical device index `0`, and passing
workloads. Keep generated reports under the ignored `runs/` directory. This
one-repeat run checks wiring only; use an idle GPU and the larger shapes in the
[profiling guide](hardware-profiling.md#5-profile-representative-shapes) for timing.

## 5. Run Strict Fused GRPO

```bash
python examples/grpo_single_gpu.py \
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

For this generic build, expect the final line:

```text
completed grpo_single_gpu steps=2 device=cuda backend=FusedLogpGenericOp
```

`--require-fused-logp` rejects fallback dispatch. Also inspect the per-step
`kernel_max_abs_error` and `train_logp_source`: this toy example validates the
fused forward path and uses the differentiable reference for training when the
selected backend has no autograd path. It does not establish fused backward,
multi-GPU alignment, or real-model rollout support. See the
[single-GPU GRPO guide](../usage/grpo-single-gpu-example.md) for details.

## Troubleshooting and Evidence

| Symptom | Check |
| --- | --- |
| `nvcc` missing or `CUDA_HOME` unset | Install/select a full CUDA toolkit and check `CUDA_HOME/bin/nvcc`. |
| Unsupported `compute_100` or PyTorch architecture warning | Check the compiler and PyTorch wheel; both must support Blackwell. |
| CUDA version mismatch while building | Match `torch.version.cuda` to the selected toolkit. |
| `_C` import fails or no kernel image is available | Rebuild in the active environment and a clean checkout with the selected B200 visible. |
| GRPO reports fallback | Check the direct `_C` import and extension smoke test before retrying. |

Record the repository commit, Python/PyTorch versions, driver and compiler
versions, physical-to-logical GPU mapping, build flags, test summary, profiler
statuses, and GRPO backend when sharing a validation result. Keep generated
logs and reports local, as in the A100 workflow.
