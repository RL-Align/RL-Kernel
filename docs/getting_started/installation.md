# Installation

RL-Kernel requires Python 3.10 or newer and PyTorch. CUDA builds require a working
CUDA toolchain; ROCm builds require a compatible ROCm environment.

## From Source

For CUDA (or ROCm) source builds, install PyTorch first (matching your CUDA/ROCm
runtime), then build with build isolation disabled so `setup.py` can access
PyTorch's extension build utilities and compile the native kernels (`rl_engine._C`):

```bash
git clone https://github.com/RL-Align/RL-Kernel.git
cd RL-Kernel
# Optional: pin the compile target. If unset, the build targets your GPU's arch.
# export TORCH_CUDA_ARCH_LIST="9.0+PTX"   # e.g. Hopper; or "8.6+PTX", "12.0+PTX"
RL_KERNEL_REQUIRE_EXT=1 python -m pip install --no-build-isolation -e .
```

`RL_KERNEL_REQUIRE_EXT=1` makes the build fail if `_C` cannot be compiled. Without
`--no-build-isolation`, PyTorch is invisible to the isolated build environment.
Confirm the compiled extension is present with:

```bash
python -c "import rl_engine._C as _C; assert hasattr(_C, 'fused_logp'); print(_C.__file__)"
```

A CPU-only install (plain `pip install -e .` on a machine with no GPU) remains
supported and runs on the pure-PyTorch backends.

## Optional Backends

The extras add optional dependencies on top of the compiled package, so they use
the same `--no-build-isolation` flag as the source build above.

```bash
RL_KERNEL_REQUIRE_EXT=1 python -m pip install --no-build-isolation -e ".[cuda]"
```

```bash
RL_KERNEL_REQUIRE_EXT=1 python -m pip install --no-build-isolation -e ".[rocm]"
```

```bash
python -m pip install --no-build-isolation -e ".[vllm]"
```

Install the vLLM extra only on rollout or benchmark environments that need the
vLLM runtime. Core CI and mocked integration tests do not require it.

For common CUDA, ROCm, vLLM, fallback, and CI questions, see the
[FAQ](faq.md).

### ROCm Backend

Use a ROCm PyTorch build that matches the installed ROCm toolchain. Then install
FlashAttention with an AMD backend:

```bash
python -m pip install ninja packaging wheel psutil einops
git clone --recurse-submodules https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
  python -m pip install --no-build-isolation --no-deps .
cd ..
```

Verify the environment from the RL-Kernel checkout:

```bash
python tools/env/check_rocm_env.py
```

RL-Kernel uses external FlashAttention as the default ROCm attention path. To
fall back to PyTorch SDPA for ROCm attention dispatch, set:

```bash
export RL_KERNEL_ROCM_ATTN_BACKEND=sdpa
```

## Development Dependencies

```bash
pip install -e ".[dev]"
pip install -r requirements/docs.txt
```

## Documentation Preview

```bash
mkdocs serve
```

Then open the local URL printed by MkDocs.
