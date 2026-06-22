# Installation

RL-Kernel requires Python 3.10 or newer and PyTorch. CUDA builds require a working
CUDA toolchain; ROCm builds require a compatible ROCm environment.

## From Source

```bash
git clone https://github.com/RL-Align/RL-Kernel.git
cd RL-Kernel
pip install -e .
```

## Optional Backends

```bash
pip install -e ".[cuda]"
```

```bash
pip install -e ".[rocm]"
```

```bash
pip install -e ".[vllm]"
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
python scripts/check_rocm_env.py
```

RL-Kernel uses external FlashAttention as the default ROCm attention path. To
fall back to PyTorch SDPA for ROCm attention dispatch, set:

```bash
export RL_KERNEL_ROCM_ATTN_BACKEND=sdpa
```

## Development Dependencies

```bash
pip install -e ".[dev]"
pip install -r requirements-docs.txt
```

## Documentation Preview

```bash
mkdocs serve
```

Then open the local URL printed by MkDocs.
