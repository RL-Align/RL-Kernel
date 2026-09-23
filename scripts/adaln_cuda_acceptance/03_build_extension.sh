#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.6}
export PATH="$CUDA_HOME/bin:$PATH"
export LIBRARY_PATH="/usr/lib/wsl/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.6}
export MAX_JOBS=${MAX_JOBS:-2}
export FORCE_CUDA=1
export RL_KERNEL_REQUIRE_EXT=1
export KERNEL_ALIGN_USE_FAST_MATH=0

uv pip install --python .venv/bin/python ninja
.venv/bin/python setup.py build_ext --inplace
.venv/bin/python - <<'PY'
import rl_engine._C as ext
for name in ("adaln_modulation_forward", "adaln_modulation_backward"):
    assert hasattr(ext, name), name
print("AdaLN CUDA extension symbols verified")
PY
