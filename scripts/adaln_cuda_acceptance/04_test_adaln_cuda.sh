#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.6}
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export RL_KERNEL_REQUIRE_EXT=1
export RLK_ADALN_REAL_SHAPES=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

.venv/bin/python - <<'PY'
import sys
import torch
import triton
assert torch.version.cuda == "12.6", torch.__version__
print(sys.version.split()[0], torch.__version__, torch.version.cuda, triton.__version__)
from rl_engine.kernels.registry import OpBackend, kernel_registry
_, trace = kernel_registry.get_adaln_modulation_op(device="cuda", hidden=3072)
assert trace["selected_backend"] == OpBackend.CUDA_ADALN_MODULATION.name, trace
assert trace["fallback"] is False, trace
print(trace)
PY
.venv/bin/python -m pytest tests/test_extension_smoke.py tests/test_adaln_modulation.py -q -rs -p no:cacheprovider
bash scripts/adaln_cuda_acceptance/06_racecheck_adaln_cuda.sh
