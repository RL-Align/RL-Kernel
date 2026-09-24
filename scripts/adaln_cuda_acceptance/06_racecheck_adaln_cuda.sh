#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.6}
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export RL_KERNEL_REQUIRE_EXT=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONDONTWRITEBYTECODE=1

.venv/bin/python - <<'PY'
import torch
assert torch.version.cuda == "12.6", torch.__version__
print(torch.__version__, torch.cuda.get_device_name(), flush=True)
PY

# Numerical equality alone does not detect shared-memory read/write hazards.
exec "$CUDA_HOME/bin/compute-sanitizer" --tool racecheck \
  --racecheck-report analysis --error-exitcode 7 \
  .venv/bin/python -m pytest -q -rs -p no:cacheprovider \
  tests/test_adaln_modulation.py::test_cuda_adaln_bit_equality_harness \
  tests/test_adaln_modulation.py::test_adaln_launch_geometry_preserves_bytes -k cuda "$@"
