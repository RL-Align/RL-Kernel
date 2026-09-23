#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cuda_home=${CUDA_HOME:-/usr/local/cuda-12.6}
nvcc="$cuda_home/bin/nvcc"

[[ -x "$nvcc" ]] || { echo "Missing $nvcc" >&2; exit 1; }
"$nvcc" --version | grep 'release 12\.6'
[[ -e /usr/lib/wsl/lib/libcuda.so ]] || { echo "Missing WSL libcuda.so" >&2; exit 1; }
"$root/.venv/bin/python" - <<'PY'
import torch
assert torch.version.cuda == "12.6", torch.version.cuda
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability() == (8, 6)
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())
PY
