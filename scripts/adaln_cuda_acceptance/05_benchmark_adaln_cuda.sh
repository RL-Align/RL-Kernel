#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
.venv/bin/python - <<'PYCODE'
from rl_engine.kernels.ops.cuda.norm.adaln_modulation import CudaAdaLNModulationOp
CudaAdaLNModulationOp()
print("AdaLN CUDA extension verified")
PYCODE
warmup=${WARMUP:-1}
repeat=${REPEAT:-3}

.venv/bin/python benchmarks/benchmark_adaln_modulation.py   --real --dtype bf16 --warmup "$warmup" --repeat "$repeat"
.venv/bin/python benchmarks/benchmark_adaln_modulation.py   --real --dtype fp32 --warmup "$warmup" --repeat "$repeat"
