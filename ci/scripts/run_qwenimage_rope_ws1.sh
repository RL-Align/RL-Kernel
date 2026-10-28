#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Operator-only gate: requires an ordinary, complete rl_engine._C installation.
set -euo pipefail

ROPE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROPE_ROOT"
ROPE_PYTHON="${ROPE_PYTHON:-python3}"
ROPE_WS1_OUTPUT="${ROPE_WS1_OUTPUT:-${TMPDIR:-/tmp}/rlk-qwenimage-rope}"
mkdir -p "$ROPE_WS1_OUTPUT"
export RL_KERNEL_REQUIRE_EXT=1

"$ROPE_PYTHON" - "$ROPE_WS1_OUTPUT/environment.json" <<'PY'
import hashlib
import json
import platform
import sys
from pathlib import Path

import torch
import triton

from build_tools.envs import env_flag
from rl_engine import _C

if env_flag("RL_KERNEL_ROPE_ONLY"):
    raise SystemExit("The complete _C build is required; unset RL_KERNEL_ROPE_ONLY")
if env_flag("KERNEL_ALIGN_USE_FAST_MATH"):
    raise SystemExit("This numerical profile does not qualify fast-math builds")
if torch.version.hip is not None or not torch.cuda.is_available():
    raise SystemExit("A real NVIDIA CUDA device is required")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("The WS1 BF16 profile requires BF16-capable hardware")
if not hasattr(torch.ops.rl_kernel_native, "multi_axis_rope"):
    raise SystemExit("The imported _C library does not register multi_axis_rope")
library = Path(_C.__file__).resolve()
report = {
    "scope": "Qwen-Image multi_axis_rope operator; ordinary full _C build",
    "python": platform.python_version(),
    "platform": platform.platform(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "triton": triton.__version__,
    "gpu": torch.cuda.get_device_name(),
    "capability": list(torch.cuda.get_device_capability()),
    "native_library": str(library),
    "native_library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
}
Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report))
PY

"$ROPE_PYTHON" -m pytest tests/ops/rope/test_multi_axis_rope.py -q -rs \
    --junitxml="$ROPE_WS1_OUTPUT/operator-tests.xml"

# GPU/backward/compile cases must execute. Only the two-device guard may skip
# on a single-GPU runner; missing CUDA/Triton support is not acceptance evidence.
"$ROPE_PYTHON" - "$ROPE_WS1_OUTPUT/operator-tests.xml" <<'PY'
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
unexpected = [case.get("name") for case in root.iter("testcase")
              if case.find("skipped") is not None and case.get("name") != "test_device_guard"]
if unexpected:
    raise SystemExit(f"Required RoPE tests skipped: {unexpected}")
PY

for backend in cuda triton; do
    "$ROPE_PYTHON" tools/validation/operators/check_operator.py \
        --op multi_axis_rope --candidate "$backend" --device cuda --dtype bf16 \
        --batch 2 --seq 7 --n-heads 3 --check-grad --json \
        > "$ROPE_WS1_OUTPUT/operator-$backend.json"
    "$ROPE_PYTHON" tools/validation/operators/validate_multi_axis_rope.py \
        --ws1 "$backend" --output "$ROPE_WS1_OUTPUT/ws1-$backend.json"
done

"$ROPE_PYTHON" - "$ROPE_WS1_OUTPUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for backend in ("cuda", "triton"):
    report = json.loads((root / f"ws1-{backend}.json").read_text(encoding="utf-8"))
    if not report["passed"] or report["trace"]["actual_backend"] != f"rlkernel.{backend}.qwenimage_multi_axis_rope.v1":
        raise SystemExit(f"WS1 failed or selected an unexpected backend: {backend}")
print("Qwen-Image RoPE full-extension operator gate passed")
PY
