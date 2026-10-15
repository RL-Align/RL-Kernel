# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Import Miles' and verl's own kernels from pinned checkouts, unmodified.

Neither repository is installed. The measured modules are imported from a
checkout of the commit named in the report, and the SHA-256 of every file used
is recorded, so a result can be tied to exact third-party bytes.

* Miles (``radixark/miles``): every ``__init__`` on the path is empty, so the
  checkout is imported as it is. Its MoE forward calls SGLang functions by an
  import path that SGLang 0.5.21 no longer has; :func:`install_sglang_aliases`
  forwards that path to the functions' new location, nothing else.
* verl (``volcengine/verl``): ``verl/__init__.py`` imports ``ray`` and the rest
  of the trainer. A temporary package keeps the two ``__init__`` files empty and
  links the real ``utils/device.py``, ``utils/kernel/`` and ``plugin/``.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
import types
from pathlib import Path

_USED = {
    "miles": (
        "miles/kernels/moe/fused_experts.py",
        "miles/kernels/moe/fused_experts_bwd.py",
        "miles/kernels/softmax/token_log_softmax.py",
        "miles/backends/training_utils/loss/hub/fused_log_probs.py",
    ),
    "verl": (
        "verl/utils/kernel/kernels.py",
        "verl/utils/kernel/linear_cross_entropy.py",
        "verl/utils/device.py",
    ),
}
_TREES: list = []  # keep temporary directories alive


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance(name: str, src: Path) -> dict:
    commit = subprocess.run(
        ["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return {
        "repository": name,
        "commit": commit,
        "files": {rel: _sha256(src / rel) for rel in _USED[name]},
    }


def load_miles(src: Path) -> dict:
    """Make ``miles.*`` importable from the checkout; returns its provenance."""
    sys.path.insert(0, str(src))
    return provenance("miles", src)


def load_verl(src: Path) -> dict:
    """Make ``verl.utils.kernel`` importable without ``verl/__init__``; returns provenance."""
    tree = tempfile.TemporaryDirectory(prefix="verl-kernel-")
    _TREES.append(tree)
    root = Path(tree.name)
    for package in ("verl", "verl/utils"):
        (root / package).mkdir(parents=True, exist_ok=True)
        (root / package / "__init__.py").write_text("")
    for link in ("verl/utils/device.py", "verl/utils/kernel", "verl/plugin"):
        (root / link).symlink_to(src / link)
    sys.path.insert(0, str(root))
    return provenance("verl", src)


def install_sglang_aliases() -> None:
    """Forward Miles' SGLang import path to SGLang 0.5.21's functions.

    Miles imports ``invoke_fused_moe_kernel``, ``moe_align_block_size``,
    ``moe_sum_reduce`` and ``silu_and_mul`` from
    ``sglang.srt.layers.moe.fused_moe_triton.fused_moe``; in 0.5.21 they live in
    ``sglang.kernels.ops`` and ``sglang.srt.layers.moe.moe_runner.triton_utils``.
    The signatures Miles calls are the same.
    """
    from benchmarks.models.qwen3_next.moe_route_candidates import _sglang_shims

    _sglang_shims()  # sgl_kernel is built for another libtorch ABI; see that function
    from sglang.kernels.ops.activation import silu_and_mul
    from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
        invoke_fused_moe_kernel,
        moe_sum_reduce_triton,
    )
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )

    name = "sglang.srt.layers.moe.fused_moe_triton.fused_moe"
    module = types.ModuleType(name)
    module.invoke_fused_moe_kernel = invoke_fused_moe_kernel
    module.moe_align_block_size = moe_align_block_size
    module.moe_sum_reduce = moe_sum_reduce_triton
    module.silu_and_mul = silu_and_mul
    parent_name = name.rsplit(".", 1)[0]
    if parent_name not in sys.modules:
        parent = types.ModuleType(parent_name)
        parent.__path__ = []
        sys.modules[parent_name] = parent
    sys.modules[name] = module
