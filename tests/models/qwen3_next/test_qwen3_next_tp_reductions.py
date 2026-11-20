# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Every Qwen3-Next cross-rank sum goes through RL-Kernel's fixed-order collective.

``torch.distributed`` reductions on NCCL change their summation order with the
NCCL algorithm, protocol and channel count, so a raw ``all_reduce`` in the
Qwen3-Next model or engine adapters would break replay/rollout logprob parity.
This scans those modules; data movement (``all_gather``) is allowed, and so is
an integer MAX of flags, whose result does not depend on order.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODULES = sorted(
    {
        *ROOT.glob("rl_engine/models/qwen3_next/*.py"),
        *ROOT.glob("rl_engine/integrations/engines/rollout/vllm/qwen3_next_*.py"),
        *ROOT.glob("rl_engine/integrations/engines/train/vllm/qwen3_next_*.py"),
        *ROOT.glob("rl_engine/integrations/orchestrators/vime/providers/qwen3_next_*.py"),
        *ROOT.glob("rl_engine/integrations/orchestrators/vime/experiments/qwen3_next/*.py"),
        ROOT / "rl_engine/integrations/common/qwen3_next_binary.py",
    }
)
REDUCTIONS = {
    "all_reduce",
    "all_reduce_coalesced",
    "reduce",
    "reduce_scatter",
    "reduce_scatter_tensor",
    "_reduce_scatter_base",
}
# (module, function): why the reduction is exact regardless of order.
ALLOWED = {
    ("qwen3_next_training.py", "_any_rank_nonzero"): "integer MAX of per-rank flags",
}


def _raw_reductions(path):
    tree = ast.parse(path.read_text())
    found = []
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(function):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in REDUCTIONS
                and isinstance(node.func.value, (ast.Name, ast.Attribute))
                and ast.unparse(node.func.value) in ("dist", "torch.distributed")
            ):
                found.append((path.name, function.name, node.lineno))
    return found


def test_scanned_modules_exist():
    assert len(MODULES) >= 20
    assert all(path.is_file() for path in MODULES)


def test_no_raw_nccl_reduction_in_qwen3_next_forward_paths():
    offenders = [
        f"{name}:{line} in {function}()"
        for path in MODULES
        for name, function, line in _raw_reductions(path)
        if (name, function) not in ALLOWED
    ]
    assert not offenders, "raw torch.distributed reductions: " + ", ".join(offenders)


def test_allowed_reduction_is_an_integer_max():
    path = ROOT / "rl_engine/integrations/orchestrators/vime/providers/qwen3_next_training.py"
    source = ast.get_source_segment(
        path.read_text(),
        next(
            node
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.FunctionDef) and node.name == "_any_rank_nonzero"
        ),
    )
    assert "dtype=torch.int32" in source and "ReduceOp.MAX" in source


def test_tp_reduction_uses_the_fixed_order_collective():
    tree = ast.parse((ROOT / "rl_engine/models/qwen3_next/qwen3_next_tp.py").read_text())
    reduce = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_reduce"
    )
    calls = {ast.unparse(n.func) for n in ast.walk(reduce) if isinstance(n, ast.Call)}
    assert "collective_for_group" in calls and "collective.all_reduce" in calls
