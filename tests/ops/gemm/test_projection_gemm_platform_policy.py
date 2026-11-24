# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CPU inspection of device marks/oracle; never import or simulate MUSA hardware."""

import ast
from pathlib import Path

import pytest
import torch

SOURCE = Path(__file__).with_name("test_projection_gemm.py")


@pytest.mark.parametrize(
    "name", ("test_native_fp32_is_tree_output_cast", "test_canonical_k_rounding_is_observable")
)
@pytest.mark.parametrize("device,skip", (("cuda", False), ("musa", True)))
def test_tree_contract_has_an_explicit_cuda_only_skip(name, device, skip):
    tree = ast.parse(SOURCE.read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    marks = [
        node
        for node in function.decorator_list
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "skipif"
    ]
    assert len(marks) == 1
    # Evaluate only the actual skip decorator, not the module or test body.
    mark = eval(
        compile(ast.Expression(marks[0]), str(SOURCE), "eval"), {"pytest": pytest, "DEVICE": device}
    )
    assert mark.args == (skip,)
    assert "CUDA" in mark.kwargs["reason"] and "MUSA" in mark.kwargs["reason"]


def test_musa_reference_policy_executes_fp64_on_cpu():
    tree = ast.parse(SOURCE.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_fp64_reference"
    )
    namespace = {"torch": torch, "DEVICE": "musa"}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
    a = torch.tensor([[256, 1, -256, 1]], dtype=torch.bfloat16)
    b = torch.ones(4, 1, dtype=torch.bfloat16)
    reference = namespace["_fp64_reference"](a, b)
    assert reference.device.type == "cpu" and reference.dtype == torch.float64
    assert reference.item() == 2.0
    generic = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "test_canonical_k_mathematical_reference"
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "skipif"
        for node in generic.decorator_list
    )
