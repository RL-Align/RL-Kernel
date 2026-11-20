# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Comparison of a bf16-executed op against its fp32 reference with the WS1 contract tolerance."""

import torch

from rl_engine.contracts.numerical import load_contract, resolve_tolerance

_CONTRACT = load_contract()


def assert_matches_fp32_reference(actual, expected, *, judgment, op_class, name=""):
    """``torch.testing.assert_close`` with the contract's (judgment, op_class) tolerance.

    The bfloat16 cell is used: the ops execute in bf16, the contract's
    execution_dtype, even where an output is fp32. Both sides are compared in fp32.
    """
    tolerance = resolve_tolerance(
        _CONTRACT, judgment=judgment, op_class=op_class, dtype=torch.bfloat16
    )
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=tolerance.atol,
        rtol=tolerance.rtol,
        msg=(lambda message: f"{name}: {message}") if name else None,
    )
