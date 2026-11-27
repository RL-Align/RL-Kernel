# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU rounding regressions for the text projection's FP32 FMA chains."""

import os
import subprocess
import sys
from pathlib import Path


def test_generic_cpu_fma_chains() -> None:
    # PyTorch selects its CPU kernels once per process. Exercise the generic
    # path even on hosts where vectorized addcmul happens to be fused.
    subprocess.run(
        [sys.executable, "-m", "tests.ops.norm.test_txt_in_reference_contract"],
        cwd=Path(__file__).resolve().parents[3],
        env={**os.environ, "ATEN_CPU_CAPABILITY": "default"},
        check=True,
    )


def _check_fma_chains() -> None:
    import torch

    from rl_engine.reference.norm.txt_in_rmsnorm_linear import (
        TXT_IN_HIDDEN,
        _TxtInRMSNormLinearFunction,
        row_sumsq_tree,
        txt_in_row_dot_tree,
    )

    terms = torch.tensor([2.0**-30, 1.0 + 2.0**-12], dtype=torch.float32)
    # (1 + 2^-12)^2 is exactly the binary32 midpoint 1 + 2^-11 + 2^-24.
    # The first term contributes 2^-60, so the correctly rounded sum is
    # the float immediately above the midpoint.
    upper_bits = 0x3F801001
    row = terms.unsqueeze(0)
    assert row_sumsq_tree(row).view(torch.int32).item() == upper_bits
    assert txt_in_row_dot_tree(row, row).view(torch.int32).item() == upper_bits

    # Select the first normalized feature with a single output channel.
    # The second coordinate keeps each row's rounded mean square at 256,
    # giving exactly representable rstd=1/16 and xhat[:, 0]=terms.
    x = torch.full((2, TXT_IN_HIDDEN), 16.0)
    x[:, 0] = 16.0 * terms
    x[:, 1] = 16.0 * torch.sqrt(2.0 - terms.square())
    gamma = torch.ones(TXT_IN_HIDDEN, requires_grad=True)
    weight = torch.zeros(1, TXT_IN_HIDDEN)
    weight[0, 0] = 1.0
    weight.requires_grad_()

    out = _TxtInRMSNormLinearFunction.apply(x, gamma, weight, None)
    torch.testing.assert_close(out[:, 0], terms, rtol=0, atol=0)
    dgamma, dw = torch.autograd.grad(out, (gamma, weight), terms.unsqueeze(1))
    assert dgamma[0].view(torch.int32).item() == upper_bits
    assert dw[0, 0].view(torch.int32).item() == upper_bits


if __name__ == "__main__":
    _check_fma_chains()
