# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Shared tolerance policy for the Muse-Glimmer [F1] norm tests.

One place rather than three, because the three [F1] test files have to agree on
what FP16 is held to or the operator is held to whichever file happens to be
stricter.

The policy: read the ``reduction`` row from the numerical contract, and where the
contract marks a dtype ``optional``, add a measured budget on top. FP32 and BF16
are ``applicable`` for both reduction judgments and are held to the row exactly --
measured over six shapes, six seeds and both GPU backends, they need nothing
added on any of the six outputs.

FP16 is ``optional`` and cannot be held to the declared 1e-3: the row sum over
6656 elements shows through a ten-bit mantissa. The worst output needs 0.135 of
one ULP at the tensor's peak beyond the row, so the budget is 0.25, about 1.9x
headroom.

Two things the budget deliberately is not.

It is not keyed to each element's own magnitude, which would read tighter. The
residual add cancels: ``residual_out`` has an element at magnitude 1.56e-02
carrying 1.95e-03 of error, which is 128 ULPs of its own value but a quarter of
one ULP at the row's scale, because the rounding comes from the operands and not
from the result. An elementwise bound calls that a failure and is wrong to.

And it is not chosen to make the tests pass. It is checked by injecting kernel
bugs and confirming the FP16 cases still go red: dropped ``(1 + w)`` 12 failed,
wrong eps 6, missing first narrowing 8, missing second narrowing 3, FP32 gradient
accumulation 2. An earlier version of this policy took one ULP at the peak, which
reached 125x the contract atol for the row-summed weight gradients and left
0.3-4% of elements with an allowance at or above their own value.
"""

from __future__ import annotations

import torch

from rl_engine.contracts.numerical import resolve_tolerance_support
from rl_engine.kernels.gtest.tolerance import load_contract, resolve_tolerance

PEAK_ULP_BUDGET = 0.25

_CONTRACT = load_contract()


def peak_ulp(reference: torch.Tensor, dtype: torch.dtype) -> float:
    """One ULP of ``dtype`` at the reference tensor's peak magnitude."""
    peak = torch.tensor(float(reference.abs().max()), dtype=dtype)
    if float(peak) == 0.0:
        return 0.0
    step = torch.nextafter(peak, torch.tensor(float("inf"), dtype=dtype)) - peak
    return float(step)


def allowance(
    judgment: str, dtype: torch.dtype, reference: torch.Tensor, *, budget: bool = True
) -> torch.Tensor:
    """Elementwise allowance: the contract row, plus the budget where optional.

    ``budget=False`` gives the contract row alone, for a negative control on a
    quantity whose own measurement shows the row already separates right from
    wrong. The budget is keyed to the tensor's peak, and for a gradient summed over
    thousands of rows that peak is large enough to swallow the signal: on
    ``d_post_w`` the plain row puts the correct eager accumulation at 0.16-0.74 of
    its allowance and the FP32 one at 1.98-40.6, and adding the budget takes the
    FP32 variant back inside. That is the looseness this policy exists to avoid, so
    the control opts out rather than the whole policy loosening.
    """
    tolerance = resolve_tolerance(_CONTRACT, judgment=judgment, op_class="reduction", dtype=dtype)
    reference_f = reference.detach().cpu().float()
    allowed = tolerance.atol + tolerance.rtol * reference_f.abs()
    support = resolve_tolerance_support(
        _CONTRACT, judgment=judgment, op_class="reduction", dtype=dtype
    )
    if budget and support.status != "applicable":
        allowed = allowed + PEAK_ULP_BUDGET * peak_ulp(reference_f, dtype)
    return allowed


def assert_within(
    judgment: str, dtype: torch.dtype, got, want, label: str = "", *, budget: bool = True
) -> None:
    """Compare at the allowance above, reporting the worst offender as a ratio.

    ``torch.testing.assert_close`` only takes scalar tolerances and the rtol term
    is elementwise, so the comparison is done by hand. Both sides are brought to
    the CPU so a caller cannot get a device mismatch by forgetting to.
    """
    got_f = got.detach().cpu().float()
    want_f = want.detach().cpu().float()
    allowed = allowance(judgment, dtype, want_f, budget=budget)
    error = (got_f - want_f).abs()
    ratio = error / allowed
    worst = int(ratio.argmax())
    assert bool((error <= allowed).all()), (
        f"{label}{judgment} outside the reduction row for {dtype}: "
        f"{int((ratio > 1).sum())} of {error.numel()} elements, worst "
        f"|error|={error.flatten()[worst]:.3e} against allowance "
        f"{allowed.flatten()[worst]:.3e} at |reference|={want_f.abs().flatten()[worst]:.3e} "
        f"(ratio {ratio.flatten()[worst]:.2f})"
    )
