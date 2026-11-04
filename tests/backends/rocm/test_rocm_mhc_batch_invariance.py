# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Batch invariance of the ROCm mHC pre, mHC post and mHC head ops (#508, #510).

This is the main acceptance check for every integrated mHC op: a token's
result must have the same bits whatever other tokens share the batch. Each op
is wrapped as a row-wise case and run through the shared checks in
rl_engine.validation.common.batch_invariance:

- forward batch invariance at M = 1, 16, 17 (RFC switch points), the WS1
  manifest's token counts 8 and 59, aiter's K-split switch points
  1024 ... 16384, and the upper bound of every bucket of the gfx942 mHC pre
  and mHC head tables, so every tuned config runs;
- chunking, reordering and padding;
- the same inputs in non-contiguous views (other strides);
- run-to-run determinism of forward and backward;
- forward with grad (training) equals forward without grad (inference);
- which launch configs ran.

Every op runs at DeepSeek-V4-Flash's C = 4096, which has tuned launch-config
tables on gfx942, and at C = 7168, which has none and is not a power of two,
so the default configs are checked too. The DeepSeek-V4 layer end is checked
in tests/models/deepseek_v4/test_layer_end_batch_invariance.py.

Backward batch invariance is checked too but is expected to fail: aiter's
backward recomputes the formula in PyTorch. It is not an acceptance item for
these tasks (the backward belongs to task 6).
"""

import pytest
import torch

from rl_engine.validation.common.batch_invariance import (
    RowwiseCase,
    differing_after_chunk_reorder_pad,
    differing_after_restride,
    differing_between,
    differing_between_runs,
    differing_in_batch,
)
from tests.helpers.deepseek_v4_mhc import (
    M_VALUES,
    N,
    aiter_mhc_available,
    forward_backward,
    randn,
    weights,
)

# (op, C): every op at the tuned C = 4096 and at an untuned C.
_OP_C = [(op, c) for c in (4096, 7168) for op in ("mhc_pre", "mhc_post", "mhc_head")]
_OP_C_IDS = [f"{op}-C{c}" for op, c in _OP_C]

pytestmark = pytest.mark.skipif(
    not aiter_mhc_available(), reason="ROCm GPU with aiter DeepSeek-V4 mHC ops required."
)


def _cases(c):
    """{op: (forward case, forward-and-backward case)} at hidden size c."""
    from rl_engine.runtime.registry import kernel_registry

    w = weights(c)
    pre = kernel_registry.get_op("mhc_pre", device="cuda")
    post = kernel_registry.get_op("mhc_post", device="cuda")
    head = kernel_registry.get_op("mhc_head", device="cuda")

    # mhc_pre rows: residual and one output gradient per output.
    def pre_rows(m, seed):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        return [
            randn(gen, m, N, c).to(torch.bfloat16),
            randn(gen, m, N),
            randn(gen, m, N, N),
            randn(gen, m, c).to(torch.bfloat16),
        ]

    def run_pre(residual):
        return list(pre(residual, w["pre_fn"], w["pre_scale"], w["pre_base"]))

    def pre_forward(rows):
        with torch.no_grad():
            return run_pre(rows[0])

    def pre_backward(rows):
        leaf = rows[0].detach().clone().requires_grad_(True)
        outputs = run_pre(leaf)
        torch.autograd.backward(outputs, rows[1:])
        return [o.detach() for o in outputs] + [leaf.grad.detach()]

    # mhc_post rows: y, residual, post_mix, comb_mix, and the output gradient.
    def post_rows(m, seed):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        return [
            randn(gen, m, c).to(torch.bfloat16),
            randn(gen, m, N, c).to(torch.bfloat16),
            2.0 * torch.sigmoid(randn(gen, m, N)),
            torch.softmax(randn(gen, m, N, N), dim=-1),
            randn(gen, m, N, c).to(torch.bfloat16),
        ]

    def post_forward(rows):
        with torch.no_grad():
            return [post(*rows[:4])]

    def post_backward(rows):
        outputs, grads = forward_backward(lambda *x: [post(*x)], rows[:4], rows[4])
        return outputs + grads

    # mhc_head rows: residual and the output gradient.
    def head_rows(m, seed):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        return [randn(gen, m, N, c).to(torch.bfloat16), randn(gen, m, c).to(torch.bfloat16)]

    def run_head(residual):
        return [head(residual, w["head_fn"], w["head_scale"], w["head_base"])]

    def head_forward(rows):
        with torch.no_grad():
            return run_head(rows[0])

    def head_backward(rows):
        outputs, grads = forward_backward(run_head, rows[:1], rows[1])
        return outputs + grads

    pre_outputs = ("post_mix", "comb_mix", "layer_input")
    return {
        "mhc_pre": (
            RowwiseCase("mhc_pre", pre_outputs, pre_forward, pre_rows),
            RowwiseCase("mhc_pre", (*pre_outputs, "dResidual"), pre_backward, pre_rows),
        ),
        "mhc_post": (
            RowwiseCase("mhc_post", ("R_new",), post_forward, post_rows),
            RowwiseCase(
                "mhc_post", ("R_new", "dy", "dR_old", "dPOST", "dC"), post_backward, post_rows
            ),
        ),
        "mhc_head": (
            RowwiseCase("mhc_head", ("head",), head_forward, head_rows),
            RowwiseCase("mhc_head", ("head", "dResidual"), head_backward, head_rows),
        ),
    }


@pytest.fixture(scope="module")
def cases():
    built = {}

    def get(op, c):
        if c not in built:
            built[c] = _cases(c)
        return built[c][op]

    return get


@pytest.mark.parametrize("m", M_VALUES)
@pytest.mark.parametrize("op,c", _OP_C, ids=_OP_C_IDS)
def test_forward_batch_invariance(cases, op, c, m):
    forward, _ = cases(op, c)
    differing = differing_in_batch(forward, m)
    # Names are in output order: the first one is where the bits diverged first.
    assert not differing, f"{op}, C={c}, M={m}: differs from M=1: {differing}"


@pytest.mark.parametrize("op,c", _OP_C, ids=_OP_C_IDS)
def test_forward_chunk_reorder_pad_invariance(cases, op, c):
    forward, _ = cases(op, c)
    differing = differing_after_chunk_reorder_pad(forward, 4096)
    assert not differing, f"{op}, C={c}: {differing}"


@pytest.mark.parametrize("op,c", _OP_C, ids=_OP_C_IDS)
def test_forward_restride_invariance(cases, op, c):
    forward, _ = cases(op, c)
    differing = differing_after_restride(forward, 1024)
    assert not differing, f"{op}, C={c}: {differing}"


@pytest.mark.parametrize("op,c", _OP_C, ids=_OP_C_IDS)
def test_forward_and_backward_run_to_run_deterministic(cases, op, c):
    _, backward = cases(op, c)
    differing = differing_between_runs(backward, 2048)
    assert not differing, f"{op}, C={c}: {differing}"


@pytest.mark.parametrize("op,c", _OP_C, ids=_OP_C_IDS)
def test_forward_with_and_without_grad_identical(cases, op, c):
    forward, backward = cases(op, c)
    differing = differing_between(backward, forward, 1024)
    assert not differing, f"{op}, C={c}: {differing}"


# Which launch configs actually ran: the tuned table at C = 4096 on gfx942, the
# default config everywhere else. Printed so the test log records them.
@pytest.mark.parametrize(
    "op,c",
    [(op, c) for op, c in _OP_C if op in ("mhc_pre", "mhc_head")],
    ids=[i for (op, _), i in zip(_OP_C, _OP_C_IDS) if op in ("mhc_pre", "mhc_head")],
)
def test_launch_configs_that_ran(cases, op, c):
    from rl_engine.backends.rocm import config_utils

    forward, _ = cases(op, c)
    forward.run(forward.make_rows(17, 0))
    records = [
        r
        for r in config_utils.selected_configs()
        if r["op"] == op
        and r["file"].endswith(f"-C={c}.json")
        and r["m_min"] <= 17 <= r["m_max"]
    ]
    print(records)
    assert len(records) == 1
    tuned = c == 4096 and records[0]["arch"] == "gfx942"
    assert (records[0]["source"] != "default") == tuned


@pytest.mark.xfail(reason="aiter recomputes backward in PyTorch; not batch invariant")
@pytest.mark.parametrize("m", [m for m in M_VALUES if m > 1])
@pytest.mark.parametrize("op,c", _OP_C, ids=_OP_C_IDS)
def test_backward_batch_invariance(cases, op, c, m):
    _, backward = cases(op, c)
    differing = differing_in_batch(backward, m)
    assert not differing, f"{op}, C={c}, M={m}: differs from M=1: {differing}"
