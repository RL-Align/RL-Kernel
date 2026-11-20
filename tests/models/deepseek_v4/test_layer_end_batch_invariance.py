# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Batch invariance of the DeepSeek-V4 layer end on ROCm (#508, #510).

mHC head -> final RMSNorm -> lm_head -> raw logprob (DSV4 RFC task 8). A
token's stages must have the same bits whatever other tokens share the batch.
The layer end is wrapped as a row-wise case and run through the shared checks
in rl_engine.validation.common.batch_invariance at DeepSeek-V4-Flash's
C = 4096:

- forward batch invariance at the same M values as the mHC ops;
- chunking, reordering and padding;
- the same inputs in non-contiguous views (other strides);
- run-to-run determinism of forward and backward;
- forward with grad (training) equals forward without grad (inference);
- backward batch invariance (d logprob / d residual).

Every backward of the chain is fixed order and row-local for the input
gradient: mHC head's Triton backward, task 5's Triton RMSNorm backward
(``_backward_dx``, one program per token), lm_head's fixed-order fp32 det_gemm
and the row-wise Triton logprob backward.
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
    V,
    forward_backward,
    randn,
    rocm_gpu_available,
    weights,
)

_C = 4096  # DeepSeek-V4-Flash-0731 hidden_size

pytestmark = pytest.mark.skipif(not rocm_gpu_available(), reason="ROCm GPU required.")


@pytest.fixture(scope="module")
def cases():
    """(forward case, forward-and-backward case) of the layer end.

    Rows are the residual and target token ids; backward is d(sum logprob)/d residual.
    Outputs are in layer-end order, so the first differing name is the first stage
    where the bits diverged (task 8).
    """
    from rl_engine.models.deepseek_v4.layer_end import LAYER_END_STAGES, mhc_layer_end

    w = weights(_C)

    def make_rows(m, seed):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        residual = randn(gen, m, N, _C).to(torch.bfloat16)
        return [residual, torch.randint(0, V, (m,), generator=gen, device="cuda")]

    def run_layer_end(residual, targets):
        stages = mhc_layer_end(
            residual,
            w["head_fn"],
            w["head_scale"],
            w["head_base"],
            w["norm_weight"],
            w["lm_head_weight"],
            targets,
            backend="rocm",
        )
        return [stages[s] for s in LAYER_END_STAGES]

    def forward(rows):
        with torch.no_grad():
            return run_layer_end(*rows)

    def backward(rows):
        outputs, grads = forward_backward(run_layer_end, rows)
        return outputs + grads

    return (
        RowwiseCase("mhc_layer_end", LAYER_END_STAGES, forward, make_rows),
        RowwiseCase("mhc_layer_end", (*LAYER_END_STAGES, "dResidual"), backward, make_rows),
    )


@pytest.mark.parametrize("m", M_VALUES)
def test_forward_batch_invariance(cases, m):
    forward, _ = cases
    differing = differing_in_batch(forward, m)
    assert not differing, f"M={m}: differs from M=1: {differing}"


def test_forward_chunk_reorder_pad_invariance(cases):
    forward, _ = cases
    differing = differing_after_chunk_reorder_pad(forward, 4096)
    assert not differing, differing


def test_forward_restride_invariance(cases):
    forward, _ = cases
    differing = differing_after_restride(forward, 1024)
    assert not differing, differing


def test_forward_and_backward_run_to_run_deterministic(cases):
    _, backward = cases
    differing = differing_between_runs(backward, 2048)
    assert not differing, differing


def test_forward_with_and_without_grad_identical(cases):
    forward, backward = cases
    differing = differing_between(backward, forward, 1024)
    assert not differing, differing


@pytest.mark.parametrize("m", [m for m in M_VALUES if m > 1])
def test_backward_batch_invariance(cases, m):
    _, backward = cases
    differing = differing_in_batch(backward, m)
    assert not differing, f"M={m}: differs from M=1: {differing}"
