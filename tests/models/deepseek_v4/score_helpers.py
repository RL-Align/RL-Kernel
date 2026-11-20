# SPDX-License-Identifier: Apache-2.0
"""Test-only oracles for the Indexer score; never a production fallback."""

import torch

from rl_engine.contracts.operators.indexer_score import SCORE_BACKEND_ID


class ReferenceScoreBackend:
    backend_id = SCORE_BACKEND_ID
    device_type = "cpu"

    @staticmethod
    def plan(contract):
        return {"reference": "cpu_sum"}

    def row_dot(self, a, b):
        return (a * b).sum(-1)


def head_partials_fp64(icv, weights, contract):
    """Math order is irrelevant here; only the result is compared, within a bound."""
    relu = torch.relu(icv.cpu().double()) * contract.softmax_scale
    w = weights.cpu().double() * contract.head_scale
    return relu * w[:, :, None]


def score_fp64(icv, weights, contract):
    return head_partials_fp64(icv, weights, contract).sum(1)


def vjp_fp64(icv, weights, grad, contract):
    x = icv.cpu().double().requires_grad_()
    w = weights.cpu().double().requires_grad_()
    score_fp64(x, w, contract).backward(grad.cpu().double())
    return x.grad, w.grad


def byte_equal(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def normalized_error(actual, expected):
    scale = expected.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    return ((actual.cpu().double() - expected).abs() / scale).max().item()
