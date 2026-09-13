# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Backward partition regressions; CPU math checks plus real NPU execution.

Only the Ascend forward extension is substituted on CPU. The production
canonical session, embedding reduction and RMSNorm backward run unchanged.
The NPU parametrization uses the compiled extension without substitutions.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rl_engine.kernels.ops.ascend.norm import rmsnorm as ascend_rms
from rl_engine.kernels.ops.canonical_backward import canonical_backward_session
from rl_engine.kernels.ops.canonical_embedding import canonical_embedding
from rl_engine.kernels.ops.canonical_rmsnorm import canonical_ascend_rmsnorm


@pytest.fixture(params=("cpu", "npu"))
def device(request, monkeypatch):
    if request.param == "cpu":

        def forward(x, weight, rstd):
            return (x.float() * rstd.unsqueeze(-1) * weight.float()).to(x.dtype)

        monkeypatch.setattr(ascend_rms, "_C_npu", SimpleNamespace(rmsnorm_ascend=forward))
        return torch.device("cpu")

    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU is unavailable")
    from rl_engine import _C_npu

    # Import again after torch_npu has loaded its shared libraries. The
    # module-level optional import may have run before device registration.
    monkeypatch.setattr(ascend_rms, "_C_npu", _C_npu)
    # If NPU is present, a missing extension is a failure, not a skip.
    assert hasattr(ascend_rms._C_npu, "rmsnorm_ascend"), "build the Ascend extension first"
    return torch.device("npu")


@pytest.mark.parametrize("hidden", (1, 33, 128, 4096))
def test_row_sum_uses_fixed_fp32_pairs(device, hidden):
    generator = torch.Generator().manual_seed(20260812)
    values = torch.randn(5, hidden, generator=generator)
    if hidden >= 4:
        values[:, :4] = torch.tensor([1.0e20, 3.0, -1.0e20, 7.0])
    # Independent scalar FP32 oracle; also exercises odd-width tails.
    expected = []
    for row in values.numpy():
        partial = list(row)
        while len(partial) > 1:
            pairs = [np.float32(partial[i] + partial[i + 1]) for i in range(0, len(partial) - 1, 2)]
            if len(partial) % 2:
                pairs.append(partial[-1])
            partial = pairs
        expected.append(partial[0])
    actual = ascend_rms._fixed_row_sum(values.to(device)).cpu()
    assert torch.equal(actual.view(torch.int32), torch.tensor(np.array(expected)).view(torch.int32))

    # Accumulation must not inherit BF16 input precision.
    low_precision = torch.tensor([[256.0, 1.0, 1.0, 1.0]], dtype=torch.bfloat16, device=device)
    assert ascend_rms._fixed_row_sum(low_precision).item() == 259.0


@pytest.mark.parametrize("hidden", (128, 4096))
def test_backward_matches_independent_float64_autograd(device, hidden):
    generator = torch.Generator().manual_seed(406)
    x_cpu = torch.randn(5, hidden, generator=generator)
    weight_cpu = torch.randn(hidden, generator=generator)
    dy_cpu = torch.randn(5, hidden, generator=generator)

    x_ref = x_cpu.double().requires_grad_()
    weight_ref = weight_cpu.double().requires_grad_()
    y_ref = x_ref * torch.rsqrt(x_ref.square().mean(-1, keepdim=True) + 1e-6) * weight_ref
    dx_ref, dw_ref = torch.autograd.grad(y_ref, (x_ref, weight_ref), dy_cpu.double())

    x, weight, dy = (t.to(device) for t in (x_cpu, weight_cpu, dy_cpu))
    rstd = ascend_rms._fixed_rstd(x, 1e-6)
    dx, dw = ascend_rms._rms_norm_backward(x, weight, rstd, dy)
    torch.testing.assert_close(dx.cpu(), dx_ref.float(), atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(dw.cpu(), dw_ref.float(), atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("hidden", (128, 4096))
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_canonical_embedding_and_norm_gradients_ignore_chunk_boundaries(device, hidden, dtype):
    generator = torch.Generator().manual_seed(20260812)
    # Match the observed 140-row workload and 7-row chunks. Repeated token
    # IDs exercise embedding aggregation, not just disjoint scatter writes.
    rows, vocab = 140, 17
    table = torch.randn(vocab, hidden, generator=generator).to(device=device, dtype=dtype)
    weights = [
        torch.randn(hidden, generator=generator).to(device=device, dtype=dtype) for _ in range(2)
    ]
    upstream = torch.randn(rows, hidden, generator=generator).to(device=device, dtype=dtype)
    ids = (torch.arange(rows, device=device) % vocab).long()
    keys = torch.stack(
        (torch.arange(rows, device=device) // 20, torch.arange(rows, device=device) % 20), dim=-1
    )
    # Masked rows may carry finite garbage but must never contribute to dW.
    keys[19::20] = -1
    upstream[19::20] = 0

    def run(partitions):
        parameters = [table.detach().clone().requires_grad_()]
        parameters.extend(w.detach().clone().requires_grad_() for w in weights)
        output_by_row = torch.empty_like(upstream)
        dx_by_row = torch.empty_like(upstream)
        outputs, gradients, inputs = [], [], []
        with canonical_backward_session() as session:
            for selection in partitions:
                row_ids = ids.index_select(0, selection)
                logical_keys = keys.index_select(0, selection)
                x = canonical_embedding(
                    row_ids,
                    parameters[0],
                    logical_keys,
                    forward_op=lambda token_ids, weight: weight[token_ids],
                    family="ascend",
                )
                x.retain_grad()
                y = x
                for layer, weight in enumerate(parameters[1:]):
                    y = canonical_ascend_rmsnorm(
                        y,
                        weight,
                        eps=1e-6,
                        logical_keys=logical_keys,
                        parameter_id=f"norm.{layer}",
                    )
                outputs.append(y)
                gradients.append(upstream.index_select(0, selection))
                inputs.append((selection, x))
                output_by_row[selection] = y.detach()
            torch.autograd.backward(outputs, gradients)
            session.validate_complete()
        for selection, x in inputs:
            dx_by_row[selection] = x.grad
        return [
            output_by_row.cpu(),
            dx_by_row.cpu(),
            *(parameter.grad.cpu() for parameter in parameters),
        ]

    indices = torch.arange(rows, device=device)
    expected = run([indices])
    variants = (
        list(indices.split(7)),
        list(reversed(indices.split(7))),
        list(torch.randperm(rows, generator=generator).to(device).split(11)),
    )
    for partitions in variants:
        actual = run(partitions)
        for name, lhs, rhs in zip(
            ("output", "dx", "embedding.dw", "norm.0.dw", "norm.1.dw"),
            expected,
            actual,
            strict=True,
        ):
            bits = torch.int32 if dtype == torch.float32 else torch.int16
            assert torch.equal(lhs.view(bits), rhs.view(bits)), name
