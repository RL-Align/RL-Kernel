# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Public Triton callable regression for caller/target device and stream ownership."""

import pytest
import torch

from rl_engine.contracts.numerical import load_contract, resolve_tolerance
from rl_engine.validation.operators.operator_specs import GtestAdaLNReference
from tests.helpers.adaln_indexed import assert_bytes, small_case


@pytest.mark.cuda_only
@pytest.mark.parametrize("indexed", [False, True], ids=["shared-559", "indexed-621"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_triton_forward_on_other_device_preserves_caller_and_target_stream(indexed, dtype):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("two NVIDIA CUDA GPUs required; not a multi-GPU PASS")
    pytest.importorskip("triton")
    from rl_engine.backends.shared.triton.norm.adaln_modulation import TritonAdaLNModulationOp

    original_device = torch.cuda.current_device()
    original_stream = torch.cuda.current_stream(1)
    stream = torch.cuda.Stream(device=1)
    try:
        torch.cuda.set_device(0)
        torch.cuda.set_stream(stream)
        torch.cuda.set_device(0)
        x, m, index, dy, dg = small_case("cpu")
        if not indexed:
            m, dg = m[:2].detach().requires_grad_(), dg[:, :1]
        rx, rm = (
            x.to(dtype).float().detach().requires_grad_(),
            m.to(dtype).float().detach().requires_grad_(),
        )
        kwargs = {"modulate_index": index} if indexed else {}
        ry, rg = GtestAdaLNReference().forward_fp32(rx.float(), rm.float(), eps=7.0, **kwargs)
        rdx, rdm = torch.autograd.grad(
            (ry, rg), (rx, rm), (dy.to(dtype).float(), dg.to(dtype).float())
        )
        # Enqueue input copies and kernel work on the target's non-default stream.
        x = x.to(device="cuda:1", dtype=dtype).detach().requires_grad_()
        m = m.to(device="cuda:1", dtype=dtype).detach().requires_grad_()
        kwargs = {"modulate_index": index.to("cuda:1")} if indexed else {}
        y, gate = TritonAdaLNModulationOp()(x, m, eps=7.0, **kwargs)
        assert torch.cuda.current_device() == 0
        assert torch.cuda.current_stream(1) == stream
        dx, dm = torch.autograd.grad((y, gate), (x, m), (dy.to(y), dg.to(gate)))
        assert torch.cuda.current_device() == 0
        stream.synchronize()
        for actual, expected, judgment in (
            (y, ry, "forward_accuracy"),
            (gate, rg, "forward_accuracy"),
            (dx, rdx, "gradient_accuracy"),
            (dm, rdm, "gradient_accuracy"),
        ):
            tolerance = resolve_tolerance(
                load_contract(), judgment=judgment, op_class="reduction", dtype=dtype
            )
            torch.testing.assert_close(
                actual.cpu().float(), expected, atol=tolerance.atol, rtol=tolerance.rtol
            )
        assert_bytes(gate, rg.to(dtype))
    finally:
        torch.cuda.set_stream(original_stream)
        torch.cuda.set_device(original_device)
