# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import gc
import weakref

import pytest
import torch

from rl_engine.kernels.dsv4.attention.mqa_joint_attention_sink import MqaJointAttentionSinkOp
from rl_engine.kernels.dsv4.attention.fixtures.catalog import make_attn_case


def _case():
    case = make_attn_case(
        "autograd_memory",
        layer_type="C0",
        tokens=1,
        n_compressed=0,
        n_recent=2,
        sink_mode="shared",
    )
    for tensor in (case.q, case.k, case.v, case.sink):
        tensor.requires_grad_()
    return case


def _forward(case):
    return MqaJointAttentionSinkOp(backend="oracle").apply_autograd(
        case.q,
        case.k,
        case.v,
        case.sink,
        case.plan,
        state_gate=case.state_gate,
        output_fp32=True,
    )


@pytest.mark.parametrize("run_backward", [False, True])
def test_output_releases_without_cyclic_gc(run_backward):
    case = _case()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        out = _forward(case)
        if run_backward:
            out.sum().backward()
        output_ref = weakref.ref(out)
        del out
        assert output_ref() is None
    finally:
        if gc_was_enabled:
            gc.enable()


def test_backward_releases_saved_probabilities_with_output_alive():
    case = _case()
    saved_refs = []

    def pack(tensor):
        saved_refs.append(weakref.ref(tensor))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        out = _forward(case)
    out.sum().backward()
    assert out.grad_fn is not None
    assert len(saved_refs) == 7
    assert saved_refs[4]() is None
    assert saved_refs[5]() is None
