# SPDX-License-Identifier: Apache-2.0
"""Task 53/54 semantics and task 55 autograd wiring on CPU."""

from dataclasses import replace

import pytest
import torch

from rl_engine.backends.rocm.moe.output_combine import RocmOutputCombine
from rl_engine.contracts.operators.fixed_output_combine import (
    prepare_fixed_output_combine as prepare,
)
from rl_engine.contracts.operators.moe import MoEOutputCombineContractError
from rl_engine.ops.moe.output_combine import output_combine
from rl_engine.reference.moe.fixed_output_combine import fixed_output_combine_reference as reference
from rl_engine.reference.moe.inverse_mapping import inverse_map_expert_output
from tests.helpers.dsv4_output_combine import request


def raw_equal(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.equal(
        a.reshape(-1).contiguous().view(torch.uint8), b.reshape(-1).contiguous().view(torch.uint8)
    )


def prepared(plan, inputs):
    return prepare(plan, inputs, expected_mapping_run_id="run-53", num_experts=16)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_reference_uses_task52_mapping_and_each_accumulation_step(dtype, shared, index_dtype):
    plan, inputs = request(dtype=dtype, shared=shared, index_dtype=index_dtype)
    p = prepared(plan, inputs)
    result = reference(p, inputs.expert_output, inputs.shared_output, mapping_run_id="run-53")
    restored = inverse_map_expert_output(
        plan, inputs, num_tokens=2, expected_mapping_run_id="run-53"
    )
    # Direct scalar-column accumulation is independent of the reference's vector loop.
    for token in range(2):
        for col in range(plan.hidden_size):
            acc = torch.tensor(0.0)
            for slot in range(6):
                acc = acc + restored[token, slot, col].float()
                raw_equal(result.slot_steps[token, slot, col], acc)
            if shared:
                acc = acc + inputs.shared_output[token, col].float()
            raw_equal(result.before_cast[token, col], acc)
            raw_equal(result.output[token, col], acc.to(dtype))
    assert p.provenance["residual_owner"] == "P1.mhc_post"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_cancellation_ulp_signed_zero_and_one_final_cast(dtype):
    plan, inputs = request(t=1, h=4, dtype=dtype, shared=False)
    p = prepared(plan, inputs)
    rows = p.canonical_rows()[0]
    x = torch.zeros_like(inputs.expert_output)
    x[rows[:4], 0] = torch.tensor([2**24, 1, -(2**24), 1], dtype=dtype)
    x[:, 1] = -0.0
    x[rows[:3], 2] = torch.tensor([1, 2**-24, 2**-24], dtype=dtype)
    x[rows[:3], 3] = torch.tensor([1, 2**-8, 2**-8], dtype=dtype)
    trace = reference(p, x, mapping_run_id="run-53")
    raw_equal(trace.before_cast, torch.tensor([[1, 0, 1, 1 + 2**-7]], dtype=torch.float32))
    assert not torch.signbit(trace.output[0, 1])  # +0 accumulator, not a copy of slot zero.
    assert trace.output[0, 3] != 1  # per-slot BF16 rounding would lose both additions.


def test_shared_added_once_after_slots_before_cast():
    plan, inputs = request(t=1, h=1, shared=True)
    p = prepared(plan, inputs)
    x = torch.zeros_like(inputs.expert_output)
    rows = p.canonical_rows()[0]
    x[rows[0], 0], x[rows[1], 0] = 1, 2**-8
    shared = torch.full((1, 1), 2**-8, dtype=torch.bfloat16)
    trace = reference(p, x, shared, mapping_run_id="run-53")
    assert trace.slot_steps[0, -1, 0] == 1 + 2**-8
    assert trace.output.item() == 1 + 2**-7
    assert trace.output.item() != (trace.slot_steps[:, -1].bfloat16() + shared).item()


@pytest.mark.parametrize(
    "case",
    [
        "weights",
        "reweight",
        "residual",
        "topk",
        "schema",
        "version",
        "unknown_id",
        "stale_id",
        "missing",
        "duplicate",
        "bad_expert",
        "empty",
        "nan_shared",
    ],
)
def test_invalid_requests_fail_before_execution(case):
    plan, inputs = request()
    expected = "run-53"
    if case == "weights":
        inputs = replace(inputs, routing_weights=torch.ones(12))
    elif case == "reweight":
        plan = replace(plan, expert_weight_applied=False)
    elif case == "residual":
        plan = replace(plan, residual_mode="external")
        inputs = replace(inputs, residual=torch.zeros(2, 33, dtype=torch.bfloat16))
    elif case == "topk":
        plan = replace(plan, top_k=2, merge_order=(0, 1))
    elif case == "schema":
        plan = replace(plan, mapping_schema_version="other")
    elif case == "version":
        plan = replace(plan, plan_version="other")
    elif case == "unknown_id":
        expected = "unknown"
    elif case == "stale_id":
        expected = "new-run"
    elif case == "missing":
        inputs = replace(inputs, expert_output=inputs.expert_output[:-1])
    elif case == "duplicate":
        inputs.token_indices[1] = inputs.token_indices[0]
        inputs.slot_indices[1] = inputs.slot_indices[0]
    elif case == "bad_expert":
        inputs.expert_indices[0] = 16
    elif case == "empty":
        inputs = replace(inputs, num_tokens=0)
    else:
        inputs.shared_output[0, 0] = float("nan")
    with pytest.raises(MoEOutputCombineContractError):
        prepare(plan, inputs, expected_mapping_run_id=expected, num_experts=16)


def test_mapping_snapshot_is_private_and_run_id_is_checked():
    plan, inputs = request()
    p = prepared(plan, inputs)
    original = p.canonical_rows()
    inputs.slot_indices.fill_(0)
    copy = p.canonical_rows()
    copy.fill_(0)
    raw_equal(p.canonical_rows(), original)
    with pytest.raises(MoEOutputCombineContractError, match="run id"):
        output_combine(p, inputs.expert_output, inputs.shared_output, mapping_run_id="new-run")
    p._rows[0, 0] = p._rows[0, 1]
    with pytest.raises(MoEOutputCombineContractError, match="modified"):
        p.validate_execution(inputs.expert_output, inputs.shared_output, "run-53")


def test_preparation_from_inference_mode():
    with torch.inference_mode():
        plan, inputs = request()
        p = prepared(plan, inputs)
        p.validate_execution(inputs.expert_output, inputs.shared_output, "run-53")
    assert not p._rows.is_inference() and not p._tokens.is_inference()


@pytest.mark.parametrize("debug", [False, True])
def test_first_order_output_vjp_and_saved_map_guard(monkeypatch, debug):
    # CPU wiring test. Native forward and VJP are covered by interpreter/GPU suites.
    def forward(p, x, shared, flag):
        r = reference(p, x, shared, mapping_run_id="run-53")
        return r.output, r.slot_steps if flag else None, r.before_cast if flag else None

    monkeypatch.setattr(RocmOutputCombine, "forward", forward)
    monkeypatch.setattr(
        RocmOutputCombine,
        "backward",
        lambda dy, tokens, dtype: dy.index_select(0, tokens).to(dtype),
    )
    plan, inputs = request()
    p = prepared(plan, inputs)
    x, shared = inputs.expert_output.requires_grad_(), inputs.shared_output.requires_grad_()
    out = output_combine(p, x, shared, mapping_run_id="run-53", debug=debug)
    if debug:
        assert not out.slot_steps.requires_grad and not out.before_cast.requires_grad
        out = out.output
    dy = torch.arange(out.numel()).reshape(out.shape).to(out.dtype)
    dx, ds = torch.autograd.grad(out, (x, shared), dy)
    raw_equal(dx, dy[inputs.token_indices])
    raw_equal(ds, dy)
    y = output_combine(p, x, shared, mapping_run_id="run-53")
    p._tokens[0] = 1 - p._tokens[0]
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        y.sum().backward()


def test_cpu_cannot_enter_rocm_execution():
    plan, inputs = request()
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        output_combine(
            prepared(plan, inputs),
            inputs.expert_output,
            inputs.shared_output,
            mapping_run_id="run-53",
        )
