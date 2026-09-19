# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from types import SimpleNamespace

import pytest
import torch

import rl_engine.kernels.ops.pytorch.ffn.ffn as ffn


def test_single_rank_tp_group_is_treated_as_local(monkeypatch):
    group = object()
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 1)

    assert ffn._require_parallel_group(group, "tensor") is None


def test_tp2_column_projection_reuses_tp4_width(monkeypatch):
    calls = []

    def fake_linear(input_value, weight, *, disable_split_k):
        calls.append((input_value, weight.clone(), disable_split_k))
        return input_value @ weight.t()

    monkeypatch.setenv("RL_KERNEL_STRICT_CANONICAL_TP", "4")
    monkeypatch.setattr(ffn, "_linear_fwd", fake_linear)
    input_value = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    weight = torch.arange(32, dtype=torch.float32).reshape(8, 4)

    actual = ffn._canonical_tp_column_projection(
        input_value,
        weight,
        tp_world=2,
        disable_split_k=True,
    )

    assert torch.equal(actual, input_value @ weight.t())
    assert len(calls) == 2
    assert all(call[1].shape == (4, 4) and call[2] for call in calls)


def test_tp2_down_projection_reuses_tp4_subtrees(monkeypatch):
    calls = []

    def fake_linear(input_value, weight, *, disable_split_k):
        calls.append((input_value.clone(), weight.clone(), disable_split_k))
        return input_value @ weight.t()

    monkeypatch.setenv("RL_KERNEL_STRICT_CANONICAL_TP", "4")
    monkeypatch.setattr(ffn, "_linear_fwd", fake_linear)
    activated = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    down_weight = torch.arange(32, dtype=torch.float32).reshape(4, 8)

    actual = ffn._canonical_tp_down_projection(
        activated,
        down_weight,
        tp_world=2,
        disable_split_k=True,
    )
    expected = activated[:, :4] @ down_weight[:, :4].t()
    expected = expected + activated[:, 4:] @ down_weight[:, 4:].t()

    assert torch.equal(actual, expected)
    assert len(calls) == 2
    assert all(call[2] for call in calls)


def test_tp4_reference_path_keeps_single_gemm(monkeypatch):
    calls = []

    def fake_linear(input_value, weight, *, disable_split_k):
        calls.append((input_value, weight, disable_split_k))
        return input_value @ weight.t()

    monkeypatch.setenv("RL_KERNEL_STRICT_CANONICAL_TP", "4")
    monkeypatch.setattr(ffn, "_linear_fwd", fake_linear)
    input_value = torch.randn(2, 8)
    column_weight = torch.randn(8, 8)
    down_weight = torch.randn(4, 8)

    column = ffn._canonical_tp_column_projection(
        input_value,
        column_weight,
        tp_world=4,
        disable_split_k=True,
    )
    ffn._canonical_tp_down_projection(
        column,
        down_weight,
        tp_world=4,
        disable_split_k=True,
    )

    assert len(calls) == 2
    assert calls[0][0] is input_value
    assert calls[0][1] is column_weight
    assert calls[1][0] is column
    assert calls[1][1] is down_weight


def test_rollout_tp4_packed_ffn_reuses_tp8_shards(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.version, "hip", None)

    def fake_packed(input_value, fused_weight, down_weight):
        calls.append((fused_weight.shape, down_weight.shape))
        return input_value.new_full(
            (*input_value.shape[:-1], down_weight.size(0)),
            float(len(calls)),
        )

    monkeypatch.setenv("RL_KERNEL_STRICT_CANONICAL_TP", "8")
    monkeypatch.setattr(ffn, "_qwen3_ffn_packed_inference", fake_packed)
    monkeypatch.setattr(
        ffn,
        "deterministic_all_reduce_inplace",
        lambda output, collective_handle: output,
    )
    hidden = torch.zeros(2, 4)
    fused_gate_up = torch.zeros(16, 4)
    down = torch.zeros(4, 8)

    output = ffn.qwen3_ffn_packed_inference(
        hidden,
        fused_gate_up,
        down,
        collective_handle=1,
        tp_world_size=4,
    )

    assert calls == [
        (torch.Size([8, 4]), torch.Size([4, 4])),
        (torch.Size([8, 4]), torch.Size([4, 4])),
    ]
    assert torch.equal(output, torch.full((2, 4), 3.0))


@pytest.mark.parametrize("rows", [2, 64])
def test_rocm_canonical_ffn_resolves_aot_slot_before_ipc_reduction(monkeypatch, rows):
    monkeypatch.setattr(torch.version, "hip", "test")
    monkeypatch.setenv("RL_KERNEL_STRICT_CANONICAL_TP", "8")
    staging, stable_output = torch.zeros(32, 4), torch.zeros(32, 4)
    monkeypatch.setitem(
        ffn._PACKED_INFERENCE_STAGING_BY_HANDLE, 7, (987654, staging, stable_output)
    )
    observed = []

    def reduce(handle, partial, output):
        observed.append(handle)
        output.copy_(partial * 4)

    def packed(hidden, gate_up, down, chunks):
        assert chunks == 2
        return hidden.new_full((rows, 4), 3)

    monkeypatch.setattr(
        ffn, "_C", SimpleNamespace(deterministic_collective_rocm_ipc_all_reduce_input=reduce)
    )
    monkeypatch.setattr(ffn, "_canonical_packed_ffn_local_output", packed)
    actual = ffn.qwen3_ffn_packed_inference(
        torch.zeros(rows, 4),
        torch.zeros(16, 4),
        torch.zeros(4, 8),
        collective_handle=7,
        tp_world_size=4,
    )
    assert observed == [987654]
    assert torch.equal(actual, torch.full((rows, 4), 12.0))
    if rows <= 32:
        assert actual.data_ptr() == stable_output.data_ptr()
