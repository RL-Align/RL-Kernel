# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""vLLM's top-k/top-p must use the sort that sampling_keep_mask reproduces."""

import sys
import types

import pytest
import torch

ops = pytest.importorskip("vllm.v1.sample.ops.topk_topp_sampler")

from rl_engine.integrations.engines.rollout.vllm import sampling_support  # noqa: E402

_IMPORTERS = sampling_support._DISPATCHER_IMPORTERS


@pytest.fixture
def fake_vllm(monkeypatch):
    """Stand-in dispatcher, sort and by-name importers; restored after the test."""

    calls = []

    def dispatcher(logits, k, p):
        calls.append("dispatcher")
        return logits

    def sort(logits, k, p, allow_cpu_sync=False):
        calls.append(("sort", allow_cpu_sync))
        return logits

    monkeypatch.setattr(ops, "apply_top_k_top_p", dispatcher)
    monkeypatch.setattr(ops, "apply_top_k_top_p_pytorch", sort)
    importers = {}
    for name in _IMPORTERS:
        module = types.ModuleType(name)
        module.apply_top_k_top_p = dispatcher
        monkeypatch.setitem(sys.modules, name, module)
        importers[name] = module
    return types.SimpleNamespace(calls=calls, dispatcher=dispatcher, importers=importers)


def test_sort_contract_routes_dispatcher_and_by_name_importers(fake_vllm):
    assert sampling_support.install_sort_contract() is True

    installed = ops.apply_top_k_top_p
    assert installed is not fake_vllm.dispatcher
    for module in fake_vllm.importers.values():
        assert module.apply_top_k_top_p is installed

    logits = torch.zeros(2, 4)
    k = torch.tensor([2, 2])
    p = torch.tensor([0.5, 0.9])
    installed(logits, k, p)
    installed(logits, None, p)
    installed(logits, k, None)
    assert fake_vllm.calls == [("sort", False)] * 3
    assert installed(logits, None, None) is logits
    assert fake_vllm.calls == [("sort", False)] * 3

    # Installing again keeps the same function and does not wrap it twice.
    assert sampling_support.install_sort_contract() is True
    assert ops.apply_top_k_top_p is installed


def test_sort_contract_leaves_rebound_importers_alone(fake_vllm):
    def other(logits, k, p):
        return logits

    rebound = _IMPORTERS[0]
    fake_vllm.importers[rebound].apply_top_k_top_p = other
    sampling_support.install_sort_contract()
    assert fake_vllm.importers[rebound].apply_top_k_top_p is other


def test_sort_contract_is_a_no_op_without_the_pytorch_variant(fake_vllm, monkeypatch):
    # vLLM 0.16 and older: apply_top_k_top_p is the sort itself.
    monkeypatch.delattr(ops, "apply_top_k_top_p_pytorch")
    assert sampling_support.install_sort_contract() is False
    assert ops.apply_top_k_top_p is fake_vllm.dispatcher


def test_capture_wraps_the_sort_and_reinstalls_cleanly(fake_vllm):
    sampling_support.install_capture()
    captured = ops.apply_top_k_top_p
    assert getattr(captured, sampling_support._MARKER)
    assert getattr(captured.__wrapped__, sampling_support._SORT_MARKER) is fake_vllm.dispatcher

    logits = torch.tensor([[0.0, float("-inf")]])
    with sampling_support.capture_support(True) as support:
        captured(logits, None, torch.tensor([0.9]))
    assert fake_vllm.calls == [("sort", False)]
    assert torch.equal(support["mask"], torch.tensor([[True, False]]))

    sampling_support.install_capture()
    assert ops.apply_top_k_top_p is captured
    assert sampling_support.install_sort_contract() is True


def test_sort_contract_refuses_to_sit_under_a_foreign_capture(fake_vllm, monkeypatch):
    def capture_without_sort(logits, k, p):
        return logits

    setattr(capture_without_sort, sampling_support._MARKER, True)
    monkeypatch.setattr(ops, "apply_top_k_top_p", capture_without_sort)
    with pytest.raises(RuntimeError, match="before the support capture"):
        sampling_support.install_sort_contract()
