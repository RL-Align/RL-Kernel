from dataclasses import replace

import pytest
import torch

from rl_engine.contracts.operators.joint_attention import (
    JointAttentionContract,
    JointAttentionError,
)
from rl_engine.models.deepseek_v4.joint_attention import JointAttentionPlan, JointMetadata
from .joint_attention_helpers import ReferenceBackend, case, record, reference, same_bytes


@pytest.mark.parametrize(
    "change",
    [
        {"num_heads": 0},
        {"num_heads": 129},
        {"head_dim": 64},
        {"max_queries": True},
        {"max_queries": 2**31},
        {"max_kv": 0},
        {"compressed_topk": -1},
        {"window_size": -1},
        {"compressed_topk": 4097},
        {"compression_ratio": 8},
        {"scale": 0.0},
        {"scale": float("inf")},
        {"scale": float("nan")},
        {"graph": 1},
        {"max_workspace_bytes": 1},
        {"profile": "other"},
    ],
)
def test_contract_errors(change):
    with pytest.raises(JointAttentionError):
        replace(JointAttentionContract(8, 128, 8, 64, 2, 4), **change)


@pytest.mark.parametrize("m", [0, 1, 4])
@pytest.mark.parametrize("window,topk", [(0, 0), (0, 2), (4, 0), (4, 2)])
def test_reference_boundaries(m, window, topk):
    plan, q, kv, sink, meta, _ = case(m=m, window=window, topk=topk, padding=True)
    result = record(plan, q, kv, sink, meta)
    out, lse = reference(q, kv[meta.permutation], sink, meta.indices, plan.contract.softmax_scale)
    if m:
        out[-1] = 0
        lse[-1] = float("-inf")
    assert same_bytes(result.output, out.bfloat16())
    assert same_bytes(result.lse, lse.float())
    result.validate_saved()


@pytest.mark.parametrize(
    "fault",
    [
        "q_duplicate",
        "kv_duplicate",
        "bad_kind",
        "partial_group",
        "cross_request",
        "future",
        "duplicate_topk",
        "bad_topk",
        "window_incomplete",
        "padding",
        "padding_topk",
    ],
)
def test_metadata_errors(fault):
    plan, q, kv, sink, meta, comp = case()
    qkeys = meta.query_keys.clone()
    # Reconstruct original loader key order.
    inverse = torch.argsort(meta.permutation)
    kvkeys = meta.kv_keys[inverse].clone()
    if fault == "q_duplicate":
        qkeys[1] = qkeys[0]
    if fault == "kv_duplicate":
        kvkeys[1] = kvkeys[0]
    if fault == "bad_kind":
        kvkeys[0, 1] = 2
    if fault == "partial_group":
        kvkeys[16, 2] = 2
    if fault == "cross_request":
        comp[0, 0] = 36
    if fault == "future":
        comp[0, 0] = 19
    if fault == "duplicate_topk":
        comp[0, :] = 16
    if fault == "bad_topk":
        comp[0, 0] = -2
    if fault == "window_incomplete":
        kvkeys[4, 2] = 100
    if fault == "padding":
        qkeys[-1] = torch.tensor([-1, 0])
    if fault == "padding_topk":
        qkeys[0] = -1
    with pytest.raises(JointAttentionError):
        JointMetadata.bind(plan.contract, qkeys, kvkeys, comp, version="v1")


@pytest.mark.parametrize(
    "fault",
    [
        "q_dtype",
        "kv_dim",
        "sink_dtype",
        "sink_nan",
        "sink_inf",
        "q_nan",
        "version",
        "mode",
        "mutated",
        "graph",
    ],
)
def test_runtime_errors(fault):
    plan, q, kv, sink, meta, _ = case()
    kwargs = {}
    if fault == "q_dtype":
        q = q.float()
    if fault == "kv_dim":
        kv = kv[:, :-1].contiguous()
    if fault == "sink_dtype":
        sink = sink.bfloat16()
    if fault == "sink_nan":
        sink[0] = float("nan")
    if fault == "sink_inf":
        sink[0] = float("inf")
    if fault == "q_nan":
        q[0, 0, 0] = float("nan")
    if fault == "version":
        meta = replace(meta, version="old")
    if fault == "mode":
        kwargs["mode"] = "invalid"
    if fault == "mutated":
        meta.indices.fill_(-1)
    if fault == "graph":
        kwargs["use_graph"] = True
    with pytest.raises(JointAttentionError):
        record(plan, q, kv, sink, meta, **kwargs)


def test_sink_has_no_value_but_changes_denominator():
    plan, q, kv, sink, meta, _ = case()
    q.zero_()
    kv.fill_(1)
    sink.zero_()
    out = record(plan, q, kv, sink, meta)
    counts = (meta.indices >= 0).sum(1)
    want = (counts / (counts + 1)).bfloat16()
    assert torch.equal(out.output[:, 0, 0], want)
    torch.testing.assert_close(out.lse[:, 0], (counts + 1).float().log())


def test_result_lifetime_and_explicit_manual_forward():
    plan, q, kv, sink, meta, _ = case()
    q.requires_grad_()
    first = record(plan, q, kv, sink, meta)
    baseline = first.output.clone()
    assert not first.output.requires_grad
    record(plan, q.detach() * 2, kv, sink, meta)
    assert same_bytes(first.output, baseline)
    first.validate_saved()
    first.lse.add_(1)
    with pytest.raises(JointAttentionError, match="mutated"):
        first.validate_saved()


def test_profile_handshake_and_capture_rejection():
    plan, *_ = case()
    with pytest.raises(JointAttentionError, match="handshake"):
        JointAttentionPlan(plan.contract, backend=ReferenceBackend(), expected_implementation="bad")
    with pytest.raises(JointAttentionError, match="disabled"):
        plan.capture()


def test_failed_workspace_is_not_reused(monkeypatch):
    plan, q, kv, sink, meta, _ = case()

    def fail(*args):
        plan._out.fill_(23)
        raise RuntimeError("injected provider failure")

    monkeypatch.setattr(plan.backend, "forward_into", fail)
    with pytest.raises(RuntimeError, match="injected"):
        record(plan, q, kv, sink, meta)
    with pytest.raises(JointAttentionError, match="poisoned"):
        record(plan, q, kv, sink, meta)


def test_source_hash_guard_and_framework_same_boundary(tmp_path):
    from rl_engine.backends.rocm.attention.joint_attention import load_pinned_kernel
    from rl_engine.integrations.engines.train.megatron.joint_attention import (
        JointAttentionPlan as Train,
    )
    from rl_engine.integrations.engines.rollout.vllm.joint_attention import (
        JointAttentionPlan as Rollout,
    )

    assert Train is Rollout
    with pytest.raises(JointAttentionError, match="pinned"):
        load_pinned_kernel(tmp_path / "missing.py")
    source = tmp_path / "bad.py"
    source.write_text("raise RuntimeError('must never be executed')")
    with pytest.raises(JointAttentionError, match="hash"):
        load_pinned_kernel(source)
