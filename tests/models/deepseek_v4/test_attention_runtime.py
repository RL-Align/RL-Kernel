import pytest
import torch

from rl_engine.contracts.operators.joint_attention import (
    JointAttentionContract,
    JointAttentionError,
)
from rl_engine.models.deepseek_v4.attention_runtime import AttentionRuntime
from rl_engine.models.deepseek_v4.joint_attention import JointAttentionPlan, JointMetadata
from .joint_attention_helpers import ReferenceBackend


def fixture(*, topk=0):
    contract = JointAttentionContract(8, 128, 8, 8, topk, 0, graph=False)
    plan = JointAttentionPlan(contract, backend=ReferenceBackend())
    kvkeys = (
        torch.tensor([[0, 0, 3]], dtype=torch.int64)
        if topk
        else torch.empty((0, 3), dtype=torch.int64)
    )
    kv = torch.ones((len(kvkeys), 128), dtype=torch.bfloat16)
    sink = torch.zeros(8, dtype=torch.float32)

    def inputs(positions, *, version="v1", value=1):
        keys = torch.tensor([[0, p] for p in positions], dtype=torch.int64).reshape(-1, 2)
        selected = torch.zeros((len(positions), topk), dtype=torch.int32)
        meta = JointMetadata.bind(contract, keys, kvkeys, selected, version=version)
        q = torch.full((len(positions), 8, 128), value, dtype=torch.bfloat16)
        return q, kv, sink, meta

    return AttentionRuntime(plan, version="v1"), inputs


def test_prefill_decode_request_lifecycle():
    runtime, inputs = fixture()
    q, kv, sink, meta = inputs([1])
    with pytest.raises(JointAttentionError, match="missing"):
        runtime.forward(q, kv, sink, metadata=meta, mode="prefill")
    q, kv, sink, meta = inputs([0, 1])
    runtime.forward(q, kv, sink, metadata=meta, mode="prefill")
    q, kv, sink, meta = inputs([2])
    result = runtime.forward(q, kv, sink, metadata=meta, mode="decode")
    assert result.mode == "decode" and result.implementation == runtime.plan.implementation
    with pytest.raises(JointAttentionError, match="position"):
        runtime.forward(q, kv, sink, metadata=meta, mode="decode")
    with pytest.raises(JointAttentionError, match="release"):
        runtime.replace_version("v2")
    runtime.release_request(0)
    runtime.replace_version("v2")
    q, kv, sink, meta = inputs([0], version="v1")
    with pytest.raises(JointAttentionError, match="stale"):
        runtime.forward(q, kv, sink, metadata=meta, mode="prefill")


def test_recompute_compares_bytes_and_retains_failure_record():
    runtime, inputs = fixture(topk=1)
    q, kv, sink, meta = inputs([4, 5])
    runtime.forward(q, kv, sink, metadata=meta, mode="train")
    q_bad, _, _, _ = inputs([4, 5], value=2)
    with pytest.raises(JointAttentionError, match="input mismatch"):
        runtime.forward(q_bad, kv, sink, metadata=meta, mode="recompute")
    runtime.forward(q, kv, sink, metadata=meta, mode="recompute")
    with pytest.raises(JointAttentionError, match="missing training"):
        runtime.forward(q, kv, sink, metadata=meta, mode="recompute")


def test_recompute_reorders_tokens_but_rejects_changed_kv():
    runtime, inputs = fixture(topk=1)
    q, kv, sink, meta = inputs([4, 5])
    q[1] = -q[1]
    runtime.forward(q, kv, sink, metadata=meta, mode="train")
    reordered_q, _, _, reversed_meta = inputs([5, 4])
    reordered_q.copy_(q.flip(0))
    altered_kv = kv.clone()
    altered_kv[0, 0] = 2
    with pytest.raises(JointAttentionError, match="input mismatch"):
        runtime.forward(reordered_q, altered_kv, sink, metadata=reversed_meta, mode="recompute")
    runtime.forward(reordered_q, kv, sink, metadata=reversed_meta, mode="recompute")


def test_failed_provider_does_not_advance_request(monkeypatch):
    runtime, inputs = fixture()
    q, kv, sink, meta = inputs([0])
    original = runtime.plan.backend.forward_into

    def fail(*args):
        raise RuntimeError("provider failure")

    monkeypatch.setattr(runtime.plan.backend, "forward_into", fail)
    with pytest.raises(RuntimeError, match="provider failure"):
        runtime.forward(q, kv, sink, metadata=meta, mode="prefill")
    assert runtime._positions == {}
    monkeypatch.setattr(runtime.plan.backend, "forward_into", original)
    # The plan itself is poisoned; callers must construct a fresh plan.
    with pytest.raises(JointAttentionError, match="poisoned"):
        runtime.forward(q, kv, sink, metadata=meta, mode="prefill")


def test_generation_rejects_grad_and_graph_without_capture():
    runtime, inputs = fixture()
    q, kv, sink, meta = inputs([0])
    q.requires_grad_()
    with pytest.raises(JointAttentionError, match="generation does not record"):
        runtime.forward(q, kv, sink, metadata=meta, mode="prefill")
    q = q.detach()
    with pytest.raises(JointAttentionError, match="capture"):
        runtime.forward(q, kv, sink, metadata=meta, mode="prefill", use_graph=True)
    assert runtime._positions == {}


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None, reason="native MI300X acceptance"
)
def test_native_mode_and_graph_bytes():
    from rl_engine.backends.rocm.attention.joint_attention import AiterJointAttentionBackend

    backend = AiterJointAttentionBackend()
    contract = JointAttentionContract(8, 128, 8, 8, 0, 0, graph=True)
    plan = JointAttentionPlan(contract, backend=backend)
    runtime = AttentionRuntime(plan, version="v1")
    kvkeys = torch.empty((0, 3), dtype=torch.int64, device=backend.device)
    kv = torch.empty((0, 128), dtype=torch.bfloat16, device=backend.device)
    sink = torch.zeros(8, dtype=torch.float32, device=backend.device)

    def inputs(position):
        keys = torch.tensor([[0, position]], dtype=torch.int64, device=backend.device)
        selected = torch.empty((1, 0), dtype=torch.int32, device=backend.device)
        meta = JointMetadata.bind(contract, keys, kvkeys, selected, version="v1")
        q = torch.randn((1, 8, 128), dtype=torch.bfloat16, device=backend.device)
        return q, meta

    q, meta = inputs(0)
    eager = runtime.forward(q, kv, sink, metadata=meta, mode="prefill")
    runtime.capture()
    q, meta = inputs(1)
    graph = runtime.forward(q, kv, sink, metadata=meta, mode="decode", use_graph=True)
    direct = plan.forward_record(q, kv, sink, metadata=meta, version="v1", mode="decode")
    assert _bytes(graph.output, direct.output) and _bytes(graph.lse, direct.lse)
    assert eager.implementation == graph.implementation


def _bytes(left, right):
    return torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None, reason="native MI300X acceptance"
)
def test_training_uses_shared_native_autograd():
    from rl_engine.backends.rocm.attention.joint_attention import AiterJointAttentionBackend
    from rl_engine.models.deepseek_v4.joint_autograd import JointAttentionAutograd

    backend = AiterJointAttentionBackend()
    contract = JointAttentionContract(8, 128, 8, 8, 1, 0, graph=False)
    plan = JointAttentionPlan(contract, backend=backend)
    operator = JointAttentionAutograd(plan, layer_id="runtime-test")
    runtime = AttentionRuntime(plan, version="v1", autograd=operator)
    device = backend.device
    keys = torch.tensor([[0, 4]], dtype=torch.int64, device=device)
    kvkeys = torch.tensor([[0, 0, 3]], dtype=torch.int64, device=device)
    selected = torch.tensor([[0]], dtype=torch.int32, device=device)
    meta = JointMetadata.bind(contract, keys, kvkeys, selected, version="v1")
    q = torch.randn((1, 8, 128), dtype=torch.bfloat16, device=device).requires_grad_()
    kv = torch.randn((1, 128), dtype=torch.bfloat16, device=device)
    sink = torch.zeros(8, dtype=torch.float32, device=device)
    first = runtime.forward(q, kv, sink, metadata=meta, mode="train")
    assert first.output.requires_grad
    first.output.float().sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    repeated = runtime.forward(q.detach(), kv, sink, metadata=meta, mode="recompute")
    assert _bytes(first.output, repeated.output) and _bytes(first.lse, repeated.lse)
