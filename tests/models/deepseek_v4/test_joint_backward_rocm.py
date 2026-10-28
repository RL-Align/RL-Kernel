"""Real native VJP acceptance; full autograd/session tests are still to follow."""

import pytest
import torch

from rl_engine.backends.rocm.attention.joint_attention import AiterJointAttentionBackend
from rl_engine.contracts.operators.joint_attention import JointAttentionError
from rl_engine.models.deepseek_v4.joint_backward import JointBackwardPlan
from .joint_attention_helpers import case, record, same_bytes

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="MI300X native VJP acceptance",
)


@pytest.fixture(scope="module")
def provider():
    return AiterJointAttentionBackend()


def oracle(saved, dy, scale):
    q, kv, sink = [t.double().detach().requires_grad_() for t in (saved.q, saved.kv, saved.sink)]
    loss = (q.sum() + kv.sum() + sink.sum()) * 0
    for row in range(q.shape[0]):
        if saved.metadata.query_keys[row, 0].item() < 0:
            continue
        values = kv[saved.indices[row][saved.indices[row] >= 0].long()]
        logits = q[row] @ values.t() * scale
        probabilities = torch.cat((logits, sink[:, None]), 1).softmax(1)[:, :-1]
        output = probabilities @ values
        loss = loss + (output * dy[row].bfloat16().double()).sum()
    loss.backward()
    return q.grad, kv.grad, sink.grad


@pytest.mark.parametrize("m", [0, 1, 3, 7])
@pytest.mark.parametrize("h,d", [(8, 128), (13, 128), (64, 512)])
def test_native_vjp_fp64_graph_repeat(m, h, d, provider):
    plan, q, kv, sink, meta, _ = case(backend=provider, m=m, heads=h, dim=d, padding=True)
    saved = record(plan, q, kv, sink, meta)
    backward = JointBackwardPlan(plan, saved)
    dy = torch.randn_like(saved.output)
    grads = backward.backward(dy)
    expected = oracle(saved, dy, plan.contract.softmax_scale)
    for actual, want in zip((grads.dq, grads.dkv, grads.dsink), expected, strict=True):
        torch.testing.assert_close(actual.double(), want, rtol=0.04, atol=0.035)
    backward.capture()
    for _ in range(3):
        captured = backward.backward(dy, use_graph=True)
        for actual, want in zip(
            (captured.dq, captured.dkv, captured.dsink),
            (grads.dq, grads.dkv, grads.dsink),
            strict=True,
        ):
            assert same_bytes(actual, want)
    changed = backward.backward(dy * 2, use_graph=True)
    eager = backward.backward(dy * 2)
    for actual, want in zip(
        (changed.dq, changed.dkv, changed.dsink), (eager.dq, eager.dkv, eager.dsink), strict=True
    ):
        assert same_bytes(actual, want)


@pytest.mark.parametrize("fault", ["saved", "csr", "budget", "dy", "graph"])
def test_backward_guards(fault, provider):
    plan, q, kv, sink, meta, _ = case(backend=provider)
    saved = record(plan, q, kv, sink, meta)
    if fault == "budget":
        with pytest.raises(JointAttentionError, match="budget"):
            JointBackwardPlan(plan, saved, max_workspace_bytes=1)
        return
    backward = JointBackwardPlan(plan, saved)
    dy = torch.ones_like(saved.output)
    if fault == "saved":
        saved.lse.add_(1)
    if fault == "csr":
        backward.ptr.add_(1)
    if fault == "dy":
        dy[0, 0, 0] = float("nan")
    with pytest.raises(JointAttentionError):
        backward.backward(dy, use_graph=fault == "graph")


def test_backward_two_streams_and_saved_results(provider):
    plan, q, kv, sink, meta, _ = case(backend=provider, m=7)
    saved = record(plan, q, kv, sink, meta)
    backward = JointBackwardPlan(plan, saved)
    backward.capture()
    dy = torch.randn_like(q)
    expected = backward.backward(dy)
    torch.cuda.synchronize()
    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    outputs = []
    for i in range(8):
        with torch.cuda.stream(streams[i % 2]):
            outputs.append(backward.backward(dy, use_graph=True))
    for stream in streams:
        stream.synchronize()
    for got in outputs:
        for a, b in zip(
            (got.dq, got.dkv, got.dsink), (expected.dq, expected.dkv, expected.dsink), strict=True
        ):
            assert same_bytes(a, b)


def test_large_native_vjp_640_kv_128_heads(provider):
    from rl_engine.contracts.operators.joint_attention import JointAttentionContract
    from rl_engine.models.deepseek_v4.joint_attention import JointAttentionPlan, JointMetadata

    c = JointAttentionContract(128, 512, 2, 640, 512, 128)
    keys = torch.tensor([[0, 2047]], device="cuda")
    kvkeys = torch.tensor(
        [(0, 0, 4 * i + 3) for i in range(512)] + [(0, 1, p) for p in range(1920, 2048)],
        device="cuda",
    )
    comp = torch.arange(512, dtype=torch.int32, device="cuda")[None]
    meta = JointMetadata.bind(c, keys, kvkeys, comp, version="v1")
    torch.manual_seed(25640)
    q = torch.randn(1, 128, 512, device="cuda").bfloat16()
    kv = torch.randn(640, 512, device="cuda").bfloat16()
    sink = torch.randn(128, device="cuda")
    plan = JointAttentionPlan(c, backend=provider)
    saved = record(plan, q, kv, sink, meta)
    dy = torch.randn_like(q)
    backward = JointBackwardPlan(plan, saved)
    got = backward.backward(dy)
    expected = oracle(saved, dy, c.softmax_scale)
    for a, b in zip((got.dq, got.dkv, got.dsink), expected, strict=True):
        torch.testing.assert_close(a.double(), b, rtol=0.04, atol=0.035)
    backward.capture()
    captured = backward.backward(dy, use_graph=True)
    for a, b in zip(
        (got.dq, got.dkv, got.dsink), (captured.dq, captured.dkv, captured.dsink), strict=True
    ):
        assert same_bytes(a, b)


def test_disabled_sink_empty_attention_returns_zero_vjp(provider):
    plan, q, kv, sink, meta, _ = case(backend=provider, topk=0, window=0)
    sink.fill_(float("-inf"))
    saved = record(plan, q, kv, sink, meta)
    backward = JointBackwardPlan(plan, saved)
    got = backward.backward(torch.randn_like(q))
    for tensor in (got.dq, got.dkv, got.dsink):
        assert torch.isfinite(tensor).all() and tensor.eq(0).all()


def test_truly_empty_kv_storage_and_explicit_backend(provider):
    from rl_engine.contracts.operators.joint_attention import JointAttentionContract
    from rl_engine.models.deepseek_v4.joint_attention import JointAttentionPlan, JointMetadata
    from rl_engine.models.deepseek_v4.joint_autograd import JointAttentionAutograd
    from rl_engine.ops.autograd.canonical_backward import canonical_backward_session

    c = JointAttentionContract(8, 128, 3, 1, 0, 0)
    keys = torch.tensor([[0, 0]], device="cuda")
    kvkeys = torch.empty((0, 3), dtype=torch.int64, device="cuda")
    comp = torch.empty((1, 0), dtype=torch.int32, device="cuda")
    meta = JointMetadata.bind(c, keys, kvkeys, comp, version="v1")
    q = torch.randn(1, 8, 128, device="cuda").bfloat16()
    kv = torch.empty((0, 128), dtype=torch.bfloat16, device="cuda")
    sink = torch.randn(8, device="cuda")
    plan = JointAttentionPlan(c, backend=provider)
    saved = record(plan, q, kv, sink, meta)
    backward = JointBackwardPlan(plan, saved)
    backward.capture()
    got = backward.backward(torch.ones_like(q), use_graph=True)
    assert got.dkv.shape == kv.shape and got.dq.eq(0).all() and got.dsink.eq(0).all()
    for value in (q, kv, sink):
        value.requires_grad_()
    adapter = JointAttentionAutograd(plan, layer_id="empty-kv", backend=backward.backend)
    with canonical_backward_session() as session:
        out = adapter(q, kv, sink, metadata=meta, version="v1")
        out.output.sum().backward()
        session.validate_complete()
    assert q.grad.eq(0).all() and sink.grad.eq(0).all() and kv.grad.shape == kv.shape
