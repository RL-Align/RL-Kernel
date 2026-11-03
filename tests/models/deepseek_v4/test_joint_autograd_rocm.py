import pytest
import torch

from rl_engine.backends.rocm.attention.joint_attention import AiterJointAttentionBackend
from rl_engine.contracts.operators.joint_attention import (
    JointAttentionError,
    JointAttentionContract,
)
from rl_engine.models.deepseek_v4.joint_attention import JointMetadata, JointAttentionPlan
from rl_engine.models.deepseek_v4.joint_autograd import JointAttentionAutograd
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session
from .joint_attention_helpers import case, record, same_bytes
from .test_joint_backward_rocm import oracle

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None, reason="MI300X autograd acceptance"
)


@pytest.fixture(scope="module")
def provider():
    return AiterJointAttentionBackend()


@pytest.mark.parametrize("m", [0, 1, 7])
@pytest.mark.parametrize("h,d", [(8, 128), (13, 512)])
def test_autograd_oracle(m, h, d, provider):
    plan, q, kv, sink, meta, _ = case(backend=provider, m=m, heads=h, dim=d, padding=True)
    q.requires_grad_()
    kv.requires_grad_()
    sink.requires_grad_()
    adapter = JointAttentionAutograd(plan, layer_id="layer25")
    saved = record(plan, q, kv, sink, meta)
    dy = torch.randn_like(saved.output)
    expected = oracle(saved, dy, plan.contract.softmax_scale)
    with canonical_backward_session() as session:
        result = adapter(q, kv, sink, metadata=meta, version="v1")
        assert not result.lse.requires_grad
        assert same_bytes(result.output, saved.output)
        result.output.backward(dy)
        session.validate_complete()
    expected = (expected[0], expected[1][torch.argsort(meta.permutation)], expected[2])
    for actual, want in zip((q.grad, kv.grad, sink.grad), expected, strict=True):
        torch.testing.assert_close(actual.double(), want, rtol=0.04, atol=0.035)


@pytest.mark.parametrize("sizes", [[1, 1, 1, 1, 1, 1, 1], [2, 3, 2], [7]])
@pytest.mark.parametrize("reverse", [False, True])
def test_microbatch_canonical_all_gradients(sizes, reverse, provider):
    plan, q, kv, sink, meta, comp = case(backend=provider, m=7, padding=True)
    q.requires_grad_()
    kv.requires_grad_()
    sink.requires_grad_()
    adapter = JointAttentionAutograd(plan, layer_id="layer25")
    dy = torch.randn_like(q)
    with canonical_backward_session() as session:
        whole = adapter(q, kv, sink, metadata=meta, version="v1")
        whole.output.backward(dy)
        session.validate_complete()
    expected = tuple(t.grad.clone() for t in (q, kv, sink))
    q.grad = kv.grad = sink.grad = None
    perm = torch.tensor([6, 4, 0, 5, 2, 1, 3], device="cuda")
    oldkeys = meta.kv_keys[torch.argsort(meta.permutation)]
    plan.capture()
    with canonical_backward_session() as session:
        pieces, start = [], 0
        for count in sizes:
            index = perm[start : start + count]
            bound = JointMetadata.bind(
                plan.contract, meta.query_keys[index], oldkeys, comp[index], version="v1"
            )
            out = adapter(
                q[index].contiguous(), kv, sink, metadata=bound, version="v1", use_graph=True
            )
            assert same_bytes(out.output, whole.output[index])
            pieces.append((out, index))
            start += count
        for result, index in reversed(pieces) if reverse else pieces:
            result.output.backward(dy[index])
        session.validate_complete()
    for actual, want in zip((q.grad, kv.grad, sink.grad), expected, strict=True):
        assert same_bytes(actual, want)
    with torch.no_grad():
        other = adapter(q, kv, sink, metadata=meta, version="v1", mode="decode")
    assert same_bytes(other.output, whole.output)


@pytest.mark.parametrize("which", [0, 1, 2])
def test_selective_gradients(which, provider):
    plan, q, kv, sink, meta, _ = case(backend=provider)
    (q, kv, sink)[which].requires_grad_()
    adapter = JointAttentionAutograd(plan, layer_id="L")
    with canonical_backward_session() as session:
        result = adapter(q, kv, sink, metadata=meta, version="v1")
        result.output.sum().backward()
        session.validate_complete()
    for i, value in enumerate((q, kv, sink)):
        assert (value.grad is not None) == (i == which)
    with torch.no_grad():
        other = adapter(q, kv, sink, metadata=meta, version="v1", mode="decode")
    assert same_bytes(other.output, result.output)


@pytest.mark.parametrize("mismatch", ["plan", "backward_profile"])
def test_canonical_parameters_require_shared_event_owner(mismatch, provider):
    plan, q, kv, sink, meta, comp = case(backend=provider)
    q.requires_grad_()
    kv.requires_grad_()
    sink.requires_grad_()
    first = JointAttentionAutograd(plan, layer_id="same-layer")
    second_plan = (
        JointAttentionPlan(plan.contract, backend=provider) if mismatch == "plan" else plan
    )
    other = JointAttentionAutograd(second_plan, layer_id="same-layer")
    if mismatch == "backward_profile":
        other.backward_implementation = "different-vjp-profile"
    original = meta.kv_keys[torch.argsort(meta.permutation)]
    left = JointMetadata.bind(plan.contract, meta.query_keys[:2], original, comp[:2], version="v1")
    right = JointMetadata.bind(plan.contract, meta.query_keys[2:], original, comp[2:], version="v1")
    with canonical_backward_session():
        first(q[:2].contiguous(), kv, sink, metadata=left, version="v1")
        with pytest.raises(JointAttentionError, match="event-owned|arithmetic"):
            other(q[2:].contiguous(), kv, sink, metadata=right, version="v1")


@pytest.mark.parametrize("fault", ["session", "duplicate", "parameter", "mutation", "higher_order"])
def test_autograd_guards(fault, provider):
    plan, q, kv, sink, meta, _ = case(backend=provider)
    q.requires_grad_()
    kv.requires_grad_()
    sink.requires_grad_()
    adapter = JointAttentionAutograd(plan, layer_id="layer25")
    if fault == "session":
        with pytest.raises(JointAttentionError, match="session"):
            adapter(q, kv, sink, metadata=meta, version="v1")
        return
    with canonical_backward_session():
        first = adapter(q, kv, sink, metadata=meta, version="v1")
        if fault == "duplicate":
            with pytest.raises(JointAttentionError, match="duplicate"):
                adapter(q, kv, sink, metadata=meta, version="v1")
        if fault == "parameter":
            with pytest.raises(JointAttentionError, match="identity"):
                adapter(q, kv.detach().clone().requires_grad_(), sink, metadata=meta, version="v1")
        if fault == "mutation":
            with torch.no_grad():
                kv.add_(1)
            with pytest.raises(RuntimeError, match="inplace"):
                first.output.sum().backward()
        if fault == "higher_order":
            with pytest.raises(JointAttentionError, match="higher-order"):
                torch.autograd.grad(first.output.sum(), q, create_graph=True)


def test_csr64_sink32_boundaries_and_microbatches(provider):
    c = JointAttentionContract(13, 128, 72, 192, 2, 4)
    query_keys = torch.tensor([[0, p] for p in range(64, 129)], device="cuda")
    kvkeys = torch.tensor([[0, 0, 3], [0, 0, 7]] + [[0, 1, p] for p in range(129)], device="cuda")
    comp = torch.tensor([[0, 1]] * 65, dtype=torch.int32, device="cuda")
    meta = JointMetadata.bind(c, query_keys, kvkeys, comp, version="v1")
    torch.manual_seed(2565)
    q = torch.randn(65, 13, 128, device="cuda").bfloat16().requires_grad_()
    kv = torch.randn(131, 128, device="cuda").bfloat16().requires_grad_()
    sink = torch.randn(13, device="cuda", requires_grad=True)
    plan = JointAttentionPlan(c, backend=provider)
    adapter = JointAttentionAutograd(plan, layer_id="L")
    dy = torch.randn_like(q)
    saved = record(plan, q, kv, sink, meta)
    expected_math = oracle(saved, dy, c.softmax_scale)
    with canonical_backward_session() as session:
        full = adapter(q, kv, sink, metadata=meta, version="v1")
        full.output.backward(dy)
        session.validate_complete()
    expected_math = (
        expected_math[0],
        expected_math[1][torch.argsort(meta.permutation)],
        expected_math[2],
    )
    for actual, want in zip((q.grad, kv.grad, sink.grad), expected_math, strict=True):
        torch.testing.assert_close(actual.double(), want, rtol=0.04, atol=0.035)
    expected = tuple(t.grad.clone() for t in (q, kv, sink))
    q.grad = kv.grad = sink.grad = None
    perm = torch.randperm(65, device="cuda")
    with canonical_backward_session() as session:
        pieces = []
        for index in (perm[:1], perm[1:32], perm[32:]):
            bound = JointMetadata.bind(c, query_keys[index], kvkeys, comp[index], version="v1")
            out = adapter(q[index].contiguous(), kv, sink, metadata=bound, version="v1")
            pieces.append((out, index))
        for out, index in reversed(pieces):
            out.output.backward(dy[index])
        session.validate_complete()
    for actual, want in zip((q.grad, kv.grad, sink.grad), expected, strict=True):
        assert same_bytes(actual, want)
