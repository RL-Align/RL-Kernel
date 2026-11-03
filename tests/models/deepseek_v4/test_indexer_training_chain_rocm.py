"""Real composition of existing projection/compressor/attention VJPs."""

import pytest
import torch

from rl_engine.contracts.operators.indexer_training import IndexerAuxContract
from rl_engine.contracts.operators.indexer_projection import IndexerProjectionContract
from rl_engine.contracts.operators.indexer_compressor_c4 import IndexerC4Contract
from rl_engine.contracts.operators.kv_compressor_c4 import C4Contract
from rl_engine.contracts.operators.kv_compressor_c128 import C128Contract
from rl_engine.contracts.operators.joint_attention import JointAttentionContract
from rl_engine.models.deepseek_v4.indexer_training import (
    IndexerAuxiliary,
    IndexerAuxMetadata,
    canonical_indexer_auxiliary_loss,
)
from rl_engine.models.deepseek_v4.indexer_projection import IndexerProjection, IndexerLogicalRows
from rl_engine.models.deepseek_v4.indexer_compressor_c4 import IndexerC4Compressor
from rl_engine.models.deepseek_v4.kv_compressor_c4 import C4Compressor
from rl_engine.models.deepseek_v4.kv_compressor_c128 import C128Compressor
from rl_engine.models.deepseek_v4.joint_attention import JointAttentionPlan, JointMetadata
from rl_engine.models.deepseek_v4.joint_autograd import JointAttentionAutograd
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session
from .indexer_c4_helpers import fixture as index_fixture
from .indexer_c4_helpers import hadamard64
from .c4_helpers import fixture as c4_fixture
from .c128_helpers import fixture as c128_fixture

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="MI300X composed training acceptance",
)


def same(a, b):
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


@pytest.mark.parametrize("precision", ["mxfp4", "fp8"])
def test_projection_query_transform_indexer_c4_auxiliary_chain(precision):
    c = IndexerAuxContract(4, 0.01, 8, 8, 512, selected_capacity=4, precision=precision)
    aux = IndexerAuxiliary(c, layer_id="chain.aux")
    index = IndexerC4Compressor(IndexerC4Contract(precision=precision), backend=aux.backend.codec)
    main = C4Compressor(C4Contract())
    project = IndexerProjection(
        IndexerProjectionContract(hidden_size=128, q_lora_rank=128, num_heads=8, head_dim=128),
        layer_id="chain.proj",
        backend=aux.backend.linear_backend,
    )
    x, ids, ape, norm, cos, sin = index_fixture(4, device="cuda", requires_grad=True)
    mx, mids, ma, mn, mc, ms = c4_fixture(4, device="cuda")
    teacher_k = main(main.bind(mx, mids, version="v1"), ma, mn, mc, ms, version="v1").values
    query_ids = ids[torch.tensor([3, 7, 11, 15], device="cuda")].contiguous()
    topk = torch.tensor(
        [[0, -1, -1, -1], [1, 0, -1, -1], [2, 1, 0, -1], [3, 0, 2, 1]], device="cuda"
    )
    torch.manual_seed(2620)
    qr = torch.randn(4, 128, device="cuda").bfloat16().requires_grad_()
    hidden = torch.randn(4, 128, device="cuda").bfloat16().requires_grad_()
    qw = (torch.randn(1024, 128, device="cuda") / 128**0.5).bfloat16().requires_grad_()
    ww = (torch.randn(8, 128, device="cuda") / 128**0.5).bfloat16().requires_grad_()
    teacher_q = torch.randn(4, 8, 512, device="cuda").bfloat16().requires_grad_()
    leaves = (x, ape, norm, qr, hidden, qw, ww)

    def run(parts):
        for leaf in leaves:
            leaf.grad = None
        with canonical_backward_session() as session:
            compressed = index(
                index.bind(x, ids, version="v1"),
                ape,
                norm,
                cos,
                sin,
                version="v1",
                parameter_id="chain.indexer_compressor",
            )
            keys = aux.prepare_operand(
                compressed.values,
                role="k",
                version="v1",
                packed=(compressed.packed, compressed.scales),
            )
            results = []
            for selection in parts:
                logical = IndexerLogicalRows.bind(query_ids[selection])
                out = project(
                    qr[selection].contiguous(),
                    hidden[selection].contiguous(),
                    qw,
                    ww,
                    logical_rows=logical,
                )
                transformed = aux.transform_queries(out.q, query_ids[selection, 1], cos, sin)
                queries = aux.prepare_operand(transformed, role="q", version="v1")
                metadata = IndexerAuxMetadata.bind(
                    c, query_ids[selection], compressed.group_ids, topk[selection], version="v1"
                )
                results.append(
                    aux.autograd(
                        queries,
                        keys,
                        out.weights,
                        teacher_q[selection].contiguous(),
                        teacher_k,
                        metadata=metadata,
                        version="v1",
                    )
                )
            loss = canonical_indexer_auxiliary_loss(results)
            loss.backward()
            session.validate_complete()
        assert teacher_q.grad is None
        return loss.detach(), tuple(t.grad.clone() for t in leaves)

    whole_loss, whole_grad = run([torch.arange(4, device="cuda")])
    loss, gradients = run(
        [
            torch.tensor([3, 1], device="cuda"),
            torch.tensor([0], device="cuda"),
            torch.tensor([2], device="cuda"),
        ]
    )
    same(loss.reshape(-1), whole_loss.reshape(-1))
    for value, target in zip(gradients, whole_grad, strict=True):
        assert torch.isfinite(value).all()
        same(value, target)


@pytest.mark.parametrize("ratio", [4, 128])
def test_main_compressor_into_native_attention_backward(ratio):
    groups = 3
    if ratio == 4:
        source, ids, ape, norm, cos, sin = c4_fixture(groups, device="cuda", requires_grad=True)
        compressor = C4Compressor(C4Contract())
    else:
        source, ids, ape, norm, cos, sin = c128_fixture(
            groups * 128, device="cuda", requires_grad=True
        )
        compressor = C128Compressor(C128Contract())
    c = JointAttentionContract(8, 512, groups, groups, groups, 0, compression_ratio=ratio)
    plan = JointAttentionPlan(c)
    attention = JointAttentionAutograd(plan, layer_id=f"chain.main{ratio}")
    query_ids = torch.tensor([[7, (g + 1) * ratio - 1] for g in range(groups)], device="cuda")
    key_ids = torch.tensor([[7, 0, (g + 1) * ratio - 1] for g in range(groups)], device="cuda")
    topk = torch.tensor([[0, -1, -1], [1, 0, -1], [2, 0, 1]], dtype=torch.int32, device="cuda")
    torch.manual_seed(26128)
    q = torch.randn(groups, 8, 512, device="cuda").bfloat16().requires_grad_()
    sink = torch.randn(8, device="cuda", requires_grad=True)
    dy = torch.randn_like(q)
    leaves = (source, ape, norm, q, sink)

    def run(parts):
        for leaf in leaves:
            leaf.grad = None
        with canonical_backward_session() as session:
            output = compressor(
                compressor.bind(source, ids, version="v1"),
                ape,
                norm,
                cos,
                sin,
                version="v1",
                parameter_id=f"chain.compressor{ratio}",
            )
            outputs, grads = [], []
            for selection in parts:
                meta = JointMetadata.bind(
                    c, query_ids[selection], key_ids, topk[selection], version="v1"
                )
                outputs.append(
                    attention(
                        q[selection].contiguous(), output.values, sink, metadata=meta, version="v1"
                    ).output
                )
                grads.append(dy[selection])
            torch.autograd.backward(outputs, grads)
            session.validate_complete()
        return tuple(t.grad.clone() for t in leaves)

    expected = run([torch.arange(groups, device="cuda")])
    actual = run([torch.tensor([2], device="cuda"), torch.tensor([0, 1], device="cuda")])
    for value, target in zip(actual, expected, strict=True):
        assert torch.isfinite(value).all()
        same(value, target)


@pytest.mark.parametrize("rows", [0, 1, 3])
def test_native_query_rope_hadamard_staged_vjp_and_tails(rows):
    c = IndexerAuxContract(max(rows, 1), 0.01, 13, 8, 128, selected_capacity=2)
    op = IndexerAuxiliary(c, layer_id="query-transform")
    torch.manual_seed(2615)
    q = torch.randn(rows, 13, 128, device="cuda").bfloat16().requires_grad_()
    positions = torch.tensor([0, 3, 7], device="cuda")[:rows]
    angles = torch.arange(8, device="cuda").double()[:, None] * (
        10000.0 ** (-torch.arange(0, 64, 2, device="cuda").double() / 64)
    )
    cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
    output = op.transform_queries(q, positions, cos, sin)
    co, si = cos[positions][:, None].double(), sin[positions][:, None].double()

    def rotate(value, inverse=False):
        a, b = value[..., -64::2].double(), value[..., -63::2].double()
        tail = torch.stack(
            (a * co + b * si, b * co - a * si) if inverse else (a * co - b * si, b * co + a * si),
            -1,
        ).flatten(-2)
        return torch.cat((value[..., :64].double(), tail), -1).bfloat16()

    h = hadamard64(q.device)
    expected = (rotate(q).double() @ h.t()).bfloat16()
    torch.testing.assert_close(output, expected, rtol=0.016, atol=0.016)
    grad = torch.randn_like(output)
    output.backward(grad)
    wanted = rotate((grad.double() @ h).bfloat16(), inverse=True)
    torch.testing.assert_close(q.grad, wanted, rtol=0.016, atol=0.016)
    for row in range(rows):
        solo = op.transform_queries(
            q[row : row + 1].contiguous(), positions[row : row + 1], cos, sin
        )
        same(solo, output[row : row + 1])
