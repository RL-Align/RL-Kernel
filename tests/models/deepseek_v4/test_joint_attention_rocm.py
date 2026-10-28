import pytest
import torch

from rl_engine.backends.rocm.attention.joint_attention import AiterJointAttentionBackend
from rl_engine.contracts.operators.joint_attention import (
    JointAttentionContract,
    JointAttentionError,
)
from rl_engine.models.deepseek_v4.joint_attention import (
    JointAttentionPlan,
    JointMetadata,
    gather_packed_joint_kv,
)
from .joint_attention_helpers import case, record, reference, same_bytes

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None, reason="native MI300X acceptance"
)


@pytest.fixture(scope="module")
def backend():
    return AiterJointAttentionBackend()


@pytest.mark.parametrize("m", [0, 1, 3, 7])
@pytest.mark.parametrize("dim,heads", [(128, 8), (128, 13), (512, 64)])
def test_native_reference_and_modes(m, dim, heads, backend):
    plan, q, kv, sink, meta, _ = case(backend=backend, m=m, dim=dim, heads=heads, padding=True)
    out = record(plan, q, kv, sink, meta)
    expected, expected_lse = reference(
        q, kv[meta.permutation], sink, meta.indices, plan.contract.softmax_scale
    )
    if m:
        expected[-1] = 0
        expected_lse[-1] = float("-inf")
    torch.testing.assert_close(out.output.double(), expected, rtol=0.025, atol=0.006)
    torch.testing.assert_close(out.lse.double(), expected_lse, rtol=1e-5, atol=2e-5)
    for mode in ("train", "recompute", "prefill", "decode"):
        other = record(plan, q, kv, sink, meta, mode=mode)
        assert same_bytes(other.output, out.output) and same_bytes(other.lse, out.lse)
    with torch.inference_mode():
        other = record(plan, q, kv, sink, meta, mode="decode")
    other.validate_saved()
    assert same_bytes(other.output, out.output)


@pytest.mark.parametrize("window,topk,ratio", [(0, 0, 4), (4, 0, 4), (0, 2, 4), (4, 2, 128)])
@pytest.mark.parametrize("sink_disabled", [False, True])
def test_empty_segments_and_sink(window, topk, ratio, sink_disabled, backend):
    plan, q, kv, sink, meta, _ = case(backend=backend, window=window, topk=topk, ratio=ratio)
    if sink_disabled:
        sink.fill_(float("-inf"))
    out = record(plan, q, kv, sink, meta)
    expected, expected_lse = reference(
        q, kv[meta.permutation], sink, meta.indices, plan.contract.softmax_scale
    )
    torch.testing.assert_close(out.output.double(), expected, rtol=0.025, atol=0.006)
    torch.testing.assert_close(out.lse.double(), expected_lse, rtol=1e-5, atol=2e-5)


def test_exact_chunks_and_loader_permutation(backend):
    plan, q, kv, sink, meta, comp = case(backend=backend, m=7, padding=True)
    full = record(plan, q, kv, sink, meta)
    original_keys = meta.kv_keys[torch.argsort(meta.permutation)]
    for index in (
        torch.tensor([4, 0, 6], device="cuda"),
        torch.tensor([2], device="cuda"),
        torch.tensor([5, 3, 1], device="cuda"),
    ):
        bound = JointMetadata.bind(
            plan.contract, meta.query_keys[index], original_keys, comp[index], version="v1"
        )
        chunk = record(plan, q[index].contiguous(), kv, sink, bound)
        assert same_bytes(chunk.output, full.output[index])
        assert same_bytes(chunk.lse, full.lse[index])
    perm = torch.randperm(kv.shape[0], device="cuda")
    remap = torch.argsort(perm)
    changed_comp = torch.where(comp >= 0, remap[comp.clamp_min(0)], -1).int()
    changed_comp = changed_comp.flip(1)
    bound = JointMetadata.bind(
        plan.contract, meta.query_keys, original_keys[perm], changed_comp, version="v1"
    )
    changed = record(plan, q, kv[perm].contiguous(), sink, bound)
    assert same_bytes(changed.output, full.output) and same_bytes(changed.lse, full.lse)


@pytest.mark.parametrize("m", [0, 1, 7])
def test_real_graph_replay_mutation_and_saved_lifetime(m, backend):
    plan, q, kv, sink, meta, _ = case(backend=backend, m=m, padding=True)
    plan.capture()
    first = record(plan, q, kv, sink, meta, use_graph=True)
    expected = record(plan, q, kv, sink, meta)
    assert same_bytes(first.output, expected.output) and same_bytes(first.lse, expected.lse)
    old = first.output.clone()
    for factor in (2.0, -1.0, 0.5):
        changed = (q * factor).contiguous()
        graph = record(plan, changed, kv, sink, meta, mode="decode", use_graph=True)
        eager = record(plan, changed, kv, sink, meta, mode="prefill")
        assert same_bytes(graph.output, eager.output) and same_bytes(graph.lse, eager.lse)
    assert same_bytes(first.output, old)
    first.validate_saved()


def test_two_streams_and_independent_graph_plans(backend):
    plan, q, kv, sink, meta, _ = case(backend=backend, m=7)
    other, *_ = case(backend=backend, m=7)
    plan.capture()
    other.capture()
    expected = record(plan, q, kv, sink, meta)
    torch.cuda.synchronize()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    records = []
    for i in range(8):
        with torch.cuda.stream(streams[i % 2]):
            records.append(record(plan if i % 3 else other, q, kv, sink, meta, use_graph=True))
    for stream in streams:
        stream.synchronize()
    for out in records:
        assert same_bytes(out.output, expected.output) and same_bytes(out.lse, expected.lse)
        out.validate_saved()


@pytest.mark.parametrize("encoding", ["ocp", "fnuz"])
@pytest.mark.parametrize("block_size", [64, 128])
def test_native_packed_cache_and_contiguous_have_same_arithmetic(encoding, block_size, backend):
    from vllm.models.deepseek_v4.common.ops.cache_utils import quantize_and_insert_k_cache

    plan, q, kv, sink, meta, _ = case(backend=backend, dim=512)
    n = kv.shape[0]
    slots = torch.randperm(block_size * 3, device="cuda")[:n].long()
    block_bytes = ((block_size * 584 + 575) // 576) * 576
    cache = torch.zeros((3, block_bytes), dtype=torch.uint8, device="cuda")
    quantize_and_insert_k_cache(
        kv, cache, slots, block_size=block_size, use_fnuz=encoding == "fnuz"
    )
    dequant = gather_packed_joint_kv(
        cache, slots, block_size=block_size, encoding=encoding, backend=backend
    )
    assert same_bytes(dequant[:, 448:], kv[:, 448:])
    torch.testing.assert_close(dequant[:, :448], kv[:, :448], rtol=0.13, atol=0.03)
    train = record(plan, q, dequant, sink, meta)
    plan.capture()
    decode = record(
        plan,
        q,
        gather_packed_joint_kv(
            cache, slots, block_size=block_size, encoding=encoding, backend=backend
        ),
        sink,
        meta,
        mode="decode",
        use_graph=True,
    )
    assert same_bytes(train.output, decode.output) and same_bytes(train.lse, decode.lse)


def test_640_selected_rows_128_heads_and_packed_order(backend):
    torch.manual_seed(24640)
    c = JointAttentionContract(128, 512, 2, 640, 512, 128)
    qkeys = torch.tensor([[0, 2047]], dtype=torch.int64, device="cuda")
    keys = [(0, 0, 4 * i + 3) for i in range(512)] + [(0, 1, i) for i in range(1920, 2048)]
    kvkeys = torch.tensor(keys, dtype=torch.int64, device="cuda")
    comp = torch.arange(512, device="cuda", dtype=torch.int32)[None]
    meta = JointMetadata.bind(c, qkeys, kvkeys, comp, version="v1")
    q = torch.randn(1, 128, 512, device="cuda").bfloat16()
    kv = torch.randn(640, 512, device="cuda").bfloat16()
    sink = torch.randn(128, device="cuda")
    plan = JointAttentionPlan(c, backend=backend)
    out = record(plan, q, kv, sink, meta)
    expected, expected_lse = reference(q, kv[meta.permutation], sink, meta.indices, c.softmax_scale)
    torch.testing.assert_close(out.output.double(), expected, rtol=0.025, atol=0.006)
    torch.testing.assert_close(out.lse.double(), expected_lse, rtol=1e-5, atol=2e-5)
    plan.capture()
    other = record(plan, q, kv, sink, meta, use_graph=True)
    assert same_bytes(other.output, out.output) and same_bytes(other.lse, out.lse)


@pytest.mark.parametrize("fault", ["range", "dtype", "encoding", "size", "stride", "budget"])
def test_packed_cache_guards(fault, backend):
    cache = torch.zeros((2, 37440), dtype=torch.uint8, device="cuda")
    slots = torch.tensor([0, 127], device="cuda")
    encoding, size, budget = "ocp", 64, 1 << 30
    if fault == "range":
        slots[1] = 128
    if fault == "dtype":
        slots = slots.int()
    if fault == "encoding":
        encoding = "auto"
    if fault == "size":
        cache = cache[:, :128].contiguous()
    if fault == "stride":
        cache = torch.zeros((2, 37441), dtype=torch.uint8, device="cuda")
    if fault == "budget":
        budget = 1
    with pytest.raises(JointAttentionError):
        gather_packed_joint_kv(
            cache, slots, block_size=size, encoding=encoding, backend=backend, max_bytes=budget
        )


def test_packed_empty_and_workspace_addresses_stable(backend):
    cache = torch.zeros((1, 37440), dtype=torch.uint8, device="cuda")
    slots = torch.empty(0, dtype=torch.int64, device="cuda")
    assert gather_packed_joint_kv(
        cache, slots, block_size=64, encoding="ocp", backend=backend
    ).shape == (0, 512)
    plan, q, kv, sink, meta, _ = case(backend=backend)
    plan.capture()
    buffers = (plan._q, plan._kv, plan._sink, plan._indices, plan._active, plan._out, plan._lse)
    pointers = tuple(t.data_ptr() for t in buffers)
    for _ in range(10):
        record(plan, q, kv, sink, meta, use_graph=True)
    assert tuple(t.data_ptr() for t in buffers) == pointers


@pytest.mark.parametrize("ratio", [4, 128])
def test_real_compressed_group_boundaries(ratio, backend):
    c = JointAttentionContract(13, 512, 4, 6, 2, 4, compression_ratio=ratio)
    pos = ratio * 2 - 1
    keys = torch.tensor([[0, pos]], device="cuda")
    kvkeys = torch.tensor(
        [[0, 0, ratio - 1], [0, 0, pos]] + [[0, 1, p] for p in range(pos - 3, pos + 1)],
        device="cuda",
    )
    comp = torch.tensor([[1, 0]], dtype=torch.int32, device="cuda")
    meta = JointMetadata.bind(c, keys, kvkeys, comp, version="v1")
    plan = JointAttentionPlan(c, backend=backend)
    q, kv = (
        torch.randn(1, 13, 512, device="cuda").bfloat16(),
        torch.randn(6, 512, device="cuda").bfloat16(),
    )
    sink = torch.randn(13, device="cuda")
    out = record(plan, q, kv, sink, meta)
    expected, expected_lse = reference(q, kv[meta.permutation], sink, meta.indices, c.softmax_scale)
    torch.testing.assert_close(out.output.double(), expected, rtol=0.025, atol=0.006)
    torch.testing.assert_close(out.lse.double(), expected_lse, rtol=1e-5, atol=2e-5)


def test_concurrent_host_threads_do_not_reuse_pending_buffers(backend):
    from concurrent.futures import ThreadPoolExecutor

    plan, q, kv, sink, meta, _ = case(backend=backend, m=7)
    plan.capture()
    expected = record(plan, q, kv, sink, meta)
    torch.cuda.synchronize()

    def worker(_):
        torch.cuda.set_device(backend.device)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            result = record(plan, q, kv, sink, meta, use_graph=True)
        stream.synchronize()
        return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(worker, range(8)))
    for result in results:
        result.validate_saved()
        assert same_bytes(result.output, expected.output) and same_bytes(result.lse, expected.lse)
