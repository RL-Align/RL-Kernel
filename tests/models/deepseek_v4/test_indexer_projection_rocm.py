# SPDX-License-Identifier: Apache-2.0
"""Real-library BI acceptance, separate from ordinary numerical tolerances."""

import ast
from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rl_engine.backends.rocm.gemm.indexer_projection import AiterIndexerProjectionBackend
from rl_engine.contracts.operators.indexer_projection import (
    IndexerProjectionContract,
    IndexerProjectionError,
)
from rl_engine.integrations.engines.rollout.vllm.indexer_projection import (
    bind_indexer_projection as bind_vllm,
)
from rl_engine.integrations.engines.train.megatron.indexer_projection import (
    bind_indexer_projection as bind_megatron,
)
from rl_engine.models.deepseek_v4.indexer_projection import (
    IndexerLogicalRows,
    IndexerProjection,
    PreparedIndexerFP8Q,
)
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session

pytestmark = [
    pytest.mark.ws1,
    pytest.mark.skipif(
        torch.version.hip is None or not torch.cuda.is_available(), reason="ROCm GPU"
    ),
]


def byte_equal(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def rand(shape, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.bfloat16).cuda()


@pytest.fixture(scope="module")
def backend():
    return AiterIndexerProjectionBackend()


@pytest.fixture(scope="module")
def real_geometry():
    c = IndexerProjectionContract()
    return c, rand((257, 1024), 1), rand((257, 4096), 2), rand((8192, 1024), 3), rand((64, 4096), 4)


@pytest.mark.parametrize("q_output", ["bf16", "fp32"])
@pytest.mark.parametrize("weights_output", ["bf16", "fp32"])
def test_real_0731_projection_geometry_batch_rows_and_reordering(
    backend, real_geometry, q_output, weights_output
):
    c, qr, hidden, wq, ww = real_geometry
    c = replace(c, q_output=q_output, weights_output=weights_output)
    op = IndexerProjection(c, layer_id="L0", backend=backend)
    baseline = op(qr, hidden, wq, ww)
    plan_before = backend.plan(4096, 64, "weights")
    for m in (0, 1, 2, 7, 16, 31, 32, 63, 64, 65, 127, 128, 129, 257):
        out = op(qr[:m], hidden[:m], wq, ww)
        byte_equal(out.q, baseline.q[:m])
        byte_equal(out.weights, baseline.weights[:m])
        assert backend.plan(4096, 64, "weights") is plan_before
    order = torch.tensor([127, 0, 65, 3, 128, 1, 256], device="cuda")
    out = op(qr[order], hidden[order], wq, ww)
    byte_equal(out.q, baseline.q[order])
    byte_equal(out.weights, baseline.weights[order])
    q_gold = qr[:2].double() @ wq[:32].double().T
    w_gold = hidden[:2].double() @ ww.double().T
    for actual, expected in (
        (baseline.q[:2].reshape(2, -1)[:, :32], q_gold),
        (baseline.weights[:2], w_gold),
    ):
        torch.testing.assert_close(
            actual.double(),
            expected,
            rtol=0.006 if actual.dtype == torch.bfloat16 else 3e-5,
            atol=2e-4,
        )
    if weights_output == "fp32":
        assert not torch.equal(baseline.weights, baseline.weights.bfloat16().float())


def test_ragged_k_and_n_and_native_vllm_reference(backend):
    from vllm.model_executor.layers.batch_invariant import linear_batch_invariant

    c = IndexerProjectionContract(hidden_size=130, q_lora_rank=67, num_heads=3, head_dim=8)
    x, h, wq, ww = rand((129, 67), 81), rand((129, 130), 82), rand((24, 67), 83), rand((3, 130), 84)
    op = IndexerProjection(c, layer_id="ragged", backend=backend)
    full = op(x, h, wq, ww)
    for m in (1, 7, 64, 65, 129):
        out = op(x[:m], h[:m], wq, ww)
        byte_equal(out.q, full.q[:m])
        byte_equal(out.weights, full.weights[:m])
    # Different upstream K-reduction contracts are accuracy references, not
    # falsely claimed bitwise-identical backends.
    torch.testing.assert_close(
        full.q.flatten(1), linear_batch_invariant(x, wq), rtol=0.02, atol=0.03
    )
    torch.testing.assert_close(
        full.weights.bfloat16(), linear_batch_invariant(h, ww), rtol=0.02, atol=0.03
    )


@pytest.mark.parametrize("q_output", ["bf16", "fp32"])
def test_real_block_fp8_q_no_bf16_substitution(backend, real_geometry, q_output):
    c, qr, hidden, master, ww = real_geometry
    c = replace(c, q_precision="block_fp8_ue8m0", q_output=q_output)
    checkpoint = master.to(torch.float8_e4m3fn)
    original = checkpoint.clone()
    source_scales = torch.ones((64, 8), dtype=torch.float32, device="cuda")
    packed = PreparedIndexerFP8Q.from_checkpoint(
        checkpoint,
        source_scales,
        version="checkpoint-1",
        backend=backend,
    )
    byte_equal(checkpoint, original)
    assert packed.data.dtype == torch.float8_e4m3fnuz
    assert packed.master is None  # Serving does not require a BF16 weight copy.
    op = IndexerProjection(c, layer_id="fp8", backend=backend)
    full = op(qr, hidden, None, ww, packed_q=packed, q_weight_version="checkpoint-1")
    full_quant, full_scales = backend.quantize_activation(qr)
    for m in (0, 1, 2, 7, 32, 65, 129, 257):
        out = op(qr[:m], hidden[:m], None, ww, packed_q=packed, q_weight_version="checkpoint-1")
        byte_equal(out.q, full.q[:m])
        byte_equal(out.weights, full.weights[:m])
        if m:
            qx, scales = backend.quantize_activation(qr[:m])
            byte_equal(qx, full_quant[:m])
            byte_equal(scales, full_scales[:m])
    order = torch.tensor([64, 3, 0, 256, 127], device="cuda")
    out = op(qr[order], hidden[order], None, ww, packed_q=packed, q_weight_version="checkpoint-1")
    byte_equal(out.q, full.q[order])
    qx = (full_quant[:2].float().view(2, 8, 128) * full_scales[:2, :, None]).reshape(2, 1024)
    qw = (
        packed.data[:128].float().view(1, 128, 8, 128) * packed.scales[:1, None, :, None]
    ).reshape(128, 1024)
    expected = qx.double() @ qw.double().T
    actual = full.q[:2].flatten(1)[:, :128]
    torch.testing.assert_close(
        actual.double(), expected, rtol=0.006 if actual.dtype == torch.bfloat16 else 3e-5, atol=2e-4
    )


def run_gradient_case(backend, *, precision, shuffled, chunks, padding):
    c = IndexerProjectionContract(
        hidden_size=256,
        q_lora_rank=128,
        num_heads=2,
        head_dim=128,
        q_precision=precision,
        q_output="fp32",
        max_rows=64,
    )
    m = 17
    x0, h0 = rand((m, 128), 51), rand((m, 256), 52)
    wq = rand((256, 128), 53).requires_grad_(True)
    ww = rand((2, 256), 54).requires_grad_(True)
    gq, gw = rand((m, 256), 55).float(), rand((m, 2), 56).float()
    packed = None
    if precision != "bf16":
        packed = PreparedIndexerFP8Q.from_checkpoint(
            wq.detach().to(torch.float8_e4m3fn),
            torch.ones((2, 1), dtype=torch.float32, device="cuda"),
            version="v1",
            backend=backend,
            master=wq,
        )
    op = IndexerProjection(c, layer_id="L1", backend=backend)
    order = list(reversed(range(m))) if shuffled else list(range(m))
    all_q = torch.empty((m, 256), device="cuda")
    all_w = torch.empty((m, 2), device="cuda")
    dx, dh = torch.empty_like(x0), torch.empty_like(h0)
    outputs, gradients, leaves = [], [], []
    cursor = 0
    with canonical_backward_session() as session:
        for count in chunks:
            ids = order[cursor : cursor + count]
            cursor += count
            index = torch.tensor(ids, dtype=torch.int64, device="cuda")
            x = torch.cat(
                (x0[index], torch.zeros((padding, 128), dtype=torch.bfloat16, device="cuda"))
            ).requires_grad_(True)
            h = torch.cat(
                (h0[index], torch.zeros((padding, 256), dtype=torch.bfloat16, device="cuda"))
            ).requires_grad_(True)
            keys = torch.tensor(
                [[7, i] for i in ids] + [[-1, -1]] * padding, dtype=torch.int64, device="cuda"
            ).reshape(-1, 2)
            rows = IndexerLogicalRows.bind(keys)
            out = op(
                x,
                h,
                wq,
                ww,
                logical_rows=rows,
                packed_q=packed,
                q_weight_version=None if packed is None else "v1",
            )
            all_q.index_copy_(0, index, out.q[:count].detach().flatten(1))
            all_w.index_copy_(0, index, out.weights[:count].detach())
            outputs.extend((out.q, out.weights))
            gradients.extend(
                (
                    torch.cat((gq[index], torch.zeros((padding, 256), device="cuda"))).view(
                        -1, 2, 128
                    ),
                    torch.cat((gw[index], torch.zeros((padding, 2), device="cuda"))),
                )
            )
            leaves.append((index, x, h))
        assert cursor == m
        torch.autograd.backward(list(reversed(outputs)), list(reversed(gradients)))
        session.validate_complete()
    for index, x, h in leaves:
        n = len(index)
        dx.index_copy_(0, index, x.grad[:n])
        dh.index_copy_(0, index, h.grad[:n])
        assert not torch.count_nonzero(x.grad[n:]) and not torch.count_nonzero(h.grad[n:])
    return all_q, all_w, dx, dh, wq.grad, ww.grad


@pytest.mark.parametrize("precision", ["bf16", "block_fp8_ue8m0"])
@pytest.mark.parametrize("chunks,padding", [([17], 3), ([4, 0, 1, 12], 2), ([1] * 17, 1)])
def test_forward_dx_dw_bitwise_under_padding_permutation_and_microbatching(
    backend,
    precision,
    chunks,
    padding,
):
    baseline = run_gradient_case(
        backend, precision=precision, shuffled=False, chunks=[17], padding=0
    )
    actual = run_gradient_case(
        backend, precision=precision, shuffled=True, chunks=chunks, padding=padding
    )
    for got, expected in zip(actual, baseline, strict=True):
        byte_equal(got, expected)


def test_native_megatron_vjp_semantics_and_numerical_accuracy(backend):
    path = os.environ.get("RLK_MEGATRON_DENSE_SOURCE")
    if not path:
        pytest.skip("set RLK_MEGATRON_DENSE_SOURCE for unmodified mLite VJP reference")
    source = ast.parse(Path(path).read_text())
    functions = [
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name == "native_linear_vjp"
    ]
    assert len(functions) == 1
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=functions, type_ignores=[]), path, "exec"), namespace)
    x, w, grad = rand((17, 128), 91), rand((64, 128), 92), rand((17, 64), 93).float()
    # Run the unchanged source on CPU, avoiding its unvalidated TE GPU branch.
    dx_native, dw_native = namespace["native_linear_vjp"](grad.cpu(), x.cpu(), w.cpu())
    dx = backend.input_gradient(grad.bfloat16(), w)
    dw = backend.weight_gradient(x, grad.bfloat16())
    torch.testing.assert_close(dx.cpu(), dx_native, rtol=0.02, atol=0.03)
    torch.testing.assert_close(dw.cpu(), dw_native, rtol=0.02, atol=0.03)
    torch.testing.assert_close(
        dx.double(), grad.bfloat16().double() @ w.double(), rtol=0.006, atol=2e-4
    )
    torch.testing.assert_close(
        dw.double(), grad.bfloat16().double().T @ x.double(), rtol=0.006, atol=2e-4
    )


@pytest.mark.parametrize("precision", ["bf16", "block_fp8_ue8m0"])
def test_train_rollout_adapters_share_raw_outputs(backend, precision):
    c = IndexerProjectionContract(
        hidden_size=256, q_lora_rank=128, num_heads=2, head_dim=128, q_precision=precision
    )
    wq = rand((256, 128), 71).requires_grad_(True)
    ww = rand((2, 256), 72).requires_grad_(True)
    packed = None
    if precision != "bf16":
        packed = PreparedIndexerFP8Q.from_checkpoint(
            wq.detach().to(torch.float8_e4m3fn),
            torch.ones((2, 1), dtype=torch.float32, device="cuda"),
            version="v1",
            backend=backend,
            master=wq,
        )
    indexer = SimpleNamespace(
        wq_b=SimpleNamespace(weight=wq, bias=None),
        weights_proj=SimpleNamespace(weight=ww, bias=None),
    )
    train = bind_megatron(indexer, contract=c, layer_id="L0", backend=backend, packed_q=packed)
    rollout = bind_vllm(indexer, contract=c, layer_id="L0", backend=backend, packed_q=packed)
    x, h = rand((31, 128), 73), rand((31, 256), 74)
    rows = IndexerLogicalRows.bind(
        torch.tensor([[9, i] for i in range(31)], dtype=torch.int64, device="cuda")
    )
    with canonical_backward_session() as session:
        a = train(x, h, logical_rows=rows)
        with torch.no_grad():
            b = rollout(x, h)
        assert a.q.requires_grad and not b.q.requires_grad
        byte_equal(a.q, b.q)
        byte_equal(a.weights, b.weights)
        (a.q.float().sum() + a.weights.sum()).backward()
        session.validate_complete()


def test_fp8_version_source_mutation_and_scale_validation(backend):
    c = IndexerProjectionContract(
        hidden_size=128, q_lora_rank=128, num_heads=1, head_dim=128, q_precision="block_fp8_ue8m0"
    )
    checkpoint = rand((128, 128), 111).to(torch.float8_e4m3fn)
    scales = torch.ones((1, 1), dtype=torch.float32, device="cuda")
    packed = PreparedIndexerFP8Q.from_checkpoint(checkpoint, scales, version="v1", backend=backend)
    op = IndexerProjection(c, layer_id="L0", backend=backend)
    x, h, ww = rand((1, 128), 112), rand((1, 128), 113), rand((1, 128), 114)
    with pytest.raises(IndexerProjectionError, match="version barrier"):
        op(x, h, None, ww, packed_q=packed, q_weight_version="v2")
    checkpoint.view(torch.uint8)[0, 0] = 0
    with pytest.raises(IndexerProjectionError, match="changed"):
        op(x, h, None, ww, packed_q=packed, q_weight_version="v1")
    with pytest.raises(IndexerProjectionError, match="power-of-two"):
        PreparedIndexerFP8Q.from_checkpoint(checkpoint, scales * 1.5, version="v2", backend=backend)
    with pytest.raises(IndexerProjectionError, match="unpacked"):
        PreparedIndexerFP8Q.from_checkpoint(
            checkpoint, scales.byte(), version="v2", backend=backend
        )


def test_hot_forward_has_no_tensor_readback_and_reuses_upstream_symbols(backend, monkeypatch):
    from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
    from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import gemm_a8w8_blockscale
    from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8

    assert backend._gemm16 is gemm_a16w16
    assert backend._gemm8 is gemm_a8w8_blockscale
    assert backend._quantize is per_token_group_quant_fp8
    c = IndexerProjectionContract(
        hidden_size=128, q_lora_rank=128, num_heads=1, head_dim=128, q_precision="block_fp8_ue8m0"
    )
    x, h, ww = rand((4, 128), 121), rand((4, 128), 122), rand((1, 128), 123)
    packed = PreparedIndexerFP8Q.from_checkpoint(
        rand((128, 128), 124).to(torch.float8_e4m3fn),
        torch.ones((1, 1), dtype=torch.float32, device="cuda"),
        version="v1",
        backend=backend,
    )
    op = IndexerProjection(c, layer_id="L0", backend=backend)
    op(x, h, None, ww, packed_q=packed, q_weight_version="v1")

    def forbidden(*args, **kwargs):
        raise AssertionError("host readback in projection hot path")

    with monkeypatch.context() as patch:
        for method in ("cpu", "item", "tolist", "__bool__"):
            patch.setattr(torch.Tensor, method, forbidden)
        op(x, h, None, ww, packed_q=packed, q_weight_version="v1")
    torch.cuda.synchronize()


def test_native_fnuz_loader_and_rollout_without_bf16_master(backend):
    from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
        normalize_e4m3fn_to_e4m3fnuz,
    )

    checkpoint = rand((128, 128), 141).to(torch.float8_e4m3fn)
    checkpoint.view(torch.uint8)[0, 0] = 128  # OCP negative zero, normalized by native vLLM.
    original = checkpoint.clone()
    scales = torch.ones((1, 1), dtype=torch.float32, device="cuda")
    fnuz, fnuz_scales, _ = normalize_e4m3fn_to_e4m3fnuz(checkpoint.clone(), scales)
    a = PreparedIndexerFP8Q.from_checkpoint(checkpoint, scales, version="v1", backend=backend)
    b = PreparedIndexerFP8Q.from_checkpoint(fnuz, fnuz_scales, version="v1", backend=backend)
    byte_equal(a.data, b.data)
    byte_equal(a.scales, b.scales)
    byte_equal(checkpoint, original)
    c = IndexerProjectionContract(
        hidden_size=128, q_lora_rank=128, num_heads=1, head_dim=128, q_precision="block_fp8_ue8m0"
    )
    indexer = SimpleNamespace(
        wq_b=SimpleNamespace(weight=fnuz, bias=None),
        weights_proj=SimpleNamespace(weight=rand((1, 128), 142), bias=None),
    )
    bound = bind_vllm(indexer, contract=c, layer_id="L0", backend=backend, packed_q=b)
    assert bound.q_weight is None
    x, h = rand((5, 128), 143), rand((5, 128), 144)
    expected = IndexerProjection(c, layer_id="L0", backend=backend)(
        x,
        h,
        None,
        indexer.weights_proj.weight,
        packed_q=a,
        q_weight_version="v1",
    )
    byte_equal(bound(x, h).q, expected.q)
    fnuz_scales.mul_(2)
    with pytest.raises(IndexerProjectionError, match="changed"):
        bound(x, h)


def test_independent_projection_hooks_preserve_native_stream_scheduling(backend):
    c = IndexerProjectionContract(hidden_size=256, q_lora_rank=128, num_heads=2, head_dim=128)
    indexer = SimpleNamespace(
        wq_b=SimpleNamespace(weight=rand((256, 128), 151), bias=None),
        weights_proj=SimpleNamespace(weight=rand((2, 256), 152), bias=None),
    )
    bound = bind_vllm(indexer, contract=c, layer_id="L0", backend=backend)
    x, h = rand((17, 128), 153), rand((17, 256), 154)
    expected = bound(x, h)
    current = torch.cuda.current_stream()
    q_stream, w_stream = torch.cuda.Stream(), torch.cuda.Stream()
    q_stream.wait_stream(current)
    w_stream.wait_stream(current)
    with torch.cuda.stream(q_stream):
        q = bound.q(x)
    with torch.cuda.stream(w_stream):
        weights = bound.weights(h)
    current.wait_stream(q_stream)
    current.wait_stream(w_stream)
    byte_equal(q, expected.q)
    byte_equal(weights, expected.weights)
