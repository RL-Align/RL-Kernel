import pytest
import torch

from rl_engine.contracts.operators.attention import AttentionContractError
from rl_engine.contracts.operators.indexer_training import IndexerAuxContract
from rl_engine.models.deepseek_v4.indexer_training import (
    IndexerAuxiliary,
    IndexerAuxMetadata,
    canonical_indexer_auxiliary_loss,
)
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="MI300X auxiliary forward validation",
)


def make_case(precision="mxfp4", rows=5):
    c = IndexerAuxContract(
        max(1, min(rows, 4)), 0.01, 8, 13, 128, selected_capacity=4, cp_size=2, precision=precision
    )
    queries = torch.tensor([[0, 7], [0, 11], [1, 3], [1, 7], [-1, -1]], device="cuda")[
        :rows
    ].contiguous()
    keys = torch.tensor([[1, 0], [0, 1], [0, 0], [1, 1], [0, 2]], device="cuda")
    topk = torch.tensor(
        [[1, 0, 1, -1], [2, 0, 1, -1], [0, -1, -1, -1], [1, 0, -1, -1], [-1, -1, -1, -1]],
        device="cuda",
    )[:rows].contiguous()
    owners = torch.tensor([1, 0, 0, 1, 0], device="cuda")
    metadata = IndexerAuxMetadata.bind(c, queries, keys, topk, version="v1", key_owners=owners)
    torch.manual_seed(2601)
    q = torch.randn(rows, 8, 128, device="cuda").bfloat16()
    k = torch.randn(5, 128, device="cuda").bfloat16()
    w = torch.randn(rows, 8, device="cuda")
    teacher_q = torch.randn(rows, 13, 128, device="cuda").bfloat16()
    teacher_k = torch.randn(5, 128, device="cuda").bfloat16()
    operator = IndexerAuxiliary(c, layer_id="aux-test")
    q = operator.prepare_operand(q, role="q", version="v1")
    k = operator.prepare_operand(k, role="k", version="v1")
    return operator, q, k, w, teacher_q, teacher_k, metadata


def oracle(operator, q, k, weights, teacher_q, teacher_k, metadata):
    c = operator.contract
    losses, predictions, targets = [], [], []
    for row in range(q.master.shape[0]):
        indices = metadata.selected_indices[row]
        selected = indices[indices >= 0].long()
        pred = torch.zeros(c.selected_capacity, dtype=torch.float64, device="cuda")
        target = torch.zeros_like(pred)
        if selected.numel():
            corr = q.real[row].double() @ k.real[selected].double().t()
            score = (
                (corr / 128**0.5).relu() * (weights[row].double() / c.num_heads**0.5)[:, None]
            ).sum(0)
            pred[: len(selected)] = score.softmax(0)
            teacher = teacher_q[row].double() @ teacher_k[selected].double().t()
            target[: len(selected)] = (teacher * c.softmax_scale).softmax(-1).mean(0)
        loss = (target * ((target + 1e-10).log() - (pred + 1e-10).log())).sum()
        losses.append(loss * c.loss_coeff / c.global_query_count)
        predictions.append(pred)
        targets.append(target)
    if not losses:
        return (
            torch.empty(0, device="cuda", dtype=torch.float64),
            torch.empty((0, c.selected_capacity), device="cuda", dtype=torch.float64),
            torch.empty((0, c.selected_capacity), device="cuda", dtype=torch.float64),
        )
    return torch.stack(losses), torch.stack(predictions), torch.stack(targets)


@pytest.mark.parametrize("precision", ["mxfp4", "fp8"])
@pytest.mark.parametrize("rows", [0, 1, 5])
def test_sparse_auxiliary_native_forward(precision, rows):
    op, q, k, w, tq, tk, metadata = make_case(precision, rows)
    saved = op.forward_record(q, k, w, tq, tk, metadata=metadata, version="v1")
    losses, pred, target = oracle(op, q, k, w, tq, tk, metadata)
    torch.testing.assert_close(saved.per_query_loss.double(), losses, rtol=3e-4, atol=2e-7)
    torch.testing.assert_close(saved.predicted.double(), pred, rtol=3e-5, atol=2e-6)
    torch.testing.assert_close(saved.target.double(), target, rtol=3e-5, atol=2e-6)
    assert not saved.per_query_loss.requires_grad
    saved.validate_saved()


def gradient_oracle(op, q, k, w, tq, tk, metadata):
    qr = q.real.double().detach().requires_grad_()
    kr = k.real.double().detach().requires_grad_()
    wr = w.double().detach().requires_grad_()
    loss = (qr.sum() + kr.sum() + wr.sum()) * 0
    c = op.contract
    for row in range(qr.shape[0]):
        selected = metadata.selected_indices[row]
        selected = selected[selected >= 0].long()
        if not len(selected):
            continue
        corr = qr[row] @ kr[selected].t()
        score = ((corr / 128**0.5).relu() * (wr[row] / c.num_heads**0.5)[:, None]).sum(0)
        pred = score.softmax(0)
        target = (
            ((tq[row].double() @ tk[selected].double().t()) * c.softmax_scale)
            .softmax(-1)
            .mean(0)
            .detach()
        )
        row_loss = (target * ((target + 1e-10).log() - (pred + 1e-10).log())).sum()
        loss = loss + row_loss * c.loss_coeff / c.global_query_count
    loss.backward()
    return qr.grad, kr.grad, wr.grad


@pytest.mark.parametrize("precision", ["mxfp4", "fp8"])
@pytest.mark.parametrize("rows", [0, 1, 5])
def test_native_auxiliary_surrogate_vjp(precision, rows):
    op, q, k, w, tq, tk, meta = make_case(precision, rows)
    for t in (q.master, k.master, w, tq, tk):
        t.requires_grad_()
    expected = gradient_oracle(op, q, k, w, tq, tk, meta)
    with canonical_backward_session() as session:
        result = op.autograd(q, k, w, tq, tk, metadata=meta, version="v1")
        canonical_indexer_auxiliary_loss([result]).backward()
        session.validate_complete()
    for actual, target in zip((q.master.grad, k.master.grad), expected[:2], strict=True):
        torch.testing.assert_close(actual.double(), target, rtol=0.03, atol=2e-6)
    torch.testing.assert_close(w.grad.double(), expected[2], rtol=3e-4, atol=2e-7)
    assert tq.grad is None and tk.grad is None


@pytest.mark.parametrize("precision", ["mxfp4", "fp8"])
def test_exact_auxiliary_microbatch_gradients_and_loss(precision):
    op, q, k, w, tq, tk, meta = make_case(precision, 5)
    for t in (q.master, k.master, w):
        t.requires_grad_()
    with canonical_backward_session() as session:
        whole = op.autograd(q, k, w, tq, tk, metadata=meta, version="v1")
        whole_loss = canonical_indexer_auxiliary_loss([whole])
        whole_loss.backward()
        session.validate_complete()
    expected = tuple(t.grad.clone() for t in (q.master, k.master, w))
    q.master.grad = k.master.grad = w.grad = None
    perm = torch.tensor([4, 1, 3, 0, 2], device="cuda")
    with canonical_backward_session() as session:
        parts = []
        for index in (perm[:2], perm[2:3], perm[3:]):
            query = op.prepare_operand(q.master[index].contiguous(), role="q", version="v1")
            bound = IndexerAuxMetadata.bind(
                op.contract,
                meta.queries.keys[index],
                meta.key_ids,
                meta.selected_group_ids[index],
                version="v1",
                key_owners=meta.key_owners,
            )
            result = op.autograd(
                query,
                k,
                w[index].contiguous(),
                tq[index].contiguous(),
                tk,
                metadata=bound,
                version="v1",
            )
            assert torch.equal(
                result.per_query_loss.view(torch.uint8),
                whole.per_query_loss[index].contiguous().view(torch.uint8),
            )
            parts.append(result)
        merged = canonical_indexer_auxiliary_loss(list(reversed(parts)))
        assert torch.equal(
            merged.reshape(-1).view(torch.uint8), whole_loss.reshape(-1).view(torch.uint8)
        )
        merged.backward()
        session.validate_complete()
    for actual, target in zip((q.master.grad, k.master.grad, w.grad), expected, strict=True):
        assert torch.equal(
            actual.contiguous().view(torch.uint8), target.contiguous().view(torch.uint8)
        )


@pytest.mark.parametrize("precision", ["mxfp4", "fp8"])
def test_unmodified_mlite_sparse_auxiliary_reference(precision):
    import ast
    import hashlib
    import os
    from pathlib import Path

    source_path = os.environ.get("RLK_MLITE_DSA_SOURCE")
    assert source_path, "set RLK_MLITE_DSA_SOURCE for the pinned mLite auxiliary reference"
    source = Path(source_path).read_bytes()
    assert hashlib.sha256(source).hexdigest() == (
        "0df650e687fa0018dd1bc1dd1f0694d8ebd3ce6a61eb9466af58678dd235fdee"
    )
    nodes = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("_sbhd_to_bshd_indexer_inputs", "cp_indexer_loss")
    ]
    assert len(nodes) == 2
    from typing import Tuple

    namespace = {"torch": torch, "Tensor": torch.Tensor, "Tuple": Tuple}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), source_path, "exec"), namespace)
    reference = namespace["cp_indexer_loss"]
    c = IndexerAuxContract(2, 0.03, 8, 13, 128, selected_capacity=2, precision=precision)
    op = IndexerAuxiliary(c, layer_id="mlite-reference")
    torch.manual_seed(26705)
    qm = torch.randn(2, 8, 128, device="cuda").bfloat16().requires_grad_()
    km = torch.randn(2, 128, device="cuda").bfloat16().requires_grad_()
    w = torch.randn(2, 8, device="cuda", requires_grad=True)
    tq = torch.randn(2, 13, 128, device="cuda").bfloat16()
    tk = torch.randn(2, 128, device="cuda").bfloat16()
    q = op.prepare_operand(qm, role="q", version="v1")
    k = op.prepare_operand(km, role="k", version="v1")
    meta = IndexerAuxMetadata.bind(
        c,
        torch.tensor([[7, 7], [7, 11]], device="cuda"),
        torch.tensor([[7, 0], [7, 1]], device="cuda"),
        torch.tensor([[1, 0], [0, 1]], device="cuda"),
        version="v1",
    )
    qr = q.real.detach().clone().requires_grad_()
    kr = k.real.detach().clone().requires_grad_()
    wr = w.detach().clone().requires_grad_()
    expected = reference(
        qr.unsqueeze(1),
        kr.unsqueeze(1),
        (wr * c.num_heads**-0.5).unsqueeze(1),
        meta.selected_indices.long().unsqueeze(0),
        tq.unsqueeze(1),
        tk.unsqueeze(1),
        mask=torch.zeros((2, 2), device="cuda"),
        softmax_scale=c.softmax_scale,
        loss_coeff=c.loss_coeff,
        sparse_loss=True,
        calculate_per_token_loss=False,
    )
    expected.backward()
    with canonical_backward_session() as session:
        out = op.autograd(q, k, w, tq, tk, metadata=meta, version="v1")
        actual = canonical_indexer_auxiliary_loss([out])
        actual.backward()
        session.validate_complete()
    torch.testing.assert_close(actual, expected.detach(), rtol=5e-4, atol=3e-7)
    for value, target in ((qm.grad, qr.grad), (km.grad, kr.grad)):
        torch.testing.assert_close(value.float(), target, rtol=0.03, atol=2e-6)
    torch.testing.assert_close(w.grad, wr.grad, rtol=5e-4, atol=3e-7)


@pytest.mark.parametrize(
    "fault",
    ["master", "packed", "decoded", "metadata", "weights_dtype", "version", "session", "saved"],
)
def test_auxiliary_lifetime_and_contract_guards(fault):
    op, q, k, w, tq, tk, meta = make_case()
    if fault == "session":
        k.master.requires_grad_()
        with pytest.raises(AttentionContractError, match="session"):
            op.autograd(q, k, w, tq, tk, metadata=meta, version="v1")
        return
    if fault == "saved":
        k.master.requires_grad_()
        with canonical_backward_session():
            result = op.autograd(q, k, w, tq, tk, metadata=meta, version="v1")
            result.predicted.add_(1)
            with pytest.raises(AttentionContractError, match="mutated"):
                canonical_indexer_auxiliary_loss([result]).backward()
        return
    version = "v1"
    if fault == "master":
        q.master.add_(1)
    if fault == "packed":
        k.quantized.data.zero_()
    if fault == "decoded":
        q.real.add_(1)
    if fault == "metadata":
        meta.selected_indices.zero_()
    if fault == "weights_dtype":
        w = w.bfloat16()
    if fault == "version":
        version = "old"
    with pytest.raises(AttentionContractError):
        op.forward_record(q, k, w, tq, tk, metadata=meta, version=version)


def test_cp_owner_gradient_partition_without_claiming_transport():
    op, q, original, w, tq, tk, meta = make_case()
    q.master.requires_grad_()
    original.master.requires_grad_()
    w.requires_grad_()
    with canonical_backward_session() as session:
        result = op.autograd(q, original, w, tq, tk, metadata=meta, version="v1")
        canonical_indexer_auxiliary_loss([result]).backward()
        session.validate_complete()
    expected = original.master.grad.clone()
    zero = torch.where(meta.key_owners == 0)[0]
    one = torch.where(meta.key_owners == 1)[0]
    k0 = original.master[zero].detach().clone().requires_grad_()
    k1 = original.master[one].detach().clone().requires_grad_()
    order = torch.cat((zero, one))
    merged = torch.cat((k0, k1))[torch.argsort(order)].contiguous()
    key = op.prepare_operand(merged, role="k", version="v1")
    with canonical_backward_session() as session:
        result = op.autograd(q, key, w, tq, tk, metadata=meta, version="v1")
        canonical_indexer_auxiliary_loss([result]).backward()
        session.validate_complete()
    assert torch.equal(k0.grad.view(torch.uint8), expected[zero].contiguous().view(torch.uint8))
    assert torch.equal(k1.grad.view(torch.uint8), expected[one].contiguous().view(torch.uint8))


@pytest.mark.parametrize("precision", ["mxfp4", "fp8"])
def test_no_visible_or_allocated_keys_is_zero_loss_and_vjp(precision):
    c = IndexerAuxContract(3, 0.01, 8, 8, 128, selected_capacity=2, precision=precision)
    op = IndexerAuxiliary(c, layer_id="empty-keys")
    qm = torch.randn(3, 8, 128, device="cuda").bfloat16().requires_grad_()
    km = torch.empty((0, 128), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    w = torch.randn(3, 8, device="cuda", requires_grad=True)
    tq = torch.randn(3, 8, 128, device="cuda").bfloat16()
    tk = torch.empty((0, 128), device="cuda", dtype=torch.bfloat16)
    meta = IndexerAuxMetadata.bind(
        c,
        torch.tensor([[7, 0], [7, 1], [7, 2]], device="cuda"),
        torch.empty((0, 2), dtype=torch.int64, device="cuda"),
        torch.full((3, 2), -1, dtype=torch.int64, device="cuda"),
        version="v1",
    )
    q, k = (
        op.prepare_operand(qm, role="q", version="v1"),
        op.prepare_operand(km, role="k", version="v1"),
    )
    with canonical_backward_session() as session:
        out = op.autograd(q, k, w, tq, tk, metadata=meta, version="v1")
        loss = canonical_indexer_auxiliary_loss([out])
        loss.backward()
        session.validate_complete()
    assert loss.item() == 0 and qm.grad.eq(0).all() and w.grad.eq(0).all()
    assert km.grad.shape == km.shape


@pytest.mark.parametrize("precision", ["mxfp4", "fp8"])
def test_auxiliary_fresh_processes(precision, tmp_path):
    import json
    import subprocess
    import sys

    reports = []
    for index in range(2):
        path = tmp_path / f"{precision}-{index}.json"
        subprocess.run(
            [
                sys.executable,
                "tests/models/deepseek_v4/indexer_training_process_probe.py",
                precision,
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        reports.append(json.loads(path.read_text()))
    assert reports[0] == reports[1]


def test_auxiliary_cross_stream_canonical_collection():
    op, q, k, w, tq, tk, meta = make_case()
    for value in (q.master, k.master, w):
        value.requires_grad_()
    with canonical_backward_session() as session:
        out = op.autograd(q, k, w, tq, tk, metadata=meta, version="v1")
        canonical_indexer_auxiliary_loss([out]).backward()
        session.validate_complete()
    expected = tuple(t.grad.clone() for t in (q.master, k.master, w))
    q.master.grad = k.master.grad = w.grad = None
    producer = torch.cuda.current_stream()
    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    selections = (torch.tensor([0, 2], device="cuda"), torch.tensor([4, 3, 1], device="cuda"))
    with canonical_backward_session() as session:
        parts = []
        for stream, selection in zip(streams, selections, strict=True):
            stream.wait_stream(producer)
            with torch.cuda.stream(stream):
                subq = op.prepare_operand(q.master[selection].contiguous(), role="q", version="v1")
                bound = IndexerAuxMetadata.bind(
                    op.contract,
                    meta.queries.keys[selection],
                    meta.key_ids,
                    meta.selected_group_ids[selection],
                    version="v1",
                    key_owners=meta.key_owners,
                )
                parts.append(
                    op.autograd(
                        subq,
                        k,
                        w[selection].contiguous(),
                        tq[selection].contiguous(),
                        tk,
                        metadata=bound,
                        version="v1",
                    )
                )
        for stream in streams:
            producer.wait_stream(stream)
        canonical_indexer_auxiliary_loss(parts).backward()
        session.validate_complete()
    for stream in streams:
        producer.wait_stream(stream)
    for actual, target in zip((q.master.grad, k.master.grad, w.grad), expected, strict=True):
        assert torch.equal(
            actual.contiguous().view(torch.uint8), target.contiguous().view(torch.uint8)
        )


def test_frozen_packed_scales_are_not_an_autograd_interface():
    op, q, k, w, tq, tk, meta = make_case("fp8")
    with pytest.raises(AttentionContractError, match="frozen"):
        op.prepare_operand(
            k.master,
            role="k",
            version="v1",
            packed=(k.quantized.data, k.quantized.scales.clone().requires_grad_()),
        )


def test_encoded_key_snapshot_cannot_change_within_session():
    op, q, k, w, tq, tk, meta = make_case()
    k.master.requires_grad_()
    second = op.prepare_operand(
        k.master, role="k", version="v1", packed=(k.quantized.data, k.quantized.scales)
    )
    with canonical_backward_session():
        for use_second, selection in ((False, slice(0, 2)), (True, slice(2, 4))):
            subq = op.prepare_operand(q.master[selection].contiguous(), role="q", version="v1")
            bound = IndexerAuxMetadata.bind(
                op.contract,
                meta.queries.keys[selection],
                meta.key_ids,
                meta.selected_group_ids[selection],
                version="v1",
                key_owners=meta.key_owners,
            )
            if use_second:
                with pytest.raises(AttentionContractError, match="encoded KV snapshot"):
                    op.autograd(
                        subq,
                        second,
                        w[selection].contiguous(),
                        tq[selection].contiguous(),
                        tk,
                        metadata=bound,
                        version="v1",
                    )
            else:
                op.autograd(
                    subq,
                    k,
                    w[selection].contiguous(),
                    tq[selection].contiguous(),
                    tk,
                    metadata=bound,
                    version="v1",
                )


def test_auxiliary_backward_is_one_submission_per_forward():
    op, q, k, w, tq, tk, meta = make_case()
    k.master.requires_grad_()
    with canonical_backward_session() as session:
        out = op.autograd(q, k, w, tq, tk, metadata=meta, version="v1")
        loss = canonical_indexer_auxiliary_loss([out])
        loss.backward(retain_graph=True)
        session.validate_complete()
        with pytest.raises(AttentionContractError, match="already submitted"):
            loss.backward()


def test_workspace_budget_checked_before_sparse_forward():
    from dataclasses import replace

    old, q, k, w, tq, tk, meta = make_case()
    c = replace(old.contract, max_workspace_bytes=q.master.numel() * 16)
    op = IndexerAuxiliary(c, layer_id="bounded")
    queries = op.prepare_operand(q.master, role="q", version="v1")
    keys = op.prepare_operand(k.master, role="k", version="v1")
    bound = IndexerAuxMetadata.bind(
        c,
        meta.queries.keys,
        meta.key_ids,
        meta.selected_group_ids,
        version="v1",
        key_owners=meta.key_owners,
    )
    with pytest.raises(AttentionContractError, match="workspace"):
        op.forward_record(queries, keys, w, tq, tk, metadata=bound, version="v1")
