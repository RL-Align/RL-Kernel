# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RFC #420 ``token_refiner_block``: the two text-refiner blocks and their final norm.

CPU tests pin the graph to the architecture fingerprint. CUDA tests replay both refiner
blocks of the pinned checkpoint (shard 14) node by node and check:

* every node is bitwise repeatable and batch-row invariant;
* nodes that promise it equal diffusers on diffusers' inputs;
* every node, the block output and the whole refiner stack are about as close to the FP32
  golden as diffusers itself; gradients reaching the input and every parameter likewise;
* the operator trace of a replay equals the declared bindings and covers the fingerprint's
  refiner paths.
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.models.minimax_h3 import fingerprint as fp
from rl_engine.models.minimax_h3 import operator_trace as ot
from rl_engine.validation.models import h3_block as hb
from rl_engine.validation.models import h3_refiner as hr
from rl_engine.validation.models.h3_weights import load_h3_tensors

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
    reason="needs an SM80+ CUDA device",
)
FORWARD_FACTOR, FORWARD_FLOOR = 1.25, 1e-3
BACKWARD_FACTOR, BACKWARD_FLOOR = 1.5, 5e-3


def test_graph_equals_the_fingerprint_vocabulary():
    assert hr.graph_matches_fingerprint()
    assert hr.NODE_NAMES == tuple(name for name, _ in fp.REFINER_NODES)
    assert len(hr.NODES) == 14


def test_graph_is_well_formed_and_owners_are_declared():
    seen = {"hidden"}
    for node in hr.NODES:
        assert set(node.inputs) <= seen, node.name
        assert node.name not in seen, node.name
        assert node.binding.status in ("row", "interim", "reference")
        seen.add(node.name)
    assert hr.OUTPUT == "residual_mlp"
    own = {n.name for n in hr.NODES if n.binding.rfc_row == "token_refiner_block"}
    assert own == {"residual_attn", "residual_mlp"}  # the row owns only the plain adds
    rows_with_kernels = {n.binding.rfc_row for n in hr.NODES if n.binding.status == "row"}
    assert rows_with_kernels == {"h3_rmsnorm", "h3_qkv_gemm", "token_refiner_block"}


def test_refiner_has_no_adaln_and_no_rope_nodes():
    assert not any("adaln" in n.binding.rfc_row or "rope" in n.binding.rfc_row for n in hr.NODES)


@pytest.fixture(scope="module")
def refiner_weights():
    names = [f"{hr.PREFIX}.{b}.{leaf}" for b in (0, 1) for leaf in hr.PARAM_LEAVES]
    names.append(hr.FINAL_NORM)
    try:
        weights = load_h3_tensors(names, "cuda")
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    table = fp.expected_tensor_table(fp.transformer_config())
    for name, tensor in weights.items():
        dtype, shape = table[name]
        assert tuple(tensor.shape) == shape and dtype == "BF16", name
    return weights


@needs_gpu
@pytest.mark.parametrize("tokens", [1, 64, 257])
@pytest.mark.parametrize("block", [0, 1])
def test_nodes_accuracy_repeat_batch_and_isolated(refiner_weights, block, tokens):
    report = hr.run_refiner_case(refiner_weights, block=block, tokens=tokens, seed=3)
    assert report["first_repeat_drift"] is None
    assert report["first_batch_drift"] is None
    for entry in report["nodes"]:
        name = entry["node"]
        assert entry["candidate_rel_err_vs_golden"] <= (
            FORWARD_FACTOR * entry["provider_rel_err_vs_golden"] + FORWARD_FLOOR
        ), name
        if entry["provider_bitwise_isolated_promised"]:
            assert entry["isolated_vs_provider"]["bitwise_equal"], name
    out = report["output"]
    assert out["candidate_rel_err_vs_golden"] <= (
        FORWARD_FACTOR * out["provider_rel_err_vs_golden"] + FORWARD_FLOOR
    )


@needs_gpu
def test_report_names_rows_status_and_backend_per_node(refiner_weights):
    report = hr.run_refiner_case(refiner_weights, block=0, tokens=16)
    assert [e["node"] for e in report["nodes"]] == list(hr.NODE_NAMES)
    by = {e["node"]: e for e in report["nodes"]}
    assert by["q_proj"]["status"] == "row" and by["q_proj"]["rfc_row"] == "h3_qkv_gemm"
    assert by["attention"]["status"] == "interim"
    assert by["residual_attn"]["rfc_row"] == "token_refiner_block"
    assert all(len(e["candidate_sha256"]) == 64 for e in report["nodes"])


@needs_gpu
def test_full_refiner_stack_against_golden_and_provider(refiner_weights):
    ops = hb.CandidateOps()
    gen = torch.Generator("cpu").manual_seed(11)
    hidden = torch.randn(1, 130, hr.HIDDEN, generator=gen).bfloat16().cuda()
    with torch.no_grad():
        cand = hr.run_refiner("candidate", refiner_weights, hidden, ops)
        again = hr.run_refiner("candidate", refiner_weights, hidden, ops)
        provider = hr.run_refiner("provider", refiner_weights, hidden)
        golden = hr.run_refiner("golden", refiner_weights, hidden)
    assert torch.equal(cand, again)
    assert cand.dtype == torch.bfloat16 and cand.shape == hidden.shape
    assert hb.rel_err(cand, golden) <= FORWARD_FACTOR * hb.rel_err(provider, golden) + FORWARD_FLOOR


@needs_gpu
def test_refiner_rows_do_not_depend_on_batch_neighbours(refiner_weights):
    ops = hb.CandidateOps()
    gen = torch.Generator("cpu").manual_seed(5)
    hidden = torch.randn(2, 65, hr.HIDDEN, generator=gen).bfloat16().cuda()
    with torch.no_grad():
        both = hr.run_refiner("candidate", refiner_weights, hidden, ops)
        first = hr.run_refiner("candidate", refiner_weights, hidden[:1].contiguous(), ops)
    assert torch.equal(both[:1], first)


@needs_gpu
def test_backward_against_fp32_golden(refiner_weights):
    ops = hb.CandidateOps()
    block = 0
    leaves = [f"{hr.PREFIX}.{block}.{leaf}" for leaf in hr.PARAM_LEAVES]
    gen = torch.Generator("cpu").manual_seed(2)
    hidden0 = torch.randn(1, 33, hr.HIDDEN, generator=gen).bfloat16().cuda()
    grad_out = torch.randn(1, 33, hr.HIDDEN, generator=gen).bfloat16().cuda()

    def grads(mode, dtype):
        weights = {
            n: refiner_weights[n].detach().to(dtype).clone().requires_grad_() for n in leaves
        }
        hidden = hidden0.detach().to(dtype).clone().requires_grad_()
        ctx = {"weights": weights, "prefix": f"{hr.PREFIX}.{block}", "hidden": hidden}
        out = hr.run_graph(mode, ctx, ops)[hr.OUTPUT]
        out.backward(grad_out.to(out.dtype))
        return {"hidden": hidden.grad, **{n: weights[n].grad for n in leaves}}

    cand = grads("candidate", torch.bfloat16)
    again = grads("candidate", torch.bfloat16)
    provider = grads("provider", torch.bfloat16)
    golden = grads("golden", torch.float32)
    for name in cand:
        assert torch.equal(cand[name], again[name]), f"{name}: backward is not repeatable"
        c_err = hb.rel_err(cand[name], golden[name])
        p_err = hb.rel_err(provider[name], golden[name])
        assert c_err <= BACKWARD_FACTOR * p_err + BACKWARD_FLOOR, (name, c_err, p_err)


@needs_gpu
def test_operator_trace_covers_the_refiner_paths(refiner_weights):
    ops = hb.CandidateOps()
    gen = torch.Generator("cpu").manual_seed(1)
    hidden = torch.randn(1, 17, hr.HIDDEN, generator=gen).bfloat16().cuda()
    trace = ot.OperatorTrace(metadata={"stage": "token_refiner"})
    with torch.no_grad():
        hr.run_refiner(
            "candidate",
            refiner_weights,
            hidden,
            ops,
            observer_factory=lambda b: ot.block_observer(
                trace, block_index=b, stage="token_refiner"
            ),
        )
    paths = [
        n.path
        for n in fp.build_graph()
        if n.path.startswith("token_refiner.refiner_blocks.") and n.stage == "token_refiner"
    ]
    assert [c.path for c in trace.calls] == paths
    assert trace.coverage(paths=paths) == {"wrong_count": [], "unexpected": []}
    bound = {n.name: n.binding for n in hr.NODES}
    for call in trace.calls:
        node = call.path.rsplit(".", 1)[1]
        assert (call.status, call.row) == (bound[node].status, bound[node].rfc_row)
    assert trace.violations(paths=paths)  # interim nodes still present
    assert trace.violations(paths=paths, allowed_statuses=("row", "interim")) == []


@needs_gpu
def test_text_path_context_projection_then_refiner(refiner_weights):
    """context_embedder (row) feeding the refiner, against the provider chain."""

    names = ["context_embedder.weight", "context_embedder.bias"]
    ctxw = load_h3_tensors(names, "cuda")
    from rl_engine.runtime.registry import KernelRegistry

    proj = KernelRegistry().get_op("text_context_projection", device="cuda")
    gen = torch.Generator("cpu").manual_seed(9)
    text = torch.randn(70, 5120, generator=gen).bfloat16().cuda()
    ops = hb.CandidateOps()
    with torch.no_grad():
        h_cand = proj.forward(text, ctxw[names[0]], ctxw[names[1]]).unsqueeze(0)
        out = hr.run_refiner("candidate", refiner_weights, h_cand, ops)
        h_prov = torch.nn.functional.linear(text, ctxw[names[0]], ctxw[names[1]]).unsqueeze(0)
        provider = hr.run_refiner("provider", refiner_weights, h_prov)
        golden = hr.run_refiner("golden", refiner_weights, h_prov)
    assert hb.rel_err(out, golden) <= 2 * hb.rel_err(provider, golden) + 2e-3
