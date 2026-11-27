# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""One real MiniMax-H3 block, forward and backward (RFC #420 ``ws1_one_h3_block``).

GPU tests run block 0 of the pinned checkpoint on packed FL2VA layouts and
check the block-level promises of ``rl_engine.validation.models.h3_block``:

* every node is bitwise repeatable, and batch row 0 of a two-row batch has the
  bytes of the single-row run (forward, and the gradient reaching every node);
* nodes that promise it are bitwise equal to diffusers on diffusers' inputs;
* every node, the block output and every gradient are about as close to the
  FP32 golden as diffusers itself.

CPU tests pin the packed layouts to the pinned diffusers pipeline and check
the graph's structure.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
import torch

from rl_engine.reference.minimax_h3 import block as gold
from rl_engine.validation.models import h3_block
from rl_engine.validation.models import h3_provider as prov
from rl_engine.validation.models.h3_cases import H3_BLOCK_LAYOUTS, h3_fl2va_layout

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
    reason="needs an SM80+ CUDA device",
)

# sha256(position_ids float64 bytes + token_tags int64 bytes) of diffusers@f53d552's
# ``MiniMaxH3PrepareLayoutStep.build_packed_sequence`` for each layout (text tags 1, patch
# (1, 2, 2), two audio channels, audio tag 2, video tag 0, no keyframe anchors).
PINNED_LAYOUTS = {
    "tiny": "5329a0a13a9996c4fea7a0e676b65aa24302d65bea2e2248ba68d12184cc7c37",
    "small": "64e4e79d8fb91d5a68ae7d15bb29049e08add38da2639030e089c6ef80968931",
    "medium": "16e448800695eedfb85e0aacbca9bd5415ed53fbbe25f701bcb60df5f4754a31",
}

# Accuracy: the candidate's error against the FP32 golden may exceed diffusers' by this
# factor plus an absolute floor (relative to the golden's max magnitude).
FORWARD_FACTOR, FORWARD_FLOOR = 1.25, 1e-3
BACKWARD_FACTOR, BACKWARD_FLOOR = 1.5, 5e-3
# A BF16 leaf gradient may instead exceed diffusers' error by this many ulps of its largest
# golden element. Its max|err| / max|golden| is set by one rounded element, so it moves in
# whole ulps (0.004 relative for attn.norm_q.weight), and the relative rule alone failed on
# seed 2 with torch 2.13 and on seed 0 with torch 2.9.1. Sized from
# tools/validation/models/h3_block_ulp_sweep.py over tiny/small x seeds 0-9 on both torch
# versions (max excess +4.15 ulp) and 34 single-node candidate/provider swaps (max +5.15 ulp);
# reports in reports/experiments/h3-one-block-b200/ulp-sweep-torch*.json.
BACKWARD_BF16_EXCESS_ULPS = 6.0


@pytest.mark.parametrize("layout", sorted(PINNED_LAYOUTS))
def test_layout_matches_pinned_pipeline(layout):
    packed = h3_fl2va_layout(**H3_BLOCK_LAYOUTS[layout], device="cpu")
    data = packed["position_ids"].numpy().tobytes() + packed["token_tags"].numpy().tobytes()
    assert hashlib.sha256(data).hexdigest() == PINNED_LAYOUTS[layout]


def test_graph_is_well_formed():
    seen = {"hidden", "temb"}
    for node in h3_block.NODES:
        assert set(node.inputs) <= seen, node.name
        assert node.name not in seen, node.name
        assert node.binding.status in ("row", "interim", "reference"), node.name
        seen.add(node.name)
    assert h3_block.OUTPUT == "residual_mlp"
    rows = {n.binding.rfc_row for n in h3_block.NODES if n.binding.status == "row"}
    assert rows == {
        "adaln_projection_3mod",
        "h3_rmsnorm",
        "adaln_gate_residual",
        "h3_qkv_gemm",
        "h3_mm_rope_3axis_partial",
        "h3_full_attention",
        *h3_block.GEMM_ROWS.values(),
    }


def test_rope_nodes_call_the_candidate_operator():
    """A missing candidate call must not pass as trivially equal provider replay."""
    x = torch.ones(1, 2, 1, 128)
    cos, sin = torch.ones(2, 96), torch.zeros(2, 96)
    output = torch.empty_like(x)
    calls = []

    def rope(tensor, cosine, sine):
        calls.append((tensor, cosine, sine))
        return output

    for node in h3_block.NODES:
        if node.name in ("rope_q", "rope_k"):
            assert node.candidate(SimpleNamespace(rope=rope), {"rope": (cos, sin)}, x) is output
    assert len(calls) == 2
    for tensor, cosine, sine in calls:
        assert tensor is x and cosine is cos and sine is sin


# (RFC row, status) the block declares per node, written out rather than read from ``NODES``
DECLARED_BINDINGS = {
    "adaln_projection": ("adaln_projection_3mod", "row"),
    "norm1": ("h3_rmsnorm", "row"),
    "q_proj": ("h3_qkv_gemm", "row"),
    "k_proj": ("h3_qkv_gemm", "row"),
    "v_proj": ("h3_qkv_gemm", "row"),
    "q_norm": ("h3_qk_rmsnorm_d128", "interim"),
    "k_norm": ("h3_qk_rmsnorm_d128", "interim"),
    "rope_q": ("h3_mm_rope_3axis_partial", "row"),
    "rope_k": ("h3_mm_rope_3axis_partial", "row"),
    "attention": ("h3_full_attention", "row"),
    "o_proj": ("h3_attention_o_gemm", "row"),
    "residual_attn": ("adaln_gate_residual", "row"),
    "norm2": ("h3_rmsnorm", "row"),
    "ffn_gate_up": ("h3_ffn_gate_up_gemm", "row"),
    "swiglu": ("h3_swiglu", "interim"),
    "ffn_down": ("h3_ffn_down_gemm", "row"),
    "residual_mlp": ("adaln_gate_residual", "row"),
}


class _Ops:
    """Stand-in ops: which GEMM rows and attention operator a run bound."""

    def __init__(self, gemm=True, attention=True):
        self.attention_h3 = object() if attention else None
        self._gemm = gemm

    def gemm_op(self, row):
        return object() if self._gemm else None


def test_bindings_report_the_operators_that_ran():
    static = h3_block.bindings(None)
    assert {n: (b.rfc_row, b.status) for n, b in static.items()} == DECLARED_BINDINGS
    for name, row in h3_block.GEMM_ROWS.items():
        assert static[name].rfc_row == row
        assert static[name].backend == h3_block.GEMM_ROW_OPS[row]
    bound = h3_block.bindings(_Ops())
    assert all(bound[n].status == "row" for n in (*h3_block.GEMM_ROWS, "attention"))
    interim = h3_block.bindings(_Ops(gemm=False, attention=False))
    for name in h3_block.GEMM_ROWS:
        assert interim[name].status == "interim"
        assert interim[name].backend == h3_block.INTERIM_GEMM
        assert interim[name].rfc_row == h3_block.GEMM_ROWS[name]
    assert interim["attention"].status == "interim"
    assert interim["attention"].backend == h3_block.INTERIM_ATTENTION
    others = set(h3_block.NODE_NAMES) - {*h3_block.GEMM_ROWS, "attention"}
    assert all(interim[n] == static[n] for n in others)


def test_interim_gemm_ops_open_no_session():
    from rl_engine.ops.autograd.canonical_backward import active_session

    with h3_block.gemm_session(_Ops(gemm=False)):
        assert active_session() is None
    with h3_block.gemm_session(_Ops()):
        session = active_session()
        assert session is not None
        with h3_block.gemm_session(_Ops()):  # an open session is reused
            assert active_session() is session
    assert active_session() is None


def test_gemm_rows_mode_is_checked():
    with pytest.raises(ValueError, match="gemm_rows"):
        h3_block.CandidateOps(gemm_rows="maybe")


def test_partial_rope_rotates_96_of_128_channels():
    packed = h3_fl2va_layout(**H3_BLOCK_LAYOUTS["tiny"], device="cpu")
    cos, sin = prov.provider_rope(packed["position_ids"])
    assert cos.shape == (packed["position_ids"].shape[0], 96) and cos.dtype == torch.float32
    gold_cos, gold_sin = gold.golden_rope_tables(packed["position_ids"])
    assert torch.equal(cos, gold_cos) and torch.equal(sin, gold_sin)
    x = torch.randn(1, cos.shape[0], 2, 128).bfloat16()
    for out in (prov.provider_apply_rotary(x, cos, sin), gold.golden_apply_rotary(x, cos, sin)):
        assert torch.equal(out[..., 96:].float(), x[..., 96:].float())
        assert not torch.equal(out[..., :96].float(), x[..., :96].float())


@pytest.fixture
def ops(require_h3_kernels):
    """Check optional extensions before constructing any candidate backend."""
    return h3_block.CandidateOps()


@needs_gpu
def test_stack_rows_dispatch_to_cuda(require_h3_kernels, ops):
    backends = ops.backends()
    for name in ("projection", "norm", "gate"):
        assert backends[name].endswith("CudaOp"), backends
    assert backends["rope"].endswith("TritonH3MMRoPEOp"), backends


@needs_gpu
@pytest.mark.parametrize("layout", ["tiny", "small"])
def test_block_forward(h3_block_weights_cuda, ops, layout):
    report = h3_block.run_block_case(h3_block_weights_cuda, layout=layout, ops=ops)
    assert report["rope_acceptance_reference"] == "isolated_model_boundary_reference"
    for entry in report["nodes"]:
        if entry["node"] in ("rope_q", "rope_k"):
            assert entry["isolated_model_boundary_accuracy"]["passed"], entry
    nodes = {node.name: node for node in h3_block.NODES}
    for entry in report["nodes"]:
        name = entry["node"]
        assert entry["repeat_bitwise_equal"], name
        assert entry["batch_row_bitwise_equal"], name
        if nodes[name].provider_bitwise_isolated:
            assert entry["isolated_vs_provider"]["bitwise_equal"], name
        bound = FORWARD_FACTOR * entry["provider_rel_err_vs_golden"] + FORWARD_FLOOR
        assert entry["candidate_rel_err_vs_golden"] <= bound, entry
    assert report["first_repeat_drift"] is None
    assert report["first_batch_drift"] is None
    drift = report["first_isolated_drift"]
    assert drift is None or not nodes[drift].provider_bitwise_isolated
    out = report["output"]
    assert out["candidate_rel_err_vs_golden"] <= (
        FORWARD_FACTOR * out["provider_rel_err_vs_golden"] + FORWARD_FLOOR
    )


@needs_gpu
def test_block_backward(h3_block_weights_cuda, ops):
    report = h3_block.run_block_backward(h3_block_weights_cuda, layout="tiny", ops=ops)
    for name, entry in report["leaves"].items():
        assert entry["repeat_bitwise_equal"], name
        bound = BACKWARD_FACTOR * entry["provider_rel_err_vs_golden"] + BACKWARD_FLOOR
        within_rel = entry["candidate_rel_err_vs_golden"] <= bound
        within_ulps = entry["storage_dtype"] == "bfloat16" and (
            entry["candidate_ulps_vs_golden"]
            <= entry["provider_ulps_vs_golden"] + BACKWARD_BF16_EXCESS_ULPS
        )
        assert within_rel or within_ulps, (name, entry)
    assert report["leaves"]["hidden"]["batch_row_bitwise_equal"]
    assert report["first_grad_repeat_drift"] is None
    assert report["first_grad_batch_drift"] is None


def test_qkv_nodes_share_one_combined_row_call():
    """``q_proj`` / ``k_proj`` / ``v_proj`` call the ``h3_qkv_gemm`` row once, so the block's
    input gradient is the row's fused backward; a copied context recomputes."""

    calls = []

    class _Ops:
        @staticmethod
        def qkv(flat, wq, wk, wv, *, logical_keys):
            """Record the keys of a fused QKV call and return the three projections."""
            calls.append(logical_keys.shape)
            return tuple(flat @ w.t() for w in (wq, wk, wv))

    nodes = {n.name: n for n in h3_block.NODES}
    weights = {f"{h3_block.BLOCK}.{w}": torch.randn(4, 6) for w in h3_block.QKV_WEIGHTS}
    ctx = h3_block.evaluation_ctx({"weights": weights})
    x = torch.randn(1, 3, 6)
    outs = [nodes[n].candidate(_Ops, ctx, x) for n in ("q_proj", "k_proj", "v_proj")]
    assert len(calls) == 1 and calls[0] == (3, 2)
    for out, w in zip(outs, h3_block.QKV_WEIGHTS):
        assert torch.equal(
            out, (x.reshape(3, 6) @ weights[f"{h3_block.BLOCK}.{w}"].t()).view(1, 3, 4)
        )
    other = {**ctx, "weights": {k: v.clone() for k, v in weights.items()}}
    nodes["k_proj"].candidate(_Ops, other, x)
    nodes["q_proj"].candidate(_Ops, ctx, torch.randn(1, 3, 6))
    assert len(calls) == 3


def test_each_graph_evaluation_makes_its_own_fused_qkv_call():
    """Two evaluations with the same ``ctx`` and inputs, or with other ops, each make their own
    combined call: a repeat check never compares an evaluation with its own cached outputs."""

    calls = []

    class _Ops:
        @staticmethod
        def qkv(flat, wq, wk, wv, *, logical_keys):
            """Record a fused QKV call and return the three projections."""
            calls.append(_Ops)
            return tuple(flat @ w.t() for w in (wq, wk, wv))

    class _OtherOps(_Ops):
        @staticmethod
        def qkv(flat, wq, wk, wv, *, logical_keys):
            """The same projections under another operator set."""
            calls.append(_OtherOps)
            return tuple(flat @ w.t() for w in (wq, wk, wv))

    nodes = {n.name: n for n in h3_block.NODES}
    weights = {f"{h3_block.BLOCK}.{w}": torch.randn(4, 6) for w in h3_block.QKV_WEIGHTS}
    ctx = {"weights": weights}
    x = torch.randn(1, 3, 6)
    provider = {"norm1": x}
    for _ in range(2):  # what ``run_isolated`` does per evaluation, with the same inputs
        run = h3_block.evaluation_ctx(ctx)
        first = [nodes[n].candidate(_Ops, run, provider["norm1"]) for n in ("q_proj", "k_proj")]
    assert calls == [_Ops, _Ops] and h3_block.FUSED_QKV_CACHE not in ctx
    second = nodes["q_proj"].candidate(_Ops, run, x)
    assert len(calls) == 2 and torch.equal(second, first[0])  # one evaluation shares its call
    nodes["v_proj"].candidate(_OtherOps, run, x)  # other ops on the same inputs recompute
    assert calls[-1] is _OtherOps


def test_fused_qkv_refuses_a_ctx_that_is_not_one_evaluation():
    """A caller's own ``ctx`` (not made by ``evaluation_ctx``) is refused, and gains no cache."""

    class _Ops:
        @staticmethod
        def qkv(flat, wq, wk, wv, *, logical_keys):
            """Never reached: the ctx is refused first."""
            raise AssertionError("no call for a refused ctx")

    nodes = {n.name: n for n in h3_block.NODES}
    weights = {f"{h3_block.BLOCK}.{w}": torch.randn(4, 6) for w in h3_block.QKV_WEIGHTS}
    ctx = {"weights": weights}
    with pytest.raises(ValueError, match="evaluation_ctx"):
        nodes["q_proj"].candidate(_Ops, ctx, torch.randn(1, 3, 6))
    assert h3_block.FUSED_QKV_CACHE not in ctx


def test_gemm_row_nodes_pass_logical_keys_and_a_parameter_id_per_block():
    calls = []

    class _RowOps:
        def gemm_op(self, row):
            def op(flat, weight, *, logical_keys, parameter_id):
                calls.append((row, tuple(logical_keys.shape), parameter_id))
                return flat @ weight.t()

            return op

    nodes = {n.name: n for n in h3_block.NODES}
    weight = torch.randn(4, 6)
    x = torch.randn(2, 3, 6)
    ctx = {"weights": {f"{h3_block.BLOCK}.ff.net.2.weight": weight}}
    out = nodes["ffn_down"].candidate(_RowOps(), ctx, x)
    assert torch.equal(out, x @ weight.t())
    ctx["parameter_prefix"] = "transformer_blocks.7"
    nodes["ffn_down"].candidate(_RowOps(), ctx, x)
    assert calls == [
        ("h3_ffn_down_gemm", (6, 2), "transformer_blocks.0.ff.net.2.weight"),
        ("h3_ffn_down_gemm", (6, 2), "transformer_blocks.7.ff.net.2.weight"),
    ]


def test_gemm_rows_need_triton_as_well_as_sm90(monkeypatch):
    from rl_engine.backends.shared.triton.gemm import det_gemm

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    monkeypatch.setattr(det_gemm, "_TRITON_AVAILABLE", False)
    assert not h3_block.gemm_rows_supported()  # "auto" then binds the interim GEMM
    monkeypatch.setattr(det_gemm, "_TRITON_AVAILABLE", True)
    assert h3_block.gemm_rows_supported()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 0))
    assert not h3_block.gemm_rows_supported()
    # weights on the CPU: no probe of a CUDA capability, and no row
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: pytest.fail("probed a CPU device")
    )
    assert not h3_block.gemm_rows_supported(torch.device("cpu"))


needs_sm90 = pytest.mark.skipif(
    not h3_block.gemm_rows_supported(), reason="the GEMM rows (#535, #554) need SM90 and Triton"
)


@needs_sm90
def test_gemm_rows_run_their_operators_on_sm90(h3_block_weights_cuda):
    ops = h3_block.CandidateOps(gemm_rows="row")
    forward = h3_block.run_block_case(h3_block_weights_cuda, layout="tiny", ops=ops)
    by = {e["node"]: e for e in forward["nodes"]}
    for name, row in h3_block.GEMM_ROWS.items():
        assert by[name]["status"] == "row" and by[name]["backend"] == h3_block.GEMM_ROW_OPS[row]
    backward = h3_block.run_block_backward(h3_block_weights_cuda, layout="tiny", ops=ops)
    for leaf in ("attn.to_out.0.weight", "ff.net.0.proj.weight", "ff.net.2.weight"):
        assert backward["leaves"][leaf]["repeat_bitwise_equal"], leaf


def test_ops_without_gemm_op_report_the_interim_gemm():
    """``cand_gemm`` runs the interim GEMM for ops that have no ``gemm_op``: so must the report."""

    class _NoAccessor:
        attention_h3 = object()

    bound = h3_block.bindings(_NoAccessor())
    for name in h3_block.GEMM_ROWS:
        assert (bound[name].status, bound[name].backend) == ("interim", h3_block.INTERIM_GEMM)
    assert bound["attention"].status == "row"


@pytest.mark.parametrize("module", ["h3_block", "h3_refiner"])
def test_isolated_replay_runs_without_tf32_and_without_grad(monkeypatch, module):
    """The GEMM rows refuse TF32, and a trainable weight outside a canonical backward session:
    isolated replay turns both off and restores the caller's settings."""

    import importlib
    from types import SimpleNamespace

    mod = importlib.import_module(f"rl_engine.validation.models.{module}")
    seen = []

    def candidate(ops, ctx, x):
        seen.append((torch.backends.cuda.matmul.allow_tf32, torch.is_grad_enabled()))
        return x

    monkeypatch.setattr(
        mod, "NODES", [SimpleNamespace(name="n", inputs=("x",), candidate=candidate)]
    )
    before = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        assert mod.run_isolated({}, None, {"x": 1}) == {"n": 1}
        assert seen == [(False, False)]
        assert torch.is_grad_enabled()  # the caller's
        assert torch.backends.cuda.matmul.allow_tf32 is True  # the caller's setting is back
    finally:
        torch.backends.cuda.matmul.allow_tf32 = before
