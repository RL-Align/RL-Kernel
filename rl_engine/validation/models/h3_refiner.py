# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""MiniMax-H3 token-refiner blocks, node by node (RFC #420 ``token_refiner_block``).

The text stream is refined by two plain pre-norm blocks and a final RMSNorm before it is
scattered into the packed sequence. A refiner block is the transformer block minus AdaLN,
minus RoPE and with plain (ungated) residual adds::

    h = h + attn(norm1(h))          attn: q/k/v -> per-head RMSNorm -> full attention -> o_proj
    h = h + ff(norm2(h))            ff:   gate_up -> SwiGLU -> down

The block is a fixed graph of 14 nodes (``NODES``, the vocabulary pinned by
``fingerprint.REFINER_NODES``) over the same operators the transformer block binds, so
every qualified row is reused and the only arithmetic owned by this row is the two plain
BF16 residual adds. Each node runs three ways, as in ``h3_block``: provider (diffusers op
sequence), candidate (RL-Kernel bindings, chained and isolated) and an FP32 golden, and
names the RFC row that owns it with its binding status (``row`` / ``interim`` /
``reference``; see ``h3_block``). The report carries repeat and batch-row bitwise checks,
agreement with the provider, error against the golden, and the first drifting node.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from rl_engine.models.minimax_h3 import fingerprint as fp
from rl_engine.reference.minimax_h3 import block as gold
from rl_engine.validation.models import h3_block as hb
from rl_engine.validation.models import h3_provider as prov

HIDDEN = hb.HIDDEN
PREFIX = "token_refiner.refiner_blocks"
FINAL_NORM = "token_refiner.final_norm.weight"
Binding = hb.Binding


@dataclass(frozen=True)
class Node:
    name: str
    inputs: tuple[str, ...]
    binding: Binding
    candidate: Callable[..., Any]  # (ops, ctx, *inputs)
    provider: Callable[..., Any]  # (ctx, *inputs)
    golden: Callable[..., Any]  # (ctx, *inputs)
    provider_bitwise_isolated: bool


def _w(ctx: dict[str, Any], name: str) -> torch.Tensor:
    return ctx["weights"][f"{ctx['prefix']}.{name}"]


def _norm_node(name: str, source: str, weight: str) -> Node:
    return Node(
        name=name,
        inputs=(source,),
        binding=Binding("h3_rmsnorm", "row", "H3RMSNormCudaOp.forward"),
        candidate=lambda ops, ctx, x: ops.norm(x, _w(ctx, weight)),
        provider=lambda ctx, x: prov.provider_head_rmsnorm(x, _w(ctx, weight)),
        golden=lambda ctx, x: gold.golden_rmsnorm(x, _w(ctx, weight)),
        provider_bitwise_isolated=True,
    )


def _qkv_node(name: str, weight: str) -> Node:
    def candidate(ops, ctx, x):
        flat = x.reshape(-1, x.shape[-1])
        return ops.qkv.project(flat, _w(ctx, weight)).view(*x.shape[:-1], -1)

    return Node(
        name=name,
        inputs=("norm1",),
        binding=Binding("h3_qkv_gemm", "row", "H3QKVGemmCudaOp.project"),
        candidate=candidate,
        provider=lambda ctx, x: prov.provider_linear(x, _w(ctx, weight)),
        golden=lambda ctx, x: gold.golden_linear(x, _w(ctx, weight)),
        provider_bitwise_isolated=False,
    )


def _head_norm_node(name: str, source: str, weight: str) -> Node:
    return Node(
        name=name,
        inputs=(source,),
        binding=Binding("h3_qk_rmsnorm_d128", "interim", "H3RMSNormCudaOp.forward"),
        candidate=lambda ops, ctx, x: ops.norm(hb._heads(x), _w(ctx, weight)),
        provider=lambda ctx, x: prov.provider_head_rmsnorm(hb._heads(x), _w(ctx, weight)),
        golden=lambda ctx, x: gold.golden_rmsnorm(hb._heads(x), _w(ctx, weight)),
        provider_bitwise_isolated=True,
    )


def _linear_node(name: str, source: str, weight: str, row: str) -> Node:
    return Node(
        name=name,
        inputs=(source,),
        binding=Binding(
            row, "interim", "triton _det_gemm_fp32_kernel (FP32 K loop, one BF16 rounding)"
        ),
        candidate=lambda ops, ctx, x: hb._cand_linear(ops, x, _w(ctx, weight)),
        provider=lambda ctx, x: prov.provider_linear(x, _w(ctx, weight)),
        golden=lambda ctx, x: gold.golden_linear(x, _w(ctx, weight)),
        provider_bitwise_isolated=False,
    )


def _residual_node(name: str, residual: str, sublayer: str) -> Node:
    """Plain residual add: one BF16 add per element, owned by this row."""

    return Node(
        name=name,
        inputs=(residual, sublayer),
        binding=Binding("token_refiner_block", "row", "torch.add (BF16 elementwise)"),
        candidate=lambda ops, ctx, r, y: r + y,
        provider=lambda ctx, r, y: r + y,
        golden=lambda ctx, r, y: r.float() + y.float(),
        provider_bitwise_isolated=True,
    )


NODES: list[Node] = [
    _norm_node("norm1", "hidden", "norm1.weight"),
    _qkv_node("q_proj", "attn.to_q.weight"),
    _qkv_node("k_proj", "attn.to_k.weight"),
    _qkv_node("v_proj", "attn.to_v.weight"),
    _head_norm_node("q_norm", "q_proj", "attn.norm_q.weight"),
    _head_norm_node("k_norm", "k_proj", "attn.norm_k.weight"),
    Node(
        name="attention",
        inputs=("q_norm", "k_norm", "v_proj"),
        binding=Binding("h3_full_attention", "interim", "DeterministicAttentionOp(causal=False)"),
        candidate=lambda ops, ctx, q, k, v: hb._cand_attention(ops, q, k, v),
        provider=lambda ctx, q, k, v: prov.provider_attention(q, k, hb._heads(v)),
        golden=lambda ctx, q, k, v: gold.golden_attention(q, k, hb._heads(v)),
        provider_bitwise_isolated=False,
    ),
    _linear_node("o_proj", "attention", "attn.to_out.0.weight", "h3_attention_o_gemm"),
    _residual_node("residual_attn", "hidden", "o_proj"),
    _norm_node("norm2", "residual_attn", "norm2.weight"),
    _linear_node("ffn_gate_up", "norm2", "ff.net.0.proj.weight", "h3_ffn_gate_up_gemm"),
    Node(
        name="swiglu",
        inputs=("ffn_gate_up",),
        binding=Binding("h3_swiglu", "interim", "SwiGLUCudaOp"),
        candidate=lambda ops, ctx, x: hb._cand_swiglu(ops, x),
        provider=lambda ctx, x: prov.provider_swiglu(x),
        golden=lambda ctx, x: gold.golden_swiglu(x),
        provider_bitwise_isolated=False,
    ),
    _linear_node("ffn_down", "swiglu", "ff.net.2.weight", "h3_ffn_down_gemm"),
    _residual_node("residual_mlp", "residual_attn", "ffn_down"),
]
NODE_NAMES = tuple(node.name for node in NODES)
OUTPUT = NODES[-1].name
PARAM_LEAVES = (
    "norm1.weight",
    "attn.to_q.weight",
    "attn.to_k.weight",
    "attn.to_v.weight",
    "attn.norm_q.weight",
    "attn.norm_k.weight",
    "attn.to_out.0.weight",
    "norm2.weight",
    "ff.net.0.proj.weight",
    "ff.net.2.weight",
)


def make_refiner_context(
    weights: dict[str, torch.Tensor], *, block: int, tokens: int, seed: int = 0, batch: int = 1
) -> dict[str, Any]:
    """Weights of refiner ``block`` and a seeded BF16 text-stream input ``(batch, tokens, 5376)``.

    Batch row 0 is the same tensor at every ``batch``; further rows get their own seeds.
    """

    rows = []
    for b in range(batch):
        gen = torch.Generator(device="cpu").manual_seed(seed if b == 0 else seed * 1009 + b)
        rows.append(torch.randn(1, tokens, HIDDEN, generator=gen))
    hidden = torch.cat(rows).to(torch.bfloat16).cuda()
    return {
        "weights": weights,
        "prefix": f"{PREFIX}.{block}",
        "block": block,
        "hidden": hidden,
        "tokens": tokens,
    }


def run_graph(mode: str, ctx: dict[str, Any], ops: hb.CandidateOps | None = None) -> dict[str, Any]:
    """Every node's output for one refiner block ("candidate", "provider" or "golden")."""

    values: dict[str, Any] = {"hidden": ctx["hidden"]}
    observer = ctx.get("node_observer")
    with hb.strict_fp32_matmul():
        for node in NODES:
            args = [values[name] for name in node.inputs]
            if mode == "candidate":
                values[node.name] = node.candidate(ops, ctx, *args)
            elif mode == "provider":
                values[node.name] = node.provider(ctx, *args)
            else:
                values[node.name] = node.golden(ctx, *args)
            if observer is not None:
                observer(node.name)
    return values


def run_isolated(
    ctx: dict[str, Any], ops: hb.CandidateOps, provider: dict[str, Any]
) -> dict[str, Any]:
    """Each candidate node fed the provider's inputs to that node."""

    return {
        node.name: node.candidate(ops, ctx, *[provider[name] for name in node.inputs])
        for node in NODES
    }


def run_refiner(
    mode: str,
    weights: dict[str, torch.Tensor],
    hidden: torch.Tensor,
    ops: hb.CandidateOps | None = None,
    *,
    num_blocks: int = 2,
    observer_factory: Callable[[int], Callable[[str], None]] | None = None,
) -> torch.Tensor:
    """The whole text refiner: ``num_blocks`` blocks then the final RMSNorm."""

    h = hidden
    for block in range(num_blocks):
        ctx = {"weights": weights, "prefix": f"{PREFIX}.{block}", "hidden": h}
        if observer_factory is not None:
            ctx["node_observer"] = observer_factory(block)
        h = run_graph(mode, ctx, ops)[OUTPUT]
    weight = weights[FINAL_NORM]
    if mode == "candidate":
        assert ops is not None
        return ops.norm(h, weight)
    if mode == "provider":
        return prov.provider_head_rmsnorm(h, weight)
    return gold.golden_rmsnorm(h, weight)


def run_refiner_case(
    weights: dict[str, torch.Tensor],
    *,
    block: int = 0,
    tokens: int = 257,
    seed: int = 0,
    ops: hb.CandidateOps | None = None,
) -> dict[str, Any]:
    """Forward replay of one refiner block: per-node agreement, invariance, accuracy, drift."""

    ops = ops or hb.CandidateOps()
    ctx = make_refiner_context(weights, block=block, tokens=tokens, seed=seed, batch=1)
    ctx2 = make_refiner_context(weights, block=block, tokens=tokens, seed=seed, batch=2)
    with torch.no_grad():
        provider = run_graph("provider", ctx)
        golden = run_graph("golden", ctx)
        chained = run_graph("candidate", ctx, ops)
        repeat = run_graph("candidate", ctx, ops)
        batched = run_graph("candidate", ctx2, ops)
        isolated = run_isolated(ctx, ops, provider)
    report: dict[str, Any] = {
        "block": block,
        "tokens": tokens,
        "seed": seed,
        "backends": ops.backends(),
        "nodes": [],
    }
    firsts: dict[str, str | None] = dict.fromkeys(
        ("first_drift", "first_isolated_drift", "first_repeat_drift", "first_batch_drift")
    )
    for node in NODES:
        name = node.name
        row0 = hb._batch_row(batched[name], 0, 2)
        entry: dict[str, Any] = {
            "node": name,
            "rfc_row": node.binding.rfc_row,
            "status": node.binding.status,
            "backend": node.binding.backend,
            "provider_bitwise_isolated_promised": node.provider_bitwise_isolated,
            "repeat_bitwise_equal": hb.bitwise(repeat[name], chained[name]),
            "batch_row_bitwise_equal": hb.bitwise(row0, chained[name]),
            "chained_vs_provider": {
                "bitwise_equal": hb.bitwise(chained[name], provider[name]),
                "max_abs": hb.max_abs(chained[name], provider[name]),
            },
            "isolated_vs_provider": {
                "bitwise_equal": hb.bitwise(isolated[name], provider[name]),
                "max_abs": hb.max_abs(isolated[name], provider[name]),
            },
            "candidate_rel_err_vs_golden": hb.rel_err(chained[name], golden[name]),
            "provider_rel_err_vs_golden": hb.rel_err(provider[name], golden[name]),
            "candidate_sha256": hb.digest(chained[name]),
        }
        checks = {
            "first_drift": entry["chained_vs_provider"]["bitwise_equal"],
            "first_isolated_drift": entry["isolated_vs_provider"]["bitwise_equal"],
            "first_repeat_drift": entry["repeat_bitwise_equal"],
            "first_batch_drift": entry["batch_row_bitwise_equal"],
        }
        for key, ok in checks.items():
            if firsts[key] is None and not ok:
                firsts[key] = name
        report["nodes"].append(entry)
    report.update(firsts)
    report["output"] = {
        "candidate_rel_err_vs_golden": hb.rel_err(chained[OUTPUT], golden[OUTPUT]),
        "provider_rel_err_vs_golden": hb.rel_err(provider[OUTPUT], golden[OUTPUT]),
        "candidate_vs_provider_max_abs": hb.max_abs(chained[OUTPUT], provider[OUTPUT]),
    }
    return report


def graph_matches_fingerprint() -> bool:
    """The node vocabulary and owners equal ``fingerprint.REFINER_NODES``."""

    return [(n.name, n.binding.rfc_row) for n in NODES] == [
        (name, row) for name, row in fp.REFINER_NODES
    ]
