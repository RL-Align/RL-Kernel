# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""One real MiniMax-H3 transformer block, node by node (RFC #420 ``ws1_one_h3_block``).

The block is a fixed graph of 17 nodes (``NODES``), from the AdaLN projection
through attention and the FFN to the second gated residual, on block 0's
pinned weights and a packed FL2VA layout. Every node runs three ways:

* provider: the diffusers op sequence (``h3_provider``), fed provider inputs;
* candidate: the RL-Kernel binding of the node, fed candidate inputs
  (chained) and, separately, provider inputs (isolated);
* golden: the FP32 reference (``rl_engine.reference.minimax_h3.block``).

Each node also names the RFC row that owns its arithmetic and how it is bound
today (``Binding.status``):

* ``row`` -- that row's operator, in this PR stack;
* ``interim`` -- an existing deterministic RL-Kernel operator standing in
  until the owning row lands (its arithmetic, speed and rounding are that
  operator's, not the row's);
* ``reference`` -- no kernel exists yet; the node runs the provider replay
  itself, so it is trivially bitwise and contributes no RL-Kernel evidence.

The report gives, per node: repeat and batch-row bitwise checks, chained and
isolated agreement with the provider, error against the golden for both the
candidate and the provider, and the backend that ran. ``first_*_drift`` name
the first node (graph order) where a comparison stops being bitwise; the
backward report does the same for the gradients reaching each node's output,
in reverse graph order.
"""

from __future__ import annotations

import hashlib
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable

import torch

from rl_engine.reference.minimax_h3 import block as gold
from rl_engine.runtime.registry import KernelRegistry
from rl_engine.validation.models import h3_provider as prov
from rl_engine.validation.models.h3_cases import H3_BLOCK_LAYOUTS, h3_fl2va_layout, h3_timesteps

HIDDEN = 5376
HEADS = 56
HEAD_DIM = 128
BLOCK = "transformer_blocks.0"
NUM_TIMESTEPS = 2


@dataclass(frozen=True)
class Binding:
    rfc_row: str
    status: str  # "row" | "interim" | "reference"
    backend: str


@dataclass(frozen=True)
class Node:
    name: str
    inputs: tuple[str, ...]
    binding: Binding
    # (ops, ctx, *inputs) -> output
    candidate: Callable[..., Any]
    # (ctx, *inputs) -> output
    provider: Callable[..., Any]
    golden: Callable[..., Any]
    # Promise checked by the tests: the isolated candidate equals the provider bitwise.
    provider_bitwise_isolated: bool


class CandidateOps:
    """The operators bound to the block's nodes, resolved once per run.

    The three rows of this stack are resolved through the registry, so the
    report records what dispatch actually returned; the interim operators are
    named directly because their registry entries (``det_gemm``,
    ``attention``, ``swiglu``) prefer backends that are not available or not
    deterministic on every GPU this replay runs on.
    """

    def __init__(self, registry: KernelRegistry | None = None) -> None:
        from rl_engine.backends.cuda.activation.swiglu import SwiGLUCudaOp
        from rl_engine.backends.cuda.attention.deterministic_attn import DeterministicAttentionOp
        from rl_engine.backends.shared.triton.gemm.det_gemm import TritonDetGemmOp

        registry = registry or KernelRegistry()
        self.projection = registry.get_op("adaln_projection_3mod", device="cuda")
        self.norm = registry.get_op("h3_rmsnorm", device="cuda")
        self.gate = registry.get_op("adaln_gate_residual", device="cuda")
        self.gemm = TritonDetGemmOp()
        self.attention = DeterministicAttentionOp()
        self.swiglu = SwiGLUCudaOp()

    def backends(self) -> dict[str, str]:
        return {
            name: f"{type(op).__module__}.{type(op).__name__}" for name, op in vars(self).items()
        }


def _w(ctx: dict[str, Any], name: str) -> torch.Tensor:
    return ctx["weights"][f"{BLOCK}.{name}"]


def _heads(x: torch.Tensor) -> torch.Tensor:
    return x.unflatten(-1, (HEADS, HEAD_DIM))


class _InterimLinear(torch.autograd.Function):
    """``y = x @ W^T`` on the Triton FP32-accumulate kernel, rounded to BF16 once.

    Interim binding for the block's GEMM rows. ``TritonDetGemmOp.__call__``
    reduces K as a tree with BF16 nodes (about 3x cuBLAS's error at these
    shapes); ``_triton_gemm_fp32`` runs one fixed-order FP32 K loop (no
    split-K), so each output row depends only on its own input row. The
    backward uses the same kernel for ``dX`` (reduction over the output
    features, row-local) and ``dW`` (reduction over tokens).
    """

    @staticmethod
    def forward(ctx, x2, weight):
        from rl_engine.backends.shared.triton.gemm.det_gemm import _triton_gemm_fp32

        ctx.save_for_backward(x2, weight)
        return _triton_gemm_fp32(x2, weight.t()).to(x2.dtype)

    @staticmethod
    def backward(ctx, grad):
        from rl_engine.backends.shared.triton.gemm.det_gemm import _triton_gemm_fp32

        x2, weight = ctx.saved_tensors
        grad = grad.contiguous()
        dx = _triton_gemm_fp32(grad, weight).to(x2.dtype)
        dw = _triton_gemm_fp32(grad.t(), x2).to(weight.dtype)
        return dx, dw


def _cand_linear(ops: CandidateOps, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    flat = x.reshape(-1, x.shape[-1])
    return _InterimLinear.apply(flat, weight).view(*x.shape[:-1], weight.shape[0])


def _cand_attention(ops: CandidateOps, q, k, v) -> torch.Tensor:
    q, k, v = (t.transpose(1, 2).contiguous() for t in (q, k, _heads(v)))
    out = ops.attention(q, k, v, causal=False)
    return out.transpose(1, 2).flatten(2, 3)


def _cand_swiglu(ops: CandidateOps, projected: torch.Tensor) -> torch.Tensor:
    hidden, gate = projected.chunk(2, dim=-1)
    return ops.swiglu(gate.contiguous(), hidden.contiguous())


def _linear_node(name: str, source: str, weight: str, row: str, status: str) -> Node:
    return Node(
        name=name,
        inputs=(source,),
        binding=Binding(
            row, status, "triton _det_gemm_fp32_kernel (FP32 K loop, one BF16 rounding)"
        ),
        candidate=lambda ops, ctx, x: _cand_linear(ops, x, _w(ctx, weight)),
        provider=lambda ctx, x: prov.provider_linear(x, _w(ctx, weight)),
        golden=lambda ctx, x: gold.golden_linear(x, _w(ctx, weight)),
        provider_bitwise_isolated=False,  # tree reduction vs cuBLAS
    )


def _norm_modulate_node(name: str, source: str, weight: str, shift: int, scale: int) -> Node:
    return Node(
        name=name,
        inputs=(source, "adaln_projection"),
        binding=Binding("h3_rmsnorm", "row", "H3RMSNormCudaOp.forward_modulated"),
        candidate=lambda ops, ctx, x, t: ops.norm.forward_modulated(
            x, _w(ctx, weight), t[shift], t[scale], ctx["index"]
        ),
        provider=lambda ctx, x, t: prov.provider_norm_modulate(
            x, _w(ctx, weight), t[shift], t[scale], ctx["index"]
        ),
        golden=lambda ctx, x, t: gold.golden_norm_modulate(
            x, _w(ctx, weight), t[shift], t[scale], ctx["index"]
        ),
        provider_bitwise_isolated=True,
    )


def _head_norm_node(name: str, source: str, weight: str) -> Node:
    return Node(
        name=name,
        inputs=(source,),
        # The row is someone else's; until it lands, this stack's plain H3 RMSNorm
        # (bitwise equal to nn.RMSNorm at any width) runs on the 128-wide heads.
        binding=Binding("h3_qk_rmsnorm_d128", "interim", "H3RMSNormCudaOp.forward"),
        candidate=lambda ops, ctx, x: ops.norm(_heads(x), _w(ctx, weight)),
        provider=lambda ctx, x: prov.provider_head_rmsnorm(_heads(x), _w(ctx, weight)),
        golden=lambda ctx, x: gold.golden_rmsnorm(_heads(x), _w(ctx, weight)),
        provider_bitwise_isolated=True,
    )


def _rope_node(name: str, source: str) -> Node:
    return Node(
        name=name,
        inputs=(source,),
        binding=Binding("h3_mm_rope_3axis_partial", "reference", "provider_apply_rotary"),
        candidate=lambda ops, ctx, x: prov.provider_apply_rotary(x, *ctx["rope"]),
        provider=lambda ctx, x: prov.provider_apply_rotary(x, *ctx["rope"]),
        golden=lambda ctx, x: gold.golden_apply_rotary(x, *ctx["rope_fp32"]),
        provider_bitwise_isolated=True,
    )


def _gate_node(name: str, residual: str, sublayer: str, gate: int) -> Node:
    return Node(
        name=name,
        inputs=(residual, sublayer, "adaln_projection"),
        binding=Binding("adaln_gate_residual", "row", "H3GateResidualCudaOp"),
        candidate=lambda ops, ctx, r, y, t: ops.gate(r, y, t[gate], ctx["index"]),
        provider=lambda ctx, r, y, t: prov.provider_gate_residual(r, t[gate], ctx["index"], y),
        golden=lambda ctx, r, y, t: gold.golden_gate_residual(r, t[gate], ctx["index"], y),
        provider_bitwise_isolated=True,
    )


def _adaln_params(ctx):
    return _w(ctx, "adaln_proj.linear.weight"), _w(ctx, "adaln_proj.linear.bias")


NODES: list[Node] = [
    Node(
        name="adaln_projection",
        inputs=("temb",),
        binding=Binding("adaln_projection_3mod", "row", "H3AdaLNProjectionCudaOp"),
        candidate=lambda ops, ctx, temb: ops.projection(temb, *_adaln_params(ctx)),
        provider=lambda ctx, temb: prov.provider_adaln_modulation(temb, *_adaln_params(ctx)),
        golden=lambda ctx, temb: gold.golden_adaln_tables(temb, *_adaln_params(ctx)),
        provider_bitwise_isolated=False,  # tensor-core tree vs cuBLAS
    ),
    _norm_modulate_node("norm1", "hidden", "norm1.weight", shift=0, scale=1),
    _linear_node("q_proj", "norm1", "attn.to_q.weight", "h3_qkv_gemm", "interim"),
    _linear_node("k_proj", "norm1", "attn.to_k.weight", "h3_qkv_gemm", "interim"),
    _linear_node("v_proj", "norm1", "attn.to_v.weight", "h3_qkv_gemm", "interim"),
    _head_norm_node("q_norm", "q_proj", "attn.norm_q.weight"),
    _head_norm_node("k_norm", "k_proj", "attn.norm_k.weight"),
    _rope_node("rope_q", "q_norm"),
    _rope_node("rope_k", "k_norm"),
    Node(
        name="attention",
        inputs=("rope_q", "rope_k", "v_proj"),
        binding=Binding("h3_full_attention", "interim", "DeterministicAttentionOp(causal=False)"),
        candidate=lambda ops, ctx, q, k, v: _cand_attention(ops, q, k, v),
        provider=lambda ctx, q, k, v: prov.provider_attention(q, k, _heads(v)),
        golden=lambda ctx, q, k, v: gold.golden_attention(q, k, _heads(v)),
        provider_bitwise_isolated=False,
    ),
    _linear_node("o_proj", "attention", "attn.to_out.0.weight", "h3_attention_o_gemm", "interim"),
    _gate_node("residual_attn", "hidden", "o_proj", gate=2),
    _norm_modulate_node("norm2", "residual_attn", "norm2.weight", shift=3, scale=4),
    _linear_node("ffn_gate_up", "norm2", "ff.net.0.proj.weight", "h3_ffn_gate_up_gemm", "interim"),
    Node(
        name="swiglu",
        inputs=("ffn_gate_up",),
        binding=Binding("h3_swiglu", "interim", "SwiGLUCudaOp"),
        candidate=lambda ops, ctx, x: _cand_swiglu(ops, x),
        provider=lambda ctx, x: prov.provider_swiglu(x),
        golden=lambda ctx, x: gold.golden_swiglu(x),
        provider_bitwise_isolated=False,  # one FP32 rounding vs the eager two
    ),
    _linear_node("ffn_down", "swiglu", "ff.net.2.weight", "h3_ffn_down_gemm", "interim"),
    _gate_node("residual_mlp", "residual_attn", "ffn_down", gate=5),
]
NODE_NAMES = tuple(node.name for node in NODES)
OUTPUT = NODES[-1].name

# Gradient leaves of the backward replay: the block input, temb and every block parameter.
PARAM_LEAVES = (
    "adaln_proj.linear.weight",
    "adaln_proj.linear.bias",
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


# --- Inputs ---------------------------------------------------------------------------------


def make_block_context(
    weights: dict[str, torch.Tensor], *, layout: str, seed: int = 0, batch: int = 1
) -> dict[str, Any]:
    """Block-0 weights, a packed FL2VA layout, the timestep embedding and a seeded input.

    ``temb`` comes from the provider's conditioning path (``h3_chain`` covers
    those rows); the block input is a seeded BF16 stand-in for the packed
    projections, which are other rows. Batch rows ``b > 0`` get their own
    seeded inputs; row 0 is the same tensor at every ``batch``.
    """

    spec = H3_BLOCK_LAYOUTS[layout]
    packed = h3_fl2va_layout(
        num_text_tokens=spec["num_text_tokens"],
        num_latent_frames=spec["num_latent_frames"],
        latent_height=spec["latent_height"],
        latent_width=spec["latent_width"],
        num_audio_latents=spec["num_audio_latents"],
    )
    seq_len = packed["position_ids"].shape[0]
    timestep = h3_timesteps(NUM_TIMESTEPS, seed=seed)
    w = weights
    with torch.no_grad():
        features = prov.provider_time_proj(timestep)
        temb = prov.provider_time_embedder(
            features,
            w["time_embedder.linear_1.weight"],
            w["time_embedder.linear_1.bias"],
            w["time_embedder.linear_2.weight"],
            w["time_embedder.linear_2.bias"],
        )
    rows = [
        torch.randn((1, seq_len, HIDDEN), generator=torch.Generator().manual_seed(seed + 2 + b))
        for b in range(batch)
    ]
    hidden = torch.cat(rows).to(torch.bfloat16).cuda()
    return {
        "layout": layout,
        "seq_len": seq_len,
        "batch": batch,
        "weights": weights,
        "temb": temb,
        "hidden": hidden,
        "index": packed["timestep_indices"] * 3 + packed["token_tags"],
        "position_ids": packed["position_ids"],
        "rope": prov.provider_rope(packed["position_ids"]),
        "rope_fp32": gold.golden_rope_tables(packed["position_ids"]),
    }


@contextmanager
def strict_fp32_matmul():
    """Golden matmuls in full FP32: no TF32, no reduced-precision reductions."""

    saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


# --- Graph execution ------------------------------------------------------------------------


def run_graph(mode: str, ctx: dict[str, Any], ops: CandidateOps | None = None) -> dict[str, Any]:
    """Every node's output for one mode ("candidate", "provider" or "golden")."""

    values: dict[str, Any] = {"hidden": ctx["hidden"], "temb": ctx["temb"]}
    # Diagnostic hybrids only: these candidate nodes run their provider replay instead.
    provider_nodes = ctx.get("provider_nodes", ())
    with strict_fp32_matmul():
        for node in NODES:
            args = [values[name] for name in node.inputs]
            if mode == "candidate" and node.name in provider_nodes:
                values[node.name] = node.provider(ctx, *args)
            elif mode == "candidate":
                values[node.name] = node.candidate(ops, ctx, *args)
            elif mode == "provider":
                values[node.name] = node.provider(ctx, *args)
            else:
                values[node.name] = node.golden(ctx, *args)
            observer = ctx.get("node_observer")
            if observer is not None:
                observer(node.name)
    return values


def run_isolated(
    ctx: dict[str, Any], ops: CandidateOps, provider: dict[str, Any]
) -> dict[str, Any]:
    """Each candidate node fed the provider's inputs to that node."""

    return {
        node.name: node.candidate(ops, ctx, *[provider[name] for name in node.inputs])
        for node in NODES
    }


def _flat(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    return [t for item in value for t in _flat(item)]


def _batch_row(value: Any, row: int, batch: int) -> Any:
    """Row ``row`` of a batched activation; AdaLN tables have no batch axis."""

    if isinstance(value, torch.Tensor):
        return value[row : row + 1] if value.dim() >= 3 and value.shape[0] == batch else value
    return tuple(_batch_row(v, row, batch) for v in value)


def bitwise(lhs: Any, rhs: Any) -> bool:
    return all(torch.equal(a, b) for a, b in zip(_flat(lhs), _flat(rhs), strict=True))


def max_abs(lhs: Any, rhs: Any) -> float:
    return max(
        float((a.float() - b.float()).abs().max()) if a.numel() else 0.0
        for a, b in zip(_flat(lhs), _flat(rhs), strict=True)
    )


def first_differing_row(lhs: Any, rhs: Any) -> int | None:
    """First packed row (sequence position) where two node outputs differ, or None."""

    for a, b in zip(_flat(lhs), _flat(rhs), strict=True):
        if a.dim() < 3:
            if not torch.equal(a, b):
                return -1  # a table row, not a sequence position
            continue
        diff = (a != b).reshape(a.shape[0], a.shape[1], -1).any(dim=-1).any(dim=0)
        rows = diff.nonzero()
        if rows.numel():
            return int(rows[0])
    return None


def rel_err(value: Any, golden: Any) -> float:
    """``max|value - golden| / max|golden|`` over a node's tensors."""

    num = den = 0.0
    for a, g in zip(_flat(value), _flat(golden), strict=True):
        g32 = g.float()
        num = max(num, float((a.float() - g32).abs().max()))
        den = max(den, float(g32.abs().max()))
    return num / max(den, 1e-30)


def ulp_err(value: torch.Tensor, golden: torch.Tensor) -> float:
    """``max|value - golden|`` in units of ``value.dtype``'s ulp at ``max|golden|``.

    ``rel_err`` of a stored gradient moves in steps of this ulp: its worst
    element is a single rounded number, so two equally valid reduction orders
    can differ by whole ulps there while every other element agrees.
    """

    g32 = golden.float()
    top = float(g32.abs().max())
    if top == 0.0:
        return 0.0
    ulp = torch.finfo(value.dtype).eps * 2.0 ** torch.floor(torch.log2(torch.tensor(top))).item()
    return float((value.float() - g32).abs().max()) / ulp


def digest(value: Any) -> str:
    """sha256 over a node's tensors' bytes, for comparing runs across processes or hosts."""

    h = hashlib.sha256()
    for t in _flat(value):
        h.update(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def _time_ms(fn: Callable[[], Any], repeats: int) -> float:
    fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return sorted(samples)[len(samples) // 2]


# --- Forward report -------------------------------------------------------------------------


def run_block_case(
    weights: dict[str, torch.Tensor],
    *,
    layout: str,
    seed: int = 0,
    ops: CandidateOps | None = None,
    timing_repeats: int = 0,
) -> dict[str, Any]:
    """Forward replay of block 0: per-node agreement, invariance, accuracy and first drift."""

    ops = ops or CandidateOps()
    ctx = make_block_context(weights, layout=layout, seed=seed, batch=1)
    ctx2 = make_block_context(weights, layout=layout, seed=seed, batch=2)
    with torch.no_grad():
        provider = run_graph("provider", ctx)
        golden = run_graph("golden", ctx)
        chained = run_graph("candidate", ctx, ops)
        repeat = run_graph("candidate", ctx, ops)
        batched = run_graph("candidate", ctx2, ops)
        isolated = run_isolated(ctx, ops, provider)

    report: dict[str, Any] = {
        "layout": layout,
        "seq_len": ctx["seq_len"],
        "seed": seed,
        "backends": ops.backends(),
        "nodes": [],
    }
    firsts: dict[str, str | None] = {"first_drift": None, "first_isolated_drift": None}
    firsts.update({"first_repeat_drift": None, "first_batch_drift": None})
    for node in NODES:
        name = node.name
        row0 = _batch_row(batched[name], 0, 2)
        entry: dict[str, Any] = {
            "node": name,
            "rfc_row": node.binding.rfc_row,
            "status": node.binding.status,
            "backend": node.binding.backend,
            "provider_bitwise_isolated_promised": node.provider_bitwise_isolated,
            "repeat_bitwise_equal": bitwise(repeat[name], chained[name]),
            "batch_row_bitwise_equal": bitwise(row0, chained[name]),
            "chained_vs_provider": {
                "bitwise_equal": bitwise(chained[name], provider[name]),
                "max_abs": max_abs(chained[name], provider[name]),
                "first_differing_row": first_differing_row(chained[name], provider[name]),
            },
            "isolated_vs_provider": {
                "bitwise_equal": bitwise(isolated[name], provider[name]),
                "max_abs": max_abs(isolated[name], provider[name]),
            },
            "candidate_rel_err_vs_golden": rel_err(chained[name], golden[name]),
            "provider_rel_err_vs_golden": rel_err(provider[name], golden[name]),
            "candidate_sha256": digest(chained[name]),
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
        "candidate_rel_err_vs_golden": rel_err(chained[OUTPUT], golden[OUTPUT]),
        "provider_rel_err_vs_golden": rel_err(provider[OUTPUT], golden[OUTPUT]),
        "candidate_vs_provider_max_abs": max_abs(chained[OUTPUT], provider[OUTPUT]),
    }
    report["permutation_probe"] = permutation_probe(ctx, ops, chained[OUTPUT], seed)
    if timing_repeats:
        with torch.no_grad():
            report["timing_ms"] = {
                "candidate_forward": _time_ms(
                    lambda: run_graph("candidate", ctx, ops), timing_repeats
                ),
                "provider_forward": _time_ms(lambda: run_graph("provider", ctx), timing_repeats),
            }
    return report


def permutation_probe(
    ctx: dict[str, Any], ops: CandidateOps, base_output: torch.Tensor, seed: int
) -> dict[str, Any]:
    """RFC #420 control H-C3, reported and not promised.

    Permute the packed rows together with every per-row tensor (positions,
    AdaLN indices); the block output should be the same permutation of the
    original. Row-local nodes keep each row's bytes, but attention sums over
    keys in packed order, so a different key order may round differently.
    """

    perm = torch.randperm(ctx["seq_len"], generator=torch.Generator().manual_seed(seed + 7)).cuda()
    permuted = {
        **ctx,
        "hidden": ctx["hidden"][:, perm],
        "index": ctx["index"][perm],
        "position_ids": ctx["position_ids"][perm],
        "rope": tuple(t[perm] for t in ctx["rope"]),
        "rope_fp32": tuple(t[perm] for t in ctx["rope_fp32"]),
    }
    with torch.no_grad():
        out = run_graph("candidate", permuted, ops)[OUTPUT]
    expected = base_output[:, perm]
    return {
        "bitwise_equal_after_permutation": bool(torch.equal(out, expected)),
        "max_abs": max_abs(out, expected),
        "differing_rows": int((out != expected).reshape(out.shape[1], -1).any(dim=-1).sum()),
    }


# --- Backward report ------------------------------------------------------------------------


def _backward(
    mode: str, ctx: dict[str, Any], ops: CandidateOps | None, upstream: torch.Tensor
) -> tuple[dict[str, torch.Tensor], dict[str, list[torch.Tensor]]]:
    """Gradients of every leaf, and the gradient reaching each node's output, for one mode.

    The golden runs on FP32 copies of the leaves, so its gradients are not
    rounded to the BF16 parameter dtype.
    """

    def leaf(t: torch.Tensor) -> torch.Tensor:
        t = t.detach().float() if mode == "golden" else t.detach()
        return t.clone().requires_grad_(True)

    leaves = {name: leaf(ctx["weights"][f"{BLOCK}.{name}"]) for name in PARAM_LEAVES}
    leaves["hidden"] = leaf(ctx["hidden"])
    leaves["temb"] = leaf(ctx["temb"])
    weights = {**ctx["weights"], **{f"{BLOCK}.{k}": leaves[k] for k in PARAM_LEAVES}}
    run_ctx = {**ctx, "weights": weights, "hidden": leaves["hidden"], "temb": leaves["temb"]}
    # ``run_graph`` makes the forward strict FP32, but the backward runs after it returns: without
    # this the golden gradients use TF32 matmuls whenever the caller enabled TF32.
    with strict_fp32_matmul():
        values = run_graph(mode, run_ctx, ops)
        for node in NODES:
            for t in _flat(values[node.name]):
                if t.requires_grad:
                    t.retain_grad()
        out = values[OUTPUT]
        torch.autograd.backward(out, upstream.to(out.dtype))
    node_grads = {
        node.name: [
            t.grad if t.grad is not None else torch.zeros_like(t) for t in _flat(values[node.name])
        ]
        for node in NODES
    }
    return {name: t.grad for name, t in leaves.items()}, node_grads


def run_block_backward(
    weights: dict[str, torch.Tensor],
    *,
    layout: str,
    seed: int = 0,
    ops: CandidateOps | None = None,
    timing_repeats: int = 0,
) -> dict[str, Any]:
    """Backward replay of block 0.

    Per leaf (block input, ``temb`` and the 12 block parameters): repeat
    bitwise, and the candidate's and provider's error against the FP32
    golden. Per node, in reverse graph order: whether the gradient reaching
    the node's output repeats bitwise, and whether batch row 0 of a two-row
    batch gets the same bytes as the batch of one (rows are separate packed
    documents, so every activation gradient is row-local). Parameter
    gradients sum over the batch and are not part of the batch check.
    """

    ops = ops or CandidateOps()
    ctx = make_block_context(weights, layout=layout, seed=seed, batch=1)
    ctx2 = make_block_context(weights, layout=layout, seed=seed, batch=2)
    generator = torch.Generator().manual_seed(seed + 11)
    up_row0 = torch.randn((1, ctx["seq_len"], HIDDEN), generator=generator)
    up_row1 = torch.randn((1, ctx["seq_len"], HIDDEN), generator=generator)
    upstream = up_row0.bfloat16().cuda()
    upstream2 = torch.cat([up_row0, up_row1]).bfloat16().cuda()

    golden, _ = _backward("golden", ctx, None, upstream)
    provider, _ = _backward("provider", ctx, None, upstream)
    first, first_nodes = _backward("candidate", ctx, ops, upstream)
    second, second_nodes = _backward("candidate", ctx, ops, upstream)
    batched, batched_nodes = _backward("candidate", ctx2, ops, upstream2)

    report: dict[str, Any] = {
        "layout": layout,
        "seq_len": ctx["seq_len"],
        "seed": seed,
        "leaves": {},
        "nodes": [],
    }
    for name in (*PARAM_LEAVES, "hidden", "temb"):
        report["leaves"][name] = {
            "repeat_bitwise_equal": bool(torch.equal(first[name], second[name])),
            "candidate_rel_err_vs_golden": rel_err(first[name], golden[name]),
            "provider_rel_err_vs_golden": rel_err(provider[name], golden[name]),
            "storage_dtype": str(first[name].dtype).removeprefix("torch."),
            "candidate_ulps_vs_golden": ulp_err(first[name], golden[name]),
            "provider_ulps_vs_golden": ulp_err(provider[name], golden[name]),
        }
    report["leaves"]["hidden"]["batch_row_bitwise_equal"] = bool(
        torch.equal(batched["hidden"][:1], first["hidden"])
    )
    first_repeat = first_batch = None
    for node in reversed(NODES):
        name = node.name
        row0 = [_batch_row(g, 0, 2) for g in batched_nodes[name]]
        entry: dict[str, Any] = {
            "node": name,
            "grad_repeat_bitwise_equal": bitwise(first_nodes[name], second_nodes[name]),
            "grad_batch_row_bitwise_equal": (
                bitwise(row0, first_nodes[name]) if node.name != "adaln_projection" else None
            ),
        }
        if first_repeat is None and not entry["grad_repeat_bitwise_equal"]:
            first_repeat = name
        if first_batch is None and entry["grad_batch_row_bitwise_equal"] is False:
            first_batch = name
        report["nodes"].append(entry)
    report["first_grad_repeat_drift"] = first_repeat
    report["first_grad_batch_drift"] = first_batch
    if timing_repeats:
        report["timing_ms"] = {
            "candidate_forward_backward": _time_ms(
                lambda: _backward("candidate", ctx, ops, upstream), timing_repeats
            ),
            "provider_forward_backward": _time_ms(
                lambda: _backward("provider", ctx, None, upstream), timing_repeats
            ),
        }
    return report
