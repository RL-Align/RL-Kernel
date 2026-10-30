# SPDX-License-Identifier: Apache-2.0
"""Stable synthetic case IDs, regenerable inputs, and independent M2 recordings."""

from __future__ import annotations

import struct
from dataclasses import asdict

import torch

from rl_engine.kernels.ops.pytorch.norm.rms_norm import NativeRMSNormOp

from . import oracle
from .contract import (
    MODES,
    OPERATORS,
    REFERENCE_PROFILE,
    Identity,
    Status,
    digest,
    integer,
    require,
)
from .planner import PlannerMock, validate_mode

# Do not rename IDs or edit recipes in place: version/checksum changes are ABI deltas.
SCENARIOS = {
    "LAYER": "c0 c4 c128 interleaved invalid_layer",
    "ROPE": "position0 position1 c4_completed c128_completed main_range index_range inverse neox",
    "C4": "first second cross_page cross_chunk cross_rank ape_order",
    "C128": "incomplete127 complete128 multiple_groups cp_boundary",
    "RECENT": "length1 length127 length128 length129 page_wrap chunked c0_only",
    "STATE4": "training prefill eager_decode graph_decode eager_to_graph strict_replay",
    "INDEX-ID": "weight ape norm state page",
    "FP4": "zero subnormal max_e2m1 scale_clip nibbles block_boundary",
    "ICV": "n0 n1 n511 n512 n513 tile fp4_expand one_ulp",
    "TOPK": "tie513 near_tie future_invalid fewer512 all_invalid",
    "ATTN": "c0 c4 c128 sink_dominates empty partial two_softmax",
    "OPROJ": "group8 concat_order inplace backward",
    "LONG": "c4_c128_pages_chunks batch_padding",
    "CP": "c4_left c128_owner recent_cross_rank global_index duplicate_owner",
    "TP": "head_shard global_scale ordered_score ordered_dkv global_topk oproj_shard",
    "BACKEND": "cuda ascend rocm unsupported_dtype unsupported_format",
    "PERF": "reference_fused debug short long tile_warp_stage",
}


def catalog() -> list[dict]:
    return [
        {
            "case_id": f"P2-F-{group}-{name}.v1",
            "group": f"P2-F-{group}",
            "scenario": name,
            "recipe_version": "p2-counter-fixture.v1",
            "fixture_checksum": digest(
                {"group": group, "scenario": name, "recipe": "p2-counter-fixture.v1"}
            ),
        }
        for group, names in SCENARIOS.items()
        for name in names.split()
    ]


def values(
    shape: tuple[int, ...], salt: int = 0, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Integer counter, exactly representable /16; independent of RNG/library seed."""
    n = 1
    for d in shape:
        n *= d
    return (
        (((torch.arange(n, dtype=torch.int64) + salt) % 31).float() / 16).reshape(shape).to(dtype)
    )


def selector_recipe(out_features: int, in_features: int, salt: int = 0) -> dict:
    integer(out_features, "out_features", 1)
    integer(in_features, "in_features", 1)
    integer(salt, "salt")
    recipe = {
        "schema": "p2-selector-weight.v1",
        "shape": [out_features, in_features],
        "dtype": "float32",
        "column": "(row+salt)%in_features",
        "salt": salt,
        "nonzero": 0.5,
        "otherwise": 0.0,
    }
    return {**recipe, "identity": digest(recipe)}


def materialize_weight(recipe: dict) -> torch.Tensor:
    require(
        recipe == selector_recipe(*recipe["shape"], recipe["salt"]),
        Status.IDENTITY_DRIFT,
        "weight recipe",
    )
    out, inn = recipe["shape"]
    w = torch.zeros(out, inn, dtype=torch.float32)
    w[torch.arange(out), (torch.arange(out) + recipe["salt"]) % inn] = 0.5
    return w


def selector_linear(x: torch.Tensor, recipe: dict) -> torch.Tensor:
    """Analytic sparse-weight oracle, not a GEMM replacement or production path."""
    require(x.shape[-1] == recipe["shape"][1], Status.SCHEMA_MISMATCH, "selector input")
    ids = (torch.arange(recipe["shape"][0]) + recipe["salt"]) % x.shape[-1]
    return x.float()[..., ids] * 0.5


def state_sequence(
    layer: str,
    tokens: int,
    mode: str,
    chunks: tuple[int, ...] | None = None,
    *,
    page_size: int = 16,
    resume_at: int | None = None,
) -> list[dict]:
    """Recorded byte-transport fixture; values are not compressed model activations."""
    validate_mode(mode)
    require(type(tokens) is int and tokens > 0, Status.SCHEMA_MISMATCH, "tokens")
    chunks = chunks or (tokens,)
    require(
        all(type(c) is int and c > 0 for c in chunks) and sum(chunks) == tokens,
        Status.SCHEMA_MISMATCH,
        "chunks cover exactly the sequence",
    )
    planner = PlannerMock(Identity(layer), page_size)
    snapshots = []
    position = 0
    for chunk in chunks:
        for _ in range(chunk):
            p = position
            cr = {"C0": 0, "C4": 4, "C128": 128}[layer]
            done = cr and (p + 1) % cr == 0
            # Distinct namespaces and bytes; all partial payloads are little-endian FP32.
            recent = struct.pack("<4f", p % 31, (p + 1) % 31, 0.0, 1.0)
            main = struct.pack("<2f", p % 17, (p + 3) % 17) if cr else None
            index = struct.pack("<2f", p % 13, (p + 5) % 13) if layer == "C4" else None
            main_row = struct.pack("<2f", p // cr, 7.0) if done else None
            index_row = bytes((p // 4 % 256, 127)) if done and layer == "C4" else None
            snapshots.append(planner.step(p, recent, main, index, main_row, index_row))
            position += 1
            if resume_at == position:
                planner = PlannerMock.restore(snapshots[-1])
    return snapshots


def boundary_recordings() -> dict[str, dict]:
    """Inputs, expected output bytes and saved boundaries for all 15 operator owners.

    Projection matrices with 67M entries are represented losslessly by selector
    recipes, not downloaded checkpoints. materialize_weight expands on demand.
    """
    r = oracle.tensor_record
    records = {}

    def add(name, inputs, outputs, saved=None):
        records[name] = {
            "operator": name,
            "profile": REFERENCE_PROFILE,
            "inputs": inputs,
            "outputs": outputs,
            "saved": saved or {},
            "evidence_level": "synthetic_reference",
        }

    u = values((2, 64))
    add(
        "indexer_scale",
        {"u": r(u)},
        {"w": r(oracle.scale(u))},
        {"du_for_unit_dw": r(oracle.scale(torch.ones_like(u), backward=True))},
    )

    # Exact orthogonal table values make fixture checksums independent of libm.
    cos = torch.ones(129, 32)
    sin = torch.zeros_like(cos)
    cos[1::2] = 0
    sin[1::2] = 1
    q = values((2, 64, 128), 1)
    pos = torch.tensor([0, 1])
    qrope = oracle.rope(q, cos, sin, pos)
    qh = oracle.hadamard(qrope)
    packed, scales = oracle.pack_mxfp4(qh)
    rope_inputs = {"x": r(q), "cos": r(cos), "sin": r(sin), "positions": r(pos)}
    add(
        "rope_gptj_interleaved_partial",
        rope_inputs,
        {"y": r(qrope)},
        {
            "table_hash": digest([r(cos), r(sin)]),
            "inverse": r(oracle.rope(qrope, cos, sin, pos, inverse=True)),
        },
    )
    add(
        "indexer_q_rope_hadamard_mxfp4",
        rope_inputs,
        {"Q_H_ref": r(qh), "packed": r(packed), "ue8m0": r(scales)},
    )
    x, qr = values((2, 4096), dtype=torch.bfloat16), values((2, 8))
    wq, ww = selector_recipe(8192, 8, 3), selector_recipe(64, 4096, 7)
    add(
        "indexer_projection",
        {"Q_r": r(qr), "X": r(x), "W_qI": wq, "W_w": ww},
        {"Q_I": r(selector_linear(qr, wq).reshape(2, 64, 128)), "u": r(selector_linear(x, ww))},
    )
    k = values((5, 128), 5)
    a = oracle.icv(q, k)
    score = oracle.relu_score(a, oracle.scale(u))
    valid = torch.tensor([[True, False, True, True, True]] * 2)
    ids = torch.arange(5, dtype=torch.int64)
    chosen, count = oracle.topk512(score, valid, ids)
    add("indexer_icv_matmul", {"Q": r(q), "K": r(k)}, {"A": r(a)})
    add(
        "indexer_relu_sum",
        {"A": r(a), "w": r(oracle.scale(u))},
        {"I": r(score)},
        {"relu_mask": r(a > 0)},
    )
    add(
        "indexer_topk512",
        {"scores": r(score), "valid": r(valid), "global_ids": r(ids)},
        {"ids": r(chosen), "valid_count": r(count)},
    )
    for name, dim, groups, cr in (
        ("kv_compressor_c4", 512, 2, 4),
        ("index_compressor_c4", 128, 2, 4),
        ("kv_compressor_c128", 512, 1, 128),
    ):
        shape = (groups, 4, 2, dim) if cr == 4 else (groups, 128, dim)
        kk, ss, ape = values(shape, 2), values(shape, 4), values(shape[1:], 7)
        extra_inputs = {}
        if name == "index_compressor_c4":
            index_x = values((8, 4096), 11, dtype=torch.bfloat16)
            index_wk, index_ws = selector_recipe(256, 4096, 13), selector_recipe(256, 4096, 17)
            kk = selector_linear(index_x, index_wk).reshape(shape)
            ss = selector_linear(index_x, index_ws).reshape(shape)
            extra_inputs = {"X": r(index_x), "W_kvI": index_wk, "W_gI": index_ws}
        kk, ss, ape = (v.requires_grad_() for v in (kk, ss, ape))
        norm_weight = torch.ones(dim, requires_grad=True)
        pooled, alpha = (oracle.c4_pool if cr == 4 else oracle.c128_pool)(kk, ss, ape)
        # Reuse the project's CPU RMSNorm reference rather than add a public primitive.
        normalized = NativeRMSNormOp().forward_fp32(pooled, norm_weight, eps=1e-6)
        completed_positions = torch.arange(groups, dtype=torch.int64) * cr
        rotated = oracle.rope(normalized, cos, sin, completed_positions)
        completed = oracle.hadamard(rotated) if name == "index_compressor_c4" else rotated
        gradients = torch.autograd.grad(completed.sum(), (kk, ss, ape, norm_weight))
        stored = {}
        if name == "index_compressor_c4":
            p4, e8 = oracle.pack_mxfp4(completed)
            stored = {"packed": r(p4), "ue8m0": r(e8)}
        add(
            name,
            {
                "K": r(kk),
                "S": r(ss),
                "APE": r(ape),
                "identity": asdict(Identity("C4" if cr == 4 else "C128")),
                "norm_weight": r(norm_weight),
                "cos": r(cos),
                "sin": r(sin),
                "positions": r(completed_positions),
                **extra_inputs,
            },
            {"pre_norm_pool": r(pooled), "completed_row_fp32": r(completed), **stored},
            {
                "alpha": r(alpha),
                "normalized": r(normalized),
                "rotated": r(rotated),
                "dK": r(gradients[0]),
                "dS": r(gradients[1]),
                "dAPE": r(gradients[2]),
                "dNorm": r(gradients[3]),
                "gradient_seed": "unit_d_completed_row",
                "production_main_fp8_store": "UNSUPPORTED_CAPABILITY",
            },
        )
    state = state_sequence("C4", 4, "training")
    for name, stream in (
        ("recent128_cache_update", "recent"),
        ("kv_compressor_state_cache_update", "main"),
        ("index_compressor_state_cache_update_fp4", "index"),
    ):
        add(
            name,
            {
                "snapshot_before": state[-2],
                "global_position": 3,
                "recent_bytes": state[-1]["recent"][-1]["data"],
                "main_partial_fp32": state[-1]["main"]["previous"][-1],
                "index_partial_fp32": state[-1]["index"]["previous"][-1],
                "main_row_bytes": state[-1]["main"]["rows"][-1]["data"],
                "index_row_bytes": state[-1]["index"]["rows"][-1]["data"],
            },
            {"snapshot_after": state[-1], "stream": stream},
            {"arithmetic_scope": "opaque transport mock, NOT compressor certification"},
        )
    aq, akv, sink = values((64, 512), 2) / 16, values((3, 512), 4) / 16, values((64,))
    out, saved = oracle.joint_attention(aq, akv, sink)
    grads = oracle.joint_attention_backward(aq, akv, saved, torch.ones_like(out))
    add(
        "mqa_joint_attention_sink",
        {"Q": r(aq), "KV": r(akv), "sink": r(sink)},
        {"O": r(out)},
        {**{key: r(v) for key, v in saved.items()}, **{key: r(v) for key, v in grads.items()}},
    )
    o = values((2, 64, 512), 3)
    oi = oracle.rope(o, cos, sin, pos, inverse=True)
    wa = [selector_recipe(1024, 4096, g) for g in range(8)]
    wb = selector_recipe(4096, 8192, 11)
    z = torch.cat(
        [selector_linear(oi[:, g * 8 : (g + 1) * 8].reshape(2, 4096), wa[g]) for g in range(8)], -1
    )
    add(
        "o_proj_grouped",
        {"O": r(o), "cos": r(cos), "sin": r(sin), "positions": r(pos), "W_a": wa, "W_b": wb},
        {"Y": r(selector_linear(z, wb))},
        {"O_tilde": r(oi), "Z": r(z)},
    )
    require(
        set(records) == {op.name for op in OPERATORS},
        Status.INCOMPLETE_ARTIFACT,
        "all 15 independent boundaries",
    )
    for record in records.values():
        record["checksum"] = digest(record)
    return records


def required_evidence() -> list[dict]:
    return [
        {
            "operator": op.name,
            "owner": op.owner,
            "mode": mode,
            "boundary": op.boundary,
            "required": [
                "forward",
                "negative",
                "provenance",
                (
                    "state"
                    if ".state." in op.boundary
                    else "backward_not_applicable"
                    if op.backward == "non_differentiable"
                    else "backward"
                ),
            ],
            "live_status": Status.UNSUPPORTED_CAPABILITY.value,
            "reason": "start kit is not a T02-T07 production implementation",
        }
        for op in OPERATORS
        for mode in MODES
    ]
