#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""``h3-parallel-v2`` against the GEMM rows' own operators, on one SM90 GPU.

For ``h3_attention_o_gemm`` / ``h3_ffn_down_gemm`` (row-parallel) and ``h3_ffn_gate_up_gemm``
(column-parallel), every TP size 1/2/4/8 and every rank, the tensor-parallel pieces are computed
as the TP paths compute them (``parallel._units_partial`` and ``fixed_fp32_gemm`` on the rank's
shard; the cross-rank sum as the balanced tree the fixed collectives use) and compared with the
rows' TP=1 operators (#554, #535), run with logical keys in a canonical backward session:

* expected bitwise: the gate/up forward columns, the O / down ``dX`` columns, every ``dW`` shard;
* reported (the 8-unit tree, not the operators' single K sweep): the O / down forward and the
  gate/up ``dX``, as the max abs difference and the count of differing elements.

Writes ``--out`` (strict JSON) and exits 1 if an expected-bitwise piece differs or any piece is not
finite.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

ROWS = {
    # row: (kind, K, N, op class)
    "h3_attention_o_gemm": (
        "row",
        7168,
        5376,
        "rl_engine.ops.gemm.h3_projections.H3AttentionOGemmOp",
    ),
    "h3_ffn_down_gemm": ("row", 14336, 5376, "rl_engine.ops.gemm.h3_ffn_down.H3FFNDownGemmOp"),
    "h3_ffn_gate_up_gemm": (
        "col",
        5376,
        28672,
        "rl_engine.ops.gemm.h3_projections.H3FFNGateUpGemmOp",
    ),
}


def _op(path):
    import importlib

    module, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module), name)()


def _reference(op, x, w, grad):
    """The row operator's ``y``, ``dX``, ``dW`` at TP=1 (logical keys, one canonical session)."""

    import torch

    from rl_engine.ops.autograd.canonical_backward import canonical_backward_session

    xs = x.detach().clone().requires_grad_(True)
    ws = w.detach().clone().requires_grad_(True)
    keys = torch.stack(
        [torch.zeros(x.shape[0], dtype=torch.int64), torch.arange(x.shape[0])], dim=1
    ).to(x.device)
    with canonical_backward_session() as session:
        y = op(xs, ws, logical_keys=keys, parameter_id="w")
        y.backward(grad)
        session.validate_complete()
    return y.detach(), xs.grad, ws.grad


def _error_bound(lhs, rhs, terms, a, b):
    """Elementwise bound on ``|a - b|`` for two FP32 sums of the same terms in different orders.

    ``lhs @ rhs`` is summed over ``terms`` terms per element. Any summation order is within
    ``gamma(terms) * (|lhs| @ |rhs|)`` of the exact value (``gamma`` with FP32 unit roundoff
    ``2**-24``), and each of the two results is then rounded once to BF16 (relative ``2**-9``).
    """

    import torch

    absolute = lhs.float().abs() @ rhs.float().abs()
    u = 2.0**-24
    gamma = (terms * u) / (1 - terms * u)
    return 2 * gamma * absolute + 2.0**-8 * torch.maximum(a.float().abs(), b.float().abs())


def _diff(a, b, bound=None):
    """``bitwise``: same dtype, shape and bytes (``torch.equal`` compares values, so ``-0.0`` and
    ``+0.0`` would match). ``finite``: both sides finite; ``max_abs`` is ``None`` otherwise (or if
    the shapes or dtypes differ), so the report stays strict JSON, and the check fails (finite
    inputs must give finite results). ``within_bound`` is ``None`` without ``bound``, else whether
    every element is within it (``_error_bound``)."""

    import torch

    same = a.dtype == b.dtype and a.shape == b.shape
    bitwise = same and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )
    finite = bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
    return {
        "bitwise": bool(bitwise),
        "finite": finite,
        "max_abs": float((a.float() - b.float()).abs().max()) if (finite and same) else None,
        "within_bound": (
            bool(((a.float() - b.float()).abs() <= bound).all())
            if bound is not None and finite and same
            else (None if bound is None else False)
        ),
        "differing": int((a != b).sum()) if same else None,
    }


def check(row: str, rows: int, seed: int) -> dict:
    import torch

    from rl_engine.backends.cuda.model_specific.minimax_h3 import fixed_fp32_gemm as fx
    from rl_engine.backends.cuda.model_specific.minimax_h3 import parallel as par
    from rl_engine.distributed.algorithms.hierarchical import tree_add

    kind, k, n, path = ROWS[row]
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, k, device="cuda", generator=gen).bfloat16()
    w = (torch.randn(n, k, device="cuda", generator=gen) / k**0.5).bfloat16()
    grad = torch.randn(rows, n, device="cuda", generator=gen).bfloat16()
    y_ref, dx_ref, dw_ref = _reference(_op(path), x, w, grad)
    out: dict = {"row": row, "kind": kind, "rows": rows, "tp": {}}
    ok = True
    for tp in par.TP_SIZES:
        entry: dict = {"expected_bitwise": {}, "reported": {}}
        if kind == "row":
            partials, dx_parts, dw_parts = [], [], []
            for r in range(tp):
                cols = slice(r * k // tp, (r + 1) * k // tp)
                ws = par.shard_cols(w, tp, r)
                partials.append(par._units_partial(x[:, cols].contiguous(), ws, tp))
                dx_parts.append(fx.gemm(grad, ws))
                dw_parts.append(fx.gemm(grad, x[:, cols].contiguous(), a_t=True))
            y = tree_add(partials).to(torch.bfloat16)
            bound = _error_bound(x, w.t(), k, y, y_ref)  # y = x @ w^T, k terms per element
            entry["reported"]["forward"] = _diff(y, y_ref, bound)
            entry["expected_bitwise"]["dx"] = _diff(torch.cat(dx_parts, 1), dx_ref)
            entry["expected_bitwise"]["dw"] = _diff(torch.cat(dw_parts, 1), dw_ref)
        else:
            half = n // 2
            local = par.UNITS // tp
            ys, dws, dx_partials = [], [], []
            for r in range(tp):
                shard = par.shard_gate_up(w, tp, r)
                hid_rows = slice(r * half // tp, (r + 1) * half // tp)
                gate_rows = slice(half + r * half // tp, half + (r + 1) * half // tp)
                g_local = [grad[:, hid_rows].contiguous(), grad[:, gate_rows].contiguous()]
                ys.append(fx.gemm(x, shard, b_t=True))
                ws_local = list(shard.chunk(2))
                dws.append([fx.gemm(gl, x, a_t=True) for gl in g_local])
                units = []
                for j in range(local):
                    acc = None
                    for gl, wl in zip(g_local, ws_local):
                        width = gl.shape[1] // local
                        piece = fx.gemm(
                            gl[:, j * width : (j + 1) * width],
                            wl[j * width : (j + 1) * width],
                            out_dtype=torch.float32,
                        )
                        acc = piece if acc is None else acc + piece
                    units.append(acc)
                dx_partials.append(tree_add(units))
            y_hid = torch.cat([t.chunk(2, 1)[0] for t in ys], 1)
            y_gate = torch.cat([t.chunk(2, 1)[1] for t in ys], 1)
            entry["expected_bitwise"]["forward"] = _diff(torch.cat([y_hid, y_gate], 1), y_ref)
            dw = torch.cat([torch.cat([d[0] for d in dws]), torch.cat([d[1] for d in dws])])
            entry["expected_bitwise"]["dw"] = _diff(dw, dw_ref)
            dx = tree_add(dx_partials).to(torch.bfloat16)
            # dX = grad @ w, n terms per element
            entry["reported"]["dx"] = _diff(dx, dx_ref, _error_bound(grad, w, n, dx, dx_ref))
        ok &= all(v["bitwise"] and v["finite"] for v in entry["expected_bitwise"].values())
        # informational order, but finite and within the summation error bound of the reference
        ok &= all(v["finite"] and v["within_bound"] for v in entry["reported"].values())
        out["tp"][str(tp)] = entry
    out["pass"] = ok
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[264, 925, 4296])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        print("the GEMM rows' operators run on SM90 only", file=sys.stderr)
        return 2
    torch.backends.cuda.matmul.allow_tf32 = False
    results = [check(row, rows, args.seed) for row in ROWS for rows in args.rows]
    report = {
        "contract": "h3-parallel-v2",
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "pass": all(r["pass"] for r in results),
        "results": results,
    }
    args.out.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    print(json.dumps({"pass": report["pass"]}))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
