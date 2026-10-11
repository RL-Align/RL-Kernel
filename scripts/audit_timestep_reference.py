#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent FP64/autograd audit; never changes the official FP32 oracle.

The boundary oracle evaluates trig at the specified FP32 phase, with the
analytic phase Jacobian, then uses FP64 F.linear/F.silu and native autograd.
A separate all-FP64 phase diagnostic measures the effect of phase rounding.
Neither oracle imports candidate reduction or custom-backward primitives.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from validate_timestep_official import inputs, NAMES, run_grad  # noqa: E402
from rl_engine.kernels.gtest.tolerance import load_contract, resolve_tolerance  # noqa: E402
from rl_engine.kernels.ops.pytorch.timestep_embed_mlp import NativeTimestepEmbedMLPOp  # noqa: E402
from rl_engine.kernels.ops.timestep_embed_mlp import TimestepEmbedMLPOp  # noqa: E402


def conventional(values, upstream, *, double, boundary):
    dtype = torch.float64 if double else torch.float32
    leaves = [values[k].cpu().to(dtype).detach().requires_grad_() for k in NAMES]
    t, w1, b1, w2, b2 = leaves
    f32 = (-math.log(10000.0) * torch.arange(128).float() / 128).exp()
    if double:
        frequency = (
            f32.double()
            if boundary
            else (-math.log(10000.0) * torch.arange(128).double() / 128).exp()
        )
        phase = (t[:, None] * frequency) * 1000
        if boundary:
            stored = ((values["timestep"].cpu().float()[:, None] * f32) * 1000).double()
            # Preserve the analytic Jacobian while anchoring the value at the
            # declared FP32 phase. This is not a full-FP64 mathematical oracle.
            phase = stored + (phase - phase.detach())
    else:
        phase = (t[:, None] * f32) * 1000
    embedding = torch.cat((phase.cos(), phase.sin()), 1)
    if double:
        y = F.linear(F.silu(F.linear(embedding, w1, b1)), w2, b2)
    else:
        # Historical per-row torch.mv reference with native autograd.
        z = torch.stack([torch.mv(w1, row) + b1 for row in embedding])
        h = z * z.sigmoid()
        y = torch.stack([torch.mv(w2, row) + b2 for row in h])
    grads = torch.autograd.grad(y, leaves, upstream.cpu().to(dtype))
    return [y.detach(), *[g.detach() for g in grads]]


def compare(actual, expected, dtype):
    rows = {}
    for i, (name, a, b) in enumerate(zip(("output", *NAMES), actual, expected, strict=True)):
        tol = resolve_tolerance(
            load_contract(),
            judgment="forward_accuracy" if i == 0 else "gradient_accuracy",
            op_class="reduction",
            dtype=dtype,
        )
        a, b = a.cpu().double(), b.cpu().double()
        error = (a - b).abs()
        limit = tol.atol + tol.rtol * b.abs()
        rows[name] = dict(
            max_abs=error.max().item(),
            rms=error.square().mean().sqrt().item(),
            max_error_over_limit=(error / limit).max().item(),
            violations=int((~torch.isfinite(error) | (error > limit)).sum()),
            atol=tol.atol,
            rtol=tol.rtol,
        )
    return dict(passed=all(r["violations"] == 0 for r in rows.values()), tensors=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = dict(
        scope="Independent reference audit at FP32 phase boundary; full FP64 is diagnostic",
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cases=[],
        passed=False,
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__).resolve(),
                ROOT / "rl_engine/kernels/ops/pytorch/timestep_embed_mlp.py",
                ROOT / "rl_engine/kernels/gtest/tolerance_contract.json",
            ]
        },
    )
    start = time.monotonic()
    try:
        for dtype in (torch.float32, torch.bfloat16):
            for batch, seed, rng in (
                (16, 9386, "cuda"),
                (16, 9386, "cpu"),
                (3, 1701, "cpu"),
                (16, 20261007, "cpu"),
            ):
                v = inputs(batch, 3072, dtype, "cpu", seed)
                dy = (
                    torch.randn(
                        (batch, 3072),
                        generator=torch.Generator(device=rng).manual_seed(seed + 1),
                        device=rng,
                    )
                    .to(dtype)
                    .cpu()
                )
                oracle = conventional(v, dy, double=True, boundary=True)
                full = conventional(v, dy, double=True, boundary=False)
                old = conventional(v, dy, double=False, boundary=True)
                new = run_grad(NativeTimestepEmbedMLPOp(), v, dy)
                row = dict(
                    batch=batch,
                    seed=seed,
                    dtype=str(dtype),
                    upstream_rng=rng,
                    input_sha256={
                        k: hashlib.sha256(
                            x.contiguous().view(torch.uint8).numpy().tobytes()
                        ).hexdigest()
                        for k, x in {**v, "upstream": dy}.items()
                    },
                    old_vs_boundary=compare(old, oracle, dtype),
                    reference_vs_boundary=compare(new, oracle, dtype),
                    boundary_vs_full_fp64=compare(oracle, full, dtype),
                    candidates={},
                )
                for backend in ("cuda", "triton"):
                    op = TimestepEmbedMLPOp(backend)
                    result = run_grad(op, {k: x.cuda() for k, x in v.items()}, dy.cuda())
                    row["candidates"][backend] = dict(
                        vs_boundary=compare(result, oracle, dtype),
                        vs_reference=compare(result, new, dtype),
                        actual_backend=op.last_trace.actual_backend,
                        fallback_reason=op.last_trace.fallback_reason,
                    )
                row["passed"] = row["reference_vs_boundary"]["passed"] and all(
                    c["vs_boundary"]["passed"]
                    and c["vs_reference"]["passed"]
                    and c["actual_backend"] == b
                    and c["fallback_reason"] is None
                    for b, c in row["candidates"].items()
                )
                report["cases"].append(row)
                print(
                    json.dumps(
                        {k: row[k] for k in ("batch", "seed", "dtype", "upstream_rng", "passed")}
                    ),
                    flush=True,
                )
                args.output.write_text(json.dumps(report, indent=2) + "\n")
        report["passed"] = len(report["cases"]) == 8 and all(c["passed"] for c in report["cases"])
    finally:
        report["seconds"] = time.monotonic() - start
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
