"""Run with pytest from repo root; reuse the isolated no-fast-math F2 fixture."""

import hashlib
import json
import os
from pathlib import Path
import sys

import pytest
import torch
import triton

sys.path.insert(0, str(Path.cwd() / "tests/ops/rope"))
from test_qknorm_scaled_rope_forward import cuda_extension  # noqa: E402,F401
from rl_engine.ops.rope.qknorm_scaled_rope import qknorm_scaled_rope, qknorm_scaled_rope_forward
from rl_engine.ops.rope.rope_table import RoPETableIdentity, load_shared_rope_table

INPUT_HASH = "30b84ed1893f5ed9ad40b21e56018737cc22c97f83f49048c2932906f5482c98"
OUTPUT_HASH = "47b5e51fdd8e2b2594dc4aff2a7c63c11df1acb37cb61a13e06d7d0a808ae349"
TABLE_HASH = "c0df1dfb156f81993c6fcfebdd50eca72ce31099436a565dc9407f8d562ac2c3"


def test_a100_baseline(cuda_extension):
    base = Path(os.environ.get("F2_BASELINE_DIR", "/workspace"))
    for name, digest in (("inputs.pt", INPUT_HASH), ("outputs.pt", OUTPUT_HASH)):
        assert hashlib.sha256((base / name).read_bytes()).hexdigest() == digest
    data = torch.load(base / "inputs.pt", weights_only=True)
    expected = torch.load(base / "outputs.pt", weights_only=True)
    c, s = load_shared_rope_table(
        data["c"], data["s"], expected=RoPETableIdentity(131072, TABLE_HASH), device="cuda"
    )
    rows, saved = [], {}
    for local in (False, True):
        for backend in ("cuda", "triton"):
            # Whole baseline, token splits, and TP-like head subsets.
            for start, end, hq, hk in (
                (0, 257, 32, 2),
                (0, 1, 32, 2),
                (1, 8, 32, 2),
                (8, 136, 32, 2),
                (136, 257, 32, 2),
                (0, 257, 4, 1),
            ):
                q, k, gq, gk = [
                    data[n][start:end, :heads].contiguous().cuda()
                    for n, heads in (("q", hq), ("k", hk), ("gq", hq), ("gk", hk))
                ]
                p = data["p"][start:end].contiguous().cuda()
                forward = qknorm_scaled_rope_forward(q, k, p, c, s, is_local=local, backend=backend)
                q.requires_grad_()
                k.requires_grad_()
                y = qknorm_scaled_rope(q, k, p, c, s, is_local=local, backend=backend)
                gradients = torch.autograd.grad(y, (q, k), (gq, gk))
                outputs = [x.cpu() for x in (*forward, *gradients)]
                counts = []
                for i, (actual, reference) in enumerate(
                    zip(outputs, expected[f"{local}/{backend}"])
                ):
                    reference = reference[start:end, : hq if i % 2 == 0 else hk].contiguous()
                    assert actual.dtype == reference.dtype and actual.shape == reference.shape
                    dtype = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
                    counts.append(
                        int((actual.contiguous().view(dtype) != reference.view(dtype)).sum())
                    )
                rows.append(
                    dict(
                        local=local,
                        backend=backend,
                        start=start,
                        end=end,
                        hq=hq,
                        hk=hk,
                        bit_mismatches=counts,
                    )
                )
                if start == 0 and end == 257 and hq == 32:
                    saved[f"{local}/{backend}"] = outputs
    report = dict(
        gpu=torch.cuda.get_device_name(),
        capability=torch.cuda.get_device_capability(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        triton=triton.__version__,
        input_sha256=INPUT_HASH,
        a100_v2_sha256=OUTPUT_HASH,
        table_sha256=TABLE_HASH,
        rows=rows,
    )
    Path("cross-gpu-results.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save(saved, "h100-outputs.pt")
    assert all(not any(row["bit_mismatches"]) for row in rows), rows
