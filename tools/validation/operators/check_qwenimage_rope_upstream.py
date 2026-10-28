# SPDX-License-Identifier: Apache-2.0
"""Audit table and rotation semantics against a checksum-pinned upstream file.

Supply transformer_qwenimage.py from Diffusers revision
40f63690da440e8c382ab53e67508835deb1ed2f. Only the two audited symbol definitions
are executed, after verifying the source hash. No Diffusers installation or model
checkpoint is required. This checks CPU semantics, not full-model integration.
"""

from __future__ import annotations

import argparse
import ast
import functools
import hashlib
import json
import logging
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
SOURCE_SHA256 = "0bad11901deae69f05aec01fd464e51e38582d455068f2851ee665036179321f"


def main():
    from rl_engine.ops.rope.multi_axis_rope import (
        build_multi_axis_cos_sin,
        get_multi_axis_rope_op,
        qwen_image_positions,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.read_text(encoding="utf-8")
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError("upstream source does not match the pinned revision checksum")
    symbols = {"QwenEmbedRope", "apply_rotary_emb_qwen"}
    nodes = [n for n in ast.parse(source).body if getattr(n, "name", None) in symbols]
    if len(nodes) != 2:
        raise ValueError("missing audited upstream definitions")
    namespace = {
        "torch": torch,
        "nn": torch.nn,
        "lru_cache_unless_export": functools.lru_cache,
        "logger": logging.getLogger("upstream_audit"),
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(args.source), "exec"),
        namespace,
    )
    embed = namespace["QwenEmbedRope"](theta=10000, axes_dim=[16, 56, 56], scale_rope=True)
    apply_upstream = namespace["apply_rotary_emb_qwen"]
    op = get_multi_axis_rope_op("reference")
    torch.manual_seed(386)
    results = []
    for height, width in ((2, 3), (3, 5), (64, 64), (83, 83), (104, 58)):
        positions = qwen_image_positions(height, width, 7)
        upstream_tables = embed((1, height, width), device=torch.device("cpu"), max_txt_seq_len=7)
        for stream, pos, upstream in zip(
            ("image", "text"), positions, upstream_tables, strict=True
        ):
            cos, sin = build_multi_axis_cos_sin(pos)
            assert torch.equal(cos, upstream.real) and torch.equal(sin, upstream.imag)
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                x = torch.randn(1, len(pos), 2, 128, dtype=dtype, requires_grad=True)
                grad = torch.randn_like(x)
                expected = apply_upstream(x, upstream, use_real=False)
                actual = op(x, cos, sin)
                eg = torch.autograd.grad(expected, x, grad)[0]
                ag = torch.autograd.grad(actual, x, grad)[0]
                tol = {torch.float32: 8e-7, torch.float16: 0.002, torch.bfloat16: 0.02}[dtype]
                torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
                torch.testing.assert_close(ag, eg, atol=tol, rtol=tol)
                results.append(
                    {
                        "grid": [height, width],
                        "stream": stream,
                        "dtype": str(dtype),
                        "table_bitwise": True,
                        "forward_max_abs": (actual.float() - expected.float()).abs().max().item(),
                        "dx_max_abs": (ag.float() - eg.float()).abs().max().item(),
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"source_sha256": digest, "cases": results}, indent=2) + "\n")
    print(
        f"{len(results)} upstream forward/backward cases passed; "
        "all image/text tables exactly equal"
    )


if __name__ == "__main__":
    main()
