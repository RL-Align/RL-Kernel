# SPDX-License-Identifier: Apache-2.0
"""F2 public API versus exact pinned native eager functions (no model weights).

Run from repo root with PYTHONPATH=. and --model-source pointing to the pinned
modeling_muse_glimmer.py. AST extraction avoids importing unrelated model deps.
"""

import argparse
import ast
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
import triton

from rl_engine.ops.rope.qknorm_scaled_rope import qknorm_scaled_rope
from rl_engine.ops.rope.rope_table import build_shared_rope_table, table_identity

SOURCE_HASH = "e69eb21d6efd815a48499d4d0e4180afda2f443beb441a734d50efc2cf83f975"


def native_functions(path):
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_HASH:
        raise ValueError("native source differs from pinned Transformers source")
    names = {"MuseGlimmerRMSNorm", "rotate_half", "apply_rotary_pos_emb"}
    nodes = [node for node in ast.parse(source).body if getattr(node, "name", None) in names]
    assert len(nodes) == 3
    for node in nodes:
        if node.name == "apply_rotary_pos_emb":
            assert [ast.unparse(d) for d in node.decorator_list] == [
                "use_kernel_forward_from_hub('rotary_pos_emb')"
            ]
            # Explicit eager baseline: disable optional external Hub replacement.
            node.decorator_list = []
        else:
            assert not node.decorator_list
    namespace = {"torch": torch, "nn": torch.nn}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return (
        namespace["MuseGlimmerRMSNorm"](128, eps=1e-5, with_scale=False),
        namespace["apply_rotary_pos_emb"],
    )


def measure(prepare, samples):
    # Preparation includes graph construction for backward-only and is untimed.
    values = []
    for index in range(samples + 5):
        call = prepare()
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = call()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1e6
        if index >= 5:
            values.append(elapsed)
        del result, call
    return dict(median_us=statistics.median(values), min_us=min(values), samples_us=values)


@torch.no_grad()
def relative_error(actual, reference):
    return float(
        (actual.double() - reference.double()).norm() / reference.double().norm().clamp_min(1e-30)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=21)
    args = parser.parse_args()
    norm, rope = native_functions(args.model_source)
    cos, sin = build_shared_rope_table(131072)
    rows = []
    for tokens in (1, 128, 2048):
        generator = torch.Generator(device="cuda").manual_seed(501)
        q = (
            torch.randn(tokens, 32, 128, device="cuda", generator=generator)
            .bfloat16()
            .requires_grad_()
        )
        k = (
            torch.randn(tokens, 2, 128, device="cuda", generator=generator)
            .bfloat16()
            .requires_grad_()
        )
        grads = tuple(
            torch.randn(x.shape, device="cuda", generator=generator).bfloat16() for x in (q, k)
        )
        p = (torch.arange(tokens, device="cuda") * 17 + 2047) % 131072
        for local in (False, True):

            def native():
                a, b = norm(q) * 3.87, norm(k)
                if local:
                    # Same prebuilt table and absolute positions as fused execution.
                    # Lookup/expansion is timed, generation and registration are not.
                    c, s = cos[p].repeat(1, 2), sin[p].repeat(1, 2)
                    a, b = rope(a, b, c, s, unsqueeze_dim=1)
                return a, b

            reference = native()
            reference_grads = torch.autograd.grad(reference, (q, k), grads)
            for backend in ("native_eager", "cuda", "triton"):
                forward = (
                    native
                    if backend == "native_eager"
                    else lambda: qknorm_scaled_rope(
                        q, k, p, cos, sin, is_local=local, backend=backend
                    )
                )
                output = forward()
                gradient = torch.autograd.grad(output, (q, k), grads)
                errors = dict(
                    forward=[relative_error(a, b) for a, b in zip(output, reference)],
                    backward=[relative_error(a, b) for a, b in zip(gradient, reference_grads)],
                )
                assert max(errors["forward"] + errors["backward"]) < 0.03, errors

                def inference():
                    with torch.no_grad():
                        return forward()

                def backward_prepare():
                    y = forward()
                    return lambda: torch.autograd.grad(y, (q, k), grads)

                def training():
                    return torch.autograd.grad(forward(), (q, k), grads)

                for mode, prepare in (
                    ("forward", lambda: inference),
                    ("backward", backward_prepare),
                    ("forward_backward", lambda: training),
                ):
                    row = dict(
                        tokens=tokens,
                        q_heads=32,
                        k_heads=2,
                        local=local,
                        backend=backend,
                        mode=mode,
                        errors=errors,
                        **measure(prepare, args.samples),
                    )
                    rows.append(row)
                    print(
                        f"T={tokens} local={local} {backend} {mode}: {row['median_us']:.1f} us",
                        flush=True,
                    )
    args.out.write_text(
        json.dumps(
            dict(
                source_sha256=SOURCE_HASH,
                torch=torch.__version__,
                cuda=torch.version.cuda,
                triton=triton.__version__,
                gpu=torch.cuda.get_device_name(),
                table_sha256=table_identity(cos, sin).sha256,
                measurement="synchronized wall time, public API; eager native, no compile/graphs",
                rows=rows,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
