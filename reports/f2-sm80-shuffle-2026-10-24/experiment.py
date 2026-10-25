"""Isolated candidate; never installs over the production extension."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.cpp_extension import load_inline
from rl_engine.backends import extension
from rl_engine.reference.rope.qknorm_scaled_rope import fixed_sum_128
from probe import bits_equal, measure


def build():
    declaration = """
std::vector<torch::Tensor> qknorm_scaled_rope_forward_cuda(
 torch::Tensor,torch::Tensor,c10::optional<torch::Tensor>,c10::optional<torch::Tensor>,c10::optional<torch::Tensor>,bool);
std::vector<torch::Tensor> qknorm_scaled_rope_backward_cuda(
 torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,
 c10::optional<torch::Tensor>,c10::optional<torch::Tensor>,c10::optional<torch::Tensor>,bool);
torch::Tensor reduce_probe(torch::Tensor);
"""
    return load_inline(
        name="rlk_f2_shuffle_candidate",
        cpp_sources=declaration,
        cuda_sources=Path(__file__).with_name("candidate.cu").read_text(),
        functions=[
            "qknorm_scaled_rope_forward_cuda",
            "qknorm_scaled_rope_backward_cuda",
            "reduce_probe",
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "--fmad=false",
            "--ftz=false",
            "--prec-div=true",
            "--prec-sqrt=true",
            "-lineinfo",
        ],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    mod = build()
    candidate = SimpleNamespace(
        qknorm_scaled_rope_forward=mod.qknorm_scaled_rope_forward_cuda,
        qknorm_scaled_rope_backward=mod.qknorm_scaled_rope_backward_cuda,
    )
    baseline = extension._C
    data = torch.load("artifacts-jit/inputs.pt", map_location="cuda", weights_only=True)
    gen = torch.Generator(device="cuda").manual_seed(12864)
    count = 128 if args.sanitize else 65536
    x = torch.randn(count, 128, device="cuda", generator=gen)
    x *= torch.exp2(torch.randint(-60, 61, x.shape, device="cuda", generator=gen).float())
    x[0] = 0
    x[0, 0], x[0, 1], x[0, 64], x[0, 65] = 2**25, 1, -(2**25), 1
    assert bits_equal(mod.reduce_probe(x), fixed_sum_128(x).flatten())
    assert mod.reduce_probe(x)[0].item() == 2
    rows = []
    for tokens in ((129,) if args.sanitize else (1, 128, 257, 2048, 8192)):
        if tokens == 257:
            q, k, gq, gk, p = (data[n] for n in ("q", "k", "gq", "gk", "p"))
        else:
            q = torch.randn(tokens, 32, 128, device="cuda", generator=gen).bfloat16()
            k = torch.randn(tokens, 2, 128, device="cuda", generator=gen).bfloat16()
            gq, gk = torch.ones_like(q), torch.ones_like(k)
            p = torch.arange(tokens, device="cuda", dtype=torch.int64)
        c, s = data["c"], data["s"]
        for local in (False, True):
            ref = baseline.qknorm_scaled_rope_forward(q, k, p, c, s, local)
            refg = baseline.qknorm_scaled_rope_backward(q, k, gq, gk, *ref[2:], p, c, s, local)
            actual = candidate.qknorm_scaled_rope_forward(q, k, p, c, s, local)
            actualg = candidate.qknorm_scaled_rope_backward(
                q, k, gq, gk, *actual[2:], p, c, s, local
            )
            assert all(bits_equal(a, b) for a, b in zip((*ref, *refg), (*actual, *actualg)))
            if args.sanitize:
                continue
            for name, impl in [("baseline", baseline), ("candidate", candidate)]:
                forward = lambda: impl.qknorm_scaled_rope_forward(q, k, p, c, s, local)
                backward = lambda: impl.qknorm_scaled_rope_backward(
                    q, k, gq, gk, *ref[2:], p, c, s, local
                )
                rows.append(
                    dict(
                        tokens=tokens,
                        local=local,
                        implementation=name,
                        forward=measure(forward),
                        backward=measure(backward),
                    )
                )
    torch.cuda.synchronize()
    if args.sanitize:
        print("Sanitizer workload: signed tree + local/global forward/backward parity passed")
        return
    Path("shuffle-results.json").write_text(
        json.dumps(
            {
                "signed_rows": count,
                "timings": rows,
                "candidate_library": mod.__file__,
                "baseline_library": baseline.__file__,
            },
            indent=2,
        )
        + "\n"
    )
    import pytest

    with patch.object(extension, "_C", candidate):
        result = pytest.main(
            [
                "tests/ops/rope/test_qknorm_scaled_rope.py",
                "tests/ops/rope/test_qknorm_scaled_rope_forward.py",
                "tests/build/test_build_platform_collectives.py",
                "-q",
            ]
        )
    raise SystemExit(result)


if __name__ == "__main__":
    main()
