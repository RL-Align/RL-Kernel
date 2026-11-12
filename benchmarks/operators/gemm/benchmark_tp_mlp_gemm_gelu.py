# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Single-GPU Qwen-Image MLP controls and shared measurement helpers."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmarks.operators.gemm.benchmark_tp_attn_out_bias_gemm import equal_bytes, measure
from rl_engine.backends.cuda.model_specific.qwenimage.tp_mlp_gemm_gelu import (
    QwenImageTPMlpGemmGeluCudaOp,
)
from rl_engine.backends.extension import _C
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session
from rl_engine.validation.fixtures.qwenimage_mlp import canonical_mlp_indices


def native_mlp(x, up_weight, up_bias, down_weight, down_bias, grad, keys):
    """Evaluate the WS1 tree using the existing pre-output and BF16 dX entries."""
    hidden, pre = _C.mlp_up_gemm_gelu_cuda_forward(x, up_weight, up_bias, True)
    _, partial = _C.mlp_up_gemm_gelu_cuda_forward(hidden, down_weight, None, True)
    y = (partial + down_bias.float()).to(x.dtype)
    dh = _C.mlp_up_gemm_gelu_cuda_dx(grad, down_weight)
    gate = _C.mlp_up_gemm_gelu_cuda_gate(dh, pre)
    dx = _C.mlp_up_gemm_gelu_cuda_dx(gate, up_weight)
    order = canonical_mlp_indices(keys)
    return (
        y[order],
        dx[order],
        _C.mlp_up_gemm_gelu_cuda_dw(gate[order], x[order]),
        _C.mlp_up_gemm_gelu_cuda_db(gate[order]).to(x.dtype),
        _C.mlp_up_gemm_gelu_cuda_dw(grad[order], hidden[order]),
        _C.mlp_up_gemm_gelu_cuda_db(grad[order]).to(x.dtype),
    )


def mlp_calls(op, x, parameters, grad, keys, cuts, parameter_id="mlp"):
    """Keep fixture preparation outside timing and retain each layout's calls."""

    def project():
        return tuple(op(x[start:stop], *parameters) for start, stop in cuts)

    def forward_backward():
        with canonical_backward_session() as session:
            ys = [
                op(
                    x[start:stop],
                    *parameters,
                    logical_keys=keys[start:stop],
                    parameter_id=parameter_id,
                )
                for start, stop in cuts
            ]
            grads = torch.autograd.grad(
                ys, (x, *parameters), [grad[start:stop] for start, stop in cuts]
            )
        session.validate_complete()
        return tuple(ys), *grads

    return torch.no_grad()(project), forward_backward


def pytorch_mlp_calls(x, parameters, grad, cuts):
    up_weight, up_bias, down_weight, down_bias = parameters

    def project():
        return tuple(
            F.linear(
                F.gelu(F.linear(x[start:stop], up_weight, up_bias), approximate="tanh"),
                down_weight,
                down_bias,
            )
            for start, stop in cuts
        )

    def forward_backward():
        ys = project()
        return ys, *torch.autograd.grad(
            ys, (x, *parameters), [grad[start:stop] for start, stop in cuts]
        )

    return torch.no_grad()(project), forward_backward


def make_inputs(rows, seed=386, device="cuda"):
    generator = torch.Generator().manual_seed(seed)

    def rand(shape, scale=1):
        return (torch.randn(shape, generator=generator) * scale).to(
            device=device, dtype=torch.bfloat16
        )

    x = rand((rows, 3072))
    parameters = (
        rand((12288, 3072), 3072**-0.5),
        rand((12288,), 0.1),
        rand((3072, 12288), 12288**-0.5),
        rand((3072,), 0.1),
    )
    grad = rand(x.shape, 3072**-0.5)
    keys = torch.stack(
        (torch.zeros(rows, device=device, dtype=torch.int64), torch.arange(rows, device=device)), 1
    )
    return x, parameters, grad, keys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 129, 4096, 6032, 6889])
    parser.add_argument("--seed", type=int, default=386)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.rows) < 1 or args.warmup < 0 or args.samples < 1:
        parser.error("rows and samples must be positive, and warmup nonnegative")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    op = QwenImageTPMlpGemmGeluCudaOp()
    results = []
    for rows in args.rows:
        x, parameters, grad, keys = make_inputs(rows, args.seed)
        expected = native_mlp(x, *parameters, grad, keys)
        x = x.requires_grad_()
        parameters = tuple(p.requires_grad_() for p in parameters)
        f, fb = mlp_calls(op, x, parameters, grad, keys, [(0, rows)])
        actual = fb()
        assert all(equal_bytes(a, b) for a, b in zip((torch.cat(actual[0]), *actual[1:]), expected))
        del actual, expected
        for backend, pair in (
            ("tp1", (f, fb)),
            ("pytorch", pytorch_mlp_calls(x, parameters, grad, [(0, rows)])),
        ):
            forward, total = [measure(call, args.warmup, args.samples) for call in pair]
            results.append(
                {"backend": backend, "rows": rows, "forward": forward, "forward_backward": total}
            )
            print(backend, rows, total["median_ms"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"seed": args.seed, "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
