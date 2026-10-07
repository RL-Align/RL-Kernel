# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Per-operator performance and accuracy measurements for the H3 evidence.

``PERF_CASES[op]`` builds the timed cases that ``benchmarks/benchmark_h3_conditioning.py``
prints and ``scripts/h3_evidence.py`` stores. ``ACCURACY[op]`` measures the
op against the provider path and its golden. Each RFC #420 row registers both.
"""

from __future__ import annotations

import statistics
from typing import Any, Callable

import torch

from rl_engine.kernels.registry import KernelRegistry
from rl_engine.testing.h3_cases import h3_packed_layout, h3_timesteps
from rl_engine.testing.h3_provider import (
    provider_adaln_modulation,
    provider_adaln_row_gather,
    provider_gate_residual,
    provider_norm_modulate,
    provider_time_embedder,
    provider_time_proj,
)
from rl_engine.testing.h3_weights import (
    WEIGHTS_ENV,
    h3_weights_dir,
    load_h3_conditioning_weights,
)

# Keys of a perf case that hold a timed callable, in display order.
TIMED_KEYS = (
    "candidate",
    "candidate_checked",
    "provider",
    "candidate_backward",
    "provider_backward",
)


def time_us(fn: Callable[[], Any], warmup: int = 20, iters: int = 200) -> float:
    """Median CUDA-event time of ``fn`` in microseconds (includes the Python wrapper)."""

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3)
    return statistics.median(samples)


def peak_mib(fn: Callable[[], Any]) -> float:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20


def measure(case: dict[str, Any], warmup: int = 20, iters: int = 200) -> dict[str, Any]:
    row = {key: case[key] for key in ("op", "case", "backend", "bytes") if key in case}
    for key in TIMED_KEYS:
        if key in case:
            us = time_us(case[key], warmup, iters)
            row[f"{key}_us"] = us
            row[f"{key}_gbps"] = case["bytes"] / (us * 1e-6) / 1e9
            row[f"{key}_peak_mib"] = peak_mib(case[key])
    return row


# --------------------------------------------------------------------------- #
# timestep_sinusoid_h3
# --------------------------------------------------------------------------- #


def _sinusoid_perf(registry: KernelRegistry) -> list[dict[str, Any]]:
    op = registry.get_op("timestep_sinusoid_h3", device="cuda")
    cases = []
    for num in (1, 2, 4, 64):
        t = h3_timesteps(num)
        cases.append(
            {
                "op": "timestep_sinusoid_h3",
                "case": f"T={num}",
                "backend": type(op).__name__,
                "bytes": num * 4 + num * 256 * 4,
                "candidate": lambda t=t: op.forward(t, check_range=False),
                "candidate_checked": lambda t=t: op.forward(t),
                "provider": lambda t=t: provider_time_proj(t),
            }
        )
    return cases


def _sinusoid_accuracy(registry: KernelRegistry) -> dict[str, Any]:
    op = registry.get_op("timestep_sinusoid_h3", device="cuda")
    golden = registry._get_or_create_backend(
        registry._priority_map["cpu"]["timestep_sinusoid_h3"][-1]
    )
    rows = []
    for num in (1, 2, 3, 7, 64, 1000, 4097):
        t = h3_timesteps(num, seed=num)
        out, ref, gold = op(t), provider_time_proj(t), golden.forward_fp32(t)
        rows.append(
            {
                "num_timesteps": num,
                "bitwise_equal_to_provider": bool(torch.equal(out, ref)),
                "max_abs_vs_fp64": float((out - gold).abs().max()),
                "provider_max_abs_vs_fp64": float((ref - gold).abs().max()),
            }
        )
    return {"contract_atol": 1e-5, "cases": rows}


def h3_params(names: list[str], shapes: list[tuple[int, ...]]) -> list[torch.Tensor]:
    """Pinned checkpoint tensors when available, otherwise same-shape random ones."""

    if h3_weights_dir() is not None:
        weights = load_h3_conditioning_weights("cuda", names)
        return [weights[name] for name in names]
    generator = torch.Generator(device="cuda").manual_seed(0)
    return [
        torch.randn(shape, device="cuda", generator=generator) * shape[-1] ** -0.5
        for shape in shapes
    ]


# --------------------------------------------------------------------------- #
# timestep_mlp_fp32
# --------------------------------------------------------------------------- #

MLP_NAMES = [f"time_embedder.linear_{i}.{p}" for i in (1, 2) for p in ("weight", "bias")]
MLP_SHAPES = [(5376, 256), (5376,), (2688, 5376), (2688,)]


def _mlp_perf(registry: KernelRegistry) -> list[dict[str, Any]]:
    op = registry.get_op("timestep_mlp_fp32", device="cuda")
    params = h3_params(MLP_NAMES, MLP_SHAPES)
    weight_bytes = sum(p.numel() * p.element_size() for p in params)
    cases = []
    for num in (1, 2, 3, 4):
        x = torch.rand(num, 256, device="cuda") * 2 - 1
        cases.append(
            {
                "op": "timestep_mlp_fp32",
                "case": f"T={num}",
                "backend": type(op).__name__,
                "bytes": weight_bytes,
                "candidate": lambda x=x: op(x, *params),
                "provider": lambda x=x: provider_time_embedder(x, *params),
            }
        )
    return cases


def _mlp_accuracy(registry: KernelRegistry, draws: int = 200) -> dict[str, Any]:
    op = registry.get_op("timestep_mlp_fp32", device="cuda")
    golden = registry._get_or_create_backend(registry._priority_map["cpu"]["timestep_mlp_fp32"][-1])
    sinusoid = registry.get_op("timestep_sinusoid_h3", device="cuda")
    params = h3_params(MLP_NAMES, MLP_SHAPES)
    cuda_err, provider_err = [], []
    for seed in range(draws):
        x = sinusoid(h3_timesteps(4, seed=seed))
        gold = golden.forward_fp32(x, *params)
        cuda_err.append(float((op(x, *params) - gold).abs().max()))
        provider_err.append(float((provider_time_embedder(x, *params) - gold).abs().max()))
    # Row invariance: each timestep alone equals its row in the batch of 9.
    x = sinusoid(h3_timesteps(9, seed=1))
    full = op(x, *params)
    invariant = all(torch.equal(op(x[i : i + 1], *params)[0], full[i]) for i in range(9))
    return {
        "contract_atol": 1e-4,
        "draws": draws,
        "num_timesteps": 4,
        "cuda_max_abs_vs_fp64": cuda_err,
        "provider_max_abs_vs_fp64": provider_err,
        "rows_batch_invariant": invariant,
    }


# --------------------------------------------------------------------------- #
# adaln_projection_3mod
# --------------------------------------------------------------------------- #

ADALN_NAMES = [
    "transformer_blocks.0.adaln_proj.linear.weight",
    "transformer_blocks.0.adaln_proj.linear.bias",
]


def _adaln_params() -> tuple[torch.Tensor, torch.Tensor]:
    weight, bias = h3_params(ADALN_NAMES, [(96768, 2688), (96768,)])
    return weight.bfloat16(), bias.bfloat16()


def _projection_perf(registry: KernelRegistry) -> list[dict[str, Any]]:
    op = registry.get_op("adaln_projection_3mod", device="cuda")
    weight, bias = _adaln_params()
    cases = []
    for num in (1, 2, 3, 4):
        temb = torch.randn(num, 2688, device="cuda")
        cases.append(
            {
                "op": "adaln_projection_3mod",
                "case": f"T={num}",
                "backend": type(op).__name__,
                "bytes": weight.numel() * weight.element_size(),
                "candidate": lambda temb=temb: op(temb, weight, bias),
                "provider": lambda temb=temb: provider_adaln_modulation(temb, weight, bias),
            }
        )
    return cases


def _flat_cat(outputs) -> torch.Tensor:
    return torch.cat([out.reshape(-1) for out in outputs])


def _projection_accuracy(registry: KernelRegistry, draws: int = 20) -> dict[str, Any]:
    """Fraction of BF16 outputs equal to the correctly rounded FP64 golden, per draw."""

    op = registry.get_op("adaln_projection_3mod", device="cuda")
    golden = registry._get_or_create_backend(
        registry._priority_map["cpu"]["adaln_projection_3mod"][-1]
    )
    weight, bias = _adaln_params()
    cuda_frac, provider_frac, early_frac = [], [], []
    for seed in range(draws):
        g = torch.Generator(device="cuda").manual_seed(seed)
        temb = torch.randn(4, 2688, device="cuda", generator=g) * 2
        gold = _flat_cat(golden.forward_fp32(temb, weight, bias)).bfloat16()
        early = _flat_cat(golden.forward_fp32(temb.bfloat16().float(), weight, bias)).bfloat16()
        ours = _flat_cat(op(temb, weight, bias))
        theirs = _flat_cat(provider_adaln_modulation(temb, weight, bias))
        cuda_frac.append(float((ours == gold).float().mean()))
        provider_frac.append(float((theirs == gold).float().mean()))
        early_frac.append(float((ours == early).float().mean()))
    temb = torch.randn(9, 2688, device="cuda")
    full = op(temb, weight, bias)
    invariant = all(
        all(
            torch.equal(s, f[3 * i : 3 * i + 3])
            for s, f in zip(op(temb[i : i + 1], weight, bias), full, strict=True)
        )
        for i in range(9)
    )
    return {
        "draws": draws,
        "num_timesteps": 4,
        "cuda_correctly_rounded": cuda_frac,
        "provider_correctly_rounded": provider_frac,
        "early_cast_golden_match": early_frac,
        "rows_batch_invariant": invariant,
    }


# --------------------------------------------------------------------------- #
# adaln_row_gather
# --------------------------------------------------------------------------- #

GATHER_SEQ_LENS = (4097, 32768, 131072)


def _gather_perf(registry: KernelRegistry) -> list[dict[str, Any]]:
    op = registry.get_op("adaln_row_gather", device="cuda")
    num_timesteps, hidden = 3, 5376
    rows = torch.randn(3 * num_timesteps, 6 * hidden, device="cuda").bfloat16()
    chunks = rows.chunk(6, dim=-1)
    cases = []
    for seq in GATHER_SEQ_LENS:
        ti, tags = h3_packed_layout(seq, num_timesteps, seed=seq)
        grads = [torch.randn(seq, hidden, device="cuda").bfloat16() for _ in range(6)]

        def backward(fn, ti=ti, tags=tags, grads=grads):
            leaf = rows.detach().requires_grad_(True)
            torch.autograd.backward(list(fn(leaf, ti, tags)), grads)

        cases.append(
            {
                "op": "adaln_row_gather",
                "case": f"S={seq}",
                "backend": type(op).__name__,
                # bytes written (six (S, H) BF16 outputs); the 3T-row table stays in L2
                "bytes": 6 * seq * hidden * 2,
                "candidate": lambda ti=ti, tags=tags: op.forward(rows, ti, tags, check_range=False),
                "provider": lambda ti=ti, tags=tags: provider_adaln_row_gather(chunks, ti, tags),
                "candidate_backward": lambda b=backward: b(
                    lambda r, ti, tags: op.forward(r, ti, tags, check_range=False)
                ),
                "provider_backward": lambda b=backward: b(
                    lambda r, ti, tags: provider_adaln_row_gather(r.chunk(6, dim=-1), ti, tags)
                ),
            }
        )
    return cases


def _gather_accuracy(registry: KernelRegistry) -> dict[str, Any]:
    from rl_engine.testing.h3_chain import run_backward_case

    op = registry.get_op("adaln_row_gather", device="cuda")
    golden = registry._get_or_create_backend(registry._priority_map["cpu"]["adaln_row_gather"][-1])
    rows = torch.randn(9, 6 * 5376, device="cuda").bfloat16()
    forward_bitwise = {}
    for seq in (1, 257, 4097, 32768):
        ti, tags = h3_packed_layout(seq, 3, seed=seq)
        ours = op(rows, ti, tags)
        theirs = provider_adaln_row_gather(rows.chunk(6, dim=-1), ti, tags)
        forward_bitwise[str(seq)] = all(
            torch.equal(a, b) for a, b in zip(ours, theirs, strict=True)
        )

    # Op-level backward: correctly rounded FP32 segment sums vs the atomic BF16 scatter-add.
    ti, tags = h3_packed_layout(4097, 3, seed=1)
    grads = [torch.randn(4097, 5376, device="cuda").bfloat16() for _ in range(6)]

    def grad_of(fn):
        leaf = rows.detach().clone().requires_grad_(True)
        torch.autograd.backward(list(fn(leaf)), grads)
        return leaf.grad

    ref = rows.detach().double().requires_grad_(True)
    torch.autograd.backward(list(golden.forward_fp32(ref, ti, tags)), [g.float() for g in grads])
    gold = ref.grad.bfloat16()
    backward = {}
    for name, fn in (
        ("cuda", lambda leaf: op(leaf, ti, tags)),
        ("provider", lambda leaf: provider_adaln_row_gather(leaf.chunk(6, dim=-1), ti, tags)),
    ):
        first, second = grad_of(fn), grad_of(fn)
        backward[name] = {
            "repeat_bitwise_equal": bool(torch.equal(first, second)),
            "correctly_rounded_fraction": float((first == gold).float().mean()),
        }

    result: dict[str, Any] = {
        "forward_bitwise_vs_index_select": forward_bitwise,
        "op_backward": backward,
        "chain_backward": [],
    }
    if h3_weights_dir() is None:
        result["chain_backward_skipped"] = (
            f"{WEIGHTS_ENV} not set; run scripts/prepare_h3_weights.py for chain measurements"
        )
    else:
        weights = load_h3_conditioning_weights("cuda")
        result["chain_backward"] = [
            run_backward_case(registry, weights, num_timesteps=t, seq_len=s)
            for t, s in ((1, 257), (3, 257), (1, 4097), (3, 4097), (4, 32768))
        ]
    return result


# --------------------------------------------------------------------------- #
# h3_rmsnorm (block norm1 + MSA modulation)
# --------------------------------------------------------------------------- #

NORM_SEQ_LENS = (4097, 32768, 131072)


def _norm_inputs(seq: int, seed: int = 0):
    weight = h3_params(["transformer_blocks.0.norm1.weight"], [(5376,)])[0].bfloat16()
    g = torch.Generator(device="cuda").manual_seed(seed)
    table = (torch.randn(9, 6 * 5376, device="cuda", generator=g) * 0.5).bfloat16()
    shift, scale = table.view(9, 6, 5376)[:, 0], table.view(9, 6, 5376)[:, 1]
    ti, tags = h3_packed_layout(seq, 3, seed=seed)
    x = (torch.randn(1, seq, 5376, device="cuda", generator=g) * 2).bfloat16()
    return x, weight, shift, scale, ti * 3 + tags


def _norm_perf(registry: KernelRegistry) -> list[dict[str, Any]]:
    op = registry.get_op("h3_rmsnorm", device="cuda")
    cases = []
    for seq in NORM_SEQ_LENS:
        x, weight, shift, scale, index = _norm_inputs(seq)
        grad = torch.randn_like(x)

        def backward(fn, x=x, weight=weight, shift=shift, scale=scale, grad=grad):
            leaves = [t.detach().requires_grad_(True) for t in (x, weight, shift, scale)]
            fn(*leaves).backward(grad)

        cases.append(
            {
                "op": "h3_rmsnorm",
                "case": f"S={seq}",
                "backend": type(op).__name__,
                "bytes": 2 * seq * 5376 * 2,  # read x, write y
                "candidate": lambda x=x, w=weight, sh=shift, sc=scale, i=index: (
                    op.forward_modulated(x, w, sh, sc, i, check_range=False)
                ),
                "provider": lambda x=x, w=weight, sh=shift, sc=scale, i=index: (
                    provider_norm_modulate(x, w, sh, sc, i)
                ),
                "candidate_backward": lambda b=backward, i=index: b(
                    lambda x_, w_, sh_, sc_: op.forward_modulated(
                        x_, w_, sh_, sc_, i, check_range=False
                    )
                ),
                "provider_backward": lambda b=backward, i=index: b(
                    lambda x_, w_, sh_, sc_: provider_norm_modulate(x_, w_, sh_, sc_, i)
                ),
            }
        )
    return cases


def _norm_accuracy(registry: KernelRegistry) -> dict[str, Any]:
    op = registry.get_op("h3_rmsnorm", device="cuda")
    weight_source = "pinned_checkpoint" if h3_weights_dir() is not None else "synthetic"
    names = [
        "transformer_blocks.0.norm1.weight",
        "transformer_blocks.0.norm2.weight",
        "token_refiner.final_norm.weight",
        "norm_out.norm.weight",
    ]
    weights = {
        name: weight.bfloat16()
        for name, weight in zip(names, h3_params(names, [(5376,)] * len(names)), strict=True)
    }
    plain = {}
    for name in names:
        x = torch.randn(2, 777, 5376, device="cuda").bfloat16()
        ref = torch.nn.functional.rms_norm(x, (5376,), weights[name], 1e-5)
        plain[name] = bool(torch.equal(op(x, weights[name]), ref))
    x, weight, shift, scale, index = _norm_inputs(4097, seed=1)
    modulated = bool(
        torch.equal(
            op.forward_modulated(x, weight, shift, scale, index),
            provider_norm_modulate(x, weight, shift, scale, index),
        )
    )
    invariant = bool(
        torch.equal(
            op.forward_modulated(x[:, 100:140], weight, shift, scale, index[100:140])[0],
            op.forward_modulated(x, weight, shift, scale, index)[0, 100:140],
        )
    )

    # Backward against FP64, for the CUDA op and the diffusers expression.
    grad = torch.randn(
        x.shape, device="cuda", generator=torch.Generator(device="cuda").manual_seed(7)
    ).to(x.dtype)

    def grads(fn, dtype=None):
        tensors = [t if dtype is None else t.to(dtype) for t in (x, weight, shift, scale)]
        leaves = [t.detach().clone().requires_grad_(True) for t in tensors]
        fn(*leaves).backward(grad if dtype is None else grad.to(dtype))
        return [leaf.grad for leaf in leaves]

    def golden(x_, w_, sh_, sc_):
        n = x_ * torch.rsqrt(x_.square().mean(-1, keepdim=True) + 1e-5) * w_
        return n * (1 + sc_.index_select(0, index)) + sh_.index_select(0, index)

    ref = grads(golden, torch.float64)
    backward = {}
    for name, fn in (
        ("cuda", lambda *t: op.forward_modulated(*t, index)),
        ("provider", lambda *t: provider_norm_modulate(*t, index)),
    ):
        first, second = grads(fn), grads(fn)
        backward[name] = {
            "repeat_bitwise_equal": all(
                torch.equal(a, b) for a, b in zip(first, second, strict=True)
            ),
            "rel_error": {
                key: float((g.double() - r).abs().max() / r.abs().max())
                for key, g, r in zip(("dx", "dweight", "dshift", "dscale"), first, ref, strict=True)
            },
        }
    return {
        "weight_source": weight_source,
        "plain_bitwise_vs_nn_rmsnorm": plain,
        "modulated_bitwise_vs_diffusers": modulated,
        "rows_batch_invariant": invariant,
        "backward": backward,
    }


# --------------------------------------------------------------------------- #
# adaln_gate_residual (gate_msa after attention)
# --------------------------------------------------------------------------- #


def _gate_inputs(seq: int, seed: int = 0, dtype=torch.bfloat16):
    g = torch.Generator(device="cuda").manual_seed(seed)
    table = (torch.randn(9, 6 * 5376, device="cuda", generator=g) * 0.5).to(dtype)
    residual = torch.randn(1, seq, 5376, device="cuda", generator=g).to(dtype)
    y = (torch.randn(1, seq, 5376, device="cuda", generator=g) * 3).to(dtype)
    ti, tags = h3_packed_layout(seq, 3, seed=seed)
    return table, residual, y, ti * 3 + tags


def _gate_view(table):
    return table.view(table.shape[0], 6, -1)[:, 2]


def _gate_perf(registry: KernelRegistry) -> list[dict[str, Any]]:
    op = registry.get_op("adaln_gate_residual", device="cuda")
    cases = []
    for seq in NORM_SEQ_LENS:
        table, residual, y, index = _gate_inputs(seq)
        gate = _gate_view(table)
        grad = torch.randn_like(residual)

        def backward(fn, residual=residual, y=y, table=table, grad=grad):
            leaves = [t.detach().requires_grad_(True) for t in (residual, y, table)]
            fn(leaves[0], leaves[1], _gate_view(leaves[2])).backward(grad)

        cases.append(
            {
                "op": "adaln_gate_residual",
                "case": f"S={seq}",
                "backend": type(op).__name__,
                "bytes": 3 * seq * 5376 * 2,  # read residual and y, write out
                "candidate": lambda r=residual, y=y, g=gate, i=index: op.forward(
                    r, y, g, i, check_range=False
                ),
                "provider": lambda r=residual, y=y, g=gate, i=index: provider_gate_residual(
                    r, g, i, y
                ),
                "candidate_backward": lambda b=backward, i=index: b(
                    lambda r, y_, g: op.forward(r, y_, g, i, check_range=False)
                ),
                "provider_backward": lambda b=backward, i=index: b(
                    lambda r, y_, g: provider_gate_residual(r, g, i, y_)
                ),
            }
        )
    return cases


def _gate_accuracy(registry: KernelRegistry) -> dict[str, Any]:
    op = registry.get_op("adaln_gate_residual", device="cuda")
    forward_bitwise = {}
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        table, residual, y, index = _gate_inputs(777, seed=1, dtype=dtype)
        gate = _gate_view(table)
        forward_bitwise[str(dtype).removeprefix("torch.")] = bool(
            torch.equal(
                op(residual, y, gate, index), provider_gate_residual(residual, gate, index, y)
            )
        )
    table, residual, y, index = _gate_inputs(4097, seed=2)
    invariant = bool(
        torch.equal(
            op(residual[:, 10:60], y[:, 10:60], _gate_view(table), index[10:60])[0],
            op(residual, y, _gate_view(table), index)[0, 10:60],
        )
    )
    grad = torch.randn(
        residual.shape, device="cuda", generator=torch.Generator(device="cuda").manual_seed(7)
    ).to(residual.dtype)

    def grads(fn, dtype=None):
        tensors = [t if dtype is None else t.to(dtype) for t in (residual, y, table)]
        leaves = [t.detach().clone().requires_grad_(True) for t in tensors]
        fn(leaves[0], leaves[1], _gate_view(leaves[2])).backward(
            grad if dtype is None else grad.to(dtype)
        )
        return [leaf.grad for leaf in leaves]

    ref = grads(lambda r, y_, g: provider_gate_residual(r, g, index, y_), torch.float64)
    backward = {}
    for name, fn in (
        ("cuda", lambda r, y_, g: op(r, y_, g, index)),
        ("provider", lambda r, y_, g: provider_gate_residual(r, g, index, y_)),
    ):
        first, second = grads(fn), grads(fn)
        backward[name] = {
            "repeat_bitwise_equal": all(
                torch.equal(a, b) for a, b in zip(first, second, strict=True)
            ),
            "rel_error": {
                key: float((g.double() - r).abs().max() / r.abs().max())
                for key, g, r in zip(
                    ("d_residual", "d_sublayer", "d_gate"), first, ref, strict=True
                )
            },
        }
    return {
        "forward_bitwise_vs_diffusers": forward_bitwise,
        "rows_batch_invariant": invariant,
        "backward": backward,
    }


PERF_CASES: dict[str, Callable[[KernelRegistry], list[dict[str, Any]]]] = {
    "timestep_sinusoid_h3": _sinusoid_perf,
    "timestep_mlp_fp32": _mlp_perf,
    "adaln_projection_3mod": _projection_perf,
    "adaln_row_gather": _gather_perf,
    "h3_rmsnorm": _norm_perf,
    "adaln_gate_residual": _gate_perf,
}
ACCURACY: dict[str, Callable[[KernelRegistry], dict[str, Any]]] = {
    "timestep_sinusoid_h3": _sinusoid_accuracy,
    "timestep_mlp_fp32": _mlp_accuracy,
    "adaln_projection_3mod": _projection_accuracy,
    "adaln_row_gather": _gather_accuracy,
    "h3_rmsnorm": _norm_accuracy,
    "adaln_gate_residual": _gate_accuracy,
}
