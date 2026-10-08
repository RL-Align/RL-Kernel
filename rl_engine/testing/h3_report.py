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
    provider_final_adaln_out,
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


def _prepared_call(
    fn: Callable[..., Any], setup: Callable[[], Any] | None = None
) -> Callable[[], Any]:
    if setup is None:
        return fn
    prepared = setup()
    return lambda: fn(prepared)


def _sample_us(fn: Callable[..., Any], setup: Callable[[], Any] | None = None) -> float:
    run = _prepared_call(fn, setup)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    run()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1e3


def time_us(
    fn: Callable[..., Any],
    warmup: int = 20,
    iters: int = 200,
    *,
    setup: Callable[[], Any] | None = None,
) -> float:
    """Median CUDA-event microseconds; optional setup runs outside the timed region.

    When supplied, ``setup`` runs before every call and its return value is passed
    to ``fn``. Backward measurements use it to build a fresh forward graph.
    """

    for _ in range(warmup):
        _prepared_call(fn, setup)()
    torch.cuda.synchronize()
    return statistics.median(_sample_us(fn, setup) for _ in range(iters))


def peak_mib(fn: Callable[..., Any], *, setup: Callable[[], Any] | None = None) -> float:
    """Incremental peak CUDA memory of the call, excluding optional setup."""

    run = _prepared_call(fn, setup)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    run()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20


def measure(case: dict[str, Any], warmup: int = 20, iters: int = 200) -> dict[str, Any]:
    """Interleave timed keys, reversing their execution order on alternate iterations.

    Record raw CUDA-event samples, executed orders, median microseconds,
    decimal GB/s from ``case['bytes']``, and peak allocated MiB per callable.
    """

    row = {key: case[key] for key in ("op", "case", "backend", "bytes") if key in case}
    keys = [key for key in TIMED_KEYS if key in case]
    if not keys:
        return row
    orders = (keys, list(reversed(keys)))
    row["execution_order"] = {
        "policy": "alternating",
        "iteration_0": orders[0],
        "iteration_1": orders[1],
    }
    if any(key.endswith("_backward") for key in keys):
        row["backward_timing_scope"] = "backward_only"
    for iteration in range(warmup):
        for key in orders[iteration % 2]:
            _prepared_call(case[key], case.get(f"{key}_setup"))()
    torch.cuda.synchronize()
    samples = {key: [] for key in keys}
    for iteration in range(iters):
        for key in orders[iteration % 2]:
            samples[key].append(_sample_us(case[key], case.get(f"{key}_setup")))
    row["timing_samples_us"] = samples
    row["timing_order"] = [orders[iteration % 2] for iteration in range(iters)]
    for key in keys:
        us = statistics.median(samples[key])
        row[f"{key}_us"] = us
        row[f"{key}_gbps"] = case["bytes"] / (us * 1e-6) / 1e9
        row[f"{key}_peak_mib"] = peak_mib(case[key], setup=case.get(f"{key}_setup"))
    return row


# --------------------------------------------------------------------------- #
# timestep_sinusoid_h3
# --------------------------------------------------------------------------- #


def _sinusoid_perf(registry: KernelRegistry) -> list[dict[str, Any]]:
    """Build CUDA sinusoid timing cases with checked, unchecked, and provider calls."""

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
    """Compare sinusoid outputs to provider and FP64 golden across timestep counts."""

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
    """Return CUDA checkpoint parameters, or seeded random tensors when weights are unset."""

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
    """Build CUDA MLP candidate/provider timing cases for one to four timesteps."""

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
    """Measure per-draw MLP error against FP64 and test single-row/batched equality."""

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
    """Return block-0 AdaLN projection weights and bias as BF16 CUDA tensors."""

    weight, bias = h3_params(ADALN_NAMES, [(96768, 2688), (96768,)])
    return weight.bfloat16(), bias.bfloat16()


def _projection_perf(registry: KernelRegistry) -> list[dict[str, Any]]:
    """Build CUDA AdaLN candidate/provider timing cases for one to four timesteps."""

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
    """Concatenate flattened modulation tensors in their output order."""

    return torch.cat([out.reshape(-1) for out in outputs])


def _projection_accuracy(registry: KernelRegistry, draws: int = 20) -> dict[str, Any]:
    """Report per-draw BF16 equality to FP64 goldens and single-row/batch invariance."""

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

        def prepare_backward(fn, ti=ti, tags=tags):
            leaf = rows.detach().requires_grad_(True)
            return list(fn(leaf, ti, tags))

        def backward(outputs, grads=grads):
            torch.autograd.backward(outputs, grads)

        cases.append(
            {
                "op": "adaln_row_gather",
                "case": f"S={seq}",
                "backend": type(op).__name__,
                # bytes written (six (S, H) BF16 outputs); the 3T-row table stays in L2
                "bytes": 6 * seq * hidden * 2,
                "candidate": lambda ti=ti, tags=tags: op.forward(rows, ti, tags, check_range=False),
                "provider": lambda ti=ti, tags=tags: provider_adaln_row_gather(chunks, ti, tags),
                "candidate_backward": backward,
                "provider_backward": backward,
                "candidate_backward_setup": lambda b=prepare_backward: b(
                    lambda r, indices, row_tags: op.forward(r, indices, row_tags, check_range=False)
                ),
                "provider_backward_setup": lambda b=prepare_backward: b(
                    lambda r, indices, row_tags: provider_adaln_row_gather(
                        r.chunk(6, dim=-1), indices, row_tags
                    )
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

        def prepare_backward(fn, x=x, weight=weight, shift=shift, scale=scale):
            leaves = [t.detach().requires_grad_(True) for t in (x, weight, shift, scale)]
            return fn(*leaves)

        def backward(output, grad=grad):
            output.backward(grad)

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
                "candidate_backward": backward,
                "provider_backward": backward,
                "candidate_backward_setup": lambda b=prepare_backward, i=index: b(
                    lambda x_, w_, sh_, sc_: op.forward_modulated(
                        x_, w_, sh_, sc_, i, check_range=False
                    )
                ),
                "provider_backward_setup": lambda b=prepare_backward, i=index: b(
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


# --------------------------------------------------------------------------- #
# final_adaln_out (norm_out)
# --------------------------------------------------------------------------- #

FINAL_NAMES = ["norm_out.norm.weight", "norm_out.linear.weight", "norm_out.linear.bias"]


def _final_inputs(seq: int, seed: int = 0):
    norm_weight, weight, bias = h3_params(FINAL_NAMES, [(5376,), (10752, 2688), (10752,)])
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(1, seq, 5376, device="cuda", generator=g) * 2).bfloat16()
    temb = torch.randn(3, 2688, device="cuda", generator=g) * 2
    ti, _ = h3_packed_layout(seq, 3, seed=seed)
    return x, norm_weight.bfloat16(), temb, weight.bfloat16(), bias.bfloat16(), ti


def _final_perf(registry: KernelRegistry) -> list[dict[str, Any]]:
    op = registry.get_op("final_adaln_out", device="cuda")
    cases = []
    for seq in NORM_SEQ_LENS:
        x, nw, temb, w, b, ti = _final_inputs(seq)
        grad = torch.randn_like(x)

        def backward(fn, tensors=(x, nw, temb, w, b), grad=grad):
            leaves = [t.detach().requires_grad_(True) for t in tensors]
            fn(*leaves).backward(grad)

        cases.append(
            {
                "op": "final_adaln_out",
                "case": f"S={seq}",
                "backend": type(op).__name__,
                # read x + norm_out.linear, write out
                "bytes": 2 * seq * 5376 * 2 + w.numel() * 2,
                "candidate": lambda t=(x, nw, temb, w, b), i=ti: op(*t, i),
                "provider": lambda t=(x, nw, temb, w, b), i=ti: provider_final_adaln_out(*t, i),
                "candidate_backward": lambda bw=backward, i=ti: bw(lambda *t: op(*t, i)),
                "provider_backward": lambda bw=backward, i=ti: bw(
                    lambda *t: provider_final_adaln_out(*t, i)
                ),
            }
        )
    return cases


def _final_accuracy(registry: KernelRegistry) -> dict[str, Any]:
    op = registry.get_op("final_adaln_out", device="cuda")
    golden_op = registry._get_or_create_backend(
        registry._priority_map["cpu"]["final_adaln_out"][-1]
    )
    x, nw, temb, w, b, ti = _final_inputs(4097, seed=1)
    ours = op(x, nw, temb, w, b, ti)
    theirs = provider_final_adaln_out(x, nw, temb, w, b, ti)
    golden = golden_op.forward_fp32(x, nw, temb, w, b, ti)
    invariant = bool(
        torch.equal(op(x[:, 100:160], nw, temb, w, b, ti[100:160])[0], ours[0, 100:160])
    )
    grad = torch.randn(
        x.shape, device="cuda", generator=torch.Generator(device="cuda").manual_seed(7)
    ).to(x.dtype)

    def grads(fn, dtype=None):
        tensors = [t if dtype is None else t.to(dtype) for t in (x, nw, temb, w, b)]
        leaves = [t.detach().clone().requires_grad_(True) for t in tensors]
        fn(*leaves).backward(grad if dtype is None else grad.to(dtype))
        return [leaf.grad for leaf in leaves]

    def golden_backward(x_, nw_, t_, w_, b_):
        # Keep the declared BF16 boundaries with FP64 leaves and identity VJPs.
        act = t_ * torch.sigmoid(t_)
        act = act + (act.to(torch.bfloat16).double() - act).detach()
        table = torch.nn.functional.linear(act, w_, b_)
        table = table + (table.to(torch.bfloat16).double() - table).detach()
        shift, scale = table.chunk(2, dim=-1)
        n = x_ * torch.rsqrt(x_.square().mean(-1, keepdim=True) + 1e-5) * nw_
        return n * (1.0 + scale.index_select(0, ti)) + shift.index_select(0, ti)

    ref = grads(golden_backward, torch.float64)
    backward = {}
    for name, fn in (
        ("cuda", lambda *t: op(*t, ti)),
        ("provider", lambda *t: provider_final_adaln_out(*t, ti)),
    ):
        first, second = grads(fn), grads(fn)
        backward[name] = {
            "repeat_bitwise_equal": all(torch.equal(a, c) for a, c in zip(first, second)),
            "rel_error": {
                key: float((g.double() - r).abs().max() / r.abs().max())
                for key, g, r in zip(("dx", "d_norm_w", "d_temb", "dW", "db"), first, ref)
            },
        }
    return {
        "equal_to_diffusers_fraction": float((ours == theirs).float().mean()),
        "max_abs_vs_golden": float((ours.float() - golden).abs().max()),
        "provider_max_abs_vs_golden": float((theirs.float() - golden).abs().max()),
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
    "final_adaln_out": _final_perf,
}
ACCURACY: dict[str, Callable[[KernelRegistry], dict[str, Any]]] = {
    "timestep_sinusoid_h3": _sinusoid_accuracy,
    "timestep_mlp_fp32": _mlp_accuracy,
    "adaln_projection_3mod": _projection_accuracy,
    "adaln_row_gather": _gather_accuracy,
    "h3_rmsnorm": _norm_accuracy,
    "adaln_gate_residual": _gate_accuracy,
    "final_adaln_out": _final_accuracy,
}
