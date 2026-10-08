#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Compare one RFC #420 operator with existing implementations: batch invariance, accuracy, latency.

The RFC #420 reuse rule asks, for every operator, whether an existing implementation is
batch-invariant (then reuse it) and for accuracy and performance comparisons. This runner
measures the operator's rl-kernel CUDA op next to the existing implementations that import in
the current environment, and writes one JSON report per operator:

    python scripts/h3_prior_art.py --op adaln_row_gather \\
        --out docs/usage/evidence/h3-prior-art-b200/adaln_row_gather.json \\
        [--megatron-src /path/to/Megatron-LM] [--quick]
    python scripts/plot_h3_prior_art.py docs/usage/evidence/h3-prior-art-b200/adaln_row_gather.json

Only operators whose rl-kernel op exists on the current branch can be selected. Optional
libraries (SGLang, Liger, Transformer Engine, vLLM, Megatron-LM from a source tree) are skipped
and recorded as unavailable when they do not import. Global batch-invariant modes (vLLM,
SGLang, Megatron) patch aten for the whole process, so each runs in its own subprocess, with
the environment variables it relies on set before CUDA initialises.

Batch invariance is bitwise, for the forward, the per-row gradients, and repeatability of every
parameter and table gradient, with three checks:

1. every row computed alone vs inside full batches of 64, 257 and 2048 rows, seeds 3-5;
2. the full workload-size batch vs sub-batches that together cover every row;
3. eight probe rows at the front, middle and back of batches of every size 1..9 and
   2^k - 1, 2^k, 2^k + 1, plus the whole batch reversed.

Sparse probes (a few rows, a few batch sizes) miss row-specific or large-batch dependence, which
is why every row is checked. Accuracy is max|err| / max|ref| against the same computation in
FP64; latency is the median of 100 CUDA-event samples after 20 warm-ups. Run on a clean tree on
an otherwise idle GPU; the report records the commit, the tree state and every library version.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata as md
import importlib.util
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

H, D, F1, T3, EPS = 5376, 2688, 256, 3, 1e-5

#: operator -> rl-kernel module that must exist on the branch
OPS = {
    "timestep_sinusoid": "rl_engine.kernels.ops.cuda.h3.timestep_sinusoid",
    "timestep_mlp": "rl_engine.kernels.ops.cuda.h3.timestep_mlp",
    "adaln_projection": "rl_engine.kernels.ops.cuda.h3.adaln_projection",
    "adaln_row_gather": "rl_engine.kernels.ops.cuda.h3.adaln_row_gather",
    "norm_modulate": "rl_engine.kernels.ops.cuda.h3.rmsnorm",
    "gate_residual": "rl_engine.kernels.ops.cuda.h3.gate_residual",
    "final_adaln_out": "rl_engine.kernels.ops.cuda.h3.final_adaln_out",
}
#: global batch-invariant modes compared for the GEMM operators, with their environment
MODES = {
    "plain": {},
    "vllm": {"CUBLAS_WORKSPACE_CONFIG": ":16:8", "CUBLASLT_WORKSPACE_SIZE": "1"},
    "sglang": {},
    "sglang_ieee": {"TRITON_F32_DEFAULT": "ieee"},
    "megatron_te_native": {"CUBLASLT_WORKSPACE_SIZE": "0"},
    "megatron_triton": {},
    "megatron_triton_ieee": {"TRITON_F32_DEFAULT": "ieee"},
}
GEMM_OPS = ("timestep_mlp", "adaln_projection")
LIBS = ("torch", "triton", "diffusers", "vllm", "flashinfer-python", "sglang", "liger-kernel")
LIBS += ("transformer_engine", "megatron-core")

DEV = "cuda"


# --------------------------------------------------------------------------- #
# Batch-invariance checks
# --------------------------------------------------------------------------- #


class _Case:
    """``fn(**inputs)`` with row arguments sliced together; fixed upstream gradients per seed."""

    def __init__(self, fn, inputs, rows, grads, out_dims, seed):
        self.fn, self.inputs, self.rows, self.grads, self.out_dims = (
            fn,
            inputs,
            rows,
            grads,
            out_dims,
        )
        self.n = next(inputs[k].shape[d] for k, d in rows.items())
        outs = self._flat(fn(**inputs))
        gen = torch.Generator(device=DEV).manual_seed(seed)
        self.ups = [torch.randn(o.shape, device=DEV, generator=gen).to(o.dtype) for o in outs]

    @staticmethod
    def _flat(out):
        return list(out) if isinstance(out, (tuple, list)) else [out]

    def run(self, idx=None):
        args = {}
        for k, v in self.inputs.items():
            if idx is not None and k in self.rows:
                v = v.index_select(self.rows[k], idx)
            if k in self.grads:
                v = v.detach().clone().requires_grad_(True)
            args[k] = v
        outs = self._flat(self.fn(**args))
        ups = (
            self.ups
            if idx is None
            else [u.index_select(d, idx) for u, d in zip(self.ups, self.out_dims)]
        )
        live = [(o, u) for o, u in zip(outs, ups) if o.requires_grad]
        if live:
            torch.autograd.backward([o for o, _ in live], [u for _, u in live])
        grads = {k: args[k].grad for k in self.grads}
        return [o.detach() for o in outs], grads


def _sweep_sizes(n: int) -> list[int]:
    sizes = set(range(1, min(n, 9) + 1))
    k = 16
    while k <= n:
        sizes.update(v for v in (k - 1, k, k + 1) if v <= n)
        k *= 2
    return sorted(sizes | {n})


def batch_invariance(cand: dict[str, Any], quick: bool) -> dict[str, Any]:
    fn, make, rows, grads = cand["fn"], cand["make"], cand["rows"], cand["grads"]
    out_dims = cand.get("out_dims", (0,))
    params = [k for k in grads if k not in rows]
    full_sizes, big, subs = cand["bi_sizes"]
    if quick:
        full_sizes, big, subs = (17, 64), 256, (1, 7, 64)
    seeds = (3, 4, 5)
    result: dict[str, Any] = {"all_rows": {}, "params_repeatable": None}
    ok = True

    def same_row(a, b, d, i):
        return torch.equal(a.select(d, 0), b.select(d, i))

    for n in full_sizes:
        fwd = grd = 0
        repeat = True
        for seed in seeds:
            case = _Case(fn, make(seed, n), rows, grads, out_dims, seed)
            fo, fg = case.run()
            for i in range(n):
                o, g = case.run(torch.tensor([i], device=DEV))
                fwd += not all(same_row(a, b, d, i) for a, b, d in zip(o, fo, out_dims))
                grd += not all(
                    same_row(g[k], fg[k], rows[k], i)
                    for k in rows
                    if k in grads and g[k] is not None
                )
            if params:
                _, again = case.run()
                repeat &= all(torch.equal(again[k], fg[k]) for k in params if fg[k] is not None)
        result["all_rows"][str(n)] = {
            "rows": len(seeds) * n,
            "fwd_differ": fwd,
            "rowgrad_differ": grd,
        }
        if params:
            result["params_repeatable"] = repeat and result["params_repeatable"] is not False
        ok &= fwd == 0 and grd == 0 and repeat

    fwd = grd = checked = 0
    for seed in seeds[:2]:
        case = _Case(fn, make(seed, big), rows, grads, out_dims, seed)
        fo, fg = case.run()
        for sb in subs:
            step = sb if sb >= 1024 else max(sb, big // 512)
            for start in range(0, big, step):
                idx = torch.arange(start, min(start + sb, big), device=DEV)
                o, g = case.run(idx)
                checked += 1
                fwd += not all(
                    torch.equal(a, b.index_select(d, idx)) for a, b, d in zip(o, fo, out_dims)
                )
                grd += not all(
                    torch.equal(g[k], fg[k].index_select(rows[k], idx))
                    for k in rows
                    if k in grads and g[k] is not None
                )
        del case, fo, fg
        torch.cuda.empty_cache()
    result["full_vs_sub_batches"] = {
        "full_batch": big,
        "sub_batch_sizes": list(subs),
        "sub_batches": checked,
        "fwd_differ": fwd,
        "rowgrad_differ": grd,
    }
    ok &= fwd == 0 and grd == 0

    n = full_sizes[-1]
    sizes = _sweep_sizes(n)
    probes = sorted({0, n - 1, *list(range(n // 8 + 3, n - 1, n // 8))[:6]})
    compared = failed = 0
    first: list[dict[str, int]] = []
    for seed in seeds:
        case = _Case(fn, make(seed, n), rows, grads, out_dims, seed)
        alone = {r: case.run(torch.tensor([r], device=DEV)) for r in probes}
        gen = torch.Generator().manual_seed(seed)

        def compare(idx, pos, r, seed=seed, case=case, alone=alone):
            nonlocal compared, failed
            o, g = case.run(idx.to(DEV))
            ao, ag = alone[r]
            good = all(
                torch.equal(a.select(d, pos), b.select(d, 0)) for a, b, d in zip(o, ao, out_dims)
            )
            good &= all(
                torch.equal(g[k].select(rows[k], pos), ag[k].select(rows[k], 0))
                for k in rows
                if k in grads and g[k] is not None
            )
            compared += 1
            if not good:
                failed += 1
                if len(first) < 8:
                    first.append(
                        {"seed": seed, "batch": int(idx.numel()), "position": pos, "row": r}
                    )

        for m in sizes:
            for r in probes:
                others = torch.randperm(n, generator=gen)
                others = others[others != r][: m - 1]
                for pos in sorted({0, (m - 1) // 2, m - 1}):
                    compare(torch.cat([others[:pos], torch.tensor([r]), others[pos:]]), pos, r)
        for r in probes:
            compare(torch.arange(n - 1, -1, -1), n - 1 - r, r)
    result["batch_size_sweep"] = {
        "batch_sizes": sizes,
        "probe_rows": probes,
        "comparisons": compared,
        "differ": failed,
        "first_failures": first,
    }
    ok &= failed == 0
    result["batch_invariant"] = bool(ok)
    return result


# --------------------------------------------------------------------------- #
# Accuracy and latency
# --------------------------------------------------------------------------- #


def _time_us(fn: Callable[[], Any], warmup: int = 20, iters: int = 100) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3)
    return statistics.median(samples)


def _rel(a: torch.Tensor, ref: torch.Tensor) -> float:
    return ((a.double() - ref).abs().max() / ref.abs().max().clamp_min(1e-300)).item()


def accuracy_latency(cand: dict[str, Any], mode_off, quick: bool) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for size in cand["perf_sizes"][:2] if quick else cand["perf_sizes"]:
        inputs = cand["make"](11, size)
        names = list(inputs)
        grads = [k for k in cand["grads"] if k in names]
        leaves = {
            k: (v.detach().clone().requires_grad_(True) if k in grads else v)
            for k, v in inputs.items()
        }
        y = cand["fn"](**leaves)
        y = torch.cat([t.reshape(-1) for t in y]) if isinstance(y, (tuple, list)) else y
        with mode_off():
            ref_in = {
                k: (v.detach().double().requires_grad_(k in grads) if v.is_floating_point() else v)
                for k, v in inputs.items()
            }
            ref = cand["ref"](**ref_in)
            ref = torch.cat([t.reshape(-1) for t in ref]) if isinstance(ref, (tuple, list)) else ref
        row: dict[str, Any] = {"fwd_rel_err": _rel(y, ref.detach())}
        if grads:
            gen = torch.Generator(device=DEV).manual_seed(9)
            up = torch.randn(y.shape, device=DEV, generator=gen).to(y.dtype)
            y.backward(up)
            with mode_off():
                ref.backward(up.double())
            row["grad_rel_err"] = {k: _rel(leaves[k].grad, ref_in[k].grad) for k in grads}
        row["fwd_us"] = _time_us(lambda inputs=inputs: cand["fn"](**inputs))
        if grads:

            def step(inputs=inputs):
                lv = {
                    k: (v.detach().clone().requires_grad_(True) if k in grads else v)
                    for k, v in inputs.items()
                }
                o = cand["fn"](**lv)
                outs = list(o) if isinstance(o, (tuple, list)) else [o]
                torch.autograd.backward(outs, [torch.ones_like(t) for t in outs])

            row["fwd_bwd_us"] = _time_us(step)
        out[str(size)] = row
        del inputs, leaves, y, ref
        torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------- #
# Candidates
# --------------------------------------------------------------------------- #


def _gen(seed: int, salt: int) -> torch.Generator:
    return torch.Generator(device=DEV).manual_seed(seed * 1000 + salt)


def _rn(seed, salt, *shape, dtype=torch.float32, scale=1.0):
    return (torch.randn(*shape, device=DEV, generator=_gen(seed, salt)) * scale).to(dtype)


def _layout(n: int, seed: int):
    from rl_engine.testing.h3_cases import h3_packed_layout

    ti, tags = h3_packed_layout(n, T3, seed=seed)
    return ti.contiguous(), tags.contiguous()


def _version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


T_BI = ((64, 257, 2048), 4096, (1, 7, 1024, 2048))
S_BI = ((64, 257, 2048), 131072, (1, 7, 4097, 32768, 65536))
T_PERF = (1, 3, 4, 64, 256, 2048)
S_PERF = (4097, 32768)


def _sinusoid_ref(t):
    half = F1 // 2
    exponent = -torch.log(torch.tensor(10000.0, dtype=torch.float64, device=DEV))
    freqs = torch.exp(exponent * torch.arange(half, device=DEV, dtype=torch.float64) / half)
    e = t.double()[:, None] * freqs[None]
    return torch.cat([torch.cos(e), torch.sin(e)], -1)


def candidates(op: str, mode: str) -> list[dict[str, Any]]:
    """Candidate dicts; a factory that raises (missing library) becomes an unavailable entry."""

    from rl_engine.testing import h3_provider as P

    plain = mode == "plain"
    found: list[tuple[str, Callable[[], dict[str, Any]]]] = []

    if op == "timestep_sinusoid":
        base = {
            "make": lambda s, n: {"t": torch.rand(n, device=DEV, generator=_gen(s, 1))},
            "rows": {"t": 0},
            "grads": [],
            "ref": _sinusoid_ref,
            "bi_sizes": ((64, 257, 2048), 8192, (1, 7, 2048, 4097)),
            "perf_sizes": T_PERF,
        }

        def diffusers():
            return {
                **base,
                "source": "diffusers get_timestep_embedding (op-for-op replay)",
                "fn": lambda t: P.provider_time_proj(t),
            }

        def sglang():
            from sglang.kernels.ops.diffusion.modulate.timestep_embedding_jit import (
                timestep_embedding,
            )

            return {
                **base,
                "source": f"SGLang {_version('sglang')} timestep_embedding",
                "fn": lambda t: timestep_embedding(
                    t, F1, flip_sin_to_cos=True, downscale_freq_shift=0.0
                ),
            }

        def ours():
            from rl_engine.kernels.ops.cuda.h3.timestep_sinusoid import H3TimestepSinusoidCudaOp

            op_ = H3TimestepSinusoidCudaOp()
            return {
                **base,
                "source": "rl-kernel H3TimestepSinusoidCudaOp (check_range=False)",
                "fn": lambda t: op_.forward(t, check_range=False),
            }

        found = [("diffusers", diffusers), ("sglang", sglang), ("rl_kernel", ours)]

    elif op == "timestep_mlp":

        def make(s, n):
            return {
                "x": _rn(s, 2, n, F1),
                "w1": _rn(1, 3, H, F1, scale=F1**-0.5),
                "b1": _rn(1, 4, H, scale=0.02),
                "w2": _rn(1, 5, D, H, scale=H**-0.5),
                "b2": _rn(1, 6, D, scale=0.02),
            }

        base = {
            "make": make,
            "rows": {"x": 0},
            "grads": ["x", "w1", "b1", "w2", "b2"],
            "ref": lambda x, w1, b1, w2, b2: F.linear(F.silu(F.linear(x, w1, b1)), w2, b2),
            "bi_sizes": T_BI,
            "perf_sizes": T_PERF,
        }

        def diffusers():
            return {
                **base,
                "source": f"diffusers TimestepEmbedding, FP32 F.linear [{mode}]",
                "fn": lambda x, w1, b1, w2, b2: P.provider_time_embedder(x, w1, b1, w2, b2),
            }

        def ours():
            from rl_engine.kernels.ops.cuda.h3.timestep_mlp import H3TimestepMLPCudaOp

            op_ = H3TimestepMLPCudaOp()
            return {
                **base,
                "source": "rl-kernel H3TimestepMLPCudaOp",
                "fn": lambda x, w1, b1, w2, b2: op_(x, w1, b1, w2, b2),
            }

        found = [(f"diffusers[{mode}]", diffusers)] + ([("rl_kernel", ours)] if plain else [])

    elif op == "adaln_projection":

        def make(s, n):
            return {
                "temb": _rn(s, 7, n, D),
                "w": _rn(1, 8, 18 * H, D, dtype=torch.bfloat16, scale=D**-0.5),
                "b": _rn(1, 9, 18 * H, dtype=torch.bfloat16, scale=0.02),
            }

        base = {
            "make": make,
            "rows": {"temb": 0},
            "grads": ["temb", "w", "b"],
            "ref": lambda temb, w, b: F.linear(F.silu(temb), w, b),
            "bi_sizes": T_BI,
            "perf_sizes": T_PERF,
        }

        def diffusers():
            return {
                **base,
                "source": f"diffusers AdaLN projection, BF16 F.linear [{mode}]",
                "fn": lambda temb, w, b: F.linear(F.silu(temb).to(w.dtype), w, b),
            }

        def ours():
            from rl_engine.kernels.ops.cuda.h3.adaln_projection import H3AdaLNProjectionCudaOp

            op_ = H3AdaLNProjectionCudaOp()
            return {
                **base,
                "source": "rl-kernel H3AdaLNProjectionCudaOp",
                "fn": lambda temb, w, b: op_.forward_table(temb, w, b),
            }

        found = [(f"diffusers[{mode}]", diffusers)] + ([("rl_kernel", ours)] if plain else [])

    elif op == "adaln_row_gather":

        def make(s, n):
            ti, tags = _layout(n, s)
            return {
                "table": _rn(1, 14, 3 * T3, 6 * H, dtype=torch.bfloat16, scale=0.1),
                "ti": ti,
                "tags": tags,
            }

        def ref(table, ti, tags):
            return tuple(table.index_select(0, ti * 3 + tags).chunk(6, dim=1))

        base = {
            "make": make,
            "rows": {"ti": 0, "tags": 0},
            "grads": ["table"],
            "out_dims": (0,) * 6,
            "ref": ref,
            "bi_sizes": S_BI,
            "perf_sizes": S_PERF,
        }

        def diffusers():
            return {**base, "source": "diffusers six index_select calls", "fn": ref}

        def ours():
            from rl_engine.kernels.ops.cuda.h3.adaln_row_gather import H3AdaLNRowGatherCudaOp

            op_ = H3AdaLNRowGatherCudaOp()
            return {
                **base,
                "source": "rl-kernel H3AdaLNRowGatherCudaOp",
                "fn": lambda table, ti, tags: op_(table, ti, tags),
            }

        found = [("diffusers", diffusers), ("rl_kernel", ours)]

    elif op == "norm_modulate":

        def make(s, n):
            ti, tags = _layout(n, s)
            w = (0.5 + torch.rand(H, device=DEV, generator=_gen(1, 11))).bfloat16()
            return {
                "x": _rn(s, 10, n, H, dtype=torch.bfloat16, scale=2.0),
                "w": w,
                "sh": _rn(1, 12, 3 * T3, H, dtype=torch.bfloat16, scale=0.1),
                "sc": _rn(1, 13, 3 * T3, H, dtype=torch.bfloat16, scale=0.1),
                "idx": ti * 3 + tags,
            }

        def ref(x, w, sh, sc, idx):
            n = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w
            return n * (1 + sc.index_select(0, idx)) + sh.index_select(0, idx)

        base = {
            "make": make,
            "rows": {"x": 0, "idx": 0},
            "grads": ["x", "w", "sh", "sc"],
            "ref": ref,
            "bi_sizes": S_BI,
            "perf_sizes": S_PERF,
        }

        def per_token(fn):
            # Liger / SGLang take per-token shift/scale rows: gather them first (index_select).
            return lambda x, w, sh, sc, idx: fn(
                x, w, sh.index_select(0, idx), sc.index_select(0, idx)
            )

        def diffusers():
            return {
                **base,
                "source": "diffusers composition (F.rms_norm + index_select modulation)",
                "fn": lambda x, w, sh, sc, idx: P.provider_norm_modulate(x, w, sh, sc, idx),
            }

        def liger():
            from liger_kernel.ops.modulated_rms_norm import LigerModulatedRMSNormFunction

            return {
                **base,
                "source": f"Liger {_version('liger-kernel')} modulated RMSNorm + index_select",
                "fn": per_token(
                    lambda x, w, shr, scr: LigerModulatedRMSNormFunction.apply(
                        x, w, scr, shr, EPS, 0.0, "llama", False
                    )
                ),
            }

        def liger_rlk_gather():
            # Same Liger kernel, with the rows gathered by rl-kernel's deterministic row gather
            # (adaln_row_gather) instead of index_select: separates Liger from the gather.
            from liger_kernel.ops.modulated_rms_norm import LigerModulatedRMSNormFunction

            from rl_engine.kernels.ops.cuda.h3.adaln_row_gather import H3AdaLNRowGatherCudaOp

            gather = H3AdaLNRowGatherCudaOp()

            def fn(x, w, sh, sc, idx):
                pad = torch.zeros_like(sh)
                table = torch.cat([sh, sc, pad, pad, pad, pad], dim=1)
                rows = gather(table, torch.div(idx, 3, rounding_mode="floor"), idx % 3)
                return LigerModulatedRMSNormFunction.apply(
                    x, w, rows[1], rows[0], EPS, 0.0, "llama", False
                )

            version = _version("liger-kernel")
            return {
                **base,
                "source": f"Liger {version} modulated RMSNorm + rl-kernel row gather",
                "fn": fn,
            }

        def sglang():
            from sglang.kernels.ops.diffusion.norm.scale_residual_norm_cutedsl import (
                fused_norm_scale_shift,
            )

            return {
                **base,
                "grads": [],
                "source": f"SGLang {_version('sglang')} fused_norm_scale_shift (forward only)",
                "fn": per_token(
                    lambda x, w, shr, scr: fused_norm_scale_shift(
                        x[None], w, None, scr[None], shr[None], "rms", EPS
                    )[0]
                ),
            }

        def ours():
            from rl_engine.kernels.ops.cuda.h3.rmsnorm import H3RMSNormCudaOp

            op_ = H3RMSNormCudaOp()
            return {
                **base,
                "source": "rl-kernel H3RMSNormCudaOp.forward_modulated",
                "fn": lambda x, w, sh, sc, idx: op_.forward_modulated(x, w, sh, sc, idx),
            }

        def norm_only(make):
            def m(s, n):
                d = make(s, n)
                return {"x": d["x"], "w": d["w"]}

            return m

        def norm_ref(x, w):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w

        def torch_norm():
            # diffusers' norm without the modulation, to show which half is batch-invariant
            return {
                **base,
                "make": norm_only(make),
                "rows": {"x": 0},
                "grads": ["x", "w"],
                "ref": norm_ref,
                "source": "torch F.rms_norm (no modulation)",
                "fn": lambda x, w: F.rms_norm(x, (H,), w, EPS),
            }

        def te_norm():
            import transformer_engine.pytorch as te

            mod = te.RMSNorm(H, eps=EPS, params_dtype=torch.bfloat16, device=DEV)

            def fn(x, w):
                # TE's backward reads its own parameter, so copy w in (functional_call
                # gives a wrong dx); dweight is then TE's parameter gradient, not checked.
                with torch.no_grad():
                    mod.weight.copy_(w)
                return mod(x)

            return {
                **base,
                "make": norm_only(make),
                "rows": {"x": 0},
                "grads": ["x"],
                "ref": norm_ref,
                "source": f"TE {_version('transformer_engine')} RMSNorm (no modulation)",
                "fn": fn,
            }

        found = [
            ("diffusers", diffusers),
            ("torch_rms_norm", torch_norm),
            ("transformer_engine", te_norm),
            ("liger", liger),
            ("liger_rlk_gather", liger_rlk_gather),
            ("sglang", sglang),
            ("rl_kernel", ours),
        ]

    elif op == "gate_residual":

        def make(s, n):
            ti, tags = _layout(n, s)
            return {
                "res": _rn(s, 15, n, H, dtype=torch.bfloat16),
                "y": _rn(s, 16, n, H, dtype=torch.bfloat16),
                "gate": _rn(1, 17, 3 * T3, H, dtype=torch.bfloat16, scale=0.1),
                "idx": ti * 3 + tags,
            }

        def ref(res, y, gate, idx):
            return res + gate.index_select(0, idx) * y

        base = {
            "make": make,
            "rows": {"res": 0, "y": 0, "idx": 0},
            "grads": ["res", "y", "gate"],
            "ref": ref,
            "bi_sizes": S_BI,
            "perf_sizes": S_PERF,
        }

        def diffusers():
            return {
                **base,
                "source": "diffusers residual + gate.index_select(...) * y",
                "fn": lambda res, y, gate, idx: P.provider_gate_residual(res, gate, idx, y),
            }

        def ours():
            from rl_engine.kernels.ops.cuda.h3.gate_residual import H3GateResidualCudaOp

            op_ = H3GateResidualCudaOp()
            return {
                **base,
                "source": "rl-kernel H3GateResidualCudaOp",
                "fn": lambda res, y, gate, idx: op_(res, y, gate, idx),
            }

        found = [("diffusers", diffusers), ("rl_kernel", ours)]

    elif op == "final_adaln_out":

        def make(s, n):
            ti, _ = _layout(n, s)
            return {
                "x": _rn(s, 18, n, H, dtype=torch.bfloat16, scale=2.0),
                "nw": (0.5 + torch.rand(H, device=DEV, generator=_gen(1, 19))).bfloat16(),
                "temb": _rn(1, 20, T3, D),
                "w": _rn(1, 21, 2 * H, D, dtype=torch.bfloat16, scale=D**-0.5),
                "b": _rn(1, 22, 2 * H, dtype=torch.bfloat16, scale=0.02),
                "ti": ti,
            }

        def ref(x, nw, temb, w, b, ti):
            return P.provider_final_adaln_out(x, nw, temb, w, b, ti)

        base = {
            "make": make,
            "rows": {"x": 0, "ti": 0},
            "grads": ["x", "nw", "temb", "w", "b"],
            "ref": ref,
            "bi_sizes": S_BI,
            "perf_sizes": S_PERF,
        }

        def diffusers():
            return {
                **base,
                "source": "diffusers MiniMaxH3AdaLayerNormOut (op-for-op replay)",
                "fn": ref,
            }

        def ours():
            from rl_engine.kernels.ops.cuda.h3.final_adaln_out import H3FinalAdaLNOutCudaOp

            op_ = H3FinalAdaLNOutCudaOp()
            return {
                **base,
                "source": "rl-kernel H3FinalAdaLNOutCudaOp",
                "fn": lambda x, nw, temb, w, b, ti: op_(x, nw, temb, w, b, ti),
            }

        found = [("diffusers", diffusers), ("rl_kernel", ours)]

    result = []
    for name, factory in found:
        try:
            result.append({"name": name, **factory()})
        except Exception as exc:  # noqa: BLE001 - optional library missing or unusable
            result.append({"name": name, "unavailable": f"{type(exc).__name__}: {exc}"[:300]})
    return result


# --------------------------------------------------------------------------- #
# Modes and driver
# --------------------------------------------------------------------------- #


def enable_mode(mode: str, megatron_src: str | None):
    """Switch on a global batch-invariant mode; returns a context manager that turns it off."""

    base = mode[:-5] if mode.endswith("_ieee") else mode
    if base == "plain":
        return contextlib.nullcontext
    if base == "vllm":
        from vllm.model_executor.determinism.batch_invariant import enable_batch_invariant_mode

        enable_batch_invariant_mode()
        return contextlib.nullcontext
    if base == "sglang":
        from sglang.srt.batch_invariant_ops import batch_invariant_ops as mod

        mod.enable_batch_invariant_mode()
        kwargs: dict[str, str] = {}
    else:
        if megatron_src:
            sys.path.insert(0, megatron_src)
        from megatron.core.transformer.custom_layers import batch_invariant_kernels as mod

        kwargs = {"backend": base.split("_", 1)[1]}
        mod.enable_batch_invariant_mode(**kwargs)
        if kwargs["backend"] != "triton":
            return contextlib.nullcontext

    @contextlib.contextmanager
    def off():
        # The Triton matmuls have no FP64 configuration: FP64 references run with the mode off.
        mod.disable_batch_invariant_mode()
        try:
            yield
        finally:
            mod.enable_batch_invariant_mode(**kwargs)

    return off


def run_mode(op: str, mode: str, megatron_src: str | None, quick: bool) -> dict[str, Any]:
    try:
        mode_off = enable_mode(mode, megatron_src)
    except Exception as exc:  # noqa: BLE001
        return {"unavailable": f"{type(exc).__name__}: {exc}"[:300]}
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out: dict[str, Any] = {}
    for cand in candidates(op, mode):
        name = cand.pop("name")
        if "unavailable" in cand:
            out[name] = cand
            print(f"  [{mode}] {name}: unavailable ({cand['unavailable'][:80]})", flush=True)
            continue
        entry: dict[str, Any] = {"source": cand["source"], "has_backward": bool(cand["grads"])}
        entry["batch_invariance"] = batch_invariance(cand, quick)
        entry["accuracy_latency"] = accuracy_latency(cand, mode_off, quick)
        out[name] = entry
        print(
            f"  [{mode}] {name}: batch_invariant={entry['batch_invariance']['batch_invariant']}",
            flush=True,
        )
        torch.cuda.empty_cache()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--op", required=True, choices=sorted(OPS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--megatron-src", default=None, help="Megatron-LM source tree (optional)")
    parser.add_argument("--quick", action="store_true", help="small sizes, for a smoke test only")
    parser.add_argument("--mode", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if importlib.util.find_spec(OPS[args.op]) is None:
        raise SystemExit(f"{args.op}: {OPS[args.op]} is not on this branch")
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device")

    if args.mode is not None:  # child process: one global mode
        args.out.write_text(json.dumps(run_mode(args.op, args.mode, args.megatron_src, args.quick)))
        return

    from rl_engine.testing.h3_chain import environment, git_state

    modes = list(MODES) if args.op in GEMM_OPS else ["plain"]
    results: dict[str, Any] = {}
    for mode in modes:
        print(f"[{args.op}] mode {mode}", flush=True)
        # The part file sits next to the report, not in /tmp: cluster job epilogs can clear a
        # user's /tmp while another of their jobs on the same node is still running.
        part = args.out.with_name(f".{args.out.stem}.{mode}.part.json")
        part.parent.mkdir(parents=True, exist_ok=True)
        part.unlink(missing_ok=True)
        cmd = [sys.executable, __file__, "--op", args.op, "--out", str(part), "--mode", mode]
        if args.megatron_src:
            cmd += ["--megatron-src", args.megatron_src]
        if args.quick:
            cmd.append("--quick")
        proc = subprocess.run(cmd, env={**os.environ, **MODES[mode]})
        results[mode] = (
            json.loads(part.read_text())
            if proc.returncode == 0 and part.exists()
            else {"unavailable": f"subprocess exited with {proc.returncode}"}
        )
        part.unlink(missing_ok=True)
    megatron_commit = None
    if args.megatron_src:
        megatron_commit = (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=args.megatron_src,
                capture_output=True,
                text=True,
            ).stdout.strip()
            or None
        )
    report = {
        "kind": "h3_prior_art",
        "rfc": "RL-Align/RL-Kernel#420",
        "op": args.op,
        **git_state(),
        "environment": {
            **environment(),
            "libraries": {lib: _version(lib) for lib in LIBS},
            "megatron_source_commit": megatron_commit,
            "mode_environment": {m: MODES[m] for m in modes},
        },
        "quick": args.quick,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"wrote {args.out} (commit {report['rl_kernel_commit'][:7]}, "
        f"dirty={report['tracked_tree_dirty']})"
    )


if __name__ == "__main__":
    main()
