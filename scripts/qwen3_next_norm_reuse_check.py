#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Compare a Qwen3-Next norm with existing implementations: batch invariance, accuracy, gates.

Checks, for the rl-kernel CUDA op and every existing implementation that imports here
(transformers, vLLM, FlashInfer, Liger, FLA, Transformer Engine, Megatron-LM from a source
tree):

* ``bi``: bitwise batch invariance of the forward and the row gradients, and repeatability of
  ``dweight``: every row computed alone vs inside full batches of three sizes (seeds 3-5);
  the full workload-size batch vs sub-batches covering every row; eight probe rows at the
  front, middle and back of batches of every size 1..9 and 2^k - 1, 2^k, 2^k + 1. Sparse probes
  miss row-specific and large-batch dependence, which is why every row is checked.
* ``perf``: accuracy against FP64 (forward: fraction equal to the correctly rounded FP64 result;
  gradients: max|err| / max|ref|) and median CUDA-event latency at the workload size.
* ``gates``: this repository's own C3/C4 gates (``scripts/check_forward_invariance.py``,
  ``scripts/check_gradient_invariance.py`` with ``qwen3_next_norm_manifest.json``, the
  arguments ``ci/run_ws1_gtest.sh`` uses), unchanged, with the CUDA candidate replaced by a
  subclass of the rl-kernel op whose forward and backward call the other library. The subclass
  keeps the op's ``parameter_vjp_contributions_fp32``, so the singleton-aggregate check
  compares that library's ``dweight`` with the same FP32 row contributions.

    python scripts/qwen3_next_norm_reuse_check.py --op qwen3_next_rms_norm \\
        --out docs/usage/evidence/qwen3-next-norm-reuse-b200/qwen3_next_rms_norm.json \\
        [--checks bi,perf,gates] [--megatron-src /path/to/Megatron-LM] [--quick]

Run on a clean tree on an otherwise idle GPU; the report records the commit, the tree state and
every library version. A figure is written next to the report when matplotlib is available.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

DEV, EPS, DT = "cuda", 1e-6, torch.bfloat16
SHAPES = {
    # op: (hidden, all-rows batch sizes, full batch, covering sub-batch sizes, perf rows)
    "qwen3_next_rms_norm": (2048, (64, 257, 2048), 65536, (1, 7, 2048, 4097, 32768), 65536),
    "rms_norm_gated": (128, (64, 257, 4097), 262144, (1, 7, 4097, 65536, 131072), 262144),
}
LIBS = ("torch", "triton", "transformers", "vllm", "flashinfer-python", "liger-kernel", "fla-core")
LIBS += ("transformer_engine", "megatron-core")
MANIFEST = "rl_engine/testing/qwen3_next_norm_manifest.json"


def _version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


# --------------------------------------------------------------------------- #
# Candidates: fn(x, w) for the zero-centred norm, fn(x, z, w) for the gated norm
# --------------------------------------------------------------------------- #


def _vllm_module(build):
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        return build()


def _c1_candidates(hidden: int) -> list[tuple[str, Any]]:
    def ours():
        from rl_engine.kernels.ops.cuda.norm.rmsnorm import Qwen3NextRMSNormCudaOp

        o = Qwen3NextRMSNormCudaOp()
        return "rl-kernel Qwen3NextRMSNormCudaOp", lambda x, w: o(x, w, eps=EPS), "full"

    def reference():
        from rl_engine.kernels.ops.pytorch.norm.qwen3_next_rms_norm import Qwen3NextRMSNormOp

        o = Qwen3NextRMSNormOp()
        return "rl-kernel PyTorch reference", lambda x, w: o(x, w, eps=EPS), "full"

    def transformers():
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRMSNorm

        m = Qwen3NextRMSNorm(hidden, eps=EPS).to(DEV, DT)

        def fn(x, w):
            return torch.func.functional_call(m, {"weight": w}, (x,))

        return f"transformers {_version('transformers')} Qwen3NextRMSNorm", fn, "full"

    def liger():
        from liger_kernel.ops.rms_norm import LigerRMSNormFunction

        def fn(x, w):
            return LigerRMSNormFunction.apply(x, w, EPS, 1.0, "gemma", False)

        return f"Liger {_version('liger-kernel')} RMSNorm, offset 1, gemma", fn, "full"

    def fla():
        from fla.modules.layernorm import rms_norm

        def fn(x, w):
            return rms_norm(x, (1.0 + w.float()).to(x.dtype), None, eps=EPS)

        return f"FLA {_version('fla-core')} rms_norm, weight passed as 1 + w", fn, "full"

    def te():
        import transformer_engine.pytorch as te_

        m = te_.RMSNorm(hidden, eps=EPS, zero_centered_gamma=True, params_dtype=DT, device=DEV)

        def fn(x, w):
            # TE's backward reads its own parameter, so copy w in (functional_call gives
            # a wrong dx); dweight is then TE's parameter gradient and is not checked.
            with torch.no_grad():
                m.weight.copy_(w)
            return m(x)

        return f"TE {_version('transformer_engine')} RMSNorm(zero_centered_gamma)", fn, "x-only"

    def megatron():
        from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
            BatchInvariantRMSNormFn,
        )

        def fn(x, w):
            return BatchInvariantRMSNormFn.apply(x, w, EPS, True)

        # Some revisions accept the flag but still multiply by the uncentered weight.
        with torch.no_grad():
            probe_x = torch.ones(2, hidden, device=DEV, dtype=DT)
            probe_w = torch.zeros(hidden, device=DEV, dtype=DT)
            expected = (probe_x.float() * (1.0 + EPS) ** -0.5).to(DT)
            if not torch.allclose(fn(probe_x, probe_w), expected):
                raise RuntimeError(
                    "Megatron zero_centered_gamma=True failed the zero-weight probe; "
                    "excluded from the zero-centered comparison"
                )

        return "Megatron BatchInvariantRMSNormFn(zero_centered_gamma=True)", fn, "full"

    def flashinfer():
        import flashinfer as fi

        def fn(x, w):
            return fi.gemma_rmsnorm(x, w, EPS)

        return f"FlashInfer {_version('flashinfer-python')} gemma_rmsnorm", fn, "none"

    def vllm():
        from vllm.model_executor.layers.layernorm import GemmaRMSNorm

        m = _vllm_module(lambda: GemmaRMSNorm(hidden, eps=EPS)).to(DEV, DT)

        def fn(x, w):
            m.weight.data.copy_(w)
            return m.forward_cuda(x)

        return f"vLLM {_version('vllm')} GemmaRMSNorm.forward_cuda", fn, "none"

    return [
        ("rl_kernel", ours),
        ("reference", reference),
        ("transformers", transformers),
        ("liger", liger),
        ("fla", fla),
        ("transformer_engine", te),
        ("megatron", megatron),
        ("flashinfer", flashinfer),
        ("vllm", vllm),
    ]


def _gated_candidates(hidden: int) -> list[tuple[str, Any]]:
    def ours():
        from rl_engine.kernels.ops.cuda.norm.rmsnorm import Qwen3NextRMSNormGatedCudaOp

        o = Qwen3NextRMSNormGatedCudaOp()
        return "rl-kernel Qwen3NextRMSNormGatedCudaOp", lambda x, z, w: o(x, w, z, eps=EPS), "full"

    def reference():
        from rl_engine.kernels.ops.pytorch.norm.qwen3_next_rms_norm import Qwen3NextRMSNormGatedOp

        o = Qwen3NextRMSNormGatedOp()
        return "rl-kernel PyTorch reference", lambda x, z, w: o(x, w, z, eps=EPS), "full"

    def transformers():
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRMSNormGated

        m = Qwen3NextRMSNormGated(hidden, eps=EPS).to(DEV, DT)

        def fn(x, z, w):
            return torch.func.functional_call(m, {"weight": w}, (x, z))

        return f"transformers {_version('transformers')} Qwen3NextRMSNormGated", fn, "full"

    def fla_lg():
        from fla.modules.layernorm_gated import rmsnorm_fn

        def fn(x, z, w):
            return rmsnorm_fn(x, w, None, z=z, eps=EPS, group_size=None, norm_before_gate=True)

        return f"FLA {_version('fla-core')} layernorm_gated.rmsnorm_fn", fn, "full"

    def fla_fused():
        from fla.modules.fused_norm_gate import rms_norm_gated

        def fn(x, z, w):
            return rms_norm_gated(x, z, w, None, activation="silu", eps=EPS)

        return f"FLA {_version('fla-core')} fused_norm_gate.rms_norm_gated", fn, "full"

    def vllm():
        from vllm.model_executor.layers.layernorm import RMSNormGated

        def build():
            return RMSNormGated(
                hidden, eps=EPS, group_size=None, norm_before_gate=True, activation="silu"
            )

        m = _vllm_module(build).to(DEV, DT)

        def fn(x, z, w):
            m.weight.data.copy_(w)
            return m.forward_cuda(x, z)

        return f"vLLM {_version('vllm')} RMSNormGated.forward_cuda", fn, "none"

    return [
        ("rl_kernel", ours),
        ("reference", reference),
        ("transformers", transformers),
        ("fla_layernorm_gated", fla_lg),
        ("fla_fused_norm_gate", fla_fused),
        ("vllm", vllm),
    ]


def candidates(op: str) -> dict[str, dict[str, Any]]:
    hidden = SHAPES[op][0]
    found = _c1_candidates(hidden) if op == "qwen3_next_rms_norm" else _gated_candidates(hidden)
    out: dict[str, dict[str, Any]] = {}
    for name, factory in found:
        try:
            source, fn, backward = factory()
            out[name] = {"source": source, "fn": fn, "backward": backward}
        except Exception as exc:  # noqa: BLE001 - optional library missing or unusable
            if name == "rl_kernel":
                raise RuntimeError(f"Required rl_kernel candidate is unavailable: {exc}") from exc
            out[name] = {"unavailable": f"{type(exc).__name__}: {exc}"[:300]}
    return out


def _inputs(op: str, seed: int, n: int) -> dict[str, torch.Tensor]:
    hidden = SHAPES[op][0]
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = (torch.randn(n, hidden, device=DEV, generator=g) * 2).to(DT)
    if op == "qwen3_next_rms_norm":
        return {"x": x, "w": (torch.randn(hidden, device=DEV, generator=g) * 0.1).to(DT)}
    z = torch.randn(n, hidden, device=DEV, generator=g).to(DT)
    return {"x": x, "z": z, "w": (1 + torch.randn(hidden, device=DEV, generator=g) * 0.1).to(DT)}


def _reference(op: str, inputs: dict[str, torch.Tensor]):
    x = inputs["x"].double()
    n = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    if op == "qwen3_next_rms_norm":
        return n * (1 + inputs["w"].double())
    return n * inputs["w"].double() * F.silu(inputs["z"].double())


# --------------------------------------------------------------------------- #
# Batch invariance
# --------------------------------------------------------------------------- #


def _run(fn, inputs, rows, grads, ups, idx=None):
    args = {}
    for k, v in inputs.items():
        if idx is not None and k in rows:
            v = v.index_select(0, idx)
        args[k] = v.detach().clone().requires_grad_(True) if k in grads else v
    out = fn(**args)
    if grads:
        out.backward(ups if idx is None else ups.index_select(0, idx))
    return out.detach(), {k: args[k].grad for k in grads}


def _sweep_sizes(n: int) -> list[int]:
    sizes = set(range(1, min(n, 9) + 1))
    k = 16
    while k <= n:
        sizes.update(v for v in (k - 1, k, k + 1) if v <= n)
        k *= 2
    return sorted(sizes | {n})


def batch_invariance(op: str, cand: dict[str, Any], quick: bool) -> dict[str, Any]:
    _, full_sizes, big, subs, _ = SHAPES[op]
    if quick:
        full_sizes, big, subs = (17, 64), 512, (1, 7, 64)
    fn = cand["fn"]
    rows = ["x"] if op == "qwen3_next_rms_norm" else ["x", "z"]
    grads = {"full": rows + ["w"], "x-only": rows, "none": []}[cand["backward"]]
    row_grads = [k for k in rows if k in grads]

    def ups_for(seed, n):
        g = torch.Generator(device=DEV).manual_seed(seed + 100)
        return torch.randn(n, SHAPES[op][0], device=DEV, generator=g).to(DT)

    res: dict[str, Any] = {"all_rows": {}, "dweight_repeatable": None}
    ok = True
    for n in full_sizes:
        fwd = grd = 0
        repeat = True
        for seed in (3, 4, 5):
            inp, ups = _inputs(op, seed, n), ups_for(seed, n)
            fo, fg = _run(fn, inp, rows, grads, ups)
            for i in range(n):
                o, g = _run(fn, inp, rows, grads, ups, torch.tensor([i], device=DEV))
                fwd += not torch.equal(o[0], fo[i])
                grd += not all(torch.equal(g[k][0], fg[k][i]) for k in row_grads)
            if "w" in grads:
                _, again = _run(fn, inp, rows, grads, ups)
                repeat &= torch.equal(again["w"], fg["w"])
        if "w" in grads:
            res["dweight_repeatable"] = repeat and res["dweight_repeatable"] is not False
        res["all_rows"][str(n)] = {"rows": 3 * n, "fwd_differ": fwd, "rowgrad_differ": grd}
        ok &= fwd == 0 and grd == 0 and repeat

    fwd = grd = checked = 0
    for seed in (3, 4):
        inp, ups = _inputs(op, seed, big), ups_for(seed, big)
        fo, fg = _run(fn, inp, rows, grads, ups)
        for sb in subs:
            for start in range(0, big, sb):
                idx = torch.arange(start, min(start + sb, big), device=DEV)
                o, g = _run(fn, inp, rows, grads, ups, idx)
                checked += 1
                fwd += not torch.equal(o, fo.index_select(0, idx))
                grd += not all(torch.equal(g[k], fg[k].index_select(0, idx)) for k in row_grads)
        del inp, ups, fo, fg
        torch.cuda.empty_cache()
    res["full_vs_sub_batches"] = {
        "full_batch": big,
        "sub_batch_sizes": list(subs),
        "coverage": "every_row_per_sub_batch_size",
        "sub_batches": checked,
        "fwd_differ": fwd,
        "rowgrad_differ": grd,
    }
    ok &= fwd == 0 and grd == 0

    n = full_sizes[-1]
    sizes = _sweep_sizes(n)
    probes = sorted({0, n - 1, *list(range(n // 8 + 3, n - 1, n // 8))[:6]})
    compared = failed = 0
    for seed in (3, 4, 5):
        inp, ups = _inputs(op, seed, n), ups_for(seed, n)
        alone = {r: _run(fn, inp, rows, grads, ups, torch.tensor([r], device=DEV)) for r in probes}
        gen = torch.Generator().manual_seed(seed)
        for m in sizes:
            for r in probes:
                others = torch.randperm(n, generator=gen)
                others = others[others != r][: m - 1]
                for pos in sorted({0, (m - 1) // 2, m - 1}):
                    idx = torch.cat([others[:pos], torch.tensor([r]), others[pos:]]).to(DEV)
                    o, g = _run(fn, inp, rows, grads, ups, idx)
                    good = torch.equal(o[pos], alone[r][0][0])
                    good &= all(torch.equal(g[k][pos], alone[r][1][k][0]) for k in row_grads)
                    compared += 1
                    failed += not good
    res["batch_size_sweep"] = {
        "batch_sizes": sizes,
        "probe_rows": probes,
        "comparisons": compared,
        "differ": failed,
    }
    ok &= failed == 0
    res["batch_invariant"] = bool(ok)
    return res


# --------------------------------------------------------------------------- #
# Accuracy and latency
# --------------------------------------------------------------------------- #


def _time_us(fn, warmup=10, iters=50) -> float:
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


def accuracy_latency(op: str, cand: dict[str, Any], quick: bool) -> dict[str, Any]:
    n = 4096 if quick else SHAPES[op][4]
    inp = _inputs(op, 11, n)
    keys = list(inp)
    gk = {"full": keys, "x-only": [k for k in keys if k != "w"], "none": []}[cand["backward"]]
    fn = cand["fn"]
    leaves64 = {k: v.double().requires_grad_(k in gk) for k, v in inp.items()}
    ref = _reference(op, leaves64)
    with torch.no_grad():
        y = fn(**inp)
    out: dict[str, Any] = {
        "rows": n,
        "fwd_correctly_rounded": (y == ref.detach().to(DT)).float().mean().item(),
        "fwd_max_abs_err": (y.double() - ref.detach()).abs().max().item(),
        "fwd_us": _time_us(lambda: fn(**inp)),
    }
    if gk:
        gen = torch.Generator(device=DEV).manual_seed(5)
        dy = torch.randn(n, SHAPES[op][0], device=DEV, generator=gen).to(DT)
        rg = torch.autograd.grad(ref, [leaves64[k] for k in gk], dy.double())
        lv = {k: v.detach().clone().requires_grad_(k in gk) for k, v in inp.items()}
        gs = torch.autograd.grad(fn(**lv), [lv[k] for k in gk], dy)
        out["grad_rel_err"] = {
            k: ((g.double() - r).abs().max() / r.abs().max()).item() for k, g, r in zip(gk, gs, rg)
        }

        def step():
            lv = {k: v.detach().clone().requires_grad_(k in gk) for k, v in inp.items()}
            torch.autograd.grad(fn(**lv), [lv[k] for k in gk], dy)

        out["fwd_bwd_us"] = _time_us(step)
    return out


# --------------------------------------------------------------------------- #
# The repository's C3/C4 gates with an existing implementation swapped in
# --------------------------------------------------------------------------- #


def _gate_child(which: str, op: str, impl: str) -> None:
    """Run scripts/check_{which}_invariance.py with the CUDA candidate replaced by ``impl``."""

    import importlib.util

    from rl_engine.kernels.ops.cuda.norm import rmsnorm as R

    path = REPO_ROOT / "scripts" / f"check_{which}_invariance.py"
    spec = importlib.util.spec_from_file_location("gate", path)
    gate = importlib.util.module_from_spec(spec)
    sys.argv = [
        str(path),
        "--manifest",
        MANIFEST,
        "--op",
        op,
        "--candidate",
        "cuda",
        "--backend-profile",
        "cuda_bf16",
        "--hidden",
        "2048",
        "--head-dim",
        "128",
    ]
    spec.loader.exec_module(gate)
    if impl != "rl_kernel":
        fn = candidates(op)[impl]["fn"]
        if op == "qwen3_next_rms_norm":

            class Swapped(R.Qwen3NextRMSNormCudaOp):
                def forward(self, x, weight, *, eps=EPS):
                    return fn(x.reshape(-1, x.shape[-1]), weight).view_as(x)

        else:

            class Swapped(R.Qwen3NextRMSNormGatedCudaOp):
                def __call__(self, x, weight, gate, *, eps=EPS):
                    return self.forward(x, weight, gate, eps=eps)

                def forward(self, x, weight, gate, *, eps=EPS):
                    return fn(x, gate, weight)

        original = gate.load_adapter_operator

        def load_adapter_operator(op_name, candidate):
            return Swapped() if candidate == "cuda" else original(op_name, candidate)

        gate.load_adapter_operator = load_adapter_operator
    gate.main()


def _clean(lines: list[str], megatron_src: str | None) -> list[str]:
    """Drop warnings and replace machine-specific paths, so reports carry no local paths."""

    subs = [(str(REPO_ROOT), "<repo>"), (sys.prefix, "<python>"), (str(Path.home()), "<home>")]
    if megatron_src:
        subs.insert(0, (str(Path(megatron_src).resolve()), "<megatron-src>"))
    out = []
    for line in lines:
        if "Warning" in line or "warnings.warn" in line:
            continue
        for path, name in subs:
            line = line.replace(path, name)
        out.append(line)
    return out


def contract_gates(op: str, impl: str, megatron_src: str | None) -> dict[str, Any]:
    if not (REPO_ROOT / MANIFEST).exists():
        # The Qwen3-Next C3/C4 gate adapters and manifest arrive with the gated-norm PR.
        return {"unavailable": f"{MANIFEST} is not on this branch"}
    out: dict[str, Any] = {}
    for which in ("forward", "gradient"):
        cmd = [
            sys.executable,
            __file__,
            "--op",
            op,
            "--out",
            os.devnull,
            "--gate-child",
            which,
            impl,
        ]
        if megatron_src:
            cmd += ["--megatron-src", megatron_src]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, env={**os.environ, "RL_KERNEL_REQUIRE_EXT": "1"}
        )
        lines = _clean(proc.stdout.splitlines(), megatron_src)
        summary = next((ln for ln in lines if ln.startswith("op=")), "")
        out[which] = {
            "returncode": proc.returncode,
            "passed": "passed=True" in summary and proc.returncode == 0,
            "failed_lines": [ln.strip() for ln in lines if "passed=False" in ln][:10],
            "singleton_aggregate": [ln.strip() for ln in lines if "singleton_aggregate pair" in ln],
            "stderr_tail": (
                _clean(proc.stderr.strip().splitlines(), megatron_src)[-3:]
                if proc.returncode
                else []
            ),
        }
    return out


# --------------------------------------------------------------------------- #
# Report and figure
# --------------------------------------------------------------------------- #


def plot(report: dict[str, Any], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    entries = [(n, e) for n, e in report["results"].items() if "unavailable" not in e]
    names = [e["source"] for _, e in entries]
    ys = list(range(len(entries)))
    fig, axes = plt.subplots(1, 3, figsize=(22, 0.55 * len(entries) + 3), layout="constrained")
    fig.suptitle(
        f"{report['op']} vs existing implementations — {report['environment']['gpu']}, "
        f"commit {report['rl_kernel_commit'][:7]}"
    )
    ax = axes[0]
    for key, off, label in (("fwd_us", -0.2, "forward"), ("fwd_bwd_us", 0.2, "fwd + bwd")):
        vals = [e.get("accuracy_latency", {}).get(key, float("nan")) for _, e in entries]
        ax.barh([y + off for y in ys], vals, 0.4, label=label)
    ax.set_xscale("log")
    ax.set_yticks(ys, names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("µs, median")
    ax.set_title("latency at the workload size")
    ax.legend(fontsize=7)
    ax.grid(True, axis="x", alpha=0.3)
    ax = axes[1]
    acc = [e.get("accuracy_latency", {}) for _, e in entries]
    cr = [a.get("fwd_correctly_rounded", float("nan")) for a in acc]
    ax.barh(ys, [max(1 - c, 1e-7) for c in cr])
    for y, c, a in zip(ys, cr, acc):
        grads = a.get("grad_rel_err", {})
        worst = f"; worst grad err {max(grads.values()):.1e}" if grads else ""
        ax.text(1.5e-7, y, f"{c:.4%} correctly rounded{worst}", va="center", fontsize=7)
    ax.set_xscale("log")
    ax.set_xlim(1e-7, 1)
    ax.set_yticks(ys, [""] * len(entries))
    ax.invert_yaxis()
    ax.set_title("forward: fraction NOT equal to the correctly rounded FP64 result")
    ax = axes[2]
    for y, (_, e) in zip(ys, entries):
        bi = e.get("batch_invariance")
        gates = e.get("contract_gates")
        text = "checks passed: " + (
            "n/a" if bi is None else ("yes" if bi["batch_invariant"] else "NO")
        )
        if bi and bi["full_vs_sub_batches"].get("coverage") == "sampled_small_sub_batches":
            text += " (sampled)"
        if gates and "unavailable" not in gates:
            text += " | C3/C4 gates: " + (
                "pass" if all(g["passed"] for g in gates.values()) else "FAIL"
            )
        good = bi is not None and bi["batch_invariant"]
        ax.text(0.02, y, text, va="center", fontsize=8, color="#3aa676" if good else "#d14b4b")
    ax.set_ylim(len(entries) - 0.5, -0.5)
    ax.axis("off")
    ax.set_title("measured batch-invariance checks and this repository's gates")
    fig.savefig(path.with_suffix(".png"), dpi=120)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--op", required=True, choices=sorted(SHAPES))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checks", default="bi,perf,gates")
    parser.add_argument("--megatron-src", default=None, help="Megatron-LM source tree (optional)")
    parser.add_argument("--quick", action="store_true", help="small sizes, for a smoke test only")
    parser.add_argument("--gate-child", nargs=2, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.megatron_src:
        sys.path.insert(0, args.megatron_src)
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device")
    if args.gate_child:
        _gate_child(args.gate_child[0], args.op, args.gate_child[1])
        return
    if args.op == "rms_norm_gated":
        from rl_engine.kernels.ops.cuda.norm import rmsnorm as R

        if not hasattr(R, "Qwen3NextRMSNormGatedCudaOp"):
            raise SystemExit("rms_norm_gated: the gated CUDA op is not on this branch")
    torch.backends.cuda.matmul.allow_tf32 = False
    checks = set(args.checks.split(","))
    results: dict[str, Any] = {}
    for name, cand in candidates(args.op).items():
        if "unavailable" in cand:
            results[name] = cand
            print(f"{name}: unavailable ({cand['unavailable'][:80]})", flush=True)
            continue
        entry: dict[str, Any] = {"source": cand["source"], "backward": cand["backward"]}
        if "bi" in checks:
            entry["batch_invariance"] = batch_invariance(args.op, cand, args.quick)
        if "perf" in checks:
            entry["accuracy_latency"] = accuracy_latency(args.op, cand, args.quick)
        if "gates" in checks and cand["backward"] == "full" and name != "reference":
            entry["contract_gates"] = contract_gates(args.op, name, args.megatron_src)
        results[name] = entry
        summary = {k: v for k, v in entry.items() if k != "source"}
        print(f"{name}: {json.dumps(summary)[:300]}", flush=True)
        torch.cuda.empty_cache()

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
        "kind": "qwen3_next_norm_reuse_check",
        "rfc": "RL-Align/RL-Kernel#428",
        "op": args.op,
        "rl_kernel_commit": _git("rev-parse", "HEAD"),
        "tracked_tree_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "cuda": torch.version.cuda,
            "libraries": {lib: _version(lib) for lib in LIBS},
            "megatron_source_commit": megatron_commit,
        },
        "quick": args.quick,
        "checks": sorted(checks),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    plot(report, args.out)
    commit, dirty = report["rl_kernel_commit"][:7], report["tracked_tree_dirty"]
    print(f"wrote {args.out} (commit {commit}, dirty={dirty})")


if __name__ == "__main__":
    main()
