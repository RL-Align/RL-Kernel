#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Evidence for the generic fused-logp backward (#174): time, memory, accuracy, invariance.

Four backward paths for selected-token logprobs at Qwen vocab size (151936), BF16:

* ``torch log_softmax + gather``: plain autograd, the existing implementation;
* ``previous VJP``: this op's VJP before #174 (FP32 softmax over the full ``[N, V]``);
* ``chunked fallback``: the bounded row-chunked FP32 VJP, kept for ``_C`` builds
  without the fused kernel;
* ``fused kernel``: ``_C.fused_logp_backward``.

The last three share ``_C.fused_logp`` for the forward. Existing implementations of the
same computation are added when importable (the RFC #420 reuse rule asks to compare against
them); logp = -loss from a per-token cross entropy, and the report measures whether each is
batch-invariant:

* Liger ``LigerCrossEntropyFunction`` (``reduction="none"``; writes the gradient into its
  input, so it gets its own copy, made outside the timed region);
* FLA ``fla.modules.fused_cross_entropy.cross_entropy_loss``;
* flash-attn's Triton ``cross_entropy_loss``, the kernel verl's ``logprobs_from_logits``
  calls when flash-attn is installed. Loaded from a flash-attention source tree given with
  ``--flash-attn-src`` (the installed package imports its CUDA extension first).

Writes one JSON report:

    python benchmarks/fused_logp_backward_evidence.py --out report.json \
        [--flash-attn-src /path/to/flash-attention]
    python benchmarks/plot_fused_logp_backward_evidence.py report.json

For a correctness-only rerun on a shared GPU, omit timing and memory measurements:

    python benchmarks/fused_logp_backward_evidence.py --out correctness.json \\
        --checks accuracy row-invariance batch-size-sweep train-rollout-invariance
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import rl_engine.backends.cuda.logprob.logp as logp_mod  # noqa: E402
from rl_engine.backends.extension import _C  # noqa: E402

VOCAB = 151936
TIMED_ROWS = (8192, 16384, 32768)
CHUNKED = logp_mod.fused_logp_backward_chunked


class _ForwardOnly:
    """A ``_C`` stand-in without ``fused_logp_backward``: selects the Python VJP."""

    @staticmethod
    def fused_logp(logits, labels):
        return _C.fused_logp(logits, labels)


def _previous_vjp(logits, labels, grad_output, out_dtype, **_):
    """The VJP before #174: FP32 softmax over the whole ``[N, V]`` at once."""

    probs = torch.softmax(logits.float(), dim=-1)
    rows = torch.arange(logits.size(0), device=logits.device)
    probs[rows, labels] -= 1.0
    return (-grad_output.reshape(-1, 1).float() * probs).to(out_dtype)


@contextlib.contextmanager
def _vjp(vjp):
    """The autograd VJP reads the module attribute at backward time, so patch it
    for the whole forward + backward."""

    previous = logp_mod.fused_logp_backward_chunked
    logp_mod.fused_logp_backward_chunked = vjp
    try:
        yield
    finally:
        logp_mod.fused_logp_backward_chunked = previous


def _autograd_fn(backend):
    return lambda logits, labels: logp_mod._FusedLogpAutograd.apply(logits, labels, backend)


def _torch_logp(logits, labels):
    return torch.log_softmax(logits.float(), dim=-1).gather(-1, labels[:, None]).squeeze(-1)


#: candidates whose forward overwrites its input (they get their own copy of the logits)
IN_PLACE: set[str] = set()
#: name -> where the implementation came from (package version or source commit)
EXTERNAL_SOURCES: dict[str, str] = {}


#: name -> (forward fn, VJP the autograd node uses when the backend has no fused kernel)
CANDIDATES: dict[str, tuple[Callable, Callable]] = {
    "torch log_softmax + gather": (_torch_logp, CHUNKED),
    "previous VJP (full FP32 softmax)": (_autograd_fn(_ForwardOnly()), _previous_vjp),
    "chunked fallback": (_autograd_fn(_ForwardOnly()), CHUNKED),
    "fused kernel": (_autograd_fn(_C), CHUNKED),
}


def _add_external_candidates(flash_attn_src: Path | None) -> None:
    """Register the existing implementations that import in this environment."""

    try:
        from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction

        name = "Liger cross_entropy"
        CANDIDATES[name] = (
            lambda x, y: -LigerCrossEntropyFunction.apply(x, y, None, -100, 0.0, 0.0, "none")[0],
            CHUNKED,
        )
        IN_PLACE.add(name)
        EXTERNAL_SOURCES[name] = "liger-kernel " + importlib.metadata.version("liger-kernel")
    except ImportError:
        pass
    try:
        from fla.modules.fused_cross_entropy import cross_entropy_loss as fla_ce

        name = "FLA fused cross_entropy"
        CANDIDATES[name] = (lambda x, y: -fla_ce(x, y)[0], CHUNKED)
        EXTERNAL_SOURCES[name] = "fla-core " + importlib.metadata.version("fla-core")
    except ImportError:
        pass
    if flash_attn_src is not None:
        path = flash_attn_src / "flash_attn" / "ops" / "triton" / "cross_entropy.py"
        spec = importlib.util.spec_from_file_location("flash_attn_triton_cross_entropy", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        name = "flash-attn cross_entropy (verl)"
        CANDIDATES[name] = (lambda x, y: -module.cross_entropy_loss(x, y)[0], CHUNKED)
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=flash_attn_src,
            capture_output=True,
            text=True,
        ).stdout.strip()
        manifest = flash_attn_src / "SOURCE.json"
        if not commit and manifest.exists():
            commit = json.loads(manifest.read_text())["commit"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        EXTERNAL_SOURCES[name] = (
            f"flash-attention source {commit or 'unknown'}, cross_entropy.py sha256 {digest}"
        )


def _leaf(name: str, logits: torch.Tensor) -> torch.Tensor:
    x = logits.detach().clone() if name in IN_PLACE else logits.detach()
    return x.requires_grad_(True)


def _effective_upstream_gradient(
    grad: torch.Tensor, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """One upstream gradient representable by every candidate output dtype.

    Quantize once at the BF16 execution boundary, then promote losslessly to FP32.
    Rounding separately by each output dtype would compare different VJPs.
    """

    return grad.to(dtype).float()


def _inputs(rows: int, seed: int, dtype=torch.bfloat16):
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = (torch.randn(rows, VOCAB, device="cuda", generator=g) * 2).to(dtype)
    labels = torch.randint(0, VOCAB, (rows,), device="cuda", generator=g)
    grad = _effective_upstream_gradient(torch.randn(rows, device="cuda", generator=g), dtype)
    return logits, labels, grad


def _step(name, logits, labels, grad, x=None):
    fn, vjp = CANDIDATES[name]
    x = _leaf(name, logits) if x is None else x
    with _vjp(vjp):
        out = fn(x, labels)
        out.backward(grad.to(out.dtype))
    return out.detach(), x.grad


def accuracy(rows: int = 257, seed: int = 1) -> dict[str, Any]:
    logits, labels, grad = _inputs(rows, seed)
    x64 = logits.double()
    probs = torch.softmax(x64, dim=-1)
    ref_out = torch.log_softmax(x64, dim=-1).gather(-1, labels[:, None]).squeeze(-1)
    one_hot = torch.zeros_like(probs).scatter_(1, labels[:, None], 1.0)
    # A single reference for all candidates, including BF16- and FP32-output paths.
    ref_grad = grad.double()[:, None] * (one_hot - probs)
    ref_absmax = ref_grad.abs().max().item()
    result = {}
    for name in CANDIDATES:
        out, g = _step(name, logits, labels, grad)
        if not torch.equal(grad.to(out.dtype).float(), grad):
            raise AssertionError(f"{name}: output dtype changes the common upstream gradient")
        err = (g.double() - ref_grad).abs()
        result[name] = {
            "output_dtype": str(out.dtype),
            "logp_max_abs": (out.double() - ref_out).abs().max().item(),
            "grad_max_abs": err.max().item(),
            "grad_max_abs_over_absmax": err.max().item() / ref_absmax,
            "grad_correctly_rounded_fraction": (g == ref_grad.to(g.dtype)).sum().item() / g.numel(),
        }
    return result


def train_rollout_invariance(rows: int = 257, seeds=(3, 4, 5)) -> dict[str, Any]:
    """Compare training providers with existing generic CUDA rollout entrypoints.

    Test BF16 inputs and the FP32-upcast inputs used by selected_logprobs_with_operator.
    Cross-provider agreement and candidate-local batch invariance are separate gates.
    This check does not exercise the separate vLLM WS2 rollout provider.
    """

    result = {}
    for dtype in (torch.bfloat16, torch.float32):
        by_name = {
            name: {
                "rows_checked": 0,
                "same_provider_no_grad_rows_differing": 0,
                "project_rollout_fp32_rows_differing": 0,
                "project_rollout_fp32_out_rows_differing": 0,
                "project_rollout_bf16_rows_differing": 0,
                "project_rollout_fp32_max_abs": 0.0,
                "project_rollout_fp32_out_max_abs": 0.0,
                "project_rollout_bf16_max_abs": 0.0,
                "batch_rows": rows,
                "seeds": list(seeds),
            }
            for name in CANDIDATES
        }
        for seed in seeds:
            logits, labels, _ = _inputs(rows, seed)
            logits = logits.to(dtype)
            with torch.no_grad():
                rollout_bf16 = _C.fused_logp(logits, labels).to(torch.bfloat16)
                rollout_fp32 = _C.fused_logp_forward_fp32(logits, labels)
                rollout_fp32_out = _C.fused_logp_forward_out(
                    logits, labels, torch.empty_like(rollout_fp32)
                )
            for name, (fn, vjp) in CANDIDATES.items():
                with _vjp(vjp):
                    train = fn(_leaf(name, logits), labels).detach()
                    # Liger overwrites its input, so modes need separate copies.
                    x = logits.detach().clone() if name in IN_PLACE else logits.detach()
                    with torch.no_grad():
                        rollout_same_provider = fn(x, labels)
                entry = by_name[name]
                entry["rows_checked"] += rows
                for key, candidate, reference in (
                    ("same_provider_no_grad", train, rollout_same_provider),
                    ("project_rollout_fp32", train.float(), rollout_fp32),
                    ("project_rollout_fp32_out", train.float(), rollout_fp32_out),
                    ("project_rollout_bf16", train.to(torch.bfloat16), rollout_bf16),
                ):
                    different_bytes = candidate.contiguous().view(
                        torch.uint8
                    ) != reference.contiguous().view(torch.uint8)
                    entry[f"{key}_rows_differing"] += (
                        different_bytes.reshape(rows, -1).any(dim=-1).sum().item()
                    )
                    if key != "same_provider_no_grad":
                        entry[f"{key}_max_abs"] = max(
                            entry[f"{key}_max_abs"],
                            (candidate.float() - reference.float()).abs().max().item(),
                        )
        result[str(dtype)] = by_name
    return result


def row_invariance(rows: int = 4096, step: int = 16, seeds=(3, 4, 5)) -> dict[str, Any]:
    """256 rows computed alone vs the same rows inside a batch, bitwise."""

    result = {}
    for name in CANDIDATES:
        out_bad = grad_bad = checked = 0
        for seed in seeds:
            logits, labels, grad = _inputs(rows, seed)
            full_out, full_grad = _step(name, logits, labels, grad)
            for i in range(0, rows, step):
                sl = slice(i, i + 1)
                out, g = _step(name, logits[sl], labels[sl], grad[sl])
                out_bad += not torch.equal(out[0], full_out[i])
                grad_bad += not torch.equal(g[0], full_grad[i])
                checked += 1
            del logits, full_grad
        result[name] = {
            "rows_checked": checked,
            "logp_rows_differing": out_bad,
            "grad_rows_differing": grad_bad,
            "batch_rows": rows,
            "seeds": list(seeds),
        }
    return result


def sweep_sizes(n: int) -> list[int]:
    """1..9, then 2^k - 1, 2^k, 2^k + 1 up to n: row counts where kernel heuristics switch."""

    sizes = set(range(1, min(n, 9) + 1))
    k = 16
    while k <= n:
        sizes.update(v for v in (k - 1, k, k + 1) if v <= n)
        k *= 2
    return sorted(sizes | {n})


def batch_size_sweep(rows: int = 2048, seeds=(3, 4, 5)) -> dict[str, Any]:
    """Probe rows alone vs at the front, middle and back of batches of every sweep size.

    Filler rows are drawn from the rest of the batch. Sparse probes (one batch size, a few
    rows) can miss heuristic switch points and report a batch-dependent kernel as invariant.
    """

    sizes = sweep_sizes(rows)
    # Batch dependence can be row-specific (FLA's cross entropy differs on a few percent of
    # rows), so probe eight rows spread over the batch, not just the ends.
    probes = tuple(sorted({0, rows - 1, *list(range(rows // 8 + 3, rows - 1, rows // 8))[:6]}))
    result = {}
    for name in CANDIDATES:
        compared = out_bad = grad_bad = 0
        failures = []
        for seed in seeds:
            logits, labels, grad = _inputs(rows, seed)
            gen = torch.Generator().manual_seed(seed)
            alone = {
                r: _step(name, logits[r : r + 1], labels[r : r + 1], grad[r : r + 1])
                for r in probes
            }
            for m in sizes:
                for r in probes:
                    others = torch.randperm(rows, generator=gen)
                    others = others[others != r][: m - 1]
                    for pos in sorted({0, (m - 1) // 2, m - 1}):
                        idx = torch.cat([others[:pos], torch.tensor([r]), others[pos:]]).cuda()
                        out, g = _step(name, logits[idx], labels[idx], grad[idx])
                        o_ok = torch.equal(out[pos], alone[r][0][0])
                        g_ok = torch.equal(g[pos], alone[r][1][0])
                        compared += 1
                        out_bad += not o_ok
                        grad_bad += not g_ok
                        if not (o_ok and g_ok) and len(failures) < 8:
                            failures.append({"seed": seed, "batch": m, "position": pos, "row": r})
            del logits
            torch.cuda.empty_cache()
        result[name] = {
            "comparisons": compared,
            "logp_differing": out_bad,
            "grad_differing": grad_bad,
            "batch_sizes": sizes,
            "probe_rows": list(probes),
            "seeds": list(seeds),
            "first_failures": failures,
        }
        print(f"  sweep {name}: {out_bad} / {grad_bad} of {compared} differ", flush=True)
    return result


def _time_ms(
    fn: Callable[..., Any],
    warmup: int = 2,
    iters: int = 5,
    setup: Callable[[], Any] | None = None,
) -> float:
    """Median CUDA-event ms; ``setup`` (outside the timed region) feeds ``fn`` its argument."""

    def call():
        return fn(setup()) if setup is not None else fn()

    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        arg = setup() if setup is not None else None
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn(arg) if setup is not None else fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def performance() -> dict[str, Any]:
    result: dict[str, Any] = {name: {} for name in CANDIDATES}
    for rows in TIMED_ROWS:
        logits, labels, grad = _inputs(rows, seed=11)
        for name in CANDIDATES:
            fn = CANDIDATES[name]

            def setup(name=name, logits=logits):
                return _leaf(name, logits)

            def step(x, name=name, logits=logits, labels=labels, grad=grad):
                _, g = _step(name, logits, labels, grad, x=x)
                return g

            row = {"step_ms": _time_ms(step, setup=setup)}
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            x = setup()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            g = step(x)
            torch.cuda.synchronize()
            row["peak_gib"] = (torch.cuda.max_memory_allocated() - base) / 2**30
            del g, x
            x = setup()
            with _vjp(fn[1]):
                out = fn[0](x, labels)
                go = grad.to(out.dtype)
                row["backward_ms"] = _time_ms(
                    lambda o=out, x=x, go=go: torch.autograd.grad(o, x, go, retain_graph=True)
                )
            del x, out
            torch.cuda.empty_cache()
            result[name][str(rows)] = row
            print(f"  N={rows} {name}: {row}", flush=True)
        del logits
        torch.cuda.empty_cache()
    return result


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--flash-attn-src",
        type=Path,
        default=None,
        help="flash-attention source tree (optional)",
    )
    parser.add_argument(
        "--checks",
        nargs="+",
        choices=(
            "accuracy",
            "row-invariance",
            "batch-size-sweep",
            "train-rollout-invariance",
            "performance",
        ),
        default=(
            "accuracy",
            "row-invariance",
            "batch-size-sweep",
            "train-rollout-invariance",
            "performance",
        ),
        help="checks to rerun; omit performance when the GPU is shared",
    )
    parser.add_argument("--accuracy-rows", type=int, default=257)
    parser.add_argument("--row-invariance-rows", type=int, default=4096)
    parser.add_argument("--sweep-rows", type=int, default=2048)
    parser.add_argument("--contract-rows", type=int, default=257)
    parser.add_argument("--seeds", type=int, nargs="+", default=(3, 4, 5))
    parser.add_argument("--build-note", default=None, help="optional extension build provenance")
    args = parser.parse_args()
    if (
        min(
            args.accuracy_rows,
            args.row_invariance_rows,
            args.sweep_rows,
            args.contract_rows,
        )
        < 16
    ):
        parser.error("row counts must be at least 16")
    _add_external_candidates(args.flash_attn_src)
    if not torch.cuda.is_available() or not hasattr(_C, "fused_logp_backward"):
        raise SystemExit("needs a CUDA device and a _C build with fused_logp_backward")
    report = {
        "kind": "fused_logp_backward_evidence",
        "issue": "RL-Align/RL-Kernel#174",
        "git_commit": _git("rev-parse", "HEAD") or "unknown",
        "git_dirty": bool(_git("status", "--porcelain")),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "triton": _version("triton"),
            "external": EXTERNAL_SOURCES,
            "extension_build_note": args.build_note,
            "extension_sha256": hashlib.sha256(Path(_C.__file__).read_bytes()).hexdigest(),
        },
        "vocab": VOCAB,
        "dtype": "bfloat16",
        "accuracy_contract": {
            "rows": args.accuracy_rows,
            "seed": 1,
            "upstream": "one shared random gradient quantized to BF16, promoted losslessly to FP32",
            "reference": (
                "one FP64 VJP from BF16 logits and the shared upstream, for every candidate"
            ),
            "identical_effective_upstream_asserted": True,
        },
        "train_rollout_contract": {
            "rollout_provider": (
                "existing _C.fused_logp, _C.fused_logp_forward_fp32 and "
                "_C.fused_logp_forward_out; separate vLLM WS2 provider not exercised"
            ),
            "input_regimes": "BF16 logits and the same logits losslessly upcast to FP32",
            "fp32_boundary": "training output promoted to FP32 vs existing FP32 rollout",
            "bf16_boundary": "training output quantized to BF16 vs existing BF16 rollout",
            "same_provider_no_grad": "training vs torch.no_grad() using the same candidate",
        },
    }
    checks = {
        "accuracy": lambda: accuracy(rows=args.accuracy_rows),
        "row-invariance": lambda: row_invariance(rows=args.row_invariance_rows, seeds=args.seeds),
        "batch-size-sweep": lambda: batch_size_sweep(rows=args.sweep_rows, seeds=args.seeds),
        "train-rollout-invariance": lambda: train_rollout_invariance(
            rows=args.contract_rows, seeds=args.seeds
        ),
        "performance": performance,
    }
    for name in dict.fromkeys(args.checks):
        print(f"running {name}", flush=True)
        report[name.replace("-", "_")] = checks[name]()
        # Save completed checks even if a later optional check is interrupted.
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"completed {name}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out} (commit {report['git_commit'][:7]}, dirty={report['git_dirty']})")


if __name__ == "__main__":
    main()
