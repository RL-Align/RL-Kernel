# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Operator-only H3 patch permutation benchmark with raw timing samples."""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import itertools
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import distribution
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_engine.reference.packing.video_patch_pack_unpack import (  # noqa: E402
    UPSTREAM_REVISION,
    UPSTREAM_SOURCE,
    UPSTREAM_SOURCE_SHA256,
    minimax_h3_patchify_video_latent,
    minimax_h3_unpatchify_video_tokens,
    pack_video_patch_reference,
    unpack_video_patch_reference,
)
from rl_engine.runtime.registry import KernelRegistry  # noqa: E402
from rl_engine.runtime.semantic_registry import OperatorRequirements  # noqa: E402

DIFFUSERS_VERSION = "0.40.0"


class DiffusersVideoPatch:
    """Execute installed upstream statements unchanged, outside pipeline plumbing.

    Diffusers' decoder entry is no_grad. Only its video tensor statements are
    extracted here; their autograd derivative is an operator reference, not an
    upstream training pipeline. No RL-Kernel permutation is used by this adapter.
    """

    backend_id = "diffusers_0.40.0_tensor_body"

    def __init__(self):
        installed = distribution("diffusers")
        if installed.version != DIFFUSERS_VERSION:
            raise RuntimeError(f"Expected diffusers=={DIFFUSERS_VERSION}; got {installed.version}")
        root = "diffusers/modular_pipelines/minimax_h3/"
        pack_path = Path(installed.locate_file(root + "before_denoise.py"))
        unpack_path = Path(installed.locate_file(root + "decoders.py"))
        pack_text, unpack_text = pack_path.read_text(), unpack_path.read_text()
        pack = next(
            n
            for n in ast.parse(pack_text).body
            if isinstance(n, ast.FunctionDef) and n.name == "patchify_video_latents"
        )
        cls = next(
            n
            for n in ast.parse(unpack_text).body
            if isinstance(n, ast.ClassDef) and n.name == "MiniMaxH3AfterDenoiseStep"
        )
        method = next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__"
        )
        # Start after get_block_state; stop at the video result, before audio work.
        start = next(
            i
            for i, n in enumerate(method.body)
            if isinstance(n, ast.Assign)
            and ast.unparse(n.targets[0]) == "(patch_t, patch_h, patch_w)"
        )
        end = next(
            i
            for i, n in enumerate(method.body)
            if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "block_state.latents"
        )
        body = method.body[start : end + 1]
        unpack = ast.parse("def unpack_video(components, block_state):\n    pass").body[0]
        unpack.body = [
            *body,
            ast.Return(value=ast.parse("block_state.latents", mode="eval").body),
        ]
        namespace = {"torch": torch}
        ast.fix_missing_locations(unpack)
        exec(
            compile(ast.Module(body=[pack], type_ignores=[]), str(pack_path), "exec"),
            namespace,
        )
        exec(
            compile(ast.Module(body=[unpack], type_ignores=[]), str(unpack_path), "exec"),
            namespace,
        )
        self._pack, self._unpack = namespace[pack.name], namespace[unpack.name]
        self.components = SimpleNamespace(patch_size=(1, 2, 2), vae_latent_channels=24)
        self.provenance = {
            "version": installed.version,
            "pack_source": root + "before_denoise.py:patchify_video_latents",
            "unpack_source": root + "decoders.py:MiniMaxH3AfterDenoiseStep.__call__",
            "pack_file_sha256": hashlib.sha256(pack_text.encode()).hexdigest(),
            "unpack_file_sha256": hashlib.sha256(unpack_text.encode()).hexdigest(),
            "unpack_statement_lines": [body[0].lineno, body[-1].end_lineno],
            "scope": "unchanged upstream tensor statements; batch views and state adapter included",
            "gradient_scope": "autograd of tensor body; upstream decoder entry is no_grad",
        }

    def pack(self, x):
        return self._pack(x, (1, 2, 2)).view(x.shape[0], -1, 96)

    def unpack(self, tokens, shape):
        state = SimpleNamespace(
            latents=tokens.view(-1, 96),
            num_condition_video_rows=0,
            num_latent_frames=shape[2],
            latent_height=shape[3],
            latent_width=shape[4],
        )
        return self._unpack(self.components, state)


class VllmOmniVideoPatch:
    """Call the pinned vendored functions directly, without RL-Kernel validation."""

    backend_id = "vllm_omni_b1fc5fc_video_tensor_body"
    provenance = {
        "revision": UPSTREAM_REVISION,
        "source": UPSTREAM_SOURCE,
        "file_sha256": UPSTREAM_SOURCE_SHA256,
        "scope": "unchanged vendored video functions with batch views",
    }

    def pack(self, x):
        return minimax_h3_patchify_video_latent(x, patch_size=(1, 2, 2)).view(x.shape[0], -1, 96)

    def unpack(self, tokens, shape):
        _, c, t, h, w = shape
        return minimax_h3_unpatchify_video_tokens(
            tokens.view(-1, 96), latent_shape=(t, h // 2, w // 2, c), patch_size=(1, 2, 2)
        )


SHAPES = {"small": (3, 6, 10), "medium": (8, 24, 42), "768p": (32, 48, 84)}
# Diffusers 0.40.0 uses spatial compression by 16. Aligned video frame counts
# F=17n+5 produce latent T=5n+2 through the VAE. Keep the measured shapes here
# so reproducing the workload does not require local experiment scripts.
ASPECTS = {
    "16x9": (48, 84),
    "9x16": (84, 48),
    "1x1": (48, 48),
    "4x3": (48, 64),
    "3x4": (64, 48),
    "21x9": (42, 96),
}
SHAPES.update(
    {f"{aspect}_t{t}": (t, h, w) for t in (37, 72, 102) for aspect, (h, w) in ASPECTS.items()}
)
SHAPES.update({f"{aspect}_t1": (1, *ASPECTS[aspect]) for aspect in ("16x9", "9x16")})


def workload_cases():
    """Return the report's 28 layouts; batch sweeps do not imply traffic frequency."""
    cases = [(f"{aspect}_t{t}", 1) for t in (37, 72, 102) for aspect in ASPECTS]
    cases += [(f"16x9_t{t}", b) for t in (37, 102) for b in (2, 3, 4, 8)]
    cases += [(f"{aspect}_t1", 1) for aspect in ("16x9", "9x16")]
    return cases


OPERATIONS = tuple(
    f"{direction}_{scope}"
    for direction in ("pack", "unpack")
    for scope in ("forward", "backward", "forward_backward")
)
BACKENDS = ("pytorch", "vllm_omni", "diffusers")
KERNEL_IDS = {
    "diffusers": "upstream.reshape_permute_contiguous",
    "vllm_omni": "vllm_omni.reshape_einsum_contiguous",
    "pytorch": "rlkernel.vllm_omni_adapter",
}


def _operator(name, dtype):
    if name in ("diffusers", "vllm_omni"):
        op = DiffusersVideoPatch() if name == "diffusers" else VllmOmniVideoPatch()
        return op, {
            "requested_backend": op.backend_id,
            "actual_backend": op.backend_id,
            "source": op.provenance,
        }
    session = KernelRegistry().semantic.session()
    resolution = session.resolve(
        semantic_op="video_patch_pack_unpack",
        requested_backend=f"{name}-video-patch-pack-unpack-v1",
        target="training",
        requirements=OperatorRequirements(
            device="cuda",
            dtype=str(dtype).removeprefix("torch."),
            topology={"world_size": 1, "sharding": "unsharded"},
            alignment_properties={
                "channels": 24,
                "patch_size": {"t": 1, "h": 2, "w": 2},
                "bitwise": True,
            },
        ),
    )
    op = session.instantiate(resolution)
    provenance = session.instance_provenance(resolution, op).to_dict()
    assert op.backend_id == provenance["backend_id"] == resolution.trace.concrete_backend
    return op, {"resolution": resolution.trace.to_dict(), "instance": provenance}


def _environment():
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        sha, dirty = "unknown", None
    try:
        driver = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
            )
            .splitlines()[0]
            .strip()
        )
    except (OSError, subprocess.CalledProcessError, IndexError):
        driver = "unknown"
    properties = torch.cuda.get_device_properties(0)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": sha,
        "git_dirty": dirty,
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "upstream_adapter_sha256": hashlib.sha256(
            inspect.getsource(DiffusersVideoPatch).encode()
        ).hexdigest(),
        "reused_upstream": VllmOmniVideoPatch.provenance,
        "source_sha256": {
            name: hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest()
            for name in (
                "rl_engine/reference/packing/video_patch_pack_unpack.py",
                "rl_engine/runtime/registry.py",
            )
        },
        "gpu": properties.name,
        "gpu_count": torch.cuda.device_count(),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "total_vram_bytes": properties.total_memory,
        "driver": driver,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": sys.version.split()[0],
        "image": os.environ.get("RLK_BENCH_IMAGE", "unknown"),
    }


def _equal_bits(actual, expected):
    return (
        actual.shape == expected.shape
        and actual.dtype == expected.dtype
        and torch.equal(
            actual.detach().contiguous().view(torch.uint8),
            expected.detach().contiguous().view(torch.uint8),
        )
    )


def _summary(values):
    ordered = sorted(values)
    index = 0.95 * (len(ordered) - 1)
    low = int(index)
    p95 = ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (index - low)
    return {
        "median": statistics.median(values),
        "p95": p95,
        "min": ordered[0],
        "max": ordered[-1],
        "samples": values,
    }


def _sample(functions, warmup, count, *, graph, graph_factory, expected, graph_calls):
    for fn in functions.values():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    names = tuple(functions)
    orders = tuple(itertools.permutations(names))
    events = {
        name: (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for name in names
    }
    # Create the underlying CUDA events outside host timing as well.
    for start, end in events.values():
        start.record()
        end.record()
    torch.cuda.synchronize()
    event_ms = {name: [] for name in names}
    wall_ms = {name: [] for name in names}
    for sample in range(count):
        for name in orders[sample % len(orders)]:
            start, end = events[name]
            torch.cuda.synchronize()
            wall_start = time.perf_counter()
            start.record()
            measured_output = functions[name]()
            end.record()
            end.synchronize()
            wall_end = time.perf_counter()
            event_ms[name].append(start.elapsed_time(end))
            wall_ms[name].append((wall_end - wall_start) * 1000)
            del measured_output
    result = {
        name: {
            "eager_event_ms": _summary(event_ms[name]),
            "eager_wall_ms": _summary(wall_ms[name]),
        }
        for name in names
    }
    if graph:
        captured = {}
        for name in functions:
            try:
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    # Autograd backward runs on the stream of its forward nodes.
                    # Prepare backward-only graphs on the capture stream too.
                    fn = graph_factory(name)
                    for _ in range(warmup):
                        fn()
                torch.cuda.current_stream().wait_stream(stream)
                cuda_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(cuda_graph, stream=stream):
                    for _ in range(graph_calls):
                        output = fn()
                cuda_graph.replay()
                torch.cuda.synchronize()
                if not _equal_bits(output, expected):
                    raise AssertionError(f"{name} CUDA Graph replay differs from reference")
                # CUDA Graph retains device pointers, not Python tensor owners.
                # Keep the closure (and its static inputs) alive through every replay.
                captured[name] = (cuda_graph, output, fn)
                result[name]["cuda_graph_status"] = "pass"
            except RuntimeError as exc:
                raise RuntimeError(f"{name} CUDA Graph capture/replay failed") from exc
        torch.cuda.synchronize()
        graph_ms = {name: [] for name in captured}
        graph_orders = tuple(itertools.permutations(captured))
        for sample in range(count):
            for name in graph_orders[sample % len(graph_orders)]:
                start, end = events[name]
                start.record()
                captured[name][0].replay()
                end.record()
                end.synchronize()
                graph_ms[name].append(start.elapsed_time(end) / graph_calls)
        for name, values in graph_ms.items():
            if not _equal_bits(captured[name][1], expected):
                raise AssertionError(f"{name} repeated Graph replay changed output")
            result[name]["cuda_graph_ms"] = _summary(values)
    return result


def _case(shape, dtype, warmup, samples, graph, graph_calls):
    x = torch.randn(shape, device="cuda", dtype=dtype).requires_grad_()
    expected_tokens = pack_video_patch_reference(x)
    tokens = torch.randn(expected_tokens.shape, device="cuda", dtype=dtype).requires_grad_()
    grad_tokens = torch.randn_like(tokens)
    grad_latents = torch.randn_like(x)
    expected_unpack = unpack_video_patch_reference(tokens, shape)
    expected_pack_grad = unpack_video_patch_reference(grad_tokens, shape)
    expected_unpack_grad = pack_video_patch_reference(grad_latents)
    selections = {name: _operator(name, dtype) for name in BACKENDS}
    operators = {name: selection[0] for name, selection in selections.items()}
    for name, op in operators.items():
        checks = (
            _equal_bits(op.pack(x), expected_tokens),
            _equal_bits(op.unpack(tokens, shape), expected_unpack),
            _equal_bits(torch.autograd.grad(op.pack(x), x, grad_tokens)[0], expected_pack_grad),
            _equal_bits(
                torch.autograd.grad(op.unpack(tokens, shape), tokens, grad_latents)[0],
                expected_unpack_grad,
            ),
        )
        if not all(checks):
            raise AssertionError(f"{name} failed bitwise checks at {shape}, {dtype}: {checks}")

    def factory(operation, op):
        # Each timed backend owns separate, equal input buffers. For backward-only,
        # build the forward graph before timing and reuse it without .grad accumulation.
        is_pack = operation.startswith("pack_")
        source = (x if is_pack else tokens).detach().clone().requires_grad_()
        gradient = (grad_tokens if is_pack else grad_latents).detach().clone()
        forward = (lambda: op.pack(source)) if is_pack else (lambda: op.unpack(source, shape))
        scope = operation.split("_", 1)[1]
        prepared = forward() if scope == "backward" else None

        def run():
            if scope == "forward":
                with torch.no_grad():
                    return forward()
            if scope == "backward":
                return torch.autograd.grad(prepared, source, gradient, retain_graph=True)[0]
            return torch.autograd.grad(forward(), source, gradient)[0]

        return run

    expected_by_operation = {
        "pack_forward": expected_tokens,
        "unpack_forward": expected_unpack,
        "pack_backward": expected_pack_grad,
        "unpack_backward": expected_unpack_grad,
        "pack_forward_backward": expected_pack_grad,
        "unpack_forward_backward": expected_unpack_grad,
    }
    rows = []
    for operation in OPERATIONS:
        bytes_moved = (4 if "forward_backward" in operation else 2) * x.numel() * x.element_size()
        functions = {name: factory(operation, operators[name]) for name in BACKENDS}
        memory = {}
        for name, fn in functions.items():
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            output = fn()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            del output
            memory[name] = {
                "peak_allocated_bytes": peak,
                "peak_extra_bytes": peak - before,
            }
        timings = _sample(
            functions,
            warmup,
            samples,
            graph=graph,
            graph_factory=lambda name, operation=operation: factory(operation, operators[name]),
            graph_calls=graph_calls,
            expected=expected_by_operation[operation],
        )
        for name in functions:
            row = timings[name]
            row.update(
                {
                    "operation": operation,
                    "requested_backend": name,
                    "actual_backend": operators[name].backend_id,
                    "kernel_id": KERNEL_IDS[name],
                    "device": str(x.device),
                    "status": "pass",
                    "bitwise_mismatch_count": 0,
                    **memory[name],
                    "logical_bytes_moved": bytes_moved,
                    "effective_bandwidth_gbps": bytes_moved
                    / (row["eager_event_ms"]["median"] * 1e-3)
                    / 1e9,
                }
            )
            rows.append(row)
    return {
        "shape": list(shape),
        "dtype": str(dtype).removeprefix("torch."),
        "backend_provenance": {name: selection[1] for name, selection in selections.items()},
        "measurements": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shapes", default="small,medium,768p")
    parser.add_argument(
        "--workloads",
        action="store_true",
        help="Run the report's 28 layouts; replaces --shapes and --batches",
    )
    parser.add_argument("--batches", default="1,2,4,8")
    parser.add_argument("--dtypes", default="float32,float16,bfloat16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--graph-calls", type=int, default=32)
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--no-graphs", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.version.hip is not None:
        parser.error("an NVIDIA CUDA GPU is required")
    shapes = tuple(args.shapes.split(","))
    batches = tuple(int(x) for x in args.batches.split(","))
    dtypes = tuple(args.dtypes.split(","))
    if not set(shapes) <= set(SHAPES) or not all(x > 0 for x in batches):
        parser.error("invalid shape or batch")
    if not set(dtypes) <= {"float32", "float16", "bfloat16"}:
        parser.error("invalid dtype")
    if args.warmup < 1 or args.samples < 1 or args.graph_calls < 1:
        parser.error("warmup and samples must be positive")
    layouts = workload_cases() if args.workloads else list(itertools.product(shapes, batches))
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "results.json"
    result = {
        "schema_version": "rlkernel.video_patch_pack_unpack.benchmark.v5",
        "command": shlex.join([sys.executable, *sys.argv]),
        "comparison_contract": {
            "baseline": "pinned vLLM-Omni video functions; Diffusers 0.40.0 second baseline",
            "accuracy_reference": "independent RL-Kernel PyTorch reference and upstream Diffusers",
            "scope": "single-GPU operator-only; no model checkpoint or VAE",
            "correctness": "bitwise forward output and first-order gradient equality",
            "timing": "same input, shape, dtype, and operation across backends",
        },
        "environment": _environment(),
        "upstream": DiffusersVideoPatch().provenance,
        "methodology": {
            "shape_presets": {name: list(SHAPES[name]) for name, _ in layouts},
            "workload_cases": workload_cases() if args.workloads else None,
            "batches": sorted({batch for _, batch in layouts}),
            "dtypes": list(dtypes),
            "operations": list(OPERATIONS),
            "warmup": args.warmup,
            "samples": args.samples,
            "seed": args.seed,
            "graph_calls_per_replay": args.graph_calls,
            "backward": (
                "autograd.grad only; forward graph prepared outside timing, retain_graph=True"
            ),
            "forward": "no_grad forward only",
            "forward_backward": "forward + autograd.grad; no accumulated .grad",
            "backend_order": "balanced 6-permutation cycle per operation",
            "eager_event": "CUDA-event device-stream interval for one Python call",
            "eager_wall": "synchronized host wall time for one Python call",
            "cuda_graph": (
                "CUDA-event interval for one replay / calls per captured graph; "
                "hot-buffer steady state"
            ),
            "peak_memory": "total and incremental allocated VRAM for one eager call",
            "effective_bandwidth": (
                "2 * input bytes for forward/backward-only, 4 * for forward+backward, "
                "divided by eager-event median"
            ),
            "graph_enabled": not args.no_graphs,
        },
        "cases": [],
    }
    for (shape_name, batch), dtype_name in itertools.product(layouts, dtypes):
        shape = (batch, 24, *SHAPES[shape_name])
        case = _case(
            shape,
            getattr(torch, dtype_name),
            args.warmup,
            args.samples,
            not args.no_graphs,
            args.graph_calls,
        )
        case["shape_preset"] = shape_name
        result["cases"].append(case)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"completed {shape_name} B={batch} {dtype_name}", flush=True)
    print(f"BENCHMARK_COMPLETE {output}", flush=True)


if __name__ == "__main__":
    main()
