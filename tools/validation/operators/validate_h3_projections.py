#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Qualify both H3 projection candidates; GPU execution is required."""
import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from rl_engine.contracts.numerical import load_contract  # noqa: E402
from rl_engine.ops.gemm.h3_projections import (  # noqa: E402
    H3_CHECKPOINT_REVISION,
    H3_MAX_ROWS,
    SPECS,
    H3ProjectionOp,
    _require_backend,
    _weight_gradient,
)
from rl_engine.validation.operators.h3_projection_validation import (  # noqa: E402
    comparison,
    logical_keys,
    reference_projection,
    training_projection,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_digest(tensor):
    return {
        "sha256": hashlib.sha256(
            tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
        ).hexdigest(),
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
    }


def provenance(args):
    names = [
        "rl_engine/ops/gemm/h3_projections.py",
        "rl_engine/backends/shared/triton/gemm/h3_projection_fp32.py",
        "rl_engine/backends/shared/triton/gemm/det_gemm.py",
        "rl_engine/ops/autograd/canonical_backward.py",
        "rl_engine/ops/autograd/backward_runtime.py",
        "rl_engine/validation/operators/h3_projection_validation.py",
        "rl_engine/runtime/registry.py",
        "rl_engine/contracts/numerical.py",
        "rl_engine/contracts/profiles/precision/ws1.json",
        "tools/validation/operators/validate_h3_projections.py",
    ]

    def git(*cmd):
        return subprocess.check_output(["git", *cmd], cwd=ROOT, text=True).strip()

    return {
        "source_revision": git("rev-parse", "HEAD"),
        "working_tree_clean": not bool(git("status", "--porcelain")),
        "source_hashes": {name: sha256(ROOT / name) for name in names},
        "checkpoint_revision": H3_CHECKPOINT_REVISION,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "contract_version": load_contract()["version"],
        "contract_sha256": sha256(ROOT / "rl_engine/contracts/profiles/precision/ws1.json"),
        "method": "FP32 reference from quantized BF16 inputs; raw-byte invariance; TF32 disabled",
        "seed": args.seed,
        "rows": args.rows,
        "families": args.families,
        "repeats": args.repeats,
        "operators": args.operators,
        "task_layouts_requested": args.task_layouts,
    }


def timed(fn, repeats):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    base_memory = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    # Wall clock includes host-side canonical-key validation, copies and reduction prep.
    for _ in range(repeats):
        start = time.perf_counter()
        value = fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
        del value
    return {
        "median_ms": statistics.median(samples),
        "samples_ms": samples,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "incremental_peak_bytes": torch.cuda.max_memory_allocated() - base_memory,
        "timing_scope": "synchronized wall time, includes wrapper/copies/padding",
    }


def add_checks(report, actual, expected, transform, *, accuracy=False):
    for i, (left, right) in enumerate(zip(actual, expected, strict=True)):
        judgment = "forward_" if i == 0 else "gradient_"
        judgment += "accuracy" if accuracy else "invariance"
        row = comparison(left, right, judgment)
        row.update(tensor=("y", "dx", "dw")[i], transform=transform)
        report["checks"].append(row)


def row_local(op, x, w, dy):
    x = x.detach().clone().requires_grad_(True)
    y = op(x, w.detach())
    (dx,) = torch.autograd.grad(y, x, dy)
    return y.detach(), dx.detach()


def run_case(x, w, dy, label, repeats, *, kind):
    input_dim, output_dim = SPECS[kind]
    rows = x.shape[0]
    keys = logical_keys(rows, x.device)
    op = H3ProjectionOp(kind)
    case = {"rows": rows, "input_family": label, "checks": [], "benchmarks": {}}
    canonical, traces = training_projection(op, x, w, dy, keys)
    case["execution_traces"] = traces
    case["output_digests"] = {
        name: tensor_digest(value) for name, value in zip(("y", "dx", "dw"), canonical, strict=True)
    }
    reference = reference_projection(x, w, dy)
    add_checks(case, canonical, reference, "fp32_reference", accuracy=True)
    del reference
    repeat, _ = training_projection(op, x, w, dy, keys)
    add_checks(case, repeat, canonical, "repeat")
    del repeat
    order = torch.arange(rows - 1, -1, -1, device=x.device)
    inverse = torch.argsort(order)
    permuted, _ = training_projection(op, x[order], w, dy[order], keys[order])
    add_checks(
        case,
        (permuted[0][inverse], permuted[1][inverse], permuted[2]),
        canonical,
        "permuted_logical_rows",
    )
    del permuted
    chunk_size = 31 if rows <= 32768 else 4093
    chunked, _ = training_projection(op, x, w, dy, keys, chunk_size=chunk_size)
    add_checks(case, chunked, canonical, f"chunk{chunk_size}_single_backward")
    del chunked
    # Force non-contiguous activations AND native-shaped non-contiguous weights.
    layout_x, layout_w = x.t().contiguous().t(), w.t().contiguous().t()
    layout, _ = training_projection(op, layout_x, layout_w, dy, keys)
    add_checks(case, layout, canonical, "strided_input_and_weight")
    del layout, layout_x, layout_w
    pad = 17
    padded_x = torch.cat((x, torch.ones(pad, input_dim, device=x.device, dtype=x.dtype)))
    padded_dy = torch.cat((dy, torch.zeros(pad, output_dim, device=x.device, dtype=dy.dtype)))
    padded_keys = torch.cat((keys, torch.full((pad, 2), -1, device=x.device, dtype=torch.int64)))
    if rows + pad <= H3_MAX_ROWS:
        padded, _ = training_projection(op, padded_x, w, padded_dy, padded_keys)
        add_checks(
            case,
            (padded[0][:rows], padded[1][:rows], padded[2]),
            canonical,
            "inactive_padding",
        )
        del padded
    del padded_x, padded_dy, padded_keys
    if rows % 2 == 0:
        batched, _ = training_projection(
            op,
            x.reshape(2, rows // 2, input_dim),
            w,
            dy.reshape(2, rows // 2, output_dim),
            keys,
        )
        add_checks(
            case,
            (
                batched[0].reshape(rows, output_dim),
                batched[1].reshape(rows, input_dim),
                batched[2],
            ),
            canonical,
            "batch2_reshape",
        )
        del batched
    # Compare only row-local tensors when the effective logical row set changes.
    one = row_local(op, x[:1], w, dy[:1])
    mutated_x, mutated_dy = x.clone(), dy.clone()
    mutated_x[1:].neg_()
    mutated_dy[1:].zero_()
    mutated = row_local(op, mutated_x, w, mutated_dy)
    for i in (0, 1):
        for name, value in (
            ("singleton_vs_batch", canonical[i][:1]),
            ("other_rows_mutated", mutated[i][:1]),
        ):
            check = comparison(
                value, one[i], "forward_invariance" if i == 0 else "gradient_invariance"
            )
            check.update(tensor=("y", "dx")[i], transform=name)
            case["checks"].append(check)
    del one, mutated, mutated_x, mutated_dy, canonical
    extension = _require_backend(x.device, op.backend)
    case["benchmarks"]["candidate_forward"] = timed(lambda: op(x, w), repeats)
    case["benchmarks"]["candidate_dx"] = timed(
        lambda: extension.det_gemm_fwd(dy.contiguous(), w.contiguous()), repeats
    )
    case["benchmarks"]["candidate_dw_canonical_order"] = timed(
        lambda: _weight_gradient(extension, x, dy), repeats
    )
    case["benchmarks"]["candidate_forward_backward"] = timed(
        lambda: training_projection(op, x, w, dy, keys), repeats
    )
    case["benchmarks"]["torch_bf16_forward"] = timed(
        lambda: torch.nn.functional.linear(x, w), repeats
    )

    def torch_training():
        tx = x.detach().requires_grad_(True)
        tw = w.detach().requires_grad_(True)
        ty = torch.nn.functional.linear(tx, tw)
        return torch.autograd.grad(ty, (tx, tw), dy)

    case["benchmarks"]["torch_bf16_forward_backward"] = timed(torch_training, repeats)
    case["passed"] = all(check["passed"] for check in case["checks"])
    return case


def make_inputs(rows, family, seed, kind):
    input_dim, output_dim = SPECS[kind]
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, input_dim, device="cuda", generator=generator)
    w = torch.randn(output_dim, input_dim, device="cuda", generator=generator) / input_dim**0.5
    if family == "cancellation":
        even_columns, odd_columns = slice(0, None, 2), slice(1, None, 2)
        x[:, odd_columns] = -x[:, even_columns]
        w[:, odd_columns] = w[:, even_columns] + 1e-5
    elif family == "normalized":
        x = x / x.square().mean(-1, keepdim=True).add(1e-5).sqrt()
    dy = torch.randn(rows, output_dim, device="cuda", generator=generator) * 0.1
    return x.bfloat16(), w.bfloat16(), dy.bfloat16()


def task_layouts():
    """Pinned diffusers v0.40.0 FL2VA geometry; synthetic activations, B=1.

    modular_pipeline.video_latent_num_frames/audio_latent_num_frames and
    before_denoise.MiniMaxH3BuildPackedSequence.build_packed_sequence define
    the frame alignment and packed-row sum. Text length is an explicit fixture
    choice, not a captured prompt. Two keyframes and stereo audio are included.
    """
    result = []
    for frames in (124, 243, 345):
        latent_frames = (frames - 5) // 17 * 5 + 2
        video_rows = latent_frames * (768 // 32) * (1344 // 32)
        audio_rows = 2 * round(frames / 24 * 40)
        conditions, text_rows = 2 * 24 * 42, 256
        result.append(
            {
                "name": f"fl2va_768p_{frames}f",
                "source": "diffusers@v0.40.0",
                "frames": frames,
                "height": 768,
                "width": 1344,
                "batch_size": 1,
                "latent_frames": latent_frames,
                "video_rows": video_rows,
                "audio_rows": audio_rows,
                "condition_rows": conditions,
                "text_rows": text_rows,
                "rows": video_rows + audio_rows + conditions + text_rows,
                "activation_kind": "synthetic",
            }
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--operators", default=",".join(SPECS))
    parser.add_argument("--rows", default="1,31,32,33,127,128,129,256,513,1024,4096,8192,32768")
    parser.add_argument("--families", default="random,normalized,cancellation")
    parser.add_argument("--task-layouts", action="store_true", help="Add pinned 768p FL2VA layouts")
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--fixture",
        action="append",
        default=[],
        help="Repeat for each weights_only fixture",
    )
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "h3-projections-qualification-v1",
        "status": "running",
        "acceptance_complete": False,
        "cases": [],
        "scope": "single SM90 CUDA device, BF16, eager, operator-only",
    }

    def save():
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    try:
        kinds = args.operators.split(",")
        rows = [int(n) for n in args.rows.split(",")]
        layouts = task_layouts() if args.task_layouts else []
        report["task_layouts"] = layouts
        rows = list(dict.fromkeys(rows + [layout["rows"] for layout in layouts]))
        families = args.families.split(",")
        if not kinds or not set(kinds) <= SPECS.keys():
            raise ValueError("invalid operator selection")
        if not rows or any(not 1 <= n <= H3_MAX_ROWS for n in rows):
            raise ValueError(f"rows must be in [1,{H3_MAX_ROWS}]")
        if args.repeats < 1 or not set(families) <= {
            "random",
            "normalized",
            "cancellation",
        }:
            raise ValueError("invalid repeats or families")
        report["provenance"] = provenance(args)
        if not torch.cuda.is_available() or torch.version.hip is not None:
            raise RuntimeError("an NVIDIA SM90 GPU is required; local CPU tests are not acceptance")
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        device = torch.cuda.get_device_properties(0)
        report["hardware"] = {
            "gpu_model": device.name,
            "architecture": list(torch.cuda.get_device_capability()),
            "device_count_used": 1,
            "memory_bytes": device.total_memory,
        }
        for kind in kinds:
            _require_backend(torch.device("cuda"), H3ProjectionOp(kind).backend)
            for family in families:
                for count in rows:
                    values = make_inputs(count, family, args.seed, kind)
                    case = run_case(*values, family, args.repeats, kind=kind)
                    case["semantic_op"] = kind
                    case["task_layouts"] = [layout for layout in layouts if layout["rows"] == count]
                    report["cases"].append(case)
                    save()
                    print(f"{kind} {family} M={count}: {case['passed']}", flush=True)
                    del values, case
        for filename in args.fixture:
            fixture = torch.load(filename, weights_only=True, map_location="cpu")
            meta = fixture["metadata"]
            kind = meta["semantic_op"]
            if kind not in kinds or meta["checkpoint_revision"] != H3_CHECKPOINT_REVISION:
                raise ValueError("fixture operator or checkpoint mismatch")
            if meta.get("activation_kind") not in {"synthetic", "captured"}:
                raise ValueError("fixture activation provenance missing")
            weight_digest = tensor_digest(fixture["weight"])["sha256"]
            if weight_digest != meta["weight_sha256"]:
                raise ValueError("fixture weight checksum mismatch")
            expected_key = (
                "transformer_blocks.0.ff.net.0.proj.weight"
                if kind == "h3_ffn_gate_up_gemm"
                else "transformer_blocks.0.attn.to_out.0.weight"
            )
            if meta.get("weight_key") != expected_key:
                raise ValueError("fixture weight identity mismatch")
            values = [fixture[k].to("cuda") for k in ("x", "weight", "grad_output")]
            case = run_case(*values, "checkpoint_fixture", args.repeats, kind=kind)
            case.update(
                semantic_op=kind,
                fixture_sha256=sha256(filename),
                fixture_metadata={
                    k: meta[k]
                    for k in (
                        "semantic_op",
                        "checkpoint_revision",
                        "weight_key",
                        "weight_sha256",
                        "activation_kind",
                        "seed",
                        "index_sha256",
                        "header_sha256",
                    )
                },
            )
            report["cases"].append(case)
            save()
        passed = bool(report["cases"]) and all(c["passed"] for c in report["cases"])
        report["status"] = "passed" if passed else "failed"
        report["limitations"] = [
            "No full-model acceptance or production promotion",
            "Captured model activations and independent environment replay were not performed",
            "PyTorch timings do not establish a byte-invariant baseline",
        ]
        return 0 if passed else 1
    except Exception as error:
        report["status"] = "error"
        # Raw reports stay local until provenance and privacy review for publication.
        report["error"] = type(error).__name__ + ": " + str(error)
        return 2
    finally:
        save()


if __name__ == "__main__":
    raise SystemExit(main())
