#!/usr/bin/env python3
# isort: skip_file
# SPDX-License-Identifier: Apache-2.0
"""Full-op reuse screening, followed by bytewise sample-batch invariance.

Run each mode in a fresh process. cublas_policy reproduces vLLM's two cuBLAS
workspace settings, not the complete vLLM integration. The vLLM, SGLang and
Megatron modes use each project's batch-invariant persistent matmul with the
same FP32 VJP adapter; vllm_liger and vllm_flashattn also replace the final
LayerNorm with Liger's or flash-attention's Triton LayerNorm. No backend math
flags are changed. FP64 comparisons use the declared storage boundaries/STE.
"""
import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument(
    "--mode",
    choices=[
        "native",
        "cublas_policy",
        "fixed",
        "vllm_local",
        "vllm_liger",
        "vllm_flashattn",
        "sglang_local",
        "megatron_local",
    ],
    required=True,
)
p.add_argument(
    "--phase",
    choices=["reuse", "bi", "workload", "accuracy", "accuracy_large", "parameters"],
    required=True,
)
p.add_argument("--model", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--capture-failures", action="store_true")
p.add_argument("--megatron-src", type=Path, help="Megatron-LM checkout for megatron_local")
p.add_argument("--flash-attn-src", type=Path, help="flash-attention checkout for vllm_flashattn")
a = p.parse_args()
if a.mode == "cublas_policy":
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["CUBLASLT_WORKSPACE_SIZE"] = "1"

if a.mode in ("vllm_local", "vllm_liger", "vllm_flashattn", "sglang_local", "megatron_local"):
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
if a.mode == "megatron_local":
    assert a.megatron_src, "--megatron-src is required for megatron_local"
    sys.path.insert(0, str(a.megatron_src))
if a.mode == "sglang_local":
    os.environ["SGLANG_ENABLE_JIT_DEEPGEMM"] = "0"
    os.environ["SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM"] = "0"

# Imports follow explicit subprocess environment setup.
import torch  # noqa: E402
from benchmarks.models.qwenimage.streams import STREAMS  # noqa: E402
from benchmarks.models.qwenimage.adaln_evidence import (  # noqa: E402
    load_weights,
    source_state,
    benchmark,
    equal_bytes,
    compare,
)
from rl_engine.backends.cuda.model_specific.qwenimage.adaln_mod_projection import (  # noqa: E402
    QwenImageAdaLNModProjectionCudaOp,
)
from rl_engine.backends.cuda.model_specific.qwenimage.final_adaln_out import (  # noqa: E402
    QwenImageFinalAdaLNOutCudaOp,
)
from rl_engine.reference.model_specific.qwenimage.adaln_mod_projection import (  # noqa: E402
    QwenImageAdaLNModProjectionReference,
    reference_projection,
)
from rl_engine.reference.model_specific.qwenimage.final_adaln_out import (  # noqa: E402
    QwenImageFinalAdaLNOutReference,
    reference_final,
)

torch.set_num_threads(8)
assert not torch.backends.cuda.matmul.allow_tf32, "TF32 must already be disabled"
report = dict(
    mode=a.mode,
    phase=a.phase,
    accuracy_seed=a.seed,
    source=source_state(),
    cases=[],
    environment=dict(
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        workspace={
            k: os.environ.get(k) for k in ["CUBLAS_WORKSPACE_CONFIG", "CUBLASLT_WORKSPACE_SIZE"]
        },
        tf32=torch.backends.cuda.matmul.allow_tf32,
    ),
    scope=(
        "sample batch BI; final uses S=3 for large B, long S tested separately; "
        "shared parameter gradients: same ordered inputs, repeat only"
    ),
)
report["packages"] = {}
for lib in [
    "torch",
    "triton",
    "diffusers",
    "vllm",
    "sglang",
    "liger-kernel",
    "transformer_engine",
    "megatron-core",
]:
    try:
        report["packages"][lib] = importlib.metadata.version(lib)
    except importlib.metadata.PackageNotFoundError:
        report["packages"][lib] = None
if a.mode == "vllm_flashattn":
    assert a.flash_attn_src, "--flash-attn-src is required for vllm_flashattn"
    report["packages"]["flash-attention-source"] = subprocess.check_output(
        ["git", "-C", str(a.flash_attn_src), "rev-parse", "HEAD"], text=True
    ).strip()
if a.mode == "megatron_local":
    report["packages"]["megatron-lm-source"] = subprocess.check_output(
        ["git", "-C", str(a.megatron_src), "rev-parse", "HEAD"], text=True
    ).strip()


def save():
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")


def run(fn, inp, upstream, parameters=False):
    names = list(inp) if parameters else [k for k in inp if k not in ("weight", "bias")]
    vals = {k: v.detach().requires_grad_(k in names) for k, v in inp.items()}
    y = fn(**vals)
    gs = torch.autograd.grad(y, tuple(vals[k] for k in names), upstream)
    return {"output": y.detach(), **{k: g.detach() for k, g in zip(names, gs)}}


def sliced(inp, idx):
    return {k: (v if k in ("weight", "bias") else v[idx]) for k, v in inp.items()}


def stats(left, right):
    return {
        k: int(
            (
                v.contiguous().view(torch.uint8).reshape(v.shape[0], -1)
                != right[k].contiguous().view(torch.uint8).reshape(v.shape[0], -1)
            )
            .any(1)
            .sum()
        )
        for k, v in left.items()
    }


def metric(x, y):
    x = x.double().cpu()
    y = y.double().cpu()
    return float((x - y).abs().max() / y.abs().max().clamp_min(1e-30))


for stream, stem in STREAMS:
    if a.mode in ("vllm_liger", "vllm_flashattn") and stream != "final":
        continue
    weights, manifest = load_weights(a.model, stem)
    final = stream == "final"
    fn = (
        (QwenImageFinalAdaLNOutCudaOp() if final else QwenImageAdaLNModProjectionCudaOp())
        if a.mode == "fixed"
        else (
            QwenImageFinalAdaLNOutReference() if final else QwenImageAdaLNModProjectionReference()
        )
    )
    if a.mode in ("vllm_local", "vllm_liger", "vllm_flashattn", "sglang_local", "megatron_local"):
        from benchmarks.models.qwenimage.adaln_reuse_candidate import Final, Projection

        if a.mode == "sglang_local":
            import benchmarks.models.qwenimage.adaln_reuse_candidate as adapter
            from sglang.srt.batch_invariant_ops.batch_invariant_ops import (
                matmul_persistent,
            )

            adapter.matmul_persistent = matmul_persistent
        if a.mode == "megatron_local":
            import benchmarks.models.qwenimage.adaln_reuse_candidate as adapter
            from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
                matmul_persistent,
            )

            adapter.matmul_persistent = matmul_persistent
        fn = Final() if final else Projection()
        if a.mode == "vllm_liger":
            from benchmarks.models.qwenimage.adaln_reuse_candidate import FinalLiger

            fn = FinalLiger()
        if a.mode == "vllm_flashattn":
            from benchmarks.models.qwenimage.adaln_reuse_candidate import FinalFlashAttn

            fn = FinalFlashAttn(a.flash_attn_src)
    for dtype in [torch.bfloat16, torch.float32]:
        w, b = [weights[k].cuda().to(dtype) for k in ("weight", "bias")]
        case = dict(stream=stream, dtype=str(dtype), weights=manifest, checks=[])
        report["cases"].append(case)

        def inputs(batch, seq=3, seed=0):
            gen = torch.Generator(device="cuda").manual_seed(seed)
            inp = dict(
                temb=torch.randn(batch, 3072, device="cuda", dtype=dtype, generator=gen),
                weight=w,
                bias=b,
            )
            if final:
                inp = {
                    "x": torch.randn(batch, seq, 3072, device="cuda", dtype=dtype, generator=gen),
                    **inp,
                }
            shape = (batch, seq, 3072) if final else (batch, w.shape[0])
            upstream = torch.randn(shape, device="cuda", dtype=dtype, generator=gen)
            return inp, upstream

        if a.phase == "parameters":
            for seed in [0, 127, 2026]:
                for batch in [1, 255, 256, 257, 2048]:
                    inp, g = inputs(batch, seed=seed)
                    left = run(fn, inp, g, True)
                    right = run(fn, inp, g, True)
                    case["checks"].append(
                        dict(
                            kind="parameter_repeat",
                            batch=batch,
                            seed=seed,
                            all_equal={
                                k: equal_bytes(left[k], right[k]) for k in ("weight", "bias")
                            },
                        )
                    )
                inp, g = inputs(257, seed=seed)
                g[255:] = 0
                base = run(fn, sliced(inp, slice(0, 255)), g[:255], True)
                for batch in [256, 257]:
                    padded = run(fn, sliced(inp, slice(0, batch)), g[:batch], True)
                    case["checks"].append(
                        dict(
                            kind="ordered_contributions_zero_padding",
                            batch=batch,
                            seed=seed,
                            all_equal={
                                k: equal_bytes(base[k], padded[k]) for k in ("weight", "bias")
                            },
                        )
                    )
                save()
            print(a.mode, stream, dtype, "parameters complete", flush=True)
        elif a.phase in ("accuracy", "accuracy_large"):
            shapes = (
                [(256, 3 if final else 1), (2048, 3 if final else 1)]
                if a.phase == "accuracy_large"
                else (
                    [(1, 3), (3, 3), (1, 4096), (1, 6889), (1, 6032)] if final else [(1, 1), (3, 1)]
                )
            )
            for batch, seq in shapes:
                inp, g = inputs(batch, seq, a.seed)
                out = run(fn, inp, g, True)
                hi = {k: v.double().cpu().requires_grad_() for k, v in inp.items()}
                gold = (reference_final if final else reference_projection)(
                    **hi, execution_dtype=dtype
                )
                gg = torch.autograd.grad(gold, tuple(hi.values()), g.double().cpu())
                expected = {"output": gold.detach(), **dict(zip(hi, gg))}
                case["checks"].append(
                    dict(
                        kind="fp64_contract",
                        batch=batch,
                        sequence=seq,
                        errors={k: metric(v, expected[k]) for k, v in out.items()},
                        contract={
                            k: compare(
                                v,
                                expected[k],
                                dtype,
                                ("forward_accuracy" if k == "output" else "gradient_accuracy"),
                            )
                            for k, v in out.items()
                        },
                    )
                )
                low = {k: v.float().cpu().requires_grad_() for k, v in inp.items()}
                low_y = (reference_final if final else reference_projection)(
                    **low, execution_dtype=dtype
                )
                low_g = torch.autograd.grad(low_y, tuple(low.values()), g.float().cpu())
                low_expected = {"output": low_y.detach(), **dict(zip(low, low_g))}
                case["checks"][-1]["cpu_fp32_contract"] = {
                    k: compare(
                        v,
                        low_expected[k],
                        dtype,
                        "forward_accuracy" if k == "output" else "gradient_accuracy",
                    )
                    for k, v in out.items()
                }
                case["checks"][-1]["cpu_fp32_vs_fp64"] = {
                    k: compare(
                        v,
                        expected[k],
                        dtype,
                        "forward_accuracy" if k == "output" else "gradient_accuracy",
                    )
                    for k, v in low_expected.items()
                }
                del low, low_y, low_g, low_expected
                if a.capture_failures:
                    details = {}
                    for name, value in out.items():
                        check = case["checks"][-1]["contract"][name]
                        actual = value.detach().float().cpu()
                        expected_value = expected[name].detach().float().cpu()
                        limit = check["atol"] + check["rtol"] * expected_value.abs()
                        coords = ((actual - expected_value).abs() > limit).nonzero()
                        details[name] = [
                            dict(
                                index=row.tolist(),
                                actual=actual[tuple(row)].item(),
                                fp64_reference=expected[name].detach().cpu()[tuple(row)].item(),
                                comparison_reference=expected_value[tuple(row)].item(),
                                tolerance=limit[tuple(row)].item(),
                            )
                            for row in coords[:1000]
                        ]
                    case["checks"][-1]["failure_elements"] = details
                save()
            print(a.mode, stream, dtype, "accuracy complete", flush=True)
        elif a.phase == "reuse":
            inp, g = inputs(3)
            out = run(fn, inp, g, True)
            hi = {k: v.double().cpu().requires_grad_() for k, v in inp.items()}
            gold = (reference_final if final else reference_projection)(**hi, execution_dtype=dtype)
            gg = torch.autograd.grad(gold, tuple(hi.values()), g.double().cpu())
            expected = {"output": gold.detach(), **dict(zip(hi, gg))}
            case["fp64_relative_max_error"] = {k: metric(v, expected[k]) for k, v in out.items()}
            case["repeat_all_gradients"] = {
                k: equal_bytes(v, run(fn, inp, g, True)[k]) for k, v in out.items()
            }
            case["performance"] = []
            for batch, seq in (
                [(1, 33), (2, 4096), (1, 6889), (1, 6032), (256, 3), (2048, 3)]
                if final
                else [(1, 1), (3, 1), (256, 1), (2048, 1)]
            ):
                inp, g = inputs(batch, seq)
                inp = {k: v.detach().requires_grad_() for k, v in inp.items()}
                case["performance"].append(
                    dict(
                        batch=batch,
                        sequence=seq,
                        measurements=benchmark({a.mode: fn}, inp, 10),
                    )
                )
                save()
            print(a.mode, stream, dtype, "reuse complete", flush=True)
        elif a.phase == "workload":
            for seed in [0, 127, 2026]:
                inp, g = inputs(4096, seed=seed)
                whole = run(fn, inp, g)
                for chunk in [7, 64, 257, 1024, 2048]:
                    counts = {k: 0 for k in whole}
                    for start in range(0, 4096, chunk):
                        stop = min(start + chunk, 4096)
                        got = run(fn, sliced(inp, slice(start, stop)), g[start:stop])
                        for k, n in stats(
                            got, {k: v[start:stop] for k, v in whole.items()}
                        ).items():
                            counts[k] += n
                    case["checks"].append(
                        dict(
                            kind="workload_4096",
                            seed=seed,
                            batch=4096,
                            chunk=chunk,
                            rows=4096,
                            mismatched_rows=counts,
                        )
                    )
                save()
                print(a.mode, stream, dtype, "workload seed", seed, flush=True)
        else:
            sizes = sorted(
                set(range(1, 10))
                | {2**k + d for k in range(4, 12) for d in [-1, 0, 1] if 2**k + d <= 2048}
            )
            for seed in [0, 127, 2026]:
                inp, g = inputs(2048, seed=seed)
                # One singleton reference per row, reused across full-batch checks.
                singles = {}
                for i in range(2048):
                    one = run(fn, sliced(inp, slice(i, i + 1)), g[i : i + 1])
                    for k, v in one.items():
                        if k not in singles:
                            singles[k] = torch.empty(
                                (2048, *v.shape[1:]), device=v.device, dtype=v.dtype
                            )
                        singles[k][i : i + 1] = v
                for batch in [64, 257, 2048]:
                    got = run(fn, sliced(inp, slice(0, batch)), g[:batch])
                    case["checks"].append(
                        dict(
                            kind="every_row_singleton",
                            seed=seed,
                            batch=batch,
                            rows=batch,
                            mismatched_rows=stats(got, {k: v[:batch] for k, v in singles.items()}),
                        )
                    )
                # Every partition covers every row; compare per-sample gradients only.
                whole = run(fn, inp, g)
                for chunk in [1, 7, 64, 257, 1024]:
                    counts = {k: 0 for k in whole}
                    for start in range(0, 2048, chunk):
                        stop = min(start + chunk, 2048)
                        got = (
                            {k: v[start:stop] for k, v in singles.items()}
                            if chunk == 1
                            else run(fn, sliced(inp, slice(start, stop)), g[start:stop])
                        )
                        for k, n in stats(
                            got, {k: v[start:stop] for k, v in whole.items()}
                        ).items():
                            counts[k] += n
                    case["checks"].append(
                        dict(
                            kind="full_partition",
                            seed=seed,
                            batch=2048,
                            chunk=chunk,
                            rows=2048,
                            mismatched_rows=counts,
                        )
                    )
                # Front/middle/back target, then reverse entire batch; raw bytes.
                for batch in sizes:
                    for pos in sorted({0, batch // 2, batch - 1}):
                        idx = torch.arange(batch, device="cuda")
                        idx[0], idx[pos] = idx[pos].clone(), idx[0].clone()
                        got = run(fn, sliced(inp, idx), g[idx])
                        case["checks"].append(
                            dict(
                                kind="dense_position",
                                seed=seed,
                                batch=batch,
                                position=pos,
                                rows=1,
                                mismatched_rows=stats(
                                    {k: v[pos : pos + 1] for k, v in got.items()},
                                    {k: v[:1] for k, v in singles.items()},
                                ),
                            )
                        )
                    idx = torch.arange(batch - 1, -1, -1, device="cuda")
                    got = run(fn, sliced(inp, idx), g[idx])
                    case["checks"].append(
                        dict(
                            kind="reverse",
                            seed=seed,
                            batch=batch,
                            rows=batch,
                            mismatched_rows=stats(got, {k: v[idx] for k, v in singles.items()}),
                        )
                    )
                small, sg = inputs(9, seed=seed)
                first = run(fn, small, sg, True)
                again = run(fn, small, sg, True)
                case["checks"].append(
                    dict(
                        kind="ordered_repeat",
                        seed=seed,
                        all_equal={k: equal_bytes(v, again[k]) for k, v in first.items()},
                    )
                )
                save()
                print(a.mode, stream, dtype, "seed", seed, "BI complete", flush=True)
            if final:
                for seq in [31, 32, 33, 4096, 6889, 6032]:
                    inp, g = inputs(2, seq, 127)
                    whole = run(fn, inp, g)
                    singles = [
                        run(fn, sliced(inp, slice(i, i + 1)), g[i : i + 1]) for i in range(2)
                    ]
                    case["checks"].append(
                        dict(
                            kind="long_sequence",
                            sequence=seq,
                            rows=2,
                            mismatched_rows=stats(
                                whole,
                                {k: torch.cat([s[k] for s in singles]) for k in whole},
                            ),
                        )
                    )
        save()
